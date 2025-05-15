from typing import Optional
import os
import random
import time
from omegaconf import MISSING
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import matplotlib
import sys
import logging
import argparse
from pathlib import Path
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import copy
import numpy as np

try:
    import multiprocessing
    multiprocessing.set_start_method("fork")  # noqa
except Exception:
    pass

from pldm_envs.wall.wall import DotWall
from pldm.qmodel import PLDMModel

# ----------  NEW: helper to remove NaN / Inf & clip  ----------
def _sanitize_pos(pos: torch.Tensor, max_x: float, max_y: float):
    """
    1) 把 NaN 统一变 0；把 ±Inf 钳到边界；
    2) 最后再 clamp 到合法范围，确保坐标永远是有限数。
    """
    pos = torch.nan_to_num(pos, nan=0.0, posinf=max_x, neginf=0.0)
    pos[..., 0].clamp_(0.0, max_x)
    pos[..., 1].clamp_(0.0, max_y)
    return pos
# --------------------------------------------------------------

#---------restrict_action 函数：这个函数会检查小球的当前位置和墙壁的位置，并根据当前方向限制动作。
def restrict_action(action, env, direction='right'):
    # 限制小球不能超出左墙和右墙的边界
    if direction == 'right' and env.dot_position[0] > env.right_wall_x - 1.0:
        action = torch.tensor([-1.0, 0.0])  # 限制不能向右
    elif direction == 'left' and env.dot_position[0] < env.left_wall_x + 1.0:
        action = torch.tensor([1.0, 0.0])   # 限制不能向左
    return action

def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns efficiently using vectorised tensor ops.

    Args:
        rewards (Sequence[float] | torch.Tensor): reward at each time-step (length T).
        gamma (float): discount factor.

    Returns
    -------
    torch.Tensor
        Tensor of shape (T,) containing the discounted return *R_t = ∑_{k≥t} γ^{k-t} r_k*.
        The returned tensor lives on the same *device* and has the same *dtype*
        as the input (when the input is a tensor); otherwise it defaults to
        ``torch.float32`` on CPU.
    """
    # Convert to tensor while preserving dtype/device when possible
    if isinstance(rewards, torch.Tensor):
        r = rewards
    else:
        r = torch.as_tensor(rewards, dtype=torch.float32)

    T = r.shape[0]
    if T == 0:
        return r.new_empty(0)

    # discounts = [1, γ, γ^2, ... γ^{T-1}]
    discounts = gamma ** torch.arange(T, dtype=r.dtype, device=r.device)
    discounted_r = r * discounts  # element-wise

    # Reverse cumulative sum, then flip back and un-discount
    returns = torch.flip(torch.cumsum(torch.flip(discounted_r, dims=[0]), dim=0), dims=[0])
    returns = returns / discounts  # undo the earlier scaling

    return returns


def rollout(model, env, max_steps: int = 100, device: str = "cpu", num_samples: int = 100, use_quadrant: bool = True):
    """
    两阶段 shaping 奖励：
      A) 未穿门：      逼近门中心
      B) 已在同房间：  逼近目标 + 方向奖励
    外加：穿门一次性 + 命中一次性 + 步罚 + 反抖动 + 撞墙罚
    """
    # ---------- 常量超参 ----------
    BONUS_CROSS_DOOR  = 100.0  # 增大穿门奖励
    BONUS_HIT_TARGET  = 2000.0  # 增大命中奖励
    STEP_PENALTY      = 1.0   # 增大步数惩罚
    WALL_PENALTY      = 2.0    # 增大撞墙惩罚
    TINY_MOVE_THRESH  = 0.3
    TINY_MOVE_PENALTY = 1.0    # 增大小动作惩罚
    DIR_REWARD_LAMBDA = 2.0    # 增大方向奖励权重
    TARGET_SCALE      = 10.0   # 增大目标接近奖励

    # ---------- 0. 环境初始化 ----------
    obs, info = env.reset()

    # 门中心（Tensor，跟随 env.device）
    door_center = torch.stack([env.wall_x, env.hole_y]).to(env.device, dtype=torch.float32)

    states, actions, rewards = [obs], [], []
    log_probs, next_goals = [], []

    # 初始化潜势
    prev_dist_to_door    = torch.norm(env.dot_position - door_center).item()
    prev_in_left = env.dot_position[0] < env.left_wall_x   # 记录在哪边
    prev_dist_to_target  = torch.norm(env.dot_position - env.target_position).item()

    # ---------- 1. 编码初始观测 ----------
    model.eval()
    with torch.no_grad():
        obs_tensor = (obs if isinstance(obs, torch.Tensor)
                      else torch.tensor(obs, dtype=torch.float32,
                                         device=device)).unsqueeze(0)
        z_t = model.encode(obs_tensor)

    done = truncated = False
    crossed_door = False            # 标记是否已经拿过穿门奖励

    for step in range(max_steps):
        step_reward = 0
        if done or truncated:
            break

        # ---------- 2. 预测下一 latent-goal ----------
        with torch.no_grad():
            z_next, log_prob = model.predict_next_goal(z_t)
            next_goals.append(z_next.squeeze(0).cpu())
            log_probs.append(log_prob.item())

        # ---------- 3. 搜索动作 ----------
        a_t = model.search_action(
            z_t.detach(), z_next.detach(),
            num_samples=num_samples,
            use_quadrant=use_quadrant
        )
        action = a_t.cpu().numpy()[0]

        # ---------- 4. 与环境交互 ----------
        obs, _, done, truncated, info = env.step(action)

        # ---------- 5. 奖励计算 ----------
        # 房间判断
        dot_left     = env.dot_position[0] < env.left_wall_x
        target_left  = env.target_position[0] < env.left_wall_x
        same_room    = bool(dot_left == target_left)

        # ---- 阶段 A：未穿门 ----
        if not same_room:
            curr_dist_to_door = torch.norm(env.dot_position - door_center).item()
            step_reward = (prev_dist_to_door - curr_dist_to_door) * 2.0      # ×2 放大
            prev_dist_to_door = curr_dist_to_door

            # 检测"刚刚跨过墙"
            crossed = (prev_in_left != (env.dot_position[0] < env.left_wall_x))
            if crossed and not crossed_door:
                step_reward += BONUS_CROSS_DOOR
                crossed_door = True

        # ---- 阶段 B：已在同房间 ----
        else:
            curr_dist_to_target = torch.norm(env.dot_position - env.target_position).item()
            # 提前终止：如果距离目标非常近，给予大额奖励并终止 episode
            if curr_dist_to_target < 2.5:
                # print(f"[DEBUG] HIT TARGET at step {step}")
                step_reward += BONUS_HIT_TARGET
                done = True
            step_reward = TARGET_SCALE * (prev_dist_to_target - curr_dist_to_target)
            
            # 增加在正确房间的持续奖励
            step_reward += 20.0 * np.exp(-curr_dist_to_target / 10)
            
            # 方向奖励使用指数函数增强近距离精确性
            v_a = action / (np.linalg.norm(action) + 1e-6)
            v_t = (env.target_position.cpu().numpy() - env.dot_position.cpu().numpy())
            dist_to_target = np.linalg.norm(v_t)
            v_t = v_t / (dist_to_target + 1e-6)
            direction_reward = DIR_REWARD_LAMBDA * np.exp(-0.1 * dist_to_target) * np.dot(v_a, v_t)
            step_reward += direction_reward
            
            # 命中奖励使用平滑函数
            hit_reward = BONUS_HIT_TARGET * np.exp(-curr_dist_to_target)
            step_reward += hit_reward if curr_dist_to_target < 3.0 else 0
            
            prev_dist_to_target = curr_dist_to_target
            prev_in_left = env.dot_position[0] < env.left_wall_x

        # ---- 通用惩罚 ----
        step_reward -= STEP_PENALTY  # 每步都扣
        if np.linalg.norm(action) < TINY_MOVE_THRESH:
            step_reward -= TINY_MOVE_PENALTY
        if env.position_history and torch.all(env.position_history[-1] == env.dot_position):
            step_reward -= WALL_PENALTY

        # ---------- 存储 ----------
        states.append(obs)
        actions.append(action)
        rewards.append(step_reward)

        # ---------- 编码下一观测 ----------
        with torch.no_grad():
            obs_tensor = (obs if isinstance(obs, torch.Tensor)
                          else torch.tensor(obs, dtype=torch.float32,
                                             device=device)).unsqueeze(0)
            z_t = model.encode(obs_tensor)

    # ---------- 8. 清理 ----------
    model.train()
    return {
        "states":   states,
        "actions":  actions,
        "rewards":  rewards,
        "log_probs": log_probs,
        "next_goals": next_goals,
        "done":     done
    }

# -----------------------------------------------------------------------------
# Utility: log gradient/update statistics
# -----------------------------------------------------------------------------
def _log_grad_update_stats(model, optimizer, step_idx=0):
    """Print average relative parameter update (Δw/|w|) for policy vs value nets.

    • Parameters belonging to `next_goal_predictor.value_mlp.*` are grouped as 'value_net'.
    • All other `next_goal_predictor.*` parameters are grouped as 'policy_net'.
    Only the mean Δw/|w| is reported for each group.
    """

    # Map each parameter id to its learning-rate for fast lookup
    param_lr = {}
    for group in optimizer.param_groups:
        lr = group.get('lr', 0.0)
        for p in group['params']:
            param_lr[id(p)] = lr

    # Accumulators
    acc = {
        'encoder':    {'sum': 0.0, 'n': 0},
        'dynamics':   {'sum': 0.0, 'n': 0},
        'policy_net': {'sum': 0.0, 'n': 0},
        'value_net':  {'sum': 0.0, 'n': 0},
    }

    for name, p in model.named_parameters():
        if p.grad is None:
            continue

        if name.startswith('encoder'):
            key = 'encoder'
        elif name.startswith('dynamics'):
            key = 'dynamics'
        elif name.startswith('next_goal_predictor'):
            if name.startswith('next_goal_predictor.value_mlp'):
                key = 'value_net'
            else:
                key = 'policy_net'
        else:
            # skip parameters outside predictor
            continue

        lr = param_lr.get(id(p), 0.0)
        grad_mean = p.grad.abs().mean().item()
        param_mean = p.abs().mean().item()
        rel_upd = (lr * grad_mean) / (param_mean + 1e-8)  # Δw/|w|

        acc[key]['sum'] += rel_upd
        acc[key]['n'] += 1

    # Compose log line
    parts = []
    for k in ('encoder','dynamics','policy_net','value_net'):
        if acc[k]['n'] == 0:
            continue
        parts.append(f"{k}: Δw/|w|={acc[k]['sum']/acc[k]['n']:.2e}")

    if parts:
        tqdm.write(f"[GradStats step={step_idx}] " + " | ".join(parts))

# -----------------------------------------------------------------------------
# Utility: log gradient contributions to encoder from individual loss terms
# -----------------------------------------------------------------------------

def _log_encoder_grad_sources(params, loss_dict, step_idx=0, prefix="EncGradSources"):
    """Print mean |grad| on *params* attributed to each loss term.

    Args:
        params: list of parameters whose gradients we want to probe.
        loss_dict: dict mapping loss_name -> loss_tensor (already scaled).
        step_idx: global step index for logging purposes.
        prefix: string prefix for the printed message (identifies the param set).
    """
    stats = {}
    for name, loss in loss_dict.items():
        if loss is None or loss.requires_grad == False:
            continue
        # autograd.grad 只能接受标量；若不是标量先取均值
        if loss.dim() != 0:
            loss = loss.mean()
        # Compute grads w.r.t params WITHOUT accumulating into .grad
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        # Compute mean absolute gradient across all params
        abs_means = []
        for g in grads:
            if g is None:
                continue
            abs_means.append(g.abs().mean())
        if len(abs_means) == 0:
            mean_grad = 0.0
        else:
            mean_grad = torch.stack(abs_means).mean().item()
        stats[name] = mean_grad

    if stats:
        parts = [f"{k}: |g|={v:.2e}" for k, v in stats.items()]
        tqdm.write(f"[{prefix} step={step_idx}] " + " | ".join(parts))

def save_experiment_info(args, output_dir, epoch, best_reward):
    """Save experiment parameters and best reward to a text file
    
    Args:
        args: Command line arguments
        output_dir: Output directory path
        epoch: Current epoch
        best_reward: Best reward achieved so far
    """
    # Create the filename
    filename = output_dir / "experiment_info.txt"
    
    with open(filename, 'w') as f:
        # Current time and epoch info
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Epoch: {epoch+1}\n")
        f.write(f"Best Avg Reward: {best_reward:.6f}\n\n")
        
        # Save all arguments
        f.write("Command Line Arguments:\n")
        # Get all arguments as a dictionary
        arg_dict = vars(args)
        # Print them sorted by key for consistency
        for key in sorted(arg_dict.keys()):
            f.write(f"  --{key}: {arg_dict[key]}\n")

# -----------------------------------------------------------------
# Helper: duplicate stdout/stderr to a log file so that every print
#         during training is also saved under <output_dir>/train.log
# -----------------------------------------------------------------
class _Tee(object):
    def __init__(self, file_path):
        self.file = open(file_path, "a", buffering=1)  # line‑buffered
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

class ParallelEpisodeCollector:
    """Collect episodes in parallel for faster training"""
    
    def __init__(self, model, env_creator, max_steps, device, 
                 num_workers=4, prefetch_queue_size=8, use_gpu_for_inference=True, 
                 num_samples=100, use_quadrant=True):
        self.model = model
        self.env_creator = env_creator
        self.max_steps = max_steps
        self.device = device
        self.num_workers = num_workers
        self.use_gpu_for_inference = use_gpu_for_inference
        self.num_samples = num_samples
        self.use_quadrant = use_quadrant
        self.prefetch_queue_size = prefetch_queue_size
        
        # Create worker environments
        self.envs = [env_creator() for _ in range(num_workers)]
        
        # Create episode queue
        self.episode_queue = queue.Queue(maxsize=prefetch_queue_size)
        
        # Create thread pool
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        # Flag to stop workers
        self.stop_flag = threading.Event()
        
        # Lock for GPU access if using GPU for inference
        self.gpu_lock = threading.Lock() if use_gpu_for_inference else None
        
        # Start workers
        self.futures = []
        for worker_id in range(num_workers):
            future = self.executor.submit(self._worker_loop, worker_id)
            self.futures.append(future)
            
    def _worker_loop(self, worker_id):
        """Worker thread loop to collect episodes"""
        env = self.envs[worker_id]
        
        while not self.stop_flag.is_set():
            try:
                # Try to collect an episode
                if self.use_gpu_for_inference:
                    with self.gpu_lock:  # Use GPU with lock
                        trajectory = rollout(
                            self.model,
                            env,
                            max_steps=self.max_steps,
                            device=self.device,
                            num_samples=self.num_samples,
                            use_quadrant=self.use_quadrant
                        )
                else:
                    # Create CPU copy for inference
                    local_model = copy.deepcopy(self.model).to('cpu')
                    local_model.eval()
                    trajectory = rollout(
                        local_model,
                        env,
                        max_steps=self.max_steps,
                        device='cpu',
                        num_samples=self.num_samples,
                        use_quadrant=self.use_quadrant
                    )
                
                # Skip empty trajectories
                if len(trajectory['states']) <= 1 or len(trajectory['log_probs']) == 0:
                    continue
                    
                # Put episode in queue
                try:
                    self.episode_queue.put(trajectory, block=True, timeout=5.0)
                except queue.Full:
                    continue
                    
            except Exception as e:
                print(f"Error in worker {worker_id}: {str(e)}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
    
    def get_batch(self, batch_size):
        """Get a batch of episodes from the queue with improved timeout handling"""
        batch = []
        max_retries = 10
        retry_count = 0
        
        while len(batch) < batch_size and retry_count < max_retries:
            try:
                episode = self.episode_queue.get(block=True, timeout=5.0)
                self.episode_queue.task_done()
                batch.append(episode)
            except queue.Empty:
                print(f"Warning: Queue timeout, retry {retry_count + 1}/{max_retries}")
                retry_count += 1
                if retry_count >= max_retries:
                    print("Max retries reached, returning partial batch")
                    break
        return batch
    
    def stop(self):
        """Stop all worker threads"""
        self.stop_flag.set()
        for future in self.futures:
            future.cancel()
        self.executor.shutdown(wait=False)

def train_pldm(args):
    """Main training function for PLDM model"""
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    # Redirect console output to log file
    sys.stdout = _Tee(output_dir / "train.log")
    sys.stderr = sys.stdout  # duplicate stderr as well
    
    # Set up device
    device = torch.device(args.device)
    
    # Environment creation function for parallel workers
    def create_env():
        return DotWall(max_step_norm=args.max_step_norm, door_space=8)
    
    # Create initial environment for the main thread
    env = create_env()
    
    # Create model
    model = PLDMModel(
        img_size=env.img_size,
        in_channels=3,  # DotWall has 3 channels: dot, wall, target
        encoding_dim=args.encoding_dim,
        action_dim=2,  # DotWall has 2D actions
        hidden_dim=args.hidden_dim,
        encoder_embedding=args.encoder_embedding,
        encoder_type=args.encoder_type,
        temperature=args.temperature,
        next_goal_temp=args.next_goal_temp,
        search_mode=args.search_mode,
        max_step_norm=args.max_step_norm
    ).to(device)
    
    # Print model parameter counts
    model.print_parameter_count()
    
    # ---------------------------------------------------------------
    # Separate parameter groups: encoder, dynamics, policy-net, value-net, decoder
    # ---------------------------------------------------------------
    policy_params = [p for n, p in model.next_goal_predictor.named_parameters() if not n.startswith('value_mlp')]
    value_params = list(model.next_goal_predictor.value_mlp.parameters())

    optimizer = optim.AdamW([
        {'params': model.encoder.parameters(),           'lr': args.encoder_lr},
        {'params': model.dynamics.parameters(),          'lr': args.dynamics_lr},
        {'params': policy_params,                        'lr': args.policy_lr},
        {'params': value_params,                         'lr': args.value_lr},
        {'params': model.decoder.parameters(),           'lr': args.decoder_lr},
    ])
    
    # Create learning rate schedulers
    # We'll implement a custom scheduler function to manage the different learning rates
    def adjust_learning_rates(epoch):
        """Adjust learning rates based on epoch and component"""
        # Get base learning rates from parameter groups
        encoder_lr  = optimizer.param_groups[0]['lr']
        dynamics_lr = optimizer.param_groups[1]['lr']
        policy_lr   = optimizer.param_groups[2]['lr']
        value_lr    = optimizer.param_groups[3]['lr']
        decoder_lr  = optimizer.param_groups[4]['lr']
        
        # # Apply encoder LR schedule: reduce to 1/3 after the first epoch
        # if epoch == 1:
        #     encoder_lr = encoder_lr / 3
            
        # # Apply dynamics LR schedule: reduce to 1/2 after the first epoch
        # if epoch == 1:
        #     dynamics_lr = dynamics_lr / 2
        
        # Update optimizer parameter groups
        optimizer.param_groups[0]['lr'] = encoder_lr
        optimizer.param_groups[1]['lr'] = dynamics_lr
        optimizer.param_groups[2]['lr'] = policy_lr
        optimizer.param_groups[3]['lr'] = value_lr
        optimizer.param_groups[4]['lr'] = decoder_lr
        
        return encoder_lr, dynamics_lr, policy_lr, value_lr, decoder_lr
    
    # Check if resume from checkpoint
    start_epoch = 0
    global_step = 0
    best_reward = float('-inf')
    checkpoint_path = output_dir / 'checkpoint.pt'
    
    if args.resume and checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Check if checkpoint has new optimizer format
        if 'optimizer_state_dict' in checkpoint:
            # New single optimizer format
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
        start_epoch = checkpoint.get('epoch', 0) + 1
        global_step = checkpoint.get('global_step', 0)
        best_reward = checkpoint.get('best_reward', float('-inf'))
        
        print(f"Resuming from epoch {start_epoch} with best reward {best_reward:.4f}")
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=str(output_dir / 'logs'))
    epoch_rewards_history = []  # store avg reward for each epoch
    
    # Determine number of data loader workers based on CPU cores
    num_workers = min(args.num_workers if hasattr(args, 'num_workers') else 4, max(1, (multiprocessing.cpu_count() - 1) // 2))
    print(f"Using {num_workers} parallel workers for episode collection")
    print(f"Action sampling strategy: {'quadrant-based' if args.use_quadrant else 'full action space'}")

    # ---------------------------------------------------------------------
    # Sanity check: batch_size must divide (num_workers * max_steps_per_episode)
    # so that each round of environment collection produces an integral
    # number of training batches.
    # ---------------------------------------------------------------------
    total_steps_per_round = num_workers * args.max_steps_per_episode
    if total_steps_per_round % args.batch_size != 0:
        raise ValueError(
            f"batch_size ({args.batch_size}) must divide num_workers*max_steps_per_episode "
            f"({total_steps_per_round}). Please choose compatible values."
        )

    # Create parallel episode collector with explicit parameters
    collector = ParallelEpisodeCollector(
        model=model,
        env_creator=create_env,
        max_steps=args.max_steps_per_episode,
        device=device,
        num_workers=num_workers,
        prefetch_queue_size=8,  # 显式设置队列大小
        use_gpu_for_inference=args.use_gpu_inference,
        num_samples=args.num_samples,
        use_quadrant=args.use_quadrant
    )

    try:
        # Training loop
        transition_buffer = []  # holds (s, s_next, a, ng, R_t)
        for epoch in range(start_epoch, args.epochs):
            total_reward = 0
            total_policy_loss = 0
            total_dynamics_loss = 0
            total_on_the_same_page_loss = 0
            total_value_loss = 0
            num_episodes = 0

            # Adjust learning rates for this epoch
            encoder_lr, dynamics_lr, policy_lr, value_lr, decoder_lr = adjust_learning_rates(epoch)

            # Log current learning rates
            print(f"Epoch {epoch+1}/{args.epochs} - Learning rates: Encoder={encoder_lr:.2e}, "
                  f"Dynamics={dynamics_lr:.2e}, Policy={policy_lr:.2e}, Value={value_lr:.2e}, Decoder={decoder_lr:.2e}")
                  
            writer.add_scalar('LearningRate/encoder', encoder_lr, epoch)
            writer.add_scalar('LearningRate/dynamics', dynamics_lr, epoch)
            writer.add_scalar('LearningRate/policy', policy_lr, epoch)
            writer.add_scalar('LearningRate/value', value_lr, epoch)
            writer.add_scalar('LearningRate/decoder', decoder_lr, epoch)

            # ------------------------------------------------------------------
            # We now define an *update* as a gradient step computed from
            # `args.batch_size` **transitions** (time-steps), not episodes.
            # The number of updates per epoch is controlled by args.updates_per_epoch.
            # ------------------------------------------------------------------
            progress_bar = tqdm(total=args.updates_per_epoch, desc=f"Epoch {epoch+1}/{args.epochs}")
            update_idx = 0

            # Ensure model is in train mode on the main thread
            model.train()

            while update_idx < args.updates_per_epoch:
                # ------------------------------------------------------------
                # 1) Ensure buffer has >= batch_size transitions
                # ------------------------------------------------------------
                while len(transition_buffer) < args.batch_size:
                    episodes = collector.get_batch(1)
                    if len(episodes) == 0:
                        time.sleep(0.05)
                        continue

                    traj = episodes[0]
                    states = traj['states']
                    actions = traj['actions']
                    rewards = traj['rewards']
                    next_goals = traj['next_goals']

                    T = len(actions)

                    # stats
                    episode_reward = sum(rewards)
                    total_reward += episode_reward
                    num_episodes += 1

                    # returns
                    returns = compute_returns(rewards, gamma=args.gamma)

                    for t in range(T):
                        s_t = torch.tensor(states[t], device=device).float() if not isinstance(states[t], torch.Tensor) else states[t].to(device).float()
                        s_next = torch.tensor(states[t+1], device=device).float() if not isinstance(states[t+1], torch.Tensor) else states[t+1].to(device).float()
                        a_t = torch.tensor(actions[t], device=device).float() if not isinstance(actions[t], torch.Tensor) else actions[t].to(device).float()
                        ng_t = torch.tensor(next_goals[t], device=device).float() if not isinstance(next_goals[t], torch.Tensor) else next_goals[t].to(device).float()
                        transition_buffer.append((s_t, s_next, a_t, ng_t, returns[t].item()))

                # ------------------------------------------------------------
                # 2) Sample batch_size transitions *without replacement*
                # ------------------------------------------------------------
                import random as _rnd
                indices = _rnd.sample(range(len(transition_buffer)), args.batch_size)
                indices.sort(reverse=True)  # sort descending for safe pop

                batch_states, batch_next_states, batch_actions, batch_next_goals, batch_returns = [], [], [], [], []
                for idx in indices:
                    s_t, s_next, a_t, ng_t, R_t = transition_buffer.pop(idx)
                    batch_states.append(s_t)
                    batch_next_states.append(s_next)
                    batch_actions.append(a_t)
                    batch_next_goals.append(ng_t)
                    batch_returns.append(R_t)

                # ------------------------------------------------------------
                # 3) Compute losses on the sampled batch
                # ------------------------------------------------------------
                # ------------------------------------------------------------------
                # (NEW) Return normalisation for a more stable value-function target.
                # This rescales *R_t* to zero-mean, unit-std on every minibatch which
                # reduces the dynamic range of the value-head gradients and makes
                # NaN explosions far less likely.
                # ------------------------------------------------------------------
                if args.use_immediate_reward:
                    # Use immediate rewards for advantage calculation and value network training
                    batch_returns_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
                else:
                    # Use discounted returns
                    batch_returns_tensor = torch.tensor(batch_returns, dtype=torch.float32, device=device)

                if args.normalize_returns_and_advantage:
                    mean_R = batch_returns_tensor.mean()
                    std_R  = batch_returns_tensor.std(unbiased=False).clamp(min=1e-4)
                    batch_returns_tensor = (batch_returns_tensor - mean_R) / std_R
                batch_returns_tensor = torch.nan_to_num(
                    batch_returns_tensor,
                    nan=0.0, posinf=0.0, neginf=0.0
                )

                # --------------------------------------------------------
                # Vectorised computation over the whole batch
                # --------------------------------------------------------

                # Stack tensors
                S_t = torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32, device=device)
                                  for s in batch_states]).float().detach()
                S_next = torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32, device=device)
                                     for s in batch_next_states]).float().detach()
                A_t = torch.stack(batch_actions).to(device).float().detach()
                NG_store = torch.stack(batch_next_goals).to(device).float().detach()

                # Encode in one shot
                Z_t = model.encode(S_t)

                if global_step % args.log_steps == 0 and args.heatmap:
                    # save a heatmap of Z_t, make the heat map larger
                    plt.figure(figsize=(10, 10))
                    plt.imshow(Z_t.cpu().detach().numpy())
                    plt.savefig(f"{args.output_dir}/heatmap_{global_step}.png")
                    plt.close()

                Z_next_actual = model.encode(S_next)

                # Policy log-prob (no grad through stored goal)
                log_probs = model.next_goal_predictor.log_prob(Z_t, NG_store)  # [B]

                # Value prediction
                V_pred = model.next_goal_predictor.value(Z_t) #grad flow from value head
                V_pred = torch.clamp(V_pred, -20.0, 20.0)        # 限定数值范围
                V_pred = torch.nan_to_num(
                    V_pred,
                    nan=0.0, posinf=20.0, neginf=-20.0
                )
                if not torch.isfinite(V_pred).all():
                    print(f"[WARN step={global_step}] NaN/Inf detected in V_pred — auto-fix to 0")
                    V_pred = torch.nan_to_num(V_pred, nan=0.0, posinf=0.0, neginf=0.0)
                    value_loss = F.smooth_l1_loss(V_pred_safe, batch_returns_tensor)

                # Dynamics prediction
                if args.mode == 'RL':
                    # print("RL mode")
                    Z_next_pred = model.dynamics(Z_t.detach(),A_t) # Control the grad flow of the world model
                else:
                    # print("JEPA mode")
                    Z_next_pred = model.dynamics(Z_t,A_t) 
                # ---------------- losses ----------------
                # KL divergence between predicted and target probability distributions
                eps = 1e-8
                if args.mode == 'RL':
                    dynamics_loss = F.mse_loss(Z_next_pred, Z_next_actual.detach())
                else:
                    dynamics_loss = F.mse_loss(Z_next_pred, Z_next_actual)  #grad flow from the world model

                # ------------------------------------------------------------------
                #  (UPDATED) Diagnostics: average L2-norm (vector magnitude) of latent tensors
                # ------------------------------------------------------------------
                avg_mag_z_t         = Z_t.norm(dim=-1).mean().item()
                avg_mag_z_next_pred = Z_next_pred.norm(dim=-1).mean().item()
                avg_mag_ng_store    = NG_store.norm(dim=-1).mean().item()

                if global_step % args.log_steps == 0:
                    tqdm.write(
                         f"[LatentStats step={global_step}] ||z_t||={avg_mag_z_t:.3f} ||z_next_pred||={avg_mag_z_next_pred:.3f} ||ng_store||={avg_mag_ng_store:.3f}"
                    )

                    # Count distinct discrete codes in this batch
                    z_codes = Z_t.argmax(dim=1)
                    ng_codes = NG_store.argmax(dim=1)
                    num_z_codes = int(torch.unique(z_codes).numel())
                    num_ng_codes = int(torch.unique(ng_codes).numel())
                    tqdm.write(f"[CodeStats step={global_step}] encoder_codes={num_z_codes} | next_goal_codes={num_ng_codes}")

                if args.use_same_page_loss:
                    on_the_same_page_loss = torch.tensor(0.0, device=device)
                    print("on_the_same_page_loss is not implemented")
                else:
                    on_the_same_page_loss = torch.tensor(0.0, device=device)

                # Advantage / policy loss – use the (possibly) normalised returns
                advantage = (batch_returns_tensor - V_pred).detach()
                if args.normalize_returns_and_advantage:
                    norm_advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + eps)
                else:
                    norm_advantage = advantage

                policy_loss = -(log_probs * norm_advantage).mean()

                # Value loss
                value_loss = F.smooth_l1_loss(V_pred, batch_returns_tensor)
                    # ---- 新增 4 行 ----
                if torch.isnan(value_loss):
                    print(f"[WARN step={global_step}] NaN in value_loss — 重新计算")
                     # 强制把 V_pred 再次钳位后重算
                    V_pred_safe = torch.clamp(torch.nan_to_num(V_pred), -20.0, 20.0)
                    value_loss = F.smooth_l1_loss(V_pred_safe, batch_returns_tensor)

                if args.lambda_entropy > 0:
                    # Entropy bonus for exploration, compute manually for numerical stability
                    dist = model.next_goal_predictor.get_numerical_stable_distribution(Z_t)
                    entropy = - (dist * (dist + eps).log()).sum(dim=-1).mean()
                    writer.add_scalar('Stats/entropy', entropy.item(), global_step)
                else:
                    entropy = torch.tensor(0.0, device=device)

                # Total loss including entropy bonus
                loss = (
                    args.lambda_policy * policy_loss
                    + args.lambda_dynamics * dynamics_loss
                    + args.lambda_value * value_loss
                    + args.lambda_same_page * on_the_same_page_loss
                    - args.lambda_entropy * entropy
                )

                # Log encoder grad sources
                encoder_params_list = [p for p in model.encoder.parameters() if p.requires_grad]
                loss_contributions = {
                    'policy': args.lambda_policy * policy_loss,
                    'dynamics': args.lambda_dynamics * dynamics_loss,
                    'entropy': - args.lambda_entropy * entropy,
                }
                # Value loss contribution (optional)
                if args.use_value_loss and args.lambda_value > 0:
                    loss_contributions['value'] = args.lambda_value * value_loss
                if args.use_same_page_loss:
                    loss_contributions['same_page'] = args.lambda_same_page * on_the_same_page_loss

                if global_step % args.log_steps == 0:
                    # Encoder gradients
                    _log_encoder_grad_sources(encoder_params_list, loss_contributions, global_step, prefix="EncGradSources")

                    # Policy network gradients (all params excluding value_mlp)
                    policy_params_list = [p for n,p in model.next_goal_predictor.named_parameters() if not n.startswith('value_mlp')]
                    _log_encoder_grad_sources(policy_params_list, loss_contributions, global_step, prefix="PolicyGradSources")

                    # Dynamics model gradients
                    dynamics_params_list = list(model.dynamics.parameters())
                    _log_encoder_grad_sources(dynamics_params_list, loss_contributions, global_step, prefix="DynamicsGradSources")

                # Optimizer step
                loss.backward()
                # First, clip gradients on the value head a bit harder — it is
                # the most common source of exploding weights.
                torch.nn.utils.clip_grad_norm_(model.next_goal_predictor.value_mlp.parameters(), max_norm=0.1)

                # Then clip the entire model to a reasonable global norm.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if global_step % args.log_steps == 0:
                    _log_grad_update_stats(model, optimizer, global_step)
                optimizer.zero_grad()

                # Logging
                writer.add_scalar('Loss/policy', policy_loss.item(), global_step)
                writer.add_scalar('Loss/dynamics', dynamics_loss.item(), global_step)
                if args.use_same_page_loss:
                    writer.add_scalar('Loss/on_the_same_page', on_the_same_page_loss.item(), global_step)
                if args.use_value_loss:
                    writer.add_scalar('Loss/value', value_loss.item(), global_step)
                writer.add_scalar('Reward/individual_episode', batch_returns[0], global_step)
                writer.add_scalar('Stats/mag_z_t', avg_mag_z_t, global_step)
                writer.add_scalar('Stats/mag_z_next_pred', avg_mag_z_next_pred, global_step)
                writer.add_scalar('Stats/mag_ng_store', avg_mag_ng_store, global_step)

                # ---------------- aggregate epoch-level stats -----------------
                total_policy_loss += policy_loss.item()
                total_dynamics_loss += dynamics_loss.item()
                if args.use_same_page_loss:
                    total_on_the_same_page_loss += on_the_same_page_loss.item()
                if args.use_value_loss:
                    total_value_loss += value_loss.item()

                # ----------------------------------------------------------------
                global_step += 1
                update_idx += 1
                progress_bar.update(1)

            progress_bar.close()
                    
            # Epoch statistics
            if num_episodes > 0:
                avg_reward = total_reward / num_episodes
                avg_policy_loss = total_policy_loss / num_episodes
                avg_dynamics_loss = total_dynamics_loss / num_episodes
                
                # Build performance report
                report = f"Epoch {epoch+1}/{args.epochs} - " \
                         f"Avg Reward: {avg_reward:.4f}, " \
                         f"Avg Policy Loss: {avg_policy_loss:.4f}, " \
                         f"Avg Dynamics Loss: {avg_dynamics_loss:.4f}"
                
                # Add conditional metrics to report
                if args.use_same_page_loss:
                    avg_on_the_same_page_loss = total_on_the_same_page_loss / num_episodes
                    report += f", Avg On-Same-Page Loss: {avg_on_the_same_page_loss:.4f}"
                
                if args.use_value_loss:
                    avg_value_loss = total_value_loss / num_episodes
                    report += f", Avg Value Loss: {avg_value_loss:.4f}"
                
                print(report)
                epoch_rewards_history.append(avg_reward)
                
                # Save best model
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    torch.save(model.state_dict(), output_dir / 'best_model.pt')
            else:
                print(f"Epoch {epoch+1}/{args.epochs} - No successful episodes.")
            
            # Save checkpoint
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_reward': best_reward,
                'global_step': global_step
            }
                
            torch.save(checkpoint_dict, output_dir / 'checkpoint.pt')
            
            # Save experiment information to text file
            save_experiment_info(args, output_dir, epoch, best_reward)
            
            # Save Avg‑Reward vs Epoch curve after each epoch
            if epoch_rewards_history:
                plt.figure(figsize=(6,4))
                plt.plot(epoch_rewards_history, marker='o')
                plt.title("Average Reward vs Epoch")
                plt.xlabel("Epoch")
                plt.ylabel("Avg Reward")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(output_dir / f"avg_reward_curve.png")
                plt.close()

            # Save epoch_rewards_history to a file after each epoch
            with open(output_dir / "epoch_rewards_history.txt", "w") as f:
                for reward in epoch_rewards_history:
                    f.write(f"{reward}\n")
        
        # -------------------------------------------------------------
        # Plot Avg‑Reward vs Epoch curve and save as PNG
        # -------------------------------------------------------------
        if epoch_rewards_history:
            plt.figure(figsize=(6,4))
            plt.plot(epoch_rewards_history, marker='o')
            plt.title("Average Reward vs Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Avg Reward")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_dir / "avg_reward_curve_final.png")
            plt.close()
        writer.close()
        
    finally:
        # Make sure to stop the collector
        collector.stop()
    
    return model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train PLDM model on DotWall environment')
    
    # Model parameters
    parser.add_argument('--encoding_dim', type=int, default=256, help='Dimension of encoded state (default 512 for discrete codes)')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Dimension of hidden layers')
    parser.add_argument('--encoder_embedding', type=int, default=192, help='Dimension of encoder embedding')
    parser.add_argument('--encoder_type', type=str, default='vit', choices=['vit','cnn'], help='Encoder architecture: vit or cnn')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--updates_per_epoch', type=int, default=32, help='Number of training updates (batches of transitions) per epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of trajectories to process in a batch')
    parser.add_argument('--max_steps_per_episode', type=int, default=200, help='Maximum steps per episode')
    parser.add_argument('--num_samples', type=int, default=8, help='Number of action samples to evaluate in parallel')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--max_step_norm', type=float, default=12, help='Maximum step norm')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of parallel workers for episode collection')
    parser.add_argument('--use_gpu_inference', action='store_true', default=True, help='Use GPU for inference during rollout')
    parser.add_argument('--log_steps', type=int, default=9999999999, help='Logging frequency for gradient statistics')
    parser.add_argument('--heatmap', action='store_false', default=False, help='Save a heatmap of Z_t')
    parser.add_argument('--use_quadrant', action='store_true', default=True, help='Use quadrant-based action sampling (True) or full action space sampling (False)')

    parser.add_argument('--use_same_page_loss', action='store_false', default=False, help='Use on-the-same-page loss between next goal and dynamics')
    parser.add_argument('--use_decoder_loss', action='store_false', default=False, help='Enable decoder reconstruction warm-up loss')
    parser.add_argument('--use_value_loss', action='store_true', default=True, help='Train value head with MSE to returns')
    parser.add_argument('--normalize_returns_and_advantage', action='store_true', default=True, help='Normalize returns and advantage to zero-mean, unit-std')
    
    parser.add_argument('--lambda_dynamics', type=float, default=100.0, help='Weight for dynamics loss')
    parser.add_argument('--lambda_policy', type=float, default=1e-2, help='Weight for policy loss')
    parser.add_argument('--lambda_value', type=float, default=1e-3, help='Weight for value loss')
    parser.add_argument('--lambda_same_page', type=float, default=0.0, help='Weight for on-the-same-page loss')
    parser.add_argument('--lambda_entropy', type=float, default=0.1, help='Weight for policy entropy bonus') # can't be larger than 1e-3 for numerical stability

    parser.add_argument('--encoder_lr', type=float, default=3e-6, help='Learning rate for encoder')
    parser.add_argument('--dynamics_lr', type=float, default=1e-5, help='Learning rate for dynamics model')
    parser.add_argument('--policy_lr', type=float, default=3e-5, help='Learning rate for policy')
    parser.add_argument('--value_lr', type=float, default=3e-5, help='Learning rate for value')
    parser.add_argument('--decoder_lr', type=float, default=1e-1, help='Learning rate for decoder')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run training on')
    parser.add_argument('--output_dir', type=str, default='output_same_page_value5143', help='Directory to save model and logs')
    parser.add_argument('--resume', action='store_false', default=False, help='Resume training from checkpoint')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for discrete softmax')
    parser.add_argument('--next_goal_temp', type=float, default=1.0, help='Temperature for next-goal predictor; if not set, uses --temperature')
    #parser.add_argument('--base_reward', type=float, default=1.0, help='Base reward for each step')
    parser.add_argument('--search_mode', type=str, default='rl', choices=['pldm','rl'], help='Action search mode: pldm or rl')
    parser.add_argument('--mode', type=str, default='JEPA', choices=['RL','JEPA'], help='block the grad flow from JEPA if mode is RL')

    # Add a new argument to choose between immediate rewards and discounted returns
    parser.add_argument('--use_immediate_reward', action='store_true', default=False, help='Use immediate reward for advantage calculation and value network training')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.autograd.set_detect_anomaly(True)     # catches backward NaNs
    train_pldm(args)
