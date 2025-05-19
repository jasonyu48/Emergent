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
from pldm.qmodel import PLDMModel, DEFAULT_ENCODING_DIM, DEFAULT_NUM_ACTIONS

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

# -----------------------------------------------------------------------------
#  Representation probing utilities
# -----------------------------------------------------------------------------

def evaluate_representation(model, env_creator, device,
                             num_samples: int = 1024,
                             train_steps: int = 200,
                             batch_size: int = 128):
    """Evaluate how linearly the latent space represents positional information.

    Returns a dict with keys 'decode_mse' and 'encode_mse'. Lower is better.
    """
    was_training = model.training
    model.eval()

    env = env_creator()
    z_list, p_list = [], []
    with torch.no_grad():
        for _ in range(num_samples):
            obs, info = env.reset()
            pos_vec = torch.stack([
                info['dot_position'][0], info['dot_position'][1],
                info['target_position'][0], info['target_position'][1]
            ]).to(device, dtype=torch.float32)
            obs_t = (obs if isinstance(obs, torch.Tensor) else torch.tensor(obs, dtype=torch.float32)).unsqueeze(0).to(device)
            z = model.encode(obs_t).squeeze(0).detach()
            z_list.append(z)
            p_list.append(pos_vec)
    Z = torch.stack(z_list)  # [N,D]
    P = torch.stack(p_list)  # [N,4]

    N = Z.size(0)
    idx = torch.randperm(N, device=device)
    split = int(0.8 * N)
    tr, val = idx[:split], idx[split:]
    Z_tr, Z_val = Z[tr], Z[val]
    P_tr, P_val = P[tr], P[val]

    mse = torch.nn.MSELoss()

    # Decode probe (z -> pos)
    lin_dec = torch.nn.Linear(Z.size(1), 4, bias=True).to(device)
    opt_dec = torch.optim.SGD(lin_dec.parameters(), lr=5e-3, momentum=0.9)
    for _ in range(train_steps):
        b = torch.randint(0, split, (batch_size,), device=device)
        loss = mse(lin_dec(Z_tr[b]), P_tr[b])
        opt_dec.zero_grad(); loss.backward(); opt_dec.step()
    with torch.no_grad():
        decode_mse = mse(lin_dec(Z_val), P_val).item()

    # Encode probe (pos -> z)
    lin_enc = torch.nn.Linear(4, Z.size(1), bias=True).to(device)
    opt_enc = torch.optim.SGD(lin_enc.parameters(), lr=5e-3, momentum=0.9)
    for _ in range(train_steps):
        b = torch.randint(0, split, (batch_size,), device=device)
        loss = mse(lin_enc(P_tr[b]), Z_tr[b])
        opt_enc.zero_grad(); loss.backward(); opt_enc.step()
    with torch.no_grad():
        encode_mse = mse(lin_enc(P_val), Z_val).item()

    if was_training:
        model.train()
    return {'decode_mse': decode_mse, 'encode_mse': encode_mse}

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

    # Store observations, continuous actions, rewards, log_probs of sampled action_idx, and one-hot action_idx
    states, continuous_actions, rewards = [obs], [], []
    log_probs_list, sampled_action_idx_one_hot_list = [], []

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

        # ---------- 2. Get action from policy ----------
        with torch.no_grad():
            # model.get_action_and_log_prob returns: continuous_action, log_prob_of_idx, action_idx
            continuous_action_tensor, log_prob, action_idx = model.get_action_and_log_prob(z_t)
            
            # Get num_actions for one-hot encoding from the model
            # It's safer to get it directly if possible, or pass as arg if PLDMModel stores it directly
            num_actions_for_one_hot = model.num_actions
            action_idx_one_hot = F.one_hot(action_idx, num_classes=num_actions_for_one_hot).float().squeeze(0) # Remove batch dim

            sampled_action_idx_one_hot_list.append(action_idx_one_hot.cpu())
            log_probs_list.append(log_prob.item())
        
        # Convert continuous_action_tensor to numpy for environment step
        action_to_step = continuous_action_tensor.cpu().numpy()[0] # Remove batch dim

        # ---------- 4. 与环境交互 ----------
        obs, _, done, truncated, info = env.step(action_to_step)

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
            v_a = action_to_step / (np.linalg.norm(action_to_step) + 1e-6)
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
        if np.linalg.norm(action_to_step) < TINY_MOVE_THRESH:
            step_reward -= TINY_MOVE_PENALTY
        if env.position_history and torch.all(env.position_history[-1] == env.dot_position):
            step_reward -= WALL_PENALTY

        # ---------- 存储 ----------
        states.append(obs)
        continuous_actions.append(action_to_step) # Store the continuous action taken
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
        "actions":  continuous_actions, # Now storing continuous actions
        "rewards":  rewards,
        "log_probs": log_probs_list, # Log probs of a_idx
        "sampled_actions_one_hot": sampled_action_idx_one_hot_list, # Store one-hot action_idx
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
        elif name.startswith('dynamics_model'):
            key = 'dynamics'
        elif name.startswith('policy_value_network'):
            if name.startswith('policy_value_network.value_mlp'):
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
                 num_workers=4, prefetch_queue_size=8, use_gpu_for_inference=True):
        self.model = model
        self.env_creator = env_creator
        self.max_steps = max_steps
        self.device = device
        self.num_workers = num_workers
        self.use_gpu_for_inference = use_gpu_for_inference
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
                            device=self.device
                        )
                else:
                    # Create CPU copy for inference
                    local_model = copy.deepcopy(self.model).to('cpu')
                    local_model.eval()
                    trajectory = rollout(
                        local_model,
                        env,
                        max_steps=self.max_steps,
                        device='cpu'
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
        return DotWall(
            max_step_norm=args.max_step_norm, 
            door_space=8,
            obs_noise_std=args.obs_noise_std # Pass noise std to env
        )
    
    # Create initial environment for the main thread
    env = create_env()
    
    # Create model
    model = PLDMModel(
        img_size=env.img_size,
        in_channels=3,  # DotWall has 3 channels: dot, wall, target
        encoding_dim=args.model_encoding_dim,
        num_actions=args.num_actions,
        action_dim_continuous=2, # DotWall has 2D continuous actions
        encoder_embedding_dim=args.encoder_embedding_dim,
        encoder_type=args.encoder_type,
        policy_temp=args.policy_temperature,
        max_step_norm=args.max_step_norm
    ).to(device)
    
    # Print model parameter counts
    model.print_parameter_count()
    
    # ---------------------------------------------------------------
    # Separate parameter groups: encoder, dynamics, policy-net, value-net, decoder
    # ---------------------------------------------------------------
    policy_params = [p for n, p in model.policy_value_network.named_parameters() if not n.startswith('value_mlp')]
    value_params = list(model.policy_value_network.value_mlp.parameters())

    optimizer = optim.AdamW([
        {'params': model.encoder.parameters(),           'lr': args.encoder_lr},
        {'params': model.dynamics_model.parameters(),      'lr': args.dynamics_lr},
        {'params': policy_params,                        'lr': args.policy_lr},
        {'params': value_params,                         'lr': args.value_lr},
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
        # decoder_lr  = optimizer.param_groups[4]['lr']
        
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
        # optimizer.param_groups[4]['lr'] = decoder_lr
        
        return encoder_lr, dynamics_lr, policy_lr, value_lr #, decoder_lr
    
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
    probe_steps_history = []    # store global_step for probe plot
    decode_mse_history = []   # store decode_mse for probe plot
    encode_mse_history = []   # store encode_mse for probe plot
    
    # Determine number of data loader workers based on CPU cores
    num_workers = min(args.num_workers if hasattr(args, 'num_workers') else 4, max(1, (multiprocessing.cpu_count() - 1) // 2))
    print(f"Using {num_workers} parallel workers for episode collection")

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
        use_gpu_for_inference=args.use_gpu_inference
    )

    try:
        # Training loop
        transition_buffer = []  # holds (s_t, s_next, a_continuous_t, a_one_hot_idx_t, R_t)
        for epoch in range(start_epoch, args.epochs):
            total_reward = 0
            total_policy_loss = 0
            total_dynamics_loss = 0
            total_on_the_same_page_loss = 0
            total_value_loss = 0
            num_episodes = 0

            # Adjust learning rates for this epoch
            encoder_lr, dynamics_lr, policy_lr, value_lr = adjust_learning_rates(epoch)

            # Log current learning rates
            # print(f"Epoch {epoch+1}/{args.epochs} - Learning rates: Encoder={encoder_lr:.2e}, Dynamics={dynamics_lr:.2e}, Policy={policy_lr:.2e}, Value={value_lr:.2e}")
                  
            writer.add_scalar('LearningRate/encoder', encoder_lr, epoch)
            writer.add_scalar('LearningRate/dynamics', dynamics_lr, epoch)
            writer.add_scalar('LearningRate/policy', policy_lr, epoch)
            writer.add_scalar('LearningRate/value', value_lr, epoch)
            # writer.add_scalar('LearningRate/decoder', decoder_lr, epoch)

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
                    states_rollout = traj['states']
                    continuous_actions_rollout = traj['actions']
                    rewards_rollout = traj['rewards']
                    # NG_store will be populated by sampled_actions_one_hot
                    sampled_actions_one_hot_rollout = traj['sampled_actions_one_hot'] 

                    T = len(continuous_actions_rollout)

                    # stats
                    episode_reward = sum(rewards_rollout)
                    total_reward += episode_reward
                    num_episodes += 1

                    # returns
                    returns = compute_returns(rewards_rollout, gamma=args.gamma)

                    for t in range(T):
                        s_t = torch.tensor(states_rollout[t], device=device).float() if not isinstance(states_rollout[t], torch.Tensor) else states_rollout[t].to(device).float()
                        s_next = torch.tensor(states_rollout[t+1], device=device).float() if not isinstance(states_rollout[t+1], torch.Tensor) else states_rollout[t+1].to(device).float()
                        # Store continuous action for dynamics model, and one-hot action for policy loss
                        a_continuous_t = torch.tensor(continuous_actions_rollout[t], device=device).float() if not isinstance(continuous_actions_rollout[t], torch.Tensor) else continuous_actions_rollout[t].to(device).float()
                        a_one_hot_idx_t = torch.tensor(sampled_actions_one_hot_rollout[t], device=device).float() if not isinstance(sampled_actions_one_hot_rollout[t], torch.Tensor) else sampled_actions_one_hot_rollout[t].to(device).float()
                        transition_buffer.append((s_t, s_next, a_continuous_t, a_one_hot_idx_t, returns[t].item()))

                # ------------------------------------------------------------
                # 2) Sample batch_size transitions *without replacement*
                # ------------------------------------------------------------
                import random as _rnd
                indices = _rnd.sample(range(len(transition_buffer)), args.batch_size)
                indices.sort(reverse=True)  # sort descending for safe pop

                batch_states, batch_next_states, batch_continuous_actions, batch_action_one_hot, batch_returns = [], [], [], [], []
                for idx in indices:
                    s_t, s_next, a_continuous_t, a_one_hot_idx_t, R_t = transition_buffer.pop(idx)
                    batch_states.append(s_t)
                    batch_next_states.append(s_next)
                    batch_continuous_actions.append(a_continuous_t)
                    batch_action_one_hot.append(a_one_hot_idx_t)
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
                    batch_returns_tensor = torch.tensor(rewards_rollout, dtype=torch.float32, device=device)
                else:
                    # Use discounted returns
                    batch_returns_tensor = torch.tensor(batch_returns, dtype=torch.float32, device=device)

                if args.normalize_returns_and_advantage:
                    mean_R = batch_returns_tensor.mean()
                    std_R  = batch_returns_tensor.std(unbiased=False).clamp(min=1e-4)
                    batch_returns_tensor = (batch_returns_tensor - mean_R) / std_R
                batch_returns_tensor = torch.nan_to_num(
                    batch_returns_tensor,
                    nan=0.0, posinf=10000.0, neginf=-10000.0
                )

                # --------------------------------------------------------
                # Vectorised computation over the whole batch
                # --------------------------------------------------------

                # Stack tensors
                S_t = torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32, device=device)
                                  for s in batch_states]).float().detach()
                S_next = torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32, device=device)
                                     for s in batch_next_states]).float().detach()
                # A_t is now continuous actions for dynamics model
                A_continuous_t = torch.stack(batch_continuous_actions).to(device).float().detach()
                # Action_OneHot_Batch is for policy loss (previously NG_store)
                Action_OneHot_Batch = torch.stack(batch_action_one_hot).to(device).float().detach()

                # Encode in one shot
                Z_t = model.encode(S_t)

                if global_step % args.log_steps == 0 and args.heatmap:
                    # save a heatmap of Z_t, make the heat map larger
                    plt.figure(figsize=(10, 10))
                    plt.imshow(Z_t.cpu().detach().numpy())
                    plt.savefig(f"{args.output_dir}/heatmap_Z_{global_step}.png")
                    plt.close()

                Z_next_actual = model.encode(S_next)

                # Policy log-prob (using stored one-hot action)
                log_probs = model.get_log_prob_of_action(Z_t, Action_OneHot_Batch)  # [B]

                # Value prediction
                V_pred = model.get_value_prediction(Z_t) #grad flow from value head
                # V_pred = torch.clamp(V_pred, -20.0, 20.0)        # 限定数值范围
                V_pred = torch.nan_to_num(
                    V_pred,
                    nan=0.0, posinf=10000.0, neginf=-10000.0
                )

                # Dynamics prediction (uses continuous actions)
                if args.mode == 'RL':
                    # print("RL mode")
                    Z_next_pred = model.predict_next_latent(Z_t.detach(), A_continuous_t) # Control the grad flow of the world model
                else:
                    # print("JEPA mode")
                    Z_next_pred = model.predict_next_latent(Z_t, A_continuous_t)
                # ---------------- losses ----------------
                # KL divergence between predicted and target probability distributions
                eps = 1e-8
                if args.mode == 'RL':
                    if args.loss_type == 1: # L1 loss
                        dynamics_loss = F.l1_loss(Z_next_pred, Z_next_actual.detach())
                    elif args.loss_type == 2: # L2 loss
                        dynamics_loss = F.mse_loss(Z_next_pred, Z_next_actual.detach())
                    elif args.loss_type == 4: # L4 loss
                        dynamics_loss = torch.mean((Z_next_pred - Z_next_actual.detach())**4)
                else:
                    if args.loss_type == 1:
                        dynamics_loss = F.l1_loss(Z_next_pred, Z_next_actual)
                    elif args.loss_type == 2:
                        dynamics_loss = F.mse_loss(Z_next_pred, Z_next_actual)  #grad flow from the world model
                    elif args.loss_type == 4:
                        dynamics_loss = torch.mean((Z_next_pred - Z_next_actual)**4)

                # ------------------------------------------------------------------
                #  (UPDATED) Diagnostics: average L2-norm (vector magnitude) of latent tensors
                # ------------------------------------------------------------------
                avg_mag_z_t         = Z_t.norm(dim=-1).mean().item()
                avg_mag_z_next_pred = Z_next_pred.norm(dim=-1).mean().item()
                avg_mag_action_one_hot = Action_OneHot_Batch.norm(dim=-1).mean().item() # Mag of one-hot actions (should be 1)

                if global_step % args.log_steps == 0:
                    tqdm.write(
                         f"[LatentStats step={global_step}] ||z_t||={avg_mag_z_t:.3f} ||z_next_pred||={avg_mag_z_next_pred:.3f} ||action_one_hot||={avg_mag_action_one_hot:.3f}"
                    )
                    # count the number of unique encoding vectors and most likely actions in the batch
                    num_z = int(torch.unique(Z_t, dim=0).size(0))
                    num_actions = int(torch.unique(Action_OneHot_Batch, dim=0).size(0))
                    tqdm.write(f"[CodeStats step={global_step}] num_encoding_vec={num_z} | num_most_likely_actions_in_batch={num_actions}")
                    if args.heatmap:
                        # save a heatmap of action distribution
                        action_distribution = model.get_action_distribution_probs(Z_t)
                        plt.figure(figsize=(10, 10))
                        plt.imshow(action_distribution.cpu().detach().numpy())
                        plt.savefig(f"{args.output_dir}/heatmap_action_distribution_{global_step}.png")
                        plt.close()

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
                    V_pred_safe = torch.clamp(torch.nan_to_num(V_pred), -10000.0, 10000.0)
                    value_loss = F.smooth_l1_loss(V_pred_safe, batch_returns_tensor)

                if args.lambda_entropy > 0:
                    # Entropy bonus for exploration, compute manually for numerical stability
                    dist_probs = model.get_action_distribution_probs(Z_t) # Renamed method in PLDMModel
                    entropy = - (dist_probs * (dist_probs + eps).log()).sum(dim=-1).mean()
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
                    policy_params_list = [p for n,p in model.policy_value_network.named_parameters() if not n.startswith('value_mlp') and p.requires_grad] # updated name
                    _log_encoder_grad_sources(policy_params_list, loss_contributions, global_step, prefix="PolicyGradSources")

                    # Dynamics model gradients
                    dynamics_params_list = [p for p in model.dynamics_model.parameters() if p.requires_grad] # updated name
                    _log_encoder_grad_sources(dynamics_params_list, loss_contributions, global_step, prefix="DynamicsGradSources")

                # Optimizer step
                loss.backward()

                # Clip gradients for each parameter group
                if args.clip_grad_encoder_norm > 0:
                    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=args.clip_grad_encoder_norm)
                if args.clip_grad_dynamics_norm > 0:
                    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[1]['params'], max_norm=args.clip_grad_dynamics_norm)
                if args.clip_grad_policy_norm > 0:
                    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[2]['params'], max_norm=args.clip_grad_policy_norm)
                if args.clip_grad_value_norm > 0:
                    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[3]['params'], max_norm=args.clip_grad_value_norm)
                
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
                writer.add_scalar('Stats/mag_action_one_hot', avg_mag_action_one_hot, global_step)

                # ---------------- aggregate epoch-level stats -----------------
                total_policy_loss += policy_loss.item()
                total_dynamics_loss += dynamics_loss.item()
                if args.use_same_page_loss:
                    total_on_the_same_page_loss += on_the_same_page_loss.item()
                if args.use_value_loss:
                    total_value_loss += value_loss.item()

                # ----------------------------------------------------------------
                global_step += 1
                # ----- Linear-probe evaluation -----
                if args.probe_eval_interval > 0 and (global_step % args.probe_eval_interval == 0):
                    probe_metrics = evaluate_representation(
                        model,
                        create_env,
                        device,
                        num_samples=args.probe_num_samples,
                        train_steps=args.probe_train_steps,
                        batch_size=args.probe_batch_size,
                    )
                    writer.add_scalar('Probe/Decode_MSE', probe_metrics['decode_mse'], global_step)
                    writer.add_scalar('Probe/Encode_MSE', probe_metrics['encode_mse'], global_step)
                    tqdm.write(f"[Probe step={global_step}] decode_mse={probe_metrics['decode_mse']:.4f} | encode_mse={probe_metrics['encode_mse']:.4f}")

                    # Append data for plotting
                    probe_steps_history.append(global_step)
                    decode_mse_history.append(probe_metrics['decode_mse'])
                    encode_mse_history.append(probe_metrics['encode_mse'])

                    # Generate and save the plot
                    if probe_steps_history:
                        plt.figure(figsize=(10, 6))
                        plt.plot(probe_steps_history, decode_mse_history, marker='o', linestyle='-', label='Decode MSE')
                        plt.plot(probe_steps_history, encode_mse_history, marker='x', linestyle='--', label='Encode MSE')
                        plt.title('Probe MSEs vs Global Step')
                        plt.xlabel('Global Step')
                        plt.ylabel('MSE')
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        plt.savefig(output_dir / "probe_metrics_vs_step.png")
                        plt.close()
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
        
            # Save probe evaluation history
            if probe_steps_history: # Check if any probe data was collected
                with open(output_dir / "probe_steps_history.txt", "w") as f:
                    for step_val in probe_steps_history:
                        f.write(f"{step_val}\n")
                with open(output_dir / "decode_mse_history.txt", "w") as f:
                    for mse_val in decode_mse_history:
                        f.write(f"{mse_val}\n")
                with open(output_dir / "encode_mse_history.txt", "w") as f:
                    for mse_val in encode_mse_history:
                        f.write(f"{mse_val}\n")

        writer.close()
        
    finally:
        # Make sure to stop the collector
        collector.stop()
    
    return model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train PLDM model on DotWall environment')
    
    # Model parameters (newly added or renamed)
    parser.add_argument('--model_encoding_dim', type=int, default=DEFAULT_ENCODING_DIM, 
                        help='Dimension of the encoded state z_t from the encoder')
    parser.add_argument('--num_actions', type=int, default=DEFAULT_NUM_ACTIONS, 
                        help='Number of discrete actions for the policy network')
    parser.add_argument('--encoder_embedding_dim', type=int, default=256, 
                        help='Internal embedding dimension for ViT encoder or similar usage in CNN')
    parser.add_argument('--encoder_type', type=str, default='vit', choices=['vit','cnn'], 
                        help='Encoder architecture: vit or cnn')
    parser.add_argument('--policy_temperature', type=float, default=1.0, 
                        help='Temperature for policy network action sampling; replaces next_goal_temp')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--updates_per_epoch', type=int, default=32, help='Number of training updates (batches of transitions) per epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of trajectories to process in a batch')
    parser.add_argument('--max_steps_per_episode', type=int, default=64, help='Maximum steps per episode')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--max_step_norm', type=float, default=12, help='Maximum step norm for action grid')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of parallel workers for episode collection')
    parser.add_argument('--use_gpu_inference', action='store_true', default=True, help='Use GPU for inference during rollout')
    parser.add_argument('--log_steps', type=int, default=64, help='Logging frequency for gradient statistics')
    parser.add_argument('--heatmap', action='store_false', default=False, help='Save a heatmap of Z_t')
    parser.add_argument('--use_same_page_loss', action='store_false', default=False, help='Use on-the-same-page loss between next goal and dynamics')
    parser.add_argument('--use_decoder_loss', action='store_false', default=False, help='Enable decoder reconstruction warm-up loss')
    parser.add_argument('--use_value_loss', action='store_true', default=True, help='Train value head with MSE to returns')
    parser.add_argument('--normalize_returns_and_advantage', action='store_true', default=True, help='Normalize returns and advantage to zero-mean, unit-std')
    
    parser.add_argument('--lambda_dynamics', type=float, default=1e0, help='Weight for dynamics loss')
    parser.add_argument('--lambda_policy', type=float, default=1e-1, help='Weight for policy loss')
    parser.add_argument('--lambda_value', type=float, default=5e-2, help='Weight for value loss')
    parser.add_argument('--lambda_same_page', type=float, default=0.0, help='Weight for on-the-same-page loss')
    parser.add_argument('--lambda_entropy', type=float, default=0.1, help='Weight for policy entropy bonus') # can't be smaller than 0.1 otherwise the policy will be constant

    parser.add_argument('--encoder_lr', type=float, default=1e-5, help='Learning rate for encoder')
    parser.add_argument('--dynamics_lr', type=float, default=5e-4, help='Learning rate for dynamics model')
    parser.add_argument('--policy_lr', type=float, default=1e-4, help='Learning rate for policy') # can't be too large otherwise the policy will be constant
    parser.add_argument('--value_lr', type=float, default=1e-4, help='Learning rate for value')
    parser.add_argument('--decoder_lr', type=float, default=1e-1, help='Learning rate for decoder')

    # Gradient clipping norms per parameter group
    parser.add_argument('--clip_grad_encoder_norm', type=float, default=1.0, help='Max grad norm for encoder parameters')
    parser.add_argument('--clip_grad_dynamics_norm', type=float, default=1.0, help='Max grad norm for dynamics model parameters')
    parser.add_argument('--clip_grad_policy_norm', type=float, default=1.0, help='Max grad norm for policy network (excluding value head) parameters')
    parser.add_argument('--clip_grad_value_norm', type=float, default=1.0, help='Max grad norm for value head parameters')
    parser.add_argument('--clip_grad_decoder_norm', type=float, default=-1.0, help='Max grad norm for decoder parameters (not used)')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run training on')
    parser.add_argument('--output_dir', type=str, default='output_JEPA_l4loss', help='Directory to save model and logs') # Updated default
    parser.add_argument('--resume', action='store_false', default=False, help='Resume training from checkpoint')
    parser.add_argument('--mode', type=str, default='JEPA', choices=['RL','JEPA'], help='block the grad flow from JEPA if mode is RL')
    parser.add_argument('--loss_type', type=int, default=4, choices=[1,2,4], help='Use L_n loss for dynamics loss')

    # Add a new argument to choose between immediate rewards and discounted returns
    parser.add_argument('--use_immediate_reward', action='store_true', default=False, help='Use immediate reward for advantage calculation and value network training')

    # Argument for observation noise
    parser.add_argument('--obs_noise_std', type=float, default=0.1, help='Standard deviation of Gaussian noise to add to observations.')

    # -------------------------------------------------------------------------
    #  Representation probe evaluation parameters
    # -------------------------------------------------------------------------
    parser.add_argument('--probe_eval_interval', type=int, default=128,
                        help='Run linear-probe evaluation every N global steps (0 disables probing).')
    parser.add_argument('--probe_num_samples', type=int, default=1024,
                        help='Number of random environment states sampled for each probe evaluation.')
    parser.add_argument('--probe_train_steps', type=int, default=2000,
                        help='SGD steps to train each linear probe during evaluation.')
    parser.add_argument('--probe_batch_size', type=int, default=128,
                        help='Mini-batch size during probe training.')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.autograd.set_detect_anomaly(True)     # catches backward NaNs
    train_pldm(args)
