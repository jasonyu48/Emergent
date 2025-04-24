from typing import Optional
import os
import shutil
from dataclasses import dataclass
import dataclasses
import random
import time
from omegaconf import MISSING

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.amp  # Add this import for mixed precision training
import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import matplotlib
import gymnasium as gym
import argparse
from pathlib import Path
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import copy

from pldm.logger import Logger, MetricTracker

try:
    import multiprocessing
    multiprocessing.set_start_method("fork")  # noqa
except Exception:
    pass

from pldm.configs import ConfigBase
from pldm.data.enums import DataConfig
from pldm.data.utils import get_optional_fields
import pldm.utils as utils

from pldm_envs.wall.wall import DotWall
from pldm.model import PLDMModel


def calculate_distance_reward(dot_position, target_position, wall_x, wall_width):
    """Calculate reward based on distance and whether dot and target are in same room"""
    
    # Calculate Euclidean distance
    distance = torch.norm(dot_position - target_position, dim=-1)
    
    # Determine if dot and target are in the same room
    half_width = wall_width // 2
    left_wall_x = wall_x - half_width
    right_wall_x = wall_x + half_width
    
    dot_in_left_room = dot_position[:, 0] < left_wall_x
    target_in_left_room = target_position[:, 0] < left_wall_x
    
    same_room = (dot_in_left_room == target_in_left_room)
    
    # Calculate reward
    distance_reward = -distance  # Negative distance as reward
    same_room_bonus = torch.where(same_room, torch.tensor(20.0, device=dot_position.device), torch.tensor(0.0, device=dot_position.device))
    
    return distance_reward + same_room_bonus + 30


def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns"""
    returns = []
    R = 0
    
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    
    returns = torch.tensor(returns)
    
    # # Normalize returns
    # if len(returns) > 1:
    #     returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    return returns


def rollout(model, env, max_steps=100, device='cpu', bf16=False, num_samples=100):
    """Perform a single rollout in the environment using the PLDM model"""
    # Reset environment
    obs, info = env.reset()
    
    # Initialize lists to store trajectory information
    states = [obs]
    actions = []
    rewards = []
    log_probs = []
    next_goals = []
    
    # Ensure model is in evaluation mode during rollout
    model.eval()
    
    # Get dtype based on bf16 setting
    dtype = torch.bfloat16 if bf16 else torch.float32
    
    # Verify model is using the correct dtype
    sample_param = next(model.parameters())
    if sample_param.dtype != dtype:
        print(f"Warning: Model dtype ({sample_param.dtype}) doesn't match requested dtype ({dtype})")
        print("Converting model to the correct dtype")
        model = model.to(dtype)
        # Verify conversion was successful
        sample_param = next(model.parameters())
        print(f"Model parameters dtype after conversion: {sample_param.dtype}")
        
        # Double-check Conv2d bias which often causes issues
        for module in model.encoder.modules():
            if isinstance(module, nn.Conv2d) and module.bias is not None:
                if module.bias.dtype != dtype:
                    print(f"Warning: Conv2d bias still has dtype {module.bias.dtype}, forcing conversion to {dtype}")
                    module.bias = nn.Parameter(module.bias.to(dtype))
                else:
                    print(f"Conv2d bias correctly has dtype {dtype}")
                break
    
    # Get initial encoding
    with torch.no_grad():
        # Convert observation to tensor properly
        if isinstance(obs, torch.Tensor):
            obs_tensor = obs.to(dtype=dtype, device=device).unsqueeze(0)
        else:
            obs_tensor = torch.tensor(obs, dtype=dtype, device=device).unsqueeze(0)
        
        # Encode current observation
        try:
            z_t = model.encode(obs_tensor)
        except RuntimeError as e:
            if "Input type" in str(e) and "bias type" in str(e):
                print(f"BFloat16 conversion error: {e}")
                print("Attempting to fix Conv2d bias dtype issue...")
                
                # Fix bias issue in specific conv layers
                for module in model.encoder.modules():
                    if isinstance(module, nn.Conv2d) and module.bias is not None:
                        module.bias = nn.Parameter(module.bias.to(dtype))
                
                # Try encoding again
                z_t = model.encode(obs_tensor)
        
    # Rollout loop
    done = False
    truncated = False
    
    for step in range(max_steps):
        if done or truncated:
            break
        
        
        # Predict next goal
        with torch.no_grad():
            # Ensure z_t has correct dtype
            if z_t.dtype != dtype:
                z_t = z_t.to(dtype)
                
            z_next, log_prob = model.predict_next_goal(z_t)
            next_goals.append(z_next.cpu())
            log_probs.append(log_prob.item())
        
        # Action search needs z_t and z_next to be detached
        z_t_detached = z_t.clone().detach()
        z_next_detached = z_next.clone().detach()
        
        # Search for action using detached tensors
        a_t = model.search_action(
            z_t_detached.to(dtype), 
            z_next_detached.to(dtype), 
            num_samples=num_samples
        )

        # Take action in environment
        # Convert to float32 before converting to NumPy since NumPy doesn't support bfloat16
        action = a_t.to(torch.float32).cpu().numpy()[0]
        obs, reward, done, truncated, info = env.step(action)

        # ----------------------------------------------------------------------------------
        # Use distance‑based custom reward (same definition as in the test script) instead
        # of the environment‑supplied reward so that train‑time and test‑time signals match.
        # ----------------------------------------------------------------------------------
        dot_position = env.dot_position.unsqueeze(0)
        target_position = env.target_position.unsqueeze(0)
        custom_reward = calculate_distance_reward(
            dot_position,
            target_position,
            env.wall_x,
            env.wall_width,
        ).item()

        # Store information
        states.append(obs)
        actions.append(action)
        rewards.append(custom_reward)
        
        # Update current encoding
        with torch.no_grad():
            # Convert observation to tensor
            if isinstance(obs, torch.Tensor):
                obs_tensor = obs.to(dtype=dtype, device=device).unsqueeze(0)
            else:
                obs_tensor = torch.tensor(obs, dtype=dtype, device=device).unsqueeze(0)
            
            # Encode next observation
            z_t = model.encode(obs_tensor)
    
    # Set model back to training mode
    model.train()
            
    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'log_probs': log_probs,
        'next_goals': next_goals,
        'done': done
    }


def calculate_custom_reward(states, env):
    """Calculate rewards based on distances rather than environment rewards"""
    rewards = []
    
    for i in range(len(states) - 1):
        # Handle case where states might already be tensors
        if isinstance(states[i], torch.Tensor):
            s_t = states[i].float().unsqueeze(0)
        else:
            s_t = torch.from_numpy(states[i]).float().unsqueeze(0)
            
        if isinstance(states[i+1], torch.Tensor):
            s_next = states[i+1].float().unsqueeze(0)
        else:
            s_next = torch.from_numpy(states[i+1]).float().unsqueeze(0)
        
        # Extract dot and target positions from states
        dot_position = env.dot_position.unsqueeze(0)
        target_position = env.target_position.unsqueeze(0)
        
        # Move tensors to the same device if needed
        device = dot_position.device
        if s_t.device != device:
            s_t = s_t.to(device)
        if s_next.device != device:
            s_next = s_next.to(device)
        
        # Calculate distance-based reward
        reward = calculate_distance_reward(
            dot_position, 
            target_position, 
            env.wall_x,
            env.wall_width
        )
        
        rewards.append(reward.item())
    
    return rewards


class ParallelEpisodeCollector:
    """Collect episodes in parallel for faster training"""
    
    def __init__(self, model, env_creator, max_steps, device, bf16_supported, 
                 num_workers=4, prefetch_queue_size=8, use_gpu_for_inference=True, num_samples=100):
        """
        Initialize parallel episode collector
        
        Args:
            model: The PLDM model
            env_creator: Function that creates a new environment instance
            max_steps: Maximum steps per episode
            device: Device to run computation on
            bf16_supported: Whether BF16 precision is supported
            num_workers: Number of parallel workers
            prefetch_queue_size: Size of the prefetch queue for episodes
            use_gpu_for_inference: Whether to use GPU for inference during rollout
            num_samples: Number of action samples to evaluate in parallel
        """
        self.model = model
        self.env_creator = env_creator
        self.max_steps = max_steps
        self.device = device
        self.bf16_supported = bf16_supported
        self.num_workers = num_workers
        self.use_gpu_for_inference = use_gpu_for_inference
        self.num_samples = num_samples
        
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
                    # Use GPU with lock to prevent race conditions
                    with self.gpu_lock:
                        # Ensure model is properly converted to BFloat16 if needed
                        if self.bf16_supported:
                            # Verify model is in BFloat16 mode
                            sample_param = next(self.model.parameters())
                            if sample_param.dtype != torch.bfloat16:
                                print(f"Converting worker {worker_id} model to BFloat16 (current dtype: {sample_param.dtype})")
                                self.model = self.model.to(torch.bfloat16)
                                # Verify conversion was successful
                                sample_param = next(self.model.parameters())
                                print(f"Worker {worker_id} model parameters dtype after conversion: {sample_param.dtype}")
                                
                                # Check a Conv2d bias specifically (they often cause issues)
                                for module in self.model.encoder.modules():
                                    if isinstance(module, nn.Conv2d) and module.bias is not None:
                                        print(f"Worker {worker_id} Conv2d bias dtype: {module.bias.dtype}")
                                        break
                        
                        trajectory = rollout(
                            self.model,  # Use the shared GPU model
                            env,
                            max_steps=self.max_steps,
                            device=self.device,  # Use GPU for inference
                            bf16=self.bf16_supported,
                            num_samples=self.num_samples
                        )
                else:
                    # Create a CPU copy for inference (original behavior)
                    local_model = copy.deepcopy(self.model).to('cpu')
                    local_model.eval()
                    trajectory = rollout(
                        local_model,
                        env,
                        max_steps=self.max_steps,
                        device='cpu',  # Use CPU for inference
                        bf16=False,    # Use FP32 on CPU for stability
                        num_samples=self.num_samples
                    )
                
                # Skip empty trajectories
                if len(trajectory['states']) <= 1 or len(trajectory['log_probs']) == 0:
                    continue
                    
                # Put episode in queue
                try:
                    self.episode_queue.put(trajectory, block=True, timeout=5.0)
                except queue.Full:
                    # If queue is full, skip this episode
                    pass
                    
            except Exception as e:
                print(f"Error in worker {worker_id}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def get_batch(self, batch_size):
        """Get a batch of episodes from the queue"""
        batch = []
        for _ in range(batch_size):
            try:
                episode = self.episode_queue.get(block=True, timeout=5.0)
                self.episode_queue.task_done()
                batch.append(episode)
            except queue.Empty:
                # If queue is empty after timeout, return whatever we have
                break
        return batch
    
    def stop(self):
        """Stop all worker threads"""
        self.stop_flag.set()
        for future in self.futures:
            future.cancel()
        self.executor.shutdown(wait=False)

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
        from tqdm.auto import tqdm as _tqdm
        _tqdm.write(f"[GradStats step={step_idx}] " + " | ".join(parts))

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
        if loss is None:
            continue
        # Compute grads w.r.t encoder params WITHOUT accumulating into .grad
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        # Compute mean absolute gradient across all encoder params
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
        from tqdm.auto import tqdm as _tqdm
        _tqdm.write(f"[{prefix} step={step_idx}] " + " | ".join(parts))

def train_pldm(args):
    """Main training function for PLDM model"""
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set up device
    device = torch.device(args.device)
    
    # Check if bf16 is supported on the current device
    bf16_supported = (
        args.bf16 and
        torch.cuda.is_available() and
        torch.cuda.is_bf16_supported()
    )
    
    if args.bf16 and not bf16_supported:
        print("Warning: BF16 precision requested but not supported on this device. Using FP32 instead.")
    
    # Set up mixed precision training
    if torch.cuda.is_available():
        if bf16_supported:
            print("Using BFloat16 mixed precision training")
            amp_dtype = torch.bfloat16
            # Optional: enable autocast globally
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            print("Using FP32 training")
            amp_dtype = None
    else:
        print("Mixed precision not available, using FP32")
        amp_dtype = None
    
    # Create scaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda', enabled=amp_dtype is not None and amp_dtype == torch.float16)
    
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
        encoder_embedding=args.encoder_embedding
    ).to(device)
    
    # Print model parameter counts
    model.print_parameter_count()
    
    # Convert model to mixed precision if supported
    if amp_dtype is not None:
        print(f"Model will use {amp_dtype} precision during forward pass")
    
    # Create single optimizer with parameter groups for different learning rates
    optimizer = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': args.encoder_lr},
        {'params': model.dynamics.parameters(), 'lr': args.dynamics_lr},
        {'params': model.next_goal_predictor.parameters(), 'lr': args.policy_lr},
        {'params': model.decoder.parameters(), 'lr': args.decoder_lr}
    ])
    
    # Create learning rate schedulers
    # We'll implement a custom scheduler function to manage the different learning rates
    def adjust_learning_rates(epoch):
        """Adjust learning rates based on epoch and component"""
        # Get base learning rates from parameter groups
        encoder_lr = optimizer.param_groups[0]['lr']
        dynamics_lr = optimizer.param_groups[1]['lr']
        policy_lr = optimizer.param_groups[2]['lr']
        decoder_lr = optimizer.param_groups[3]['lr']
        
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
        optimizer.param_groups[3]['lr'] = decoder_lr
        
        return encoder_lr, dynamics_lr, policy_lr, decoder_lr
    
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
            
            # If using FP16, load scaler state
            if 'scaler_state_dict' in checkpoint and amp_dtype == torch.float16:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
        start_epoch = checkpoint.get('epoch', 0) + 1
        global_step = checkpoint.get('global_step', 0)
        best_reward = checkpoint.get('best_reward', float('-inf'))
        
        print(f"Resuming from epoch {start_epoch} with best reward {best_reward:.4f}")
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=str(output_dir / 'logs'))
    
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

    # Create parallel episode collector
    collector = ParallelEpisodeCollector(
        model=model, 
        env_creator=create_env,
        max_steps=args.max_steps_per_episode,
        device=device,
        bf16_supported=bf16_supported,
        num_workers=num_workers,
        use_gpu_for_inference=args.use_gpu_inference,  # Use GPU for inference since we have a powerful A100
        num_samples=args.num_samples
    )

    try:
        # Training loop
        transition_buffer = []  # holds (s, s_next, a, ng, R_t)
        for epoch in range(start_epoch, args.epochs):
            total_reward = 0
            total_policy_loss = 0
            total_dynamics_loss = 0
            total_next_state_loss = 0
            total_on_the_same_page_loss = 0
            total_reconstruction_loss = 0
            total_variance_loss = 0
            total_value_loss = 0
            num_episodes = 0

            # Adjust learning rates for this epoch
            encoder_lr, dynamics_lr, policy_lr, decoder_lr = adjust_learning_rates(epoch)

            # Log current learning rates
            print(f"Epoch {epoch+1}/{args.epochs} - Learning rates: Encoder={encoder_lr:.2e}, "
                  f"Dynamics={dynamics_lr:.2e}, Policy={policy_lr:.2e}, Decoder={decoder_lr:.2e}")
                  
            writer.add_scalar('LearningRate/encoder', encoder_lr, epoch)
            writer.add_scalar('LearningRate/dynamics', dynamics_lr, epoch)
            writer.add_scalar('LearningRate/policy', policy_lr, epoch)
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
                batch_returns_tensor = torch.tensor(batch_returns, dtype=torch.float32, device=device)

                with torch.amp.autocast('cuda', enabled=amp_dtype is not None, dtype=amp_dtype):
                    policy_log_probs = []
                    dynamics_loss = 0.0
                    next_state_loss = 0.0
                    on_the_same_page_loss = 0.0
                    reconstruction_loss = 0.0
                    variance_loss = 0.0
                    value_loss = 0.0

                    # Keep all z_t to compute batch variance later
                    batch_encodings = []

                    for state, next_state, action, next_goal in zip(batch_states, batch_next_states, batch_actions, batch_next_goals):
                        z_t = model.encode(state.unsqueeze(0))
                        z_next_actual = model.encode(next_state.unsqueeze(0))

                        # Accumulate z_t for variance computation
                        batch_encodings.append(z_t.squeeze(0))

                        # Decoder warm-up reconstruction loss
                        if epoch < args.warmup_epochs_decoder and args.use_decoder_loss:
                            recon = model.decode(z_t)
                            target_img = state.unsqueeze(0) / 255.0 if state.max() > 1.0 else state.unsqueeze(0)
                            reconstruction_loss += F.mse_loss(recon, target_img)

                        # Policy log-prob
                        log_prob = model.next_goal_predictor.log_prob(z_t, next_goal.unsqueeze(0))
                        policy_log_probs.append(log_prob.squeeze(0))

                        # Value prediction
                        value_pred = model.next_goal_predictor.value(z_t).squeeze(0)
                        value_loss += F.mse_loss(value_pred, torch.tensor(R_t, device=device, dtype=value_pred.dtype))

                        # Dynamics prediction
                        a_t = action.unsqueeze(0)
                        z_next_pred = model.dynamics(z_t, a_t)
                        dynamics_loss += F.mse_loss(z_next_pred, z_next_actual)

                        if args.use_next_state_loss:
                            next_state_loss += 0  # placeholder
                        if args.use_same_page_loss:
                            on_the_same_page_loss += F.mse_loss(next_goal, z_next_pred)

                    # Average losses over batch transitions (those accumulated inside loop)
                    dynamics_loss = dynamics_loss / len(batch_states)
                    if epoch < args.warmup_epochs_decoder and args.use_decoder_loss:
                        reconstruction_loss = reconstruction_loss / len(batch_states)
                    if args.use_next_state_loss:
                        next_state_loss = next_state_loss / len(batch_states)
                    if args.use_same_page_loss:
                        on_the_same_page_loss = on_the_same_page_loss / len(batch_states)

                    # ------------------------------------------------------------------
                    # Variance regularization (VICReg-style): encourage each latent dim to
                    # have std ≥ 1 across the batch. Penalise only during warm-up.
                    # ------------------------------------------------------------------
                    if epoch < args.warmup_epochs_decoder and args.lambda_variance > 0:
                        enc_stack = torch.stack(batch_encodings, dim=0)  # [B, D]
                        enc_stack = F.normalize(enc_stack, p=2, dim=1)     # unit-norm vectors
                        var = enc_stack.var(dim=0, unbiased=False) + 1e-4
                        std = torch.sqrt(var)
                        variance_loss = F.relu(1.0 - std).mean()

                    # --------------------- Value head & policy advantage ------------------
                    enc_stack = torch.stack(batch_encodings, dim=0)  # [B, D]
                    value_preds_batch = model.next_goal_predictor.value(enc_stack)

                    # Advantage = R - V (detach V for policy gradient)
                    advantage = batch_returns_tensor - value_preds_batch.detach()
                    # Normalize advantage
                    adv_mean = advantage.mean()
                    adv_std = advantage.std(unbiased=False) + 1e-8
                    norm_advantage = (advantage - adv_mean) / adv_std
                    policy_log_probs_tensor = torch.stack(policy_log_probs)
                    policy_loss = -(policy_log_probs_tensor * norm_advantage).mean()

                    # Value loss (MSE)
                    value_loss = F.mse_loss(value_preds_batch, batch_returns_tensor)

                    # Total loss
                    loss = args.lambda_policy * policy_loss + args.lambda_dynamics * dynamics_loss
                    if epoch < args.warmup_epochs_decoder and args.use_decoder_loss:
                        loss += args.lambda_recon * reconstruction_loss
                    if args.use_next_state_loss:
                        loss += args.lambda_dynamics * next_state_loss
                    if args.use_same_page_loss:
                        loss += args.lambda_dynamics * on_the_same_page_loss
                    if epoch < args.warmup_epochs_decoder and args.lambda_variance > 0:
                        loss += args.lambda_variance * variance_loss
                    if args.use_value_loss:
                        loss += args.lambda_value * value_loss

                # Log encoder grad sources
                encoder_params_list = [p for p in model.encoder.parameters() if p.requires_grad]
                loss_contributions = {
                    'policy': args.lambda_policy * policy_loss,
                    'dynamics': args.lambda_dynamics * dynamics_loss,
                    'value': args.lambda_value * value_loss if args.use_value_loss else 0.0,
                }
                if epoch < args.warmup_epochs_decoder and args.use_decoder_loss:
                    loss_contributions['recon'] = args.lambda_recon * reconstruction_loss
                if args.use_next_state_loss:
                    loss_contributions['next_state'] = args.lambda_dynamics * next_state_loss
                if args.use_same_page_loss:
                    loss_contributions['same_page'] = args.lambda_dynamics * on_the_same_page_loss
                if epoch < args.warmup_epochs_decoder and args.lambda_variance > 0:
                    loss_contributions['variance'] = args.lambda_variance * variance_loss
                if global_step % args.log_steps == 0:
                    # Encoder gradients
                    _log_encoder_grad_sources(encoder_params_list, loss_contributions, global_step, prefix="EncGradSources")

                # Optimizer step
                if amp_dtype == torch.float16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if global_step % args.log_steps == 0:
                    _log_grad_update_stats(model, optimizer, global_step)
                optimizer.zero_grad()

                # Logging
                writer.add_scalar('Loss/policy', policy_loss.item(), global_step)
                writer.add_scalar('Loss/dynamics', dynamics_loss.item(), global_step)
                if args.use_next_state_loss:
                    writer.add_scalar('Loss/next_state', next_state_loss.item(), global_step)
                if args.use_same_page_loss:
                    writer.add_scalar('Loss/on_the_same_page', on_the_same_page_loss.item(), global_step)
                if args.use_decoder_loss and epoch < args.warmup_epochs_decoder:
                    writer.add_scalar('Loss/reconstruction', reconstruction_loss.item(), global_step)
                if epoch < args.warmup_epochs_decoder and args.lambda_variance > 0:
                    writer.add_scalar('Loss/variance', variance_loss.item(), global_step)
                if args.use_value_loss:
                    writer.add_scalar('Loss/value', value_loss.item(), global_step)
                writer.add_scalar('Loss/total', loss.item(), global_step)
                writer.add_scalar('Reward/individual_episode', batch_returns[0], global_step)

                # ---------------- aggregate epoch-level stats -----------------
                total_policy_loss += policy_loss.item()
                total_dynamics_loss += dynamics_loss.item()
                if args.use_next_state_loss:
                    total_next_state_loss += next_state_loss.item()
                if args.use_same_page_loss:
                    total_on_the_same_page_loss += on_the_same_page_loss.item()
                if args.use_decoder_loss and epoch < args.warmup_epochs_decoder:
                    total_reconstruction_loss += reconstruction_loss.item()
                if epoch < args.warmup_epochs_decoder and args.lambda_variance > 0:
                    # accumulate for reporting
                    pass  # variance loss is diagnostic; not included in epoch stats for now
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
                if args.use_next_state_loss:
                    avg_next_state_loss = total_next_state_loss / num_episodes
                    report += f", Avg Next State Loss: {avg_next_state_loss:.4f}"
                
                if args.use_same_page_loss:
                    avg_on_the_same_page_loss = total_on_the_same_page_loss / num_episodes
                    report += f", Avg On-Same-Page Loss: {avg_on_the_same_page_loss:.4f}"
                
                if args.use_decoder_loss and epoch < args.warmup_epochs_decoder:
                    avg_reconstruction_loss = total_reconstruction_loss / num_episodes
                    report += f", Avg Reconstruction Loss: {avg_reconstruction_loss:.4f}"
                
                if args.use_value_loss:
                    avg_value_loss = total_value_loss / num_episodes
                    report += f", Avg Value Loss: {avg_value_loss:.4f}"
                
                print(report)
                
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
            
            # Add scaler state dict if using FP16
            if amp_dtype == torch.float16:
                checkpoint_dict['scaler_state_dict'] = scaler.state_dict()
                
            torch.save(checkpoint_dict, output_dir / 'checkpoint.pt')
        
        writer.close()
        
    finally:
        # Make sure to stop the collector
        collector.stop()
    
    return model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train PLDM model on DotWall environment')
    
    # Model parameters
    parser.add_argument('--encoding_dim', type=int, default=32, help='Dimension of encoded state')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Dimension of hidden layers')
    parser.add_argument('--encoder_embedding', type=int, default=256, help='Dimension of encoder embedding')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--updates_per_epoch', type=int, default=32, help='Number of training updates (batches of transitions) per epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of trajectories to process in a batch')
    parser.add_argument('--max_steps_per_episode', type=int, default=32, help='Maximum steps per episode')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of action samples to evaluate in parallel')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--max_step_norm', type=float, default=15, help='Maximum step norm')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of parallel workers for episode collection')
    parser.add_argument('--use_gpu_inference', type=bool, default=True, help='Use GPU for inference during rollout')
    parser.add_argument('--log_steps', type=int, default=16, help='Logging frequency for gradient statistics')

    parser.add_argument('--use_next_state_loss', type=bool, default=False, help='Use next state prediction loss')
    parser.add_argument('--use_same_page_loss', type=bool, default=False, help='Use on-the-same-page loss between next goal and dynamics')
    parser.add_argument('--use_decoder_loss', type=bool, default=False, help='Enable decoder reconstruction warm-up loss')
    parser.add_argument('--use_value_loss', type=bool, default=True, help='Train value head with MSE to returns')
    parser.add_argument('--warmup_epochs_decoder', type=int, default=0, help='Number of epochs to train decoder')
    
    parser.add_argument('--lambda_dynamics', type=float, default=1e0, help='Weight for dynamics loss')
    parser.add_argument('--lambda_policy', type=float, default=1e0, help='Weight for policy loss')
    parser.add_argument('--lambda_value', type=float, default=1e-2, help='Weight for value loss')
    parser.add_argument('--lambda_recon', type=float, default=1e5, help='Weight for reconstruction loss during warm-up')
    parser.add_argument('--lambda_variance', type=float, default=1e1, help='Weight for encoder variance loss during warm-up')

    parser.add_argument('--encoder_lr', type=float, default=5e-7, help='Learning rate for encoder')
    parser.add_argument('--dynamics_lr', type=float, default=1e+1, help='Learning rate for dynamics model')
    parser.add_argument('--policy_lr', type=float, default=2e-1, help='Learning rate for policy')
    parser.add_argument('--decoder_lr', type=float, default=2e-4, help='Learning rate for decoder')
    
    # Precision parameters
    parser.add_argument('--bf16', type=bool, default=False, help='Use BFloat16 mixed precision for training')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run training on')
    parser.add_argument('--output_dir', type=str, default='output_v', help='Directory to save model and logs')
    parser.add_argument('--resume', type=bool, default=False, help='Resume training from checkpoint')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_pldm(args)
