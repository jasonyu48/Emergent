import multiprocessing
import warnings

warnings.filterwarnings("ignore", message="Ill-formed record")

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
    multiprocessing.set_start_method("fork")  # noqa
except:
    pass

from pldm.configs import ConfigBase
from pldm.data.enums import DataConfig
from pldm.data.dataset_factory import DatasetFactory
from pldm.data.utils import get_optional_fields
from pldm.optimizers.schedulers import Scheduler, LRSchedule
from pldm.optimizers.optimizer_factory import OptimizerFactory, OptimizerType
from pldm.evaluation.evaluator import EvalConfig, Evaluator

# if "AMD" not in torch.cuda.get_device_name(0):

from pldm.models.hjepa import HJEPA, HJEPAConfig

from pldm.objectives import ObjectivesConfig
import pldm.utils as utils

from pldm_envs.wall.wall import DotWall
from pldm.model import PLDMModel


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@dataclass
class TrainConfig(ConfigBase):
    env_name: str = MISSING
    n_steps: int = 17
    val_n_steps: int = 17
    l1_n_steps: int = 17
    wandb: bool = True
    run_name: Optional[str] = None
    run_group: Optional[str] = None
    run_project: str = "PLDM"
    output_root: Optional[str] = None
    output_dir: Optional[str] = None
    eval_mpcs: int = 20
    quick_debug: bool = False
    seed: int = 42
    load_checkpoint_path: Optional[str] = None
    load_l1_only: bool = False
    eval_only: bool = False
    train_only: bool = False
    epochs: int = 100
    base_lr: float = 0.2
    disable_l2: bool = True
    optimizer_type: OptimizerType = OptimizerType.LARS
    optimizer_schedule: LRSchedule = LRSchedule.Cosine

    data: DataConfig = DataConfig()

    objectives_l1: ObjectivesConfig = ObjectivesConfig()

    eval_at_beginning: bool = False
    eval_during_training: bool = False

    save_every_n_epochs: int = 5
    eval_every_n_epochs: int = 20

    hjepa: HJEPAConfig = HJEPAConfig()

    resume_if_possible: bool = True
    compile_model: bool = True

    eval_cfg: EvalConfig = EvalConfig()

    def __post_init__(self):
        if self.quick_debug:
            self.data.quick_debug = True

            # Wall stuff
            self.data.dot_config.size = self.data.dot_config.batch_size
            self.data.wall_config.size = self.data.wall_config.batch_size
            self.eval_cfg.wall_planning.n_envs = 7
            self.eval_cfg.wall_planning.n_steps = 4
            self.eval_cfg.wall_planning.level1.sgd.n_iters = 2
            self.data.offline_wall_config.lazy_load = True

            # D4RL stuff
            self.data.d4rl_config.quick_debug = True
            self.data.d4rl_config.num_workers = 1
            self.eval_cfg.d4rl_planning.n_envs = 5
            self.eval_cfg.d4rl_planning.n_envs_batch_size = 2
            self.eval_cfg.d4rl_planning.replan_every = 1
            self.eval_cfg.d4rl_planning.n_steps = 6
            self.eval_cfg.d4rl_planning.plot_every = 1

        # Wall stuff
        self.eval_cfg.wall_planning.fix_wall = self.data.wall_config.fix_wall
        self.data.dot_config.n_steps = self.n_steps
        self.data.wall_config.n_steps = self.n_steps
        self.eval_cfg.wall_planning.padding = self.data.wall_config.border_wall_loc

        # D4RL stuff
        if self.hjepa.level1.backbone.arch in ["resnet18", "menet5"]:
            self.eval_cfg.d4rl_planning.image_obs = True
        # assert (
        #     self.eval_cfg.d4rl_planning.plot_every
        #     % self.eval_cfg.d4rl_planning.replan_every
        #     == 0
        # )
        self.eval_cfg.d4rl_planning.stack_states = self.data.d4rl_config.stack_states
        self.eval_cfg.d4rl_planning.img_size = self.data.d4rl_config.img_size

        # general
        self.val_n_steps = self.n_steps
        self.eval_cfg.eval_l2 = not self.hjepa.disable_l2

        self.output_path = os.path.join(
            self.output_root.rstrip("/"), self.output_dir.lstrip("/")
        )
        self.run_group = self.output_dir

        if self.train_only:
            self.eval_cfg.eval_l1 = False
            self.eval_cfg.probe_preds = False
            self.eval_cfg.probe_encoder = False
            self.eval_cfg.disable_planning = True

        if "test" in self.output_dir:
            test_dir = os.path.join(self.output_root.rstrip("/"), "test")
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

        self.objectives_l1.idm.action_dim = self.hjepa.level1.action_dim


class LimitedStepLR(torch.optim.lr_scheduler.StepLR):
    """Modified StepLR that stops decreasing after a maximum number of epochs"""
    def __init__(self, optimizer, step_size, gamma, max_epochs, last_epoch=-1, verbose=False):
        self.max_epochs = max_epochs
        super().__init__(optimizer, step_size, gamma, last_epoch, verbose)
    
    def get_lr(self):
        if self.last_epoch < self.max_epochs:
            return super().get_lr()
        else:
            # Keep LR the same after max_epochs
            return [group['lr'] for group in self.optimizer.param_groups]


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config

        Logger.run().initialize(
            output_path=self.config.output_path,
            wandb_enabled=self.config.wandb,
            project=config.run_project,
            name=config.run_name,
            group=config.run_group,
            config=dataclasses.asdict(config),
        )

        seed_everything(config.seed)

        self.sample_step = 0
        self.epoch = 0
        self.step = 0

        # create data
        datasets = DatasetFactory(
            config.data,
            probing_cfg=config.eval_cfg.probing,
            disable_l2=config.hjepa.disable_l2,
        ).create_datasets()

        self.datasets = datasets

        self.ds = datasets.ds
        self.val_ds = datasets.val_ds

        # infer obs shape
        sample_data = next(iter(self.ds))
        input_dim = sample_data.states.shape[2:]
        print("Inferred input_dim:", input_dim)
        if len(input_dim) == 1:
            input_dim = input_dim[0]

        # check if proprioceptive states are used
        use_propio_pos = (
            hasattr(sample_data, "propio_pos")
            and sample_data.propio_pos is not None
            and bool(sample_data.propio_pos.shape[-1])
        )
        use_propio_vel = (
            hasattr(sample_data, "propio_vel")
            and sample_data.propio_vel is not None
            and bool(sample_data.propio_vel.shape[-1])
        )

        # create model
        self.model = HJEPA(
            config.hjepa,
            input_dim=input_dim,
            normalizer=self.ds.normalizer,
            use_propio_pos=use_propio_pos,
            use_propio_vel=use_propio_vel,
        )

        self.model = self.model.cuda()

        # create objectives
        self.objectives_l1 = self.config.objectives_l1.build_objectives_list(
            name_prefix="l1", repr_dim=self.model.level1.spatial_repr_dim
        )
        # other stuff...

        load_result = self.maybe_load_model()

        if (
            config.eval_only
            and not config.eval_cfg.probing.full_finetune
            and not load_result
        ):
            print("WARN: probing a random network. Is that intentional?")

        assert not (self.config.hjepa.train_l1 and self.config.hjepa.freeze_l1)

        if self.config.hjepa.freeze_l1:
            print("freezing first level weights")
            for m in self.model.level1.modules():
                for p in m.parameters():
                    p.requires_grad = False

        print(self.model)
        self.n_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print("number of params:", self.n_parameters)

        l1_predictor_n_parameters = sum(
            p.numel()
            for p in self.model.level1.predictor.parameters()
            if p.requires_grad
        )
        print("number of l1 predictor params:", l1_predictor_n_parameters)

        l1_backbone_n_parameters = sum(
            p.numel()
            for p in self.model.level1.backbone.parameters()
            if p.requires_grad
        )
        print("number of l1 backbone params:", l1_backbone_n_parameters)

        Logger.run().log_summary(
            {
                "n_params": self.n_parameters,
            }
        )

        self.metric_tracker = MetricTracker(window_size=100)

        if self.config.compile_model:
            print("compiling model")
            c_time = time.time()
            self.model = torch.compile(self.model)
            print(f"compilation finished after {time.time() - c_time:.3f}s")

    def maybe_resume(self):
        if not os.path.exists(self.config.output_path):
            return False
        latest_checkpoint = utils.pick_latest_model(self.config.output_path)
        if latest_checkpoint is None:
            return False
        print("resuming from", latest_checkpoint)
        checkpoint = torch.load(latest_checkpoint)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]
        self.sample_step = checkpoint["sample_step"]
        print("resumed from epoch", self.epoch, "step", self.step)

    def maybe_load_model(self):
        if self.config.load_checkpoint_path is not None:
            checkpoint = torch.load(self.config.load_checkpoint_path)
            state_dict = checkpoint["model_state_dict"]
            # remove "_orig_mod." prefix from the keys
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            # remove all posterior parameters (incompatible because we don't
            # use it in l1).

            if (
                "backbone.layer1.0.weight" in state_dict
            ):  # this is jepa only model (legacy)
                res = self.model.level1.load_state_dict(state_dict)
            else:
                if self.config.load_l1_only:
                    for k in list(state_dict.keys()):
                        # 1. remove all posterior parameters
                        # (incompatible because we don't use it in l1).
                        # 2. remove everything belonging to l2
                        if "decoder" in k:  # this is for loading RSSM
                            del state_dict[k]
                res = self.model.load_state_dict(state_dict, strict=False)
            assert (
                len(res.unexpected_keys) == 0
            ), f"Unexpected keys when loading weights: {res.unexpected_keys}"
            print(f"loaded model from {self.config.load_checkpoint_path}")
            return True
        return False

    def train(self):
        self.optimizer = OptimizerFactory(
            model=self.model,
            optimizer_type=self.config.optimizer_type,
            base_lr=self.config.base_lr,
        ).create_optimizer()

        if self.config.resume_if_possible:
            if self.maybe_resume():
                print("resuming training")

        scheduler = Scheduler(
            schedule=self.config.optimizer_schedule,
            base_lr=self.config.base_lr,
            data_loader=self.ds,
            epochs=self.config.epochs,
            optimizer=self.optimizer,
        )

        first_step = None

        if self.config.eval_at_beginning and not self.config.quick_debug:
            self.validate()

        for epoch in tqdm(range(self.epoch, self.config.epochs + 1), desc="Epoch"):
            self.epoch = epoch
            end_time = time.time()
            for step, batch in (
                pbar := tqdm(
                    enumerate(self.ds, start=epoch * len(self.ds)),
                    desc="Batch",
                    total=len(self.ds),
                    maxinterval=10,
                )
            ):
                if first_step is None:
                    first_step = step
                start_time = time.time()
                if end_time is not None:
                    # data time is the time it took to load the data
                    # (which is the time between the end of the previous
                    # batch and the start of this batch)
                    data_time = start_time - end_time
                else:
                    data_time = None

                # move to cuda and swap batch and time
                s = batch.states.cuda().transpose(0, 1)
                a = batch.actions.cuda().transpose(0, 1)

                lr = scheduler.adjust_learning_rate(step)

                self.sample_step += s.shape[1]
                self.step = step

                self.optimizer.zero_grad()

                optional_fields = get_optional_fields(batch, device=s.device)

                forward_result = self.model.forward_posterior(s, a, **optional_fields)

                loss_infos = []

                if self.config.hjepa.train_l1:
                    loss_infos += [
                        objective(batch, [forward_result.level1])
                        for objective in self.objectives_l1
                    ]

                total_loss = sum([loss_info.total_loss for loss_info in loss_infos])
                if total_loss.isnan():
                    raise RuntimeError("NaN loss")
                total_loss.backward()
                self.optimizer.step()
                _log_grad_update_stats(self.model, self.optimizer, self.step)
                self.model.update_ema()  # if ema is enabled, update ema encoder

                train_time = time.time() - start_time
                log_start_time = time.time()

                self.metric_tracker.update("train_time", train_time)
                self.metric_tracker.update("data_time", data_time)

                if self.config.quick_debug or (step % 100 == 0):
                    metric_log = self.metric_tracker.build_log_dict()
                    pbar.set_description(
                        f"Loss: {total_loss.item():.4f}, "
                        f"train: {metric_log['train_time/mean']:.3f}s, "
                        f"data: {metric_log['data_time/mean']:.3f}s, "
                        f"log: {metric_log['log_time/mean'] if 'log_time/mean' in metric_log else 0:.3f}s"  # noqa
                    )
                    log_dict = {}

                    for loss_info in loss_infos:
                        if hasattr(loss_info, "build_log_dict"):
                            log_dict.update(loss_info.build_log_dict())

                    if data_time is not None:
                        log_dict["data_time"] = data_time

                    Logger.run().log(
                        {
                            "sample_step": self.sample_step,
                            "loss": total_loss.item(),
                            "learning_rate": lr,
                            "custom_step": step,
                            "epoch": epoch,
                            **log_dict,
                            **metric_log,
                        },
                        commit=False,
                    )
                    Logger.run().commit()

                    if step - first_step == 5:
                        return

                self.metric_tracker.update("log_time", time.time() - log_start_time)
                end_time = time.time()

            if (
                self.epoch % self.config.save_every_n_epochs == 0 and self.epoch > 0
            ) or self.epoch >= self.config.epochs:
                self.save_model()

            if (
                self.epoch % self.config.eval_every_n_epochs == 0
                and self.config.eval_during_training
            ) or self.epoch >= self.config.epochs:
                self.validate()

    @torch.no_grad()
    def eval_on_objectives(self):
        if self.val_ds is None:
            return

        losses = {}

        for step, batch in tqdm(enumerate(self.val_ds)):
            # move to cuda and swap batch and time
            s = batch.states.cuda().transpose(0, 1)
            a = batch.actions.cuda().transpose(0, 1)

            optional_fields = get_optional_fields(batch, device=s.device)

            forward_result = self.model.forward_posterior(s, a, **optional_fields)

            loss_infos = []

            if self.config.hjepa.train_l1:
                loss_infos += [
                    objective(batch, [forward_result.level1])
                    for objective in self.objectives_l1
                ]

            for loss_info in loss_infos:
                for attr in loss_info._fields:
                    val = getattr(loss_info, attr)
                    if isinstance(val, str):
                        continue

                    assert isinstance(val, torch.Tensor)
                    assert len(val.shape) == 0

                    key = f"val_epoch_{self.epoch}_{loss_info.name_prefix}/{loss_info.loss_name}_{attr}"

                    if key in losses:
                        losses[key].append(val.item())
                    else:
                        losses[key] = [val.item()]

            if step > 2:
                break

        # take mean over batches
        for key, val in losses.items():
            losses[key] = sum(val) / len(val)

        Logger.run().log(losses)
        Logger.run().commit()

    def validate(self):
        training = self.model.training
        self.model.eval()

        # evals on the same objectives used for training
        self.eval_on_objectives()

        # create evaluator (for both probing and planning)
        self.evaluator = Evaluator(
            config=self.config.eval_cfg,
            model=self.model,
            quick_debug=self.config.quick_debug,
            normalizer=self.ds.normalizer,
            epoch=self.epoch,
            probing_datasets=self.datasets.probing_datasets,
            l2_probing_datasets=self.datasets.l2_probing_datasets,
            load_checkpoint_path=self.config.load_checkpoint_path,
            output_path=self.config.output_path,
            data_config=self.config.data.wall_config,  # TODO: refactor name to data_config
        )

        log_dict = self.evaluator.evaluate()
        log_dict["custom_step"] = self.step

        Logger.run().log(log_dict)
        Logger.run().log_summary(log_dict)
        Logger.run().save_summary(
            f"summary_epoch={self.epoch}_sample_step={self.sample_step}.json"
        )

        for v in log_dict.values():
            if isinstance(v, matplotlib.figure.Figure):
                plt.close(v)

        if training:
            # if model is previously in training
            self.model.train()

        return

    def save_model(self):
        if self.config.output_path is not None:
            os.makedirs(self.config.output_path, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": self.epoch,
                    "step": self.step,
                    "sample_step": self.sample_step,
                },
                os.path.join(
                    self.config.output_path,
                    f"epoch={self.epoch}_sample_step={self.sample_step}.ckpt",
                ),
            )


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
    same_room_bonus = torch.where(same_room, torch.tensor(50.0, device=dot_position.device), torch.tensor(0.0, device=dot_position.device))
    
    return distance_reward + same_room_bonus


def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns"""
    returns = []
    R = 0
    
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    
    returns = torch.tensor(returns)
    
    # Normalize returns
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    return returns


def compute_advantages(rewards, values, gamma=0.99, lambda_=0.95):
    """Compute generalized advantage estimates"""
    advantages = []
    advantage = 0
    next_value = 0
    
    for r, v in zip(reversed(rewards), reversed(values)):
        td_error = r + gamma * next_value - v
        advantage = td_error + gamma * lambda_ * advantage
        next_value = v
        advantages.insert(0, advantage)
    
    advantages = torch.tensor(advantages)
    
    # Normalize advantages
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages


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
        {'params': model.next_goal_predictor.parameters(), 'lr': args.policy_lr}
    ])
    
    # Create learning rate schedulers
    # We'll implement a custom scheduler function to manage the different learning rates
    def adjust_learning_rates(epoch):
        """Adjust learning rates based on epoch and component"""
        # Get base learning rates from parameter groups
        encoder_lr = optimizer.param_groups[0]['lr']
        dynamics_lr = optimizer.param_groups[1]['lr']
        policy_lr = optimizer.param_groups[2]['lr']
        
        # Apply encoder LR schedule: reduce to 1/3 after the first epoch
        if epoch == 1:
            encoder_lr = encoder_lr / 3
            
        # Apply dynamics LR schedule: reduce to 1/2 after the first epoch
        if epoch == 1:
            dynamics_lr = dynamics_lr / 2
        
        # Update optimizer parameter groups
        optimizer.param_groups[0]['lr'] = encoder_lr
        optimizer.param_groups[1]['lr'] = dynamics_lr
        optimizer.param_groups[2]['lr'] = policy_lr
        
        return encoder_lr, dynamics_lr, policy_lr
    
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
        for epoch in range(start_epoch, args.epochs):
            total_reward = 0
            total_policy_loss = 0
            total_dynamics_loss = 0
            total_next_state_loss = 0
            total_on_the_same_page_loss = 0
            num_episodes = 0
            
            # Adjust learning rates for this epoch
            encoder_lr, dynamics_lr, policy_lr = adjust_learning_rates(epoch)
            
            # Log current learning rates
            print(f"Epoch {epoch+1}/{args.epochs} - Learning rates: Encoder={encoder_lr:.2e}, "
                  f"Dynamics={dynamics_lr:.2e}, Policy={policy_lr:.2e}")
                  
            writer.add_scalar('LearningRate/encoder', encoder_lr, epoch)
            writer.add_scalar('LearningRate/dynamics', dynamics_lr, epoch)
            writer.add_scalar('LearningRate/policy', policy_lr, epoch)
            
            # Process episodes in batches
            progress_bar = tqdm(total=args.episodes_per_epoch, desc=f"Epoch {epoch+1}/{args.epochs}")
            episode_idx = 0
            
            # Set model to train mode for the main thread
            model.train()
            
            while episode_idx < args.episodes_per_epoch:
                # Collect batch_size trajectories (or whatever remains in the epoch)
                batch_size = min(args.batch_size, args.episodes_per_epoch - episode_idx)
                
                # Get batch from parallel collector (this is prefetched in parallel)
                trajectories = collector.get_batch(batch_size)
                valid_episodes = len(trajectories)
                
                # Skip if no valid episodes
                if valid_episodes == 0:
                    # Just in case we're not getting episodes, sleep briefly
                    time.sleep(0.1)
                    continue
                
                # Process collected trajectories
                batch_states = []
                batch_next_states = []
                batch_actions = []
                batch_log_probs = []
                batch_returns = []
                batch_rewards = []
                batch_next_goals = []
                
                for trajectory in trajectories:
                    states = trajectory['states']
                    actions = trajectory['actions']
                    log_probs = trajectory['log_probs']
                    next_goals = trajectory['next_goals']
                    rewards = trajectory['rewards']
                    
                    # Calculate rewards for this episode
                    episode_reward = sum(rewards)
                    total_reward += episode_reward
                    
                    # Calculate returns for this episode
                    returns = compute_returns(rewards, gamma=args.gamma)
                    
                    # Convert data to tensors and move to device with non-blocking transfers
                    states_tensor = []
                    for s in states[:-1]:  # All states except the last one
                        if isinstance(s, torch.Tensor):
                            s_tensor = s.to(device=device, non_blocking=True)
                        else:
                            s_tensor = torch.tensor(s, device=device)
                        
                        # No need to cast to bf16 - autocast will handle precision
                        s_tensor = s_tensor.float()
                        
                        states_tensor.append(s_tensor)
                    
                    next_states_tensor = []
                    for s in states[1:]:  # All states except the first one
                        if isinstance(s, torch.Tensor):
                            s_tensor = s.to(device=device, non_blocking=True)
                        else:
                            s_tensor = torch.tensor(s, device=device)
                        
                        # No need to cast to bf16 - autocast will handle precision
                        s_tensor = s_tensor.float()
                        
                        next_states_tensor.append(s_tensor)
                    
                    actions_tensor = []
                    for a in actions:
                        if isinstance(a, torch.Tensor):
                            a_tensor = a.to(device=device, non_blocking=True)
                        else:
                            a_tensor = torch.tensor(a, device=device)
                        
                        # No need to cast to bf16 - autocast will handle precision
                        a_tensor = a_tensor.float()
                            
                        actions_tensor.append(a_tensor)

                    next_goals_tensor = []
                    for ng in next_goals:
                        if isinstance(ng, torch.Tensor):
                            ng_tensor = ng.to(device=device, non_blocking=True)
                        else:
                            ng_tensor = torch.tensor(ng, device=device)

                        # No need to cast to bf16 - autocast will handle precision
                        ng_tensor = ng_tensor.float()
                            
                        next_goals_tensor.append(ng_tensor)
                    
                    # Add to batch data
                    batch_states.extend(states_tensor)
                    batch_next_states.extend(next_states_tensor)
                    batch_actions.extend(actions_tensor)
                    batch_log_probs.extend(log_probs)
                    batch_returns.extend(returns.tolist())
                    batch_rewards.append(episode_reward)
                    batch_next_goals.extend(next_goals_tensor)
                
                # Prefetch done, update episode count
                episode_idx += valid_episodes
                num_episodes += valid_episodes
                progress_bar.update(valid_episodes)
                
                # Log individual episode metrics for the first episode in the batch
                writer.add_scalar('Reward/individual_episode', batch_rewards[0], global_step + episode_idx)
                
                # Convert batch data to tensors - always use float32 for loss computation tensors
                # Autocast will handle the conversion to lower precision during forward pass
                batch_returns_tensor = torch.tensor(batch_returns, dtype=torch.float32, device=device)

                # We will compute log‑probs inside the dynamics loop to avoid
                # encoding the same observation twice.
                
                # Use autocast for mixed precision training
                with torch.amp.autocast('cuda', enabled=amp_dtype is not None, dtype=amp_dtype):
                    # Policy loss (policy gradient)
                    policy_log_probs = []
                    dynamics_loss = 0
                    next_state_loss = 0
                    on_the_same_page_loss = 0

                    for state, next_state, action, next_goal in zip(batch_states, batch_next_states, batch_actions, batch_next_goals):
                        # Encode current and next state
                        z_t = model.encode(state.unsqueeze(0))
                        z_next_actual = model.encode(next_state.unsqueeze(0))

                        # Log‑prob of stored goal
                        log_prob = model.next_goal_predictor.log_prob(z_t, next_goal.unsqueeze(0))
                        policy_log_probs.append(log_prob.squeeze(0))

                        # Predict next state using dynamics model directly
                        a_t = action.unsqueeze(0)
                        z_next_pred = model.dynamics(z_t, a_t)

                        # Add to dynamics loss
                        dynamics_loss += F.mse_loss(z_next_pred, z_next_actual)

                        # Conditional losses based on args
                        if args.use_next_state_loss:
                            next_state_loss += 0 # not implemented yet
                            print("Next state loss is not implemented yet")

                        if args.use_same_page_loss:
                            on_the_same_page_loss += F.mse_loss(next_goal, z_next_pred)

                    batch_log_probs_tensor = torch.stack(policy_log_probs)
                    policy_loss = -(batch_log_probs_tensor * batch_returns_tensor).mean()

                    if len(batch_states) > 0:
                        dynamics_loss = dynamics_loss / len(batch_states)
                        if args.use_next_state_loss:
                            next_state_loss = next_state_loss / len(batch_states)
                        if args.use_same_page_loss:
                            on_the_same_page_loss = on_the_same_page_loss / len(batch_states)

                    # Total loss
                    loss = policy_loss + args.lambda_dynamics * dynamics_loss
                    
                    # Add conditional losses
                    if args.use_next_state_loss:
                        loss += args.lambda_dynamics * next_state_loss
                    
                    if args.use_same_page_loss:
                        loss += args.lambda_dynamics * on_the_same_page_loss
                
                # Backward pass with scaler for FP16 or regular backward for BF16/FP32
                if amp_dtype == torch.float16:
                    # Use scaler for FP16 precision
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    _log_grad_update_stats(model, optimizer, global_step)
                else:
                    # Regular backward for BF16 or FP32
                    loss.backward()
                    optimizer.step()
                    _log_grad_update_stats(model, optimizer, global_step)
                
                # Update statistics
                total_policy_loss += policy_loss.item() * valid_episodes
                total_dynamics_loss += dynamics_loss.item() * valid_episodes
                if args.use_next_state_loss:
                    total_next_state_loss += next_state_loss.item() * valid_episodes
                if args.use_same_page_loss:
                    total_on_the_same_page_loss += on_the_same_page_loss.item() * valid_episodes
                
                # Log batch metrics
                writer.add_scalar('Loss/policy', policy_loss.item(), global_step)
                writer.add_scalar('Loss/dynamics', dynamics_loss.item(), global_step)
                if args.use_next_state_loss:
                    writer.add_scalar('Loss/next_state', next_state_loss.item(), global_step)
                if args.use_same_page_loss:
                    writer.add_scalar('Loss/on_the_same_page', on_the_same_page_loss.item(), global_step)
                writer.add_scalar('Loss/total', loss.item(), global_step)
                writer.add_scalar('Reward/batch_mean', sum(batch_rewards)/len(batch_rewards), global_step)
                
                global_step += 1
            
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
    parser.add_argument('--episodes_per_epoch', type=int, default=128, help='Number of episodes per epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of trajectories to process in a batch')
    parser.add_argument('--max_steps_per_episode', type=int, default=40, help='Maximum steps per episode')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of action samples to evaluate in parallel')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--lambda_dynamics', type=float, default=0.5, help='Weight for dynamics loss')
    parser.add_argument('--max_step_norm', type=float, default=15, help='Maximum step norm')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of parallel workers for episode collection')
    parser.add_argument('--use_gpu_inference', type=bool, default=True, help='Use GPU for inference during rollout')
    parser.add_argument('--use_next_state_loss', type=bool, default=False, help='Use next state prediction loss')
    parser.add_argument('--use_same_page_loss', type=bool, default=False, help='Use on-the-same-page loss between next goal and dynamics')
    
    # Optimizer parameters
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='Learning rate for encoder')
    parser.add_argument('--dynamics_lr', type=float, default=5e-4, help='Learning rate for dynamics model')
    parser.add_argument('--policy_lr', type=float, default=1e-3, help='Learning rate for policy')
    
    # Precision parameters
    parser.add_argument('--bf16', type=bool, default=False, help='Use BFloat16 mixed precision for training')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run training on')
    parser.add_argument('--output_dir', type=str, default='output_large_model3', help='Directory to save model and logs')
    parser.add_argument('--resume', type=bool, default=False, help='Resume training from checkpoint')
    
    return parser.parse_args()


def main(config: TrainConfig):
    torch.set_num_threads(1)
    trainer = Trainer(config)

    if config.eval_only and not config.quick_debug:
        trainer.validate()
    else:
        trainer.train()

    if config.quick_debug:
        trainer.validate()


if __name__ == "__main__":
    args = parse_args()
    train_pldm(args)

# -----------------------------------------------------------------------------
# Utility: log gradient/update statistics
# -----------------------------------------------------------------------------
def _log_grad_update_stats(model, optimizer, step_idx=0):
    """Print average |grad| and |update| (lr*|grad|) for main components.

    Components are inferred from parameter names:
      encoder.*  -> 'encoder'
      dynamics.* -> 'dynamics'
      next_goal_predictor.* -> 'predictor'
      level1.*  -> 'level1'
      level2.*  -> 'level2'
      otherwise -> 'other'
    """
    # Map param -> lr for quick lookup
    param_lr = {}
    for group in optimizer.param_groups:
        lr = group.get('lr', 0.0)
        for p in group['params']:
            param_lr[id(p)] = lr

    stats = {}
    def _add(comp, g, u):
        if comp not in stats:
            stats[comp] = {'g': 0.0, 'u': 0.0, 'n': 0}
        stats[comp]['g'] += g
        stats[comp]['u'] += u
        stats[comp]['n'] += 1

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        comp = 'other'
        if name.startswith('encoder'):
            comp = 'encoder'
        elif name.startswith('dynamics'):
            comp = 'dynamics'
        elif name.startswith('next_goal_predictor'):
            comp = 'predictor'
        elif name.startswith('level1'):
            comp = 'level1'
        elif name.startswith('level2'):
            comp = 'level2'

        lr = param_lr.get(id(p), 0.0)
        grad_mean = p.grad.abs().mean().item()
        upd_mean = lr * grad_mean
        _add(comp, grad_mean, upd_mean)

    parts = []
    for comp, d in stats.items():
        if d['n'] == 0:
            continue
        parts.append(f"{comp}: g={d['g']/d['n']:.2e}, u={d['u']/d['n']:.2e}")
    if parts:
        print(f"[GradStats step={step_idx}] " + " | ".join(parts))
