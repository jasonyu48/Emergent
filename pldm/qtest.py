import os
import torch
import torch.nn.functional as F
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio
from torchvision import transforms as T
from PIL import Image
import torch.nn as nn
import configparser # For reading experiment_info.txt

from pldm_envs.wall.wall import DotWall
from pldm.qmodel import PLDMModel, DEFAULT_ENCODING_DIM, DEFAULT_NUM_ACTIONS # Import defaults


def check_for_nan(tensor, name="tensor", raise_error=True):
    """Check if a tensor contains any NaN values and raise an exception if found."""
    if isinstance(tensor, torch.Tensor) and torch.isnan(tensor).any():
        message = f"NaN detected in {name}"
        if raise_error:
            raise ValueError(message)
        else:
            print(f"WARNING: {message}")
    return tensor


def make_gif(frames, filename, fps=10):
    """Convert a list of frames into a GIF and save it to a file."""
    images = []
    transform = T.ToPILImage()

    for frame in frames:
        if isinstance(frame, torch.Tensor):
            if frame.ndimension() == 2:
                frame = frame.unsqueeze(0).repeat(3, 1, 1)
            elif frame.shape[0] == 1:
                frame = frame.repeat(3, 1, 1)
            image = transform(frame.cpu())
        elif isinstance(frame, np.ndarray):
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=2)
            elif frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            if frame.dtype in (np.float32, np.float64) and frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            image = Image.fromarray(frame.astype(np.uint8))
        else:
            raise TypeError(f"Unsupported frame type: {type(frame)}")
        images.append(image)

    imageio.mimsave(filename, images, fps=fps)


def visualize_trajectory(frames, recon_frames, dot_positions, target_position, wall_x, output_path):
    """Visualize the trajectory with dot positions and reconstruction frames."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(exist_ok=True, parents=True)

    frames_np = [frame.cpu().numpy() for frame in frames]
    # Ensure recon_frames are also numpy and on CPU before processing
    recon_frames_np = [(frame.cpu().numpy() if isinstance(frame, torch.Tensor) else frame) for frame in recon_frames]
    plt_frames = []

    for i, (frame, recon_frame) in enumerate(zip(frames_np, recon_frames_np)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        frame_norm = frame / 255.0 if frame.max() > 1.0 else frame
        axes[0].imshow(frame_norm.transpose(1, 2, 0))
        axes[0].set_title(f"Current Observation (Step {i})")
        axes[0].axis('off')

        recon_norm = recon_frame / 255.0 if recon_frame.max() > 1.0 else recon_frame
        axes[1].imshow(recon_norm.transpose(1, 2, 0))
        axes[1].set_title(f"Reconstruction (Step {i})")
        axes[1].axis('off')

        axes[2].imshow(np.zeros((64, 64, 3)), cmap='gray')
        wall_width = 3
        half_width = wall_width // 2
        axes[2].axvline(x=wall_x - half_width, color='white', linewidth=2)
        axes[2].axvline(x=wall_x + half_width, color='white', linewidth=2)

        for j, pos in enumerate(dot_positions[:i + 1]):
            alpha = 0.3 + 0.7 * (j / (i + 1))
            axes[2].scatter(pos[0], pos[1], color='blue', alpha=alpha, s=30)
            if j > 0:
                prev_pos = dot_positions[j - 1]
                axes[2].arrow(prev_pos[0], prev_pos[1],
                              pos[0] - prev_pos[0], pos[1] - prev_pos[1],
                              color='blue', alpha=alpha, width=0.5, head_width=2)

        axes[2].scatter(target_position[0], target_position[1], color='red', s=50, marker='x')
        axes[2].set_title("Trajectory")
        axes[2].axis('off')

        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        plt_frames.append(img)
        plt.close(fig)

    make_gif(plt_frames, output_path, fps=2)


def rollout_episode(model, env, max_steps, device, max_step_norm):
    """Rollout one episode using the same reward scheme as qtrain."""
    BONUS_CROSS_DOOR = 100.0
    BONUS_HIT_TARGET = 2000.0
    STEP_PENALTY = 0.05
    WALL_PENALTY = 2.0
    TINY_MOVE_THRESH = 0.3
    TINY_MOVE_PENALTY = 1.0
    DIR_REWARD_LAMBDA = 2.0
    TARGET_SCALE = 10.0

    obs, info = env.reset()
    door_center = torch.stack([env.wall_x, env.hole_y]).to(env.device, dtype=torch.float32)

    states, actions, rewards = [obs], [], []
    reconstructions = []
    dot_positions = [env.dot_position.cpu().numpy()]

    crossed_door = False
    prev_dist_to_door = torch.norm(env.dot_position - door_center).item()
    prev_dist_to_target = torch.norm(env.dot_position - env.target_position).item()
    prev_in_left = env.dot_position[0] < env.left_wall_x

    with torch.no_grad():
        obs_tensor = (obs if isinstance(obs, torch.Tensor) else
                      torch.tensor(obs, dtype=torch.float32, device=device)).unsqueeze(0)
        z_t = model.encode(obs_tensor)
        reconstructions.append(model.decode(z_t).squeeze(0).cpu())

    done = truncated = False
    episode_reward = 0.0

    for step in range(max_steps):
        if done or truncated:
            break

        with torch.no_grad():
            # Get continuous action directly from the policy via PLDMModel
            continuous_action_tensor, _, _ = model.get_action_and_log_prob(z_t, sample=True) # Sample=True for eval
            # No separate search_action needed
        
        action_to_step = continuous_action_tensor.cpu().numpy()[0] # Get batch 0, convert to numpy

        obs, _, done, truncated, info = env.step(action_to_step)
        dot_positions.append(env.dot_position.cpu().numpy())

        dot_left = env.dot_position[0] < env.left_wall_x
        target_left = env.target_position[0] < env.left_wall_x
        same_room = bool(dot_left == target_left)

        if not same_room:  # -------- 阶段 A
            curr_dist_to_door = torch.norm(env.dot_position - door_center).item()
            step_reward = 2.0 * (prev_dist_to_door - curr_dist_to_door)
            prev_dist_to_door = curr_dist_to_door

            crossed = (prev_in_left != (env.dot_position[0] < env.left_wall_x))
            if crossed and not crossed_door:
                step_reward += BONUS_CROSS_DOOR
                crossed_door = True

        else:  # -------- 阶段 B
            curr_dist_to_target = torch.norm(env.dot_position - env.target_position).item()
            step_reward = TARGET_SCALE * (prev_dist_to_target - curr_dist_to_target)
            step_reward += 20.0 * np.exp(-curr_dist_to_target / 10)

            v_a = action_to_step / (np.linalg.norm(action_to_step) + 1e-6) # Use action_to_step
            v_t = (env.target_position.cpu().numpy() - env.dot_position.cpu().numpy())
            dist_to_target = np.linalg.norm(v_t)
            v_t = v_t / (dist_to_target + 1e-6)
            direction_reward = DIR_REWARD_LAMBDA * np.exp(-0.1 * dist_to_target) * np.dot(v_a, v_t)
            step_reward += direction_reward

            hit_reward = BONUS_HIT_TARGET * np.exp(-curr_dist_to_target)
            step_reward += hit_reward if curr_dist_to_target < 3.0 else 0
            if curr_dist_to_target < 2.5:
                done = True

            prev_dist_to_target = curr_dist_to_target

        step_reward -= STEP_PENALTY
        if np.linalg.norm(action_to_step) < TINY_MOVE_THRESH:
            step_reward -= TINY_MOVE_PENALTY
        if env.position_history and torch.all(env.position_history[-1] == env.dot_position):
            step_reward -= WALL_PENALTY

        rewards.append(step_reward)
        episode_reward += step_reward

        with torch.no_grad():
            obs_tensor = (obs if isinstance(obs, torch.Tensor) else
                          torch.tensor(obs, dtype=torch.float32, device=device)).unsqueeze(0)
            z_t = model.encode(obs_tensor)
            # Check if decoder exists and is not None before calling
            if hasattr(model, 'decoder') and model.decoder is not None:
                 reconstructions.append(model.decode(z_t).squeeze(0).cpu())
            else:
                 # Append a placeholder if no decoder, e.g., a zero tensor or skip reconstruction visualization
                 placeholder_recon = torch.zeros_like(obs_tensor.squeeze(0).cpu()) # Match shape of obs
                 reconstructions.append(placeholder_recon)

        states.append(obs)
        actions.append(action_to_step) # Store the continuous action taken
        prev_in_left = env.dot_position[0] < env.left_wall_x

    final_distance = torch.norm(env.dot_position - env.target_position).item()
    return {
        "states": states,
        "actions": actions,
        "dot_positions": dot_positions,
        "rewards": rewards,
        "reconstructions": reconstructions,
        "total_reward": episode_reward,
        "done": done,
        "length": len(states) - 1,
        "final_distance": final_distance,
        "same_room": same_room
    }


class NanDetector:
    """Context manager to detect NaNs in model outputs during forward passes."""
    def __init__(self, model, verbose=True):
        self.model = model
        self.hooks = []
        self.verbose = verbose

    def __enter__(self):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor) and torch.isnan(output).any():
                module_name = module.__class__.__name__
                raise ValueError(f"NaN detected in output of {module_name}")
            return output
        for _, module in self.model.named_modules():
            self.hooks.append(module.register_forward_hook(hook))
        if self.verbose:
            print("NaN detection enabled for all model modules")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for hook in self.hooks:
            hook.remove()
        if self.verbose:
            print("NaN detection disabled")


def load_config_from_experiment_info(exp_info_path: Path) -> dict:
    """Loads configuration from experiment_info.txt file."""
    config = {}
    if not exp_info_path.exists():
        print(f"Warning: experiment_info.txt not found at {exp_info_path}. Using defaults or CLI args where possible.")
        return config

    print(f"Loading configuration from {exp_info_path}")
    with open(exp_info_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("--"):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().lstrip('--')
                    value_str = parts[1].strip()
                    # Attempt to infer type
                    if value_str.lower() == 'true':
                        config[key] = True
                    elif value_str.lower() == 'false':
                        config[key] = False
                    elif value_str.lower() == 'none':
                        config[key] = None
                    else:
                        try:
                            if '.' in value_str or 'e' in value_str.lower():
                                config[key] = float(value_str)
                            else:
                                config[key] = int(value_str)
                        except ValueError:
                            config[key] = value_str # Store as string if conversion fails
    return config


def evaluate_model(model_path_str: str, output_dir_str:str ='test_output', device_str:str='cpu', 
                   num_episodes:int=5, max_steps:int=50, eval_max_step_norm:float=None):
    """Evaluate the trained model on the DotWall environment."""
    model_path = Path(model_path_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device(device_str)
    
    # Load experiment config to get model parameters
    exp_info_path = model_path.parent / "experiment_info.txt"
    train_args = load_config_from_experiment_info(exp_info_path)

    # Determine model parameters from loaded config or use defaults
    model_encoding_dim = train_args.get('model_encoding_dim', DEFAULT_ENCODING_DIM)
    num_actions = train_args.get('num_actions', DEFAULT_NUM_ACTIONS)
    encoder_embedding_dim = train_args.get('encoder_embedding_dim', 256) # Default from old qtrain
    encoder_type = train_args.get('encoder_type', 'vit')
    policy_temperature = train_args.get('policy_temperature', 1.0)
    # max_step_norm for action grid in PLDMModel should come from training config
    # If eval_max_step_norm is provided for env, use that, otherwise use training's max_step_norm
    pldm_max_step_norm = train_args.get('max_step_norm', 15.0)
    env_max_step_norm = eval_max_step_norm if eval_max_step_norm is not None else pldm_max_step_norm

    env = DotWall(max_step_norm=env_max_step_norm, door_space=8)

    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    print(f"Creating model with: model_encoding_dim={model_encoding_dim}, num_actions={num_actions}, encoder_type={encoder_type}")
    model = PLDMModel(
        img_size=env.img_size, # Get from env
        in_channels=3,         # DotWall specific
        encoding_dim=model_encoding_dim,
        num_actions=num_actions,
        encoder_embedding_dim=encoder_embedding_dim,
        encoder_type=encoder_type,
        policy_temp=policy_temperature,
        max_step_norm=pldm_max_step_norm # For action_grid generation
        # Other params like action_dim_continuous, multipliers use defaults in PLDMModel
    ).to(device)

    model.print_parameter_count()

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model from model_state_dict")
    elif checkpoint:
        try:
            model.load_state_dict(checkpoint)
            print("Loaded model directly from checkpoint")
        except RuntimeError as e:
            print(f"Error loading state_dict directly: {e}")
            print("This might be due to model structure changes. Ensure the checkpoint matches the current model structure or was saved as 'model_state_dict'.")
            return 0.0, 0.0 # Cannot proceed
    else:
        print("Error: Checkpoint is empty or invalid.")
        return 0.0, 0.0

    model.eval()
    # print("Using parallel action search with", num_samples, "samples per step") # Obsolete
    # print(f"Action sampling strategy: {'quadrant-based' if use_quadrant else 'full action space'}") # Obsolete

    success_count = 0
    total_rewards = []

    with NanDetector(model):
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            try:
                result = rollout_episode(
                    model=model,
                    env=env,
                    max_steps=max_steps,
                    device=device,
                    max_step_norm=env_max_step_norm # Pass env's max_step_norm for rollout logic if needed
                )

                states = result['states']
                dot_positions = result['dot_positions']
                reconstructions = result['reconstructions']
                done = result['done']
                episode_reward = result['total_reward']
                final_distance = result['final_distance']
                same_room = result['same_room']

                if done:
                    success_count += 1
                total_rewards.append(episode_reward)

                target_position = env.target_position.cpu().numpy()
                visualize_trajectory(
                    states,
                    reconstructions,
                    dot_positions,
                    target_position,
                    env.wall_x.cpu().numpy(),
                    output_dir / f"episode_{episode + 1}.gif"
                )

                print(f"\nEpisode {episode + 1} summary:")
                print(f"  Length: {result['length']} steps")
                print(f"  Success: {done}")
                print(f"  Final distance to target: {final_distance:.4f}")
                print(f"  Reward total: {episode_reward:.2f}")
                print(f"  In same room as target: {same_room}")

                print("\nDetailed diagnostics:")
                print(f"  Average step reward: {np.mean(result['rewards']):.4f}")
                print(f"  Min step reward: {min(result['rewards']):.4f}")
                print(f"  Max step reward: {max(result['rewards']):.4f}")
                print(f"  Steps in correct room: {sum(1 for r in result['rewards'] if r > 0)}")
                print(f"  Average action magnitude: {np.mean([np.linalg.norm(a) for a in result['actions']]):.4f}")

            except ValueError as e:
                if "NaN detected" in str(e):
                    print(f"Episode {episode + 1} failed with error: {e}")
                    print("Skipping to next episode...")
                else:
                    raise

    if total_rewards:
        success_rate = success_count / num_episodes
        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"\nOverall performance:")
        print(f"  Success rate: {success_rate:.2f} ({success_count}/{num_episodes})")
        print(f"  Average reward: {avg_reward:.2f}")
        return success_rate, avg_reward
    else:
        print("\nNo successful evaluations completed.")
        return 0.0, 0.0


def parse_args():
    parser = argparse.ArgumentParser(description='Test PLDM model on DotWall environment')
    parser.add_argument('--model_path', type=str, default='output_pldm_refactored/best_model.pt')
    parser.add_argument('--output_dir', type=str, default='output_pldm_refactored_test')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--eval_max_step_norm', type=float, default=None, 
                        help='Max step norm for DotWall env during evaluation. If None, uses value from training config.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate_model(
        model_path_str=args.model_path,
        output_dir_str=args.output_dir,
        device_str=args.device,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        eval_max_step_norm=args.eval_max_step_norm
    )