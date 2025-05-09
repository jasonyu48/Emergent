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

from pldm_envs.wall.wall import DotWall
from pldm.qmodel import PLDMModel


def check_for_nan(tensor, name="tensor", raise_error=True):
    """Check if a tensor contains any NaN values and raise an exception if found.
    
    Args:
        tensor: The tensor to check
        name: Name of the tensor for error reporting
        raise_error: If True, raise an exception when NaN is found
        
    Returns:
        The input tensor
    """
    if isinstance(tensor, torch.Tensor) and torch.isnan(tensor).any():
        message = f"NaN detected in {name}"
        if raise_error:
            raise ValueError(message)
        else:
            print(f"WARNING: {message}")
    return tensor


def make_gif(frames, filename, fps=10):
    """
    Convert a list of frames into a GIF and save it to a file.

    Args:
        frames (list): List of image frames (either PyTorch tensors or NumPy arrays)
        filename (str or Path): Path to save the GIF.
        fps (int): Frames per second for the GIF.
    """
    images = []
    transform = T.ToPILImage()

    for frame in frames:
        if isinstance(frame, torch.Tensor):  # Handle PyTorch tensors
            if frame.ndimension() == 2:  # Convert grayscale to 3-channel RGB for consistency
                frame = frame.unsqueeze(0).repeat(3, 1, 1)
            elif frame.shape[0] == 1:  # Single channel, expand to RGB
                frame = frame.repeat(3, 1, 1)
            image = transform(frame.cpu())  # Convert to PIL Image
        elif isinstance(frame, np.ndarray):  # Handle NumPy arrays
            if frame.ndim == 2:  # Grayscale
                # Expand to RGB
                frame = np.stack([frame] * 3, axis=2)
            elif frame.ndim == 3 and frame.shape[2] == 4:  # RGBA
                # Convert to RGB
                frame = frame[:, :, :3]
            
            # Ensure values are in the right range for PIL
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
            
            # Create PIL image
            image = Image.fromarray(frame.astype(np.uint8))
        else:
            raise TypeError(f"Unsupported frame type: {type(frame)}")
            
        images.append(image)

    imageio.mimsave(filename, images, fps=fps)


def visualize_trajectory(frames, recon_frames, dot_positions, target_position, wall_x, output_path):
    """Visualize the trajectory with dot positions and reconstruction frames"""
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert frames to numpy arrays for visualization
    frames_np = [frame.cpu().numpy() for frame in frames]
    recon_frames_np = [frame.cpu().numpy() for frame in recon_frames]
    
    # Create GIF frames
    plt_frames = []
    
    for i, (frame, recon_frame) in enumerate(zip(frames_np, recon_frames_np)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot current observation - normalize to [0,1] range for imshow
        if frame.max() > 1.0:
            frame_norm = frame / 255.0
        else:
            frame_norm = frame
        axes[0].imshow(frame_norm.transpose(1, 2, 0))
        axes[0].set_title(f"Current Observation (Step {i})")
        axes[0].axis('off')
        
        # Plot reconstruction
        if recon_frame.max() > 1.0:
            recon_norm = recon_frame / 255.0
        else:
            recon_norm = recon_frame
        axes[1].imshow(recon_norm.transpose(1, 2, 0))
        axes[1].set_title(f"Reconstruction (Step {i})")
        axes[1].axis('off')
        
        # Plot trajectory
        axes[2].imshow(np.zeros((64, 64, 3)), cmap='gray')
        
        # Draw wall
        wall_width = 3  # Assuming wall width is 3
        half_width = wall_width // 2
        axes[2].axvline(x=wall_x - half_width, color='white', linewidth=2)
        axes[2].axvline(x=wall_x + half_width, color='white', linewidth=2)
        
        # Plot previous dot positions
        for j, pos in enumerate(dot_positions[:i+1]):
            alpha = 0.3 + 0.7 * (j / (i + 1))
            axes[2].scatter(pos[0], pos[1], color='blue', alpha=alpha, s=30)
            
            # Draw arrows for consecutive positions
            if j > 0:
                prev_pos = dot_positions[j-1]
                axes[2].arrow(
                    prev_pos[0], prev_pos[1],
                    pos[0] - prev_pos[0], pos[1] - prev_pos[1],
                    color='blue', alpha=alpha, width=0.5, head_width=2
                )
        
        # Plot target position
        axes[2].scatter(target_position[0], target_position[1], color='red', s=50, marker='x')
        axes[2].set_title("Trajectory")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Convert plot to image - fix for Matplotlib 3.8+ deprecation
        fig.canvas.draw()
        # Use buffer_rgba instead of tostring_rgb
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        
        plt_frames.append(img)
        
        plt.close(fig)
    
    # Save as GIF
    make_gif(plt_frames, output_path, fps=2)


def calculate_distance_reward(dot_position, target_position, wall_x, wall_width, base_reward):
    """Calculate reward based on distance and whether dot and target are in same room"""
    
    # Calculate Euclidean distance
    distance = torch.norm(dot_position - target_position, dim=-1)
    
    # Check for NaN in distance
    check_for_nan(distance, "distance calculation")
    
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
    
    result = distance_reward + same_room_bonus + base_reward
    return check_for_nan(result, "reward calculation")


def rollout_episode(model, env, max_steps, num_samples, device, max_step_norm, use_quadrant=True, base_reward=64.0):
    """Roll out a single episode using the model with parallel action search"""
    # Reset environment
    obs, info = env.reset()
    
    # Initialize trajectory data
    states = [obs]
    actions = []
    dot_positions = [env.dot_position.cpu().numpy()]
    rewards = []
    next_goals = []
    reconstructions = []  # store decoded images
    
    # Encode initial state
    with torch.no_grad():
        if isinstance(obs, torch.Tensor):
            obs_tensor = obs.to(device=device).unsqueeze(0)
        else:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Ensure batch size is 1
        if obs_tensor.shape[0] != 1:
            obs_tensor = obs_tensor[:1]
            
        # Check for NaN in observation
        check_for_nan(obs_tensor, "observation tensor")
        
        z_t = model.encode(obs_tensor)
        
        # Check for NaN in encoding
        z_t = check_for_nan(z_t, "initial state encoding")
        
        # Decode reconstruction and store
        with torch.no_grad():
            recon_img = model.decode(z_t).squeeze(0).cpu()
            # Check for NaN in reconstruction
            recon_img = check_for_nan(recon_img, "initial reconstruction")
        reconstructions.append(recon_img)
    
    # Rollout loop
    done = False
    truncated = False
    episode_reward = 0
    
    for step in range(max_steps):
        if done or truncated:
            break
        
        # Predict next goal
        with torch.no_grad():
            z_next, log_prob = model.predict_next_goal(z_t)
            
            # Check for NaN in next goal prediction
            z_next = check_for_nan(z_next, f"next goal at step {step}")
            log_prob = check_for_nan(log_prob, f"log probability at step {step}")
            
            next_goals.append(z_next.cpu().numpy())
        
        # Search for action
        with torch.no_grad():
            a_t = model.search_action(
                z_t.detach(),
                z_next.detach(),
                num_samples=num_samples,
                max_step_norm=max_step_norm,
                verbose=(step == 0),  # Verbose only on first step
                use_quadrant=use_quadrant  # Use quadrant-based sampling if specified
            )
            
            # Check for NaN in action
            a_t = check_for_nan(a_t, f"action at step {step}")
        
        # Take action in environment
        action = a_t.cpu().numpy()[0]
        obs, env_reward, done, truncated, info = env.step(action)
        
        # Calculate custom reward based on distance instead of using environment reward
        # Create tensor versions of dot and target positions for the reward calculation
        dot_position = env.dot_position.unsqueeze(0)  # Add batch dimension
        target_position = env.target_position.unsqueeze(0)  # Add batch dimension
        
        # Check for NaN in positions
        check_for_nan(dot_position, f"dot position at step {step}")
        check_for_nan(target_position, f"target position at step {step}")
        
        # Calculate reward using the distance-based function
        custom_reward = calculate_distance_reward(
            dot_position, 
            target_position, 
            env.wall_x, 
            env.wall_width,
            base_reward
        ).item()
        
        # Update episode reward with custom reward
        episode_reward += custom_reward
        
        # Store trajectory data
        states.append(obs)
        actions.append(action)
        dot_positions.append(env.dot_position.cpu().numpy())
        rewards.append(custom_reward)  # Store custom reward instead of env reward
        
        # Update encoding
        with torch.no_grad():
            if isinstance(obs, torch.Tensor):
                obs_tensor = obs.to(device=device).unsqueeze(0)
            else:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Ensure batch size is 1
            if obs_tensor.shape[0] != 1:
                obs_tensor = obs_tensor[:1]
            
            # Check for NaN in observation
            check_for_nan(obs_tensor, f"observation tensor at step {step}")
            
            z_t = model.encode(obs_tensor)
            
            # Check for NaN in encoding
            z_t = check_for_nan(z_t, f"state encoding at step {step}")
            
            # Decode reconstruction for this new state
            with torch.no_grad():
                recon_img = model.decode(z_t).squeeze(0).cpu()
                # Check for NaN in reconstruction
                recon_img = check_for_nan(recon_img, f"reconstruction at step {step}")
            reconstructions.append(recon_img)
    
    # Calculate final distance
    final_distance = torch.norm(env.dot_position - env.target_position).item()
    same_room = (env.dot_position[0] < (env.wall_x - env.wall_width//2)) == (env.target_position[0] < (env.wall_x - env.wall_width//2))
    
    return {
        'states': states,
        'actions': actions,
        'dot_positions': dot_positions,
        'rewards': rewards,
        'next_goals': next_goals,
        'reconstructions': reconstructions,
        'total_reward': episode_reward,
        'done': done,
        'length': len(states) - 1,
        'final_distance': final_distance,
        'same_room': same_room
    }


class NanDetector:
    """Context manager to detect NaNs in model outputs during forward passes"""
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
        
        for name, module in self.model.named_modules():
            self.hooks.append(module.register_forward_hook(hook))
        
        if self.verbose:
            print("NaN detection enabled for all model modules")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        for hook in self.hooks:
            hook.remove()
        if self.verbose:
            print("NaN detection disabled")


def evaluate_model(model_path, output_dir='test_output', device='cpu', num_episodes=5, max_steps=50, num_samples=100, max_step_norm=15, encoder_embedding=200, encoding_dim=32, hidden_dim=409, use_quadrant=True, temperature=1.0, encoder_type='cnn', next_goal_temp=None, base_reward=64.0, search_mode='pldm'):
    """Evaluate the trained model on the DotWall environment"""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set up device
    device = torch.device(device)
    
    # Create environment
    env = DotWall(max_step_norm=max_step_norm, door_space=8)
    
    # Load model
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model with specified architecture
    print(f"Creating model with: encoding_dim={encoding_dim}, hidden_dim={hidden_dim}, encoder_embedding={encoder_embedding}")
    model = PLDMModel(
        img_size=env.img_size,
        in_channels=3,
        encoding_dim=encoding_dim,
        action_dim=2,
        hidden_dim=hidden_dim,
        encoder_embedding=encoder_embedding,
        encoder_type=encoder_type,
        temperature=temperature,
        next_goal_temp=next_goal_temp,
        search_mode=search_mode,
        max_step_norm=max_step_norm
    ).to(device)
    
    # Print model parameter counts
    model.print_parameter_count()
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model from model_state_dict")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model directly from checkpoint")
    
    # Set model to evaluation mode
    model.eval()
    print("Using parallel action search with", num_samples, "samples per step")
    print(f"Action sampling strategy: {'quadrant-based' if use_quadrant else 'full action space'}")
    
    # Enable NaN detection globally for the model
    with NanDetector(model):
        # Evaluate model for multiple episodes
        success_count = 0
        total_rewards = []
        
        for episode in range(num_episodes):
            print(f"\nEpisode {episode+1}/{num_episodes}")
            
            try:
                # Run the episode using our rollout function
                result = rollout_episode(
                    model=model,
                    env=env,
                    max_steps=max_steps,
                    num_samples=num_samples,
                    device=device,
                    max_step_norm=max_step_norm,
                    use_quadrant=use_quadrant,
                    base_reward=base_reward
                )
                
                # Extract data from result
                states = result['states']
                actions = result['actions']
                dot_positions = result['dot_positions']
                rewards = result['rewards']
                next_goals = result['next_goals']
                reconstructions = result['reconstructions']
                done = result['done']
                episode_reward = result['total_reward']
                final_distance = result['final_distance']
                same_room = result['same_room']
                
                # Count success
                if done:
                    success_count += 1
                
                # Track rewards
                total_rewards.append(episode_reward)
                
                # Create visualization frames for states and goals
                states_tensor = []
                for s in states:
                    if isinstance(s, torch.Tensor):
                        states_tensor.append(s.float())
                    else:
                        states_tensor.append(torch.from_numpy(s).float())
                    
                # For visualization - create empty goal tensors (same shape as state)
                if len(states_tensor) > 0:
                    goal_states = [torch.zeros_like(states_tensor[0]) for _ in range(len(states_tensor)-1)]
                
                # Visualize trajectory
                target_position = env.target_position.cpu().numpy()
                
                visualize_trajectory(
                    states_tensor,
                    reconstructions,
                    dot_positions,
                    target_position,
                    env.wall_x.cpu().numpy(),
                    output_dir / f"episode_{episode+1}.gif"
                )
                
                # Print episode summary
                print(f"\nEpisode {episode+1} summary:")
                print(f"  Length: {result['length']} steps")
                print(f"  Success: {done}")
                print(f"  Final distance to target: {final_distance:.4f}")
                print(f"  Reward total: {episode_reward:.2f}")
                print(f"  In same room as target: {same_room}")
                
            except ValueError as e:
                if "NaN detected" in str(e):
                    print(f"Episode {episode+1} failed with error: {e}")
                    print("Skipping to next episode...")
                else:
                    raise
    
    # Report overall performance
    if len(total_rewards) > 0:
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
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test PLDM model on DotWall environment')
    
    # Model path and output directory
    parser.add_argument('--model_path', type=str, default='output_same_page_value11/best_model.pt', help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='output_same_page_value11_best2', help='Directory to save test results')
    
    # Device and evaluation parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       help='Device to run on')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes to evaluate')
    parser.add_argument('--max_steps', type=int, default=32, help='Maximum steps per episode')
    
    # Action parameters
    parser.add_argument('--num_samples', type=int, default=8, help='Number of action samples to evaluate in parallel')
    parser.add_argument('--max_step_norm', type=float, default=8, help='Maximum step norm')
    parser.add_argument('--use_quadrant', type=bool, default=True, help='Use quadrant-based action sampling (True) or full action space sampling (False)')
    
    # Model architecture parameters
    parser.add_argument('--encoding_dim', type=int, default=512, help='Dimension of encoded state')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Dimension of hidden layers')
    parser.add_argument('--encoder_embedding', type=int, default=200, help='Dimension of encoder embedding')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for discrete softmax')
    parser.add_argument('--encoder_type', type=str, default='cnn', choices=['vit','cnn'], help='Encoder architecture: vit or cnn')
    parser.add_argument('--next_goal_temp', type=float, default=1.0, help='Temperature for next-goal predictor; if not set, uses --temperature')
    parser.add_argument('--base_reward', type=float, default=1.0, help='Base reward for each step')
    parser.add_argument('--search_mode', type=str, default='rl', choices=['pldm','rl'], help='Action search mode: pldm or rl')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        num_samples=args.num_samples,
        max_step_norm=args.max_step_norm,
        encoder_embedding=args.encoder_embedding,
        encoding_dim=args.encoding_dim,
        hidden_dim=args.hidden_dim,
        use_quadrant=args.use_quadrant,
        temperature=args.temperature,
        encoder_type=args.encoder_type,
        next_goal_temp=args.next_goal_temp,
        base_reward=args.base_reward,
        search_mode=args.search_mode
    ) 