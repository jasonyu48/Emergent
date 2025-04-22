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

from pldm_envs.wall.wall import DotWall
from pldm.model import PLDMModel


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


def visualize_trajectory(frames, goal_frames, dot_positions, target_position, wall_x, output_path):
    """Visualize the trajectory with dot positions and goal frames"""
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert frames to numpy arrays for visualization
    frames_np = [frame.cpu().numpy() for frame in frames]
    goal_frames_np = [frame.cpu().numpy() for frame in goal_frames]
    
    # Create GIF frames
    plt_frames = []
    
    for i, (frame, goal_frame) in enumerate(zip(frames_np, goal_frames_np)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot current observation - normalize to [0,1] range for imshow
        if frame.max() > 1.0:
            frame_norm = frame / 255.0
        else:
            frame_norm = frame
        axes[0].imshow(frame_norm.transpose(1, 2, 0))
        axes[0].set_title(f"Current Observation (Step {i})")
        axes[0].axis('off')
        
        # Plot goal prediction - normalize to [0,1] range for imshow
        if goal_frame.max() > 1.0:
            goal_frame_norm = goal_frame / 255.0
        else:
            goal_frame_norm = goal_frame
        axes[1].imshow(goal_frame_norm.transpose(1, 2, 0))
        axes[1].set_title(f"Goal Prediction (Step {i})")
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
        try:
            # For newer Matplotlib versions
            buf = fig.canvas.buffer_rgba()
            img = np.asarray(buf)
        except AttributeError:
            # Fallback for older Matplotlib versions
            buf = fig.canvas.tostring_rgb()
            img = np.frombuffer(buf, dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt_frames.append(img)
        
        plt.close(fig)
    
    # Save as GIF
    make_gif(plt_frames, output_path, fps=2)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test PLDM model on DotWall environment')
    
    parser.add_argument('--model_path', type=str, required=True, default='output/model.pt', help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='test_output', help='Directory to save test results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       help='Device to run on')
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of episodes to evaluate')
    parser.add_argument('--max_steps', type=int, default=20, help='Maximum steps per episode')
    parser.add_argument('--search_steps', type=int, default=30, help='Number of steps for action search')
    parser.add_argument('--max_step_norm', type=float, default=15, help='Maximum step norm')
    parser.add_argument('--bf16', action='store_true', help='Use BFloat16 precision for evaluation')
    
    return parser.parse_args()


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
    same_room_bonus = torch.where(same_room, torch.tensor(10.0, device=dot_position.device), torch.tensor(0.0, device=dot_position.device))
    
    return distance_reward + same_room_bonus


def evaluate_model(model_path, output_dir='test_output', device='cpu', num_episodes=5, max_steps=50, search_steps=10, use_bf16=False, max_step_norm=12.25):
    """Evaluate the trained model on the DotWall environment"""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set up device
    device = torch.device(device)
    
    # Check if bf16 is supported on the current device
    bf16_supported = (
        use_bf16 and
        torch.cuda.is_available() and
        torch.cuda.is_bf16_supported()
    )
    
    if use_bf16 and not bf16_supported:
        print("Warning: BF16 precision requested but not supported on this device. Using FP32 instead.")
    
    if bf16_supported:
        print("Using BFloat16 precision for evaluation")
    
    # Create environment
    env = DotWall(max_step_norm=max_step_norm)
    
    # Load model
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model
    model = PLDMModel(
        img_size=env.img_size,
        in_channels=3,
        encoding_dim=16,  # Updated to match training default
        action_dim=2,
        hidden_dim=128    # Updated to match training default
    ).to(device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model from model_state_dict")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model directly from checkpoint")
    
    # Convert model to bf16 if supported
    if bf16_supported:
        model = model.to(torch.bfloat16)
        print("Converted model to BFloat16")
    
    # Print model parameter norms to check if they're reasonable
    total_params = 0
    print("\nModel summary:")
    
    # Check encoder parameters
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    dynamics_params = sum(p.numel() for p in model.dynamics.parameters())
    
    print(f"Encoder: {encoder_params} parameters")
    print(f"Dynamics: {dynamics_params} parameters")
    
    # Calculate total parameters
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
    print(f"Total trainable parameters: {total_params}")
    
    # Set model to evaluation mode
    model.eval()
    print("Model set to evaluation mode")
    
    # Evaluate model for multiple episodes
    success_count = 0
    total_rewards = []
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode+1}/{num_episodes}")
        
        try:
            # Reset environment
            obs, info = env.reset()
            
            # Initialize trajectory data
            states = [obs]
            actions = []
            dot_positions = [env.dot_position.cpu().numpy()]
            goal_states = []
            rewards = []
            
            # Get initial encoding
            with torch.no_grad():
                # Get appropriate dtype
                dtype = torch.bfloat16 if bf16_supported else torch.float32
                
                if isinstance(obs, torch.Tensor):
                    obs_tensor = obs.to(dtype=dtype, device=device).unsqueeze(0)
                else:
                    obs_tensor = torch.tensor(obs, dtype=dtype, device=device).unsqueeze(0)
                
                z_t = model.encode(obs_tensor)
            
            # Rollout loop
            done = False
            truncated = False
            
            for step in range(max_steps):
                if done or truncated:
                    break
                
                # Predict next goal
                with torch.no_grad():
                    z_next, transition_loss = model.predict_next_goal(z_t)
                    
                    # Ensure tensors are detached for action search
                    z_t_detached = z_t.clone().detach()
                    z_next_detached = z_next.clone().detach()
                    
                    # Search for action to reach predicted next goal
                    try:
                        # Enable verbose mode for better debugging
                        verbose = False  # Set verbose to False to reduce output
                        
                        a_t = model.search_action(
                            z_t_detached, 
                            z_next_detached, 
                            num_steps=search_steps,
                            verbose=verbose,
                            max_step_norm=max_step_norm  # Pass the max_step_norm to constrain initial actions
                        )
                    except Exception as e:
                        print(f"Error in search_action: {e}")
                        # Fallback to a random action if search fails
                        a_t = torch.randn(z_t.shape[0], 2, device=device)
                        if bf16_supported:
                            a_t = a_t.to(torch.bfloat16)
                        print("Using fallback random action")
                
                # Take action in environment
                # Convert to float32 before converting to NumPy since NumPy doesn't support bfloat16
                action = a_t.to(torch.float32).cpu().numpy()[0]
                obs, env_reward, done, truncated, info = env.step(action)
                
                # Calculate custom reward (like during training)
                dot_position = env.dot_position.unsqueeze(0)
                target_position = env.target_position.unsqueeze(0)
                custom_reward = calculate_distance_reward(
                    dot_position, 
                    target_position, 
                    env.wall_x,
                    env.wall_width
                ).item()
                
                # Store trajectory data
                states.append(obs)
                actions.append(action)
                dot_positions.append(env.dot_position.cpu().numpy())
                rewards.append(custom_reward)  # Use custom reward
                
                # Print step-level information every 10 steps
                if step % 10 == 0 or step == max_steps-1:
                    distance = torch.norm(dot_position - target_position).item()
                    print(f"  Step {step}: Distance: {distance:.2f}, Reward: {custom_reward:.2f}")
                
                # Store goal prediction visualization
                # Create a fake observation with the goal position
                if isinstance(obs, torch.Tensor):
                    goal_obs = torch.zeros_like(obs.to(dtype=torch.float32))
                else:
                    goal_obs = torch.zeros_like(torch.tensor(obs, dtype=torch.float32))
                
                # Reconstruct the goal state (this is just for visualization)
                goal_states.append(goal_obs)
                
                # Update current encoding
                with torch.no_grad():
                    if isinstance(obs, torch.Tensor):
                        obs_tensor = obs.to(dtype=dtype, device=device).unsqueeze(0)
                    else:
                        obs_tensor = torch.tensor(obs, dtype=dtype, device=device).unsqueeze(0)
                    
                    z_t = model.encode(obs_tensor)
            
            # Compute total reward
            total_reward = sum(rewards)
            total_rewards.append(total_reward)
            
            # Check if episode was successful
            if done:
                success_count += 1
            
            # Calculate final distance to target
            final_distance = torch.norm(env.dot_position - env.target_position).item()
            
            # Convert states to tensors
            states_tensor = []
            for s in states:
                if isinstance(s, torch.Tensor):
                    states_tensor.append(s.float())
                else:
                    states_tensor.append(torch.from_numpy(s).float())
                
            # Goal states are already tensors as we created them that way
            goal_states_tensor = goal_states
            
            # Visualize trajectory
            target_position = env.target_position.cpu().numpy()
            
            visualize_trajectory(
                states_tensor,
                goal_states_tensor,
                dot_positions,
                target_position,
                env.wall_x.cpu().numpy(),
                output_dir / f"episode_{episode+1}.gif"
            )
            
            print(f"\nEpisode {episode+1} summary:")
            print(f"  Length: {len(states)-1} steps")
            print(f"  Success: {done}")
            print(f"  Final distance to target: {final_distance:.4f}")
            print(f"  Custom reward total: {total_reward:.2f}")
            same_room = (env.dot_position[0] < (env.wall_x - env.wall_width//2)) == (env.target_position[0] < (env.wall_x - env.wall_width//2))
            print(f"  In same room as target: {same_room}")
            
        except Exception as e:
            print(f"Error during episode {episode}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Report overall performance
    if len(total_rewards) > 0:
        success_rate = success_count / num_episodes
        avg_reward = sum(total_rewards) / len(total_rewards)
        
        print(f"\nOverall performance:")
        print(f"  Success rate: {success_rate:.2f} ({success_count}/{num_episodes})")
        print(f"  Average custom reward: {avg_reward:.2f}")
        print(f"  Note: Custom rewards include distance penalties and same-room bonuses")
        
        return success_rate, avg_reward
    else:
        print("\nNo successful evaluations completed.")
        return 0.0, 0.0


if __name__ == '__main__':
    args = parse_args()
    evaluate_model(
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        search_steps=args.search_steps,
        use_bf16=args.bf16,
        max_step_norm=args.max_step_norm
    ) 