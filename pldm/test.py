import os
import torch
import torch.nn.functional as F
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio

from pldm_envs.wall.wall import DotWall
from pldm.model import PLDMModel


def make_gif(frames, filename, fps=10):
    """Create a GIF from a list of frames"""
    imageio.mimsave(filename, frames, fps=fps)


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
        
        # Plot current observation
        axes[0].imshow(frame.transpose(1, 2, 0))
        axes[0].set_title(f"Current Observation (Step {i})")
        axes[0].axis('off')
        
        # Plot goal prediction
        axes[1].imshow(goal_frame.transpose(1, 2, 0))
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
        
        # Convert plot to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt_frames.append(img)
        
        plt.close(fig)
    
    # Save as GIF
    make_gif(plt_frames, output_path, fps=2)


def evaluate_model(model_path, output_dir='test_output', device='cpu', num_episodes=5, max_steps=100, search_steps=10):
    """Evaluate the trained model on the DotWall environment"""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set device
    device = torch.device(device)
    
    # Create environment
    env = DotWall()
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model
    model = PLDMModel(
        img_size=env.img_size,
        in_channels=3,
        encoding_dim=128,
        action_dim=2,
        hidden_dim=256
    ).to(device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Set model to evaluation mode
    model.eval()
    
    # Evaluate model for multiple episodes
    success_count = 0
    total_rewards = []
    
    for episode in range(num_episodes):
        print(f"Episode {episode+1}/{num_episodes}")
        
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
                if isinstance(obs, torch.Tensor):
                    obs_tensor = obs.float().unsqueeze(0).to(device)
                else:
                    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                z_t = model.encode(obs_tensor)
            
            # Rollout loop
            done = False
            truncated = False
            
            for step in range(max_steps):
                if done or truncated:
                    break
                
                # Predict next goal
                with torch.no_grad():
                    z_next, _ = model.predict_next_goal(z_t)
                    
                    # Search for action to reach predicted next goal
                    a_t = model.search_action(z_t, z_next, num_steps=search_steps)
                
                # Take action in environment
                action = a_t.cpu().numpy()[0]
                obs, reward, done, truncated, info = env.step(action)
                
                # Store trajectory data
                states.append(obs)
                actions.append(action)
                dot_positions.append(env.dot_position.cpu().numpy())
                rewards.append(reward)
                
                # Store goal prediction visualization
                # Create a fake observation with the goal position
                if isinstance(obs, torch.Tensor):
                    goal_obs = torch.zeros_like(obs.float())
                else:
                    goal_obs = torch.zeros_like(torch.from_numpy(obs).float())
                
                # Reconstruct the goal state (this is just for visualization)
                goal_states.append(goal_obs)
                
                # Update current encoding
                with torch.no_grad():
                    if isinstance(obs, torch.Tensor):
                        obs_tensor = obs.float().unsqueeze(0).to(device)
                    else:
                        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                    z_t = model.encode(obs_tensor)
            
            # Compute total reward
            total_reward = sum(rewards)
            total_rewards.append(total_reward)
            
            # Check if episode was successful
            if done:
                success_count += 1
            
            # Convert states to tensors
            states_tensor = []
            for s in states:
                if isinstance(s, torch.Tensor):
                    states_tensor.append(s.float())
                else:
                    states_tensor.append(torch.from_numpy(s).float())
            
            # Visualize trajectory
            target_position = env.target_position.cpu().numpy()
            
            visualize_trajectory(
                states_tensor,
                goal_states,
                dot_positions,
                target_position,
                env.wall_x.cpu().numpy(),
                output_dir / f"episode_{episode+1}.gif"
            )
            
            print(f"  Episode length: {len(states)-1}")
            print(f"  Success: {done}")
            print(f"  Reward: {total_reward:.4f}")
            
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
        print(f"  Average reward: {avg_reward:.4f}")
        
        return success_rate, avg_reward
    else:
        print("\nNo successful evaluations completed.")
        return 0.0, 0.0


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test PLDM model on DotWall environment')
    
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='test_output', help='Directory to save test results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       help='Device to run on')
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of episodes to evaluate')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps per episode')
    parser.add_argument('--search_steps', type=int, default=10, help='Number of steps for action search')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate_model(
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        search_steps=args.search_steps
    ) 