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
    
    return distance_reward + same_room_bonus + 64


def rollout_episode(model, env, max_steps, num_samples, device, use_bf16, max_step_norm, use_quadrant=True):
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
    
    # Set up dtype
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    
    # Verify model is using the correct dtype
    sample_param = next(model.parameters())
    expected_dtype = torch.bfloat16 if use_bf16 else torch.float32
    if sample_param.dtype != expected_dtype:
        print(f"Warning: Model parameters have dtype {sample_param.dtype}, but expected {expected_dtype}")
        print("Converting model to the correct dtype")
        model = model.to(dtype)
        # Verify conversion was successful
        sample_param = next(model.parameters())
        print(f"Model parameters dtype after conversion: {sample_param.dtype}")
    
    # Encode initial state
    with torch.no_grad():
        if isinstance(obs, torch.Tensor):
            obs_tensor = obs.to(dtype=dtype, device=device).unsqueeze(0)
        else:
            obs_tensor = torch.tensor(obs, dtype=dtype, device=device).unsqueeze(0)
        
        # Ensure batch size is 1
        if obs_tensor.shape[0] != 1:
            obs_tensor = obs_tensor[:1]
            
        # Verify tensor dtype before passing to model
        if obs_tensor.dtype != dtype:
            print(f"Warning: Input tensor dtype {obs_tensor.dtype} doesn't match expected {dtype}")
            obs_tensor = obs_tensor.to(dtype)
            
        z_t = model.encode(obs_tensor)
        
        # Verify encoding dtype
        if z_t.dtype != dtype:
            print(f"Warning: Encoded state z_t has dtype {z_t.dtype}, converting to {dtype}")
            z_t = z_t.to(dtype)
        
        # Decode reconstruction and store
        with torch.no_grad():
            recon_img = model.decode(z_t).squeeze(0).cpu()
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
            # Ensure z_t has the correct dtype
            if z_t.dtype != dtype:
                z_t = z_t.to(dtype)
            
            z_next, log_prob = model.predict_next_goal(z_t)
            
            # Ensure z_next has the correct dtype
            if z_next.dtype != dtype:
                print(f"Warning: z_next has dtype {z_next.dtype}, converting to {dtype}")
                z_next = z_next.to(dtype)
                
            next_goals.append(z_next.cpu().numpy())
        
        # Search for action
        with torch.no_grad():
            a_t = model.search_action(
                z_t.detach().to(dtype),  # Ensure correct dtype
                z_next.detach().to(dtype),  # Ensure correct dtype
                num_samples=num_samples,
                max_step_norm=max_step_norm,
                verbose=(step == 0),  # Verbose only on first step
                use_quadrant=use_quadrant  # Use quadrant-based sampling if specified
            )
            
            # Ensure action has the correct dtype
            if a_t.dtype != dtype:
                print(f"Warning: Action a_t has dtype {a_t.dtype}, converting to {dtype}")
                a_t = a_t.to(dtype)
        
        # Take action in environment
        # Convert to float32 for CPU numpy operations, regardless of model dtype
        action = a_t.to(torch.float32).cpu().numpy()[0]
        obs, env_reward, done, truncated, info = env.step(action)
        
        # Calculate custom reward based on distance instead of using environment reward
        # Create tensor versions of dot and target positions for the reward calculation
        dot_position = env.dot_position.unsqueeze(0)  # Add batch dimension
        target_position = env.target_position.unsqueeze(0)  # Add batch dimension
        
        # Calculate reward using the distance-based function
        custom_reward = calculate_distance_reward(
            dot_position, 
            target_position, 
            env.wall_x, 
            env.wall_width
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
                obs_tensor = obs.to(dtype=dtype, device=device).unsqueeze(0)
            else:
                obs_tensor = torch.tensor(obs, dtype=dtype, device=device).unsqueeze(0)
            
            # Ensure batch size is 1
            if obs_tensor.shape[0] != 1:
                obs_tensor = obs_tensor[:1]
            
            # Verify tensor dtype before passing to model
            if obs_tensor.dtype != dtype:
                obs_tensor = obs_tensor.to(dtype)
                
            z_t = model.encode(obs_tensor)
            
            # Verify encoding dtype
            if z_t.dtype != dtype:
                z_t = z_t.to(dtype)
            
            # Decode reconstruction for this new state
            with torch.no_grad():
                recon_img = model.decode(z_t).squeeze(0).cpu()
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


def evaluate_model(model_path, output_dir='test_output', device='cpu', num_episodes=5, max_steps=50, num_samples=100, use_bf16=False, max_step_norm=15, encoder_embedding=200, encoding_dim=32, hidden_dim=409, use_quadrant=True, temperature=1.0):
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
        temperature=temperature
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
    
    # Convert model to bf16 if supported
    if bf16_supported:
        # Use the model's to() method to properly convert all parameters and buffers
        model = model.to(torch.bfloat16)
        print("Converted model to BFloat16")
        
        # Verify model is in BFloat16 mode by checking parameter dtype
        sample_param = next(model.parameters())
        print(f"Model parameters dtype: {sample_param.dtype}")
        
        # Verify all components of the model are in BFloat16
        sample_encoder_param = next(model.encoder.parameters())
        sample_dynamics_param = next(model.dynamics.parameters())
        sample_goal_param = next(model.next_goal_predictor.parameters())
        
        print(f"Encoder parameters dtype: {sample_encoder_param.dtype}")
        print(f"Dynamics parameters dtype: {sample_dynamics_param.dtype}")
        print(f"NextGoalPredictor parameters dtype: {sample_goal_param.dtype}")
        
        # Explicitly check and convert Conv2d biases
        for module in model.modules():
            if isinstance(module, nn.Conv2d) and module.bias is not None:
                if module.bias.dtype != torch.bfloat16:
                    print(f"Converting Conv2d bias from {module.bias.dtype} to BFloat16")
                    module.bias = nn.Parameter(module.bias.to(torch.bfloat16))
                else:
                    print(f"Conv2d bias already in BFloat16")
                    
        # Verify conversion one more time
        for module in model.encoder.modules():
            if isinstance(module, nn.Conv2d) and module.bias is not None:
                print(f"Final Conv2d bias check - dtype: {module.bias.dtype}")
                break
    
    # Set model to evaluation mode
    model.eval()
    print("Using parallel action search with", num_samples, "samples per step")
    print(f"Action sampling strategy: {'quadrant-based' if use_quadrant else 'full action space'}")
    
    # Evaluate model for multiple episodes
    success_count = 0
    total_rewards = []
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode+1}/{num_episodes}")
        
        # Run the episode using our rollout function
        result = rollout_episode(
            model=model,
            env=env,
            max_steps=max_steps,
            num_samples=num_samples,
            device=device,
            use_bf16=bf16_supported,
            max_step_norm=max_step_norm,
            use_quadrant=use_quadrant
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
    parser.add_argument('--model_path', type=str, default='output_same_page_value6/best_model.pt', help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='output_same_page_value6', help='Directory to save test results')
    
    # Device and evaluation parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       help='Device to run on')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes to evaluate')
    parser.add_argument('--max_steps', type=int, default=32, help='Maximum steps per episode')
    parser.add_argument('--bf16', type=bool, default=False, help='Use BFloat16 precision for evaluation')
    
    # Action parameters
    parser.add_argument('--num_samples', type=int, default=8, help='Number of action samples to evaluate in parallel')
    parser.add_argument('--max_step_norm', type=float, default=8, help='Maximum step norm')
    parser.add_argument('--use_quadrant', type=bool, default=True, help='Use quadrant-based action sampling (True) or full action space sampling (False)')
    
    # Model architecture parameters
    parser.add_argument('--encoding_dim', type=int, default=512, help='Dimension of encoded state')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Dimension of hidden layers')
    parser.add_argument('--encoder_embedding', type=int, default=200, help='Dimension of encoder embedding')
    parser.add_argument('--temperature', type=float, default=0.001, help='Temperature for discrete softmax')
    
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
        use_bf16=args.bf16,
        max_step_norm=args.max_step_norm,
        encoder_embedding=args.encoder_embedding,
        encoding_dim=args.encoding_dim,
        hidden_dim=args.hidden_dim,
        use_quadrant=args.use_quadrant,
        temperature=args.temperature
    ) 