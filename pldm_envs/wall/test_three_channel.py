from wall import DotWall
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import imageio
import torchvision.transforms as T

from pldm_envs.wall.data.wall import WallDataset, WallDatasetConfig


OUTPUT_DIR = Path("debug_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def make_gif(frames, filename, fps=10):
    """
    Convert a list of PyTorch tensors into a GIF and save it to a file.

    Args:
        frames (list of torch.Tensor): List of image tensors (C, H, W) or (H, W).
        filename (str or Path): Path to save the GIF.
        fps (int): Frames per second for the GIF.
    """
    images = []
    transform = T.ToPILImage()

    for frame in frames:
        if (
            frame.ndimension() == 2
        ):  # Convert grayscale to 3-channel RGB for consistency
            frame = frame.unsqueeze(0).repeat(3, 1, 1)
        elif frame.shape[0] == 1:  # Single channel, expand to RGB
            frame = frame.repeat(3, 1, 1)

        image = transform(frame.cpu())  # Convert to PIL Image
        images.append(image)

    imageio.mimsave(filename, images, fps=fps)


def test_three_channel_visualization():
    """Test the 3-channel observation with visualization."""
    env = DotWall()
    obs, info = env.reset()
    
    # Create a figure with subplots
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    
    # Channel 0: Dot position
    axs[0].imshow(obs[0].cpu(), cmap='gray')
    axs[0].set_title('Dot Position (Channel 0)')
    axs[0].axis('off')
    
    # Channel 1: Wall
    axs[1].imshow(obs[1].cpu(), cmap='gray')
    axs[1].set_title('Wall (Channel 1)')
    axs[1].axis('off')
    
    # Channel 2: Target position
    axs[2].imshow(obs[2].cpu(), cmap='gray')
    axs[2].set_title('Target Position (Channel 2)')
    axs[2].axis('off')
    
    # Combined visualization (max across channels)
    combined = obs.max(dim=0)[0]
    axs[3].imshow(combined.cpu(), cmap='gray')
    axs[3].set_title('Combined (Max across channels)')
    axs[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "three_channel_obs.png")
    plt.close()
    
    # Now create an animated trajectory
    frames = []
    done = truncated = False
    
    for _ in range(50):  # Collect 50 steps maximum
        if done or truncated:
            break
        
        # Take a random action
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        # Create a visualization for this frame
        combined = obs.max(dim=0)[0]
        frames.append(combined)
    
    # Make a GIF from the frames
    make_gif(frames, OUTPUT_DIR / "three_channel_trajectory.gif")
    
    print(f"Trajectory ended with done={done}, truncated={truncated}")
    print(f"Final reward: {reward}")
    print(f"Dot position: {info['dot_position']}")
    print(f"Target position: {info['target_position']}")


if __name__ == "__main__":
    test_three_channel_visualization() 