import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Constants
NUM_TRIALS = 6
OUTPUT_DIR = Path('output4')

# Function to run qtrain.py with specified mode
def run_qtrain():
    for i in range(NUM_TRIALS):
        mode = 'JEPA'
        print(f"Running trial {i+1}/{NUM_TRIALS} for {mode} mode...")
        subprocess.run([
            'python', 'pldm/qtrain.py',
            '--mode', str(mode),
            '--output_dir', str(OUTPUT_DIR / f'{mode}_trial_{i+1}')
        ])
        mode = 'RL'
        print(f"Running trial {i+1}/{NUM_TRIALS} for {mode} mode...")
        subprocess.run([
            'python', 'pldm/qtrain.py',
            '--mode', str(mode),
            '--output_dir', str(OUTPUT_DIR / f'{mode}_trial_{i+1}')
        ])


# Function to read rewards from a file
def read_rewards(file_path):
    with open(file_path, 'r') as f:
        return [float(line.strip()) for line in f]

# Run experiments
run_qtrain()

# Collect and plot results
rl_rewards = []
jepa_rewards = []

for i in range(NUM_TRIALS):
    rl_rewards.append(read_rewards(OUTPUT_DIR / f'RL_trial_{i+1}' / 'epoch_rewards_history.txt'))
    jepa_rewards.append(read_rewards(OUTPUT_DIR / f'JEPA_trial_{i+1}' / 'epoch_rewards_history.txt'))

# Convert to numpy arrays for easier manipulation
rl_rewards = np.array(rl_rewards)
jepa_rewards = np.array(jepa_rewards)

# Calculate means and standard deviations
rl_mean = rl_rewards.mean(axis=0)
rl_std = rl_rewards.std(axis=0)
jepa_mean = jepa_rewards.mean(axis=0)
jepa_std = jepa_rewards.std(axis=0)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(rl_mean, label='RL', color='blue')
plt.fill_between(range(len(rl_mean)), rl_mean - rl_std, rl_mean + rl_std, color='blue', alpha=0.2)
plt.plot(jepa_mean, label='JEPA + RL', color='green')
plt.fill_between(range(len(jepa_mean)), jepa_mean - jepa_std, jepa_mean + jepa_std, color='green', alpha=0.2)
plt.title('Average Reward vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Reward')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/comparison_plot.png")
plt.show() 