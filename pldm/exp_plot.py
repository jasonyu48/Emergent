import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Constants
NUM_TRIALS = 10
OUTPUT_DIR = Path('output519_l1loss')

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

# Function to read generic metric data from a file
def read_metric_data(file_path):
    # Check if file exists, return empty list if not, to handle cases where probing might not have run for all trials
    if not file_path.exists():
        print(f"Warning: Metric file {file_path} not found. Returning empty list.")
        return []
    with open(file_path, 'r') as f:
        return [float(line.strip()) for line in f]

# Run experiments
run_qtrain()

# Collect and plot results
rl_rewards = []
jepa_rewards = []

rl_decode_mses = []
jepa_decode_mses = []
rl_encode_mses = []
jepa_encode_mses = []

# Assume probe_steps_history will be the same for all trials of a given mode if probe_eval_interval is constant
# Read it once, e.g., from the first RL trial if available
probe_steps = []
first_rl_trial_steps_path = OUTPUT_DIR / 'RL_trial_1' / 'probe_steps_history.txt'
if first_rl_trial_steps_path.exists():
    probe_steps = read_metric_data(first_rl_trial_steps_path)
else:
    print(f"Warning: {first_rl_trial_steps_path} not found. MSE plots will not have correct x-axis.")

for i in range(NUM_TRIALS):
    trial_num = i + 1
    rl_trial_dir = OUTPUT_DIR / f'RL_trial_{trial_num}'
    jepa_trial_dir = OUTPUT_DIR / f'JEPA_trial_{trial_num}'

    rl_rewards.append(read_rewards(rl_trial_dir / 'epoch_rewards_history.txt'))
    jepa_rewards.append(read_rewards(jepa_trial_dir / 'epoch_rewards_history.txt'))

    rl_decode_mses.append(read_metric_data(rl_trial_dir / 'decode_mse_history.txt'))
    jepa_decode_mses.append(read_metric_data(jepa_trial_dir / 'decode_mse_history.txt'))
    rl_encode_mses.append(read_metric_data(rl_trial_dir / 'encode_mse_history.txt'))
    jepa_encode_mses.append(read_metric_data(jepa_trial_dir / 'encode_mse_history.txt'))

# Convert to numpy arrays for easier manipulation
# For MSEs, we need to handle potentially different lengths if some trials didn't complete or probing was off
# Truncate to the minimum length found across trials for each metric if probe_steps exist
if probe_steps:
    min_len_decode_rl = min(len(l) for l in rl_decode_mses if l) if any(rl_decode_mses) else 0
    min_len_decode_jepa = min(len(l) for l in jepa_decode_mses if l) if any(jepa_decode_mses) else 0
    min_len_encode_rl = min(len(l) for l in rl_encode_mses if l) if any(rl_encode_mses) else 0
    min_len_encode_jepa = min(len(l) for l in jepa_encode_mses if l) if any(jepa_encode_mses) else 0

    # Use the minimum of probe_steps length and the metric data length
    min_len_decode_rl = min(min_len_decode_rl, len(probe_steps))
    min_len_decode_jepa = min(min_len_decode_jepa, len(probe_steps))
    min_len_encode_rl = min(min_len_encode_rl, len(probe_steps))
    min_len_encode_jepa = min(min_len_encode_jepa, len(probe_steps))

    probe_steps_truncated = np.array(probe_steps[:min(min_len_decode_rl, min_len_decode_jepa, min_len_encode_rl, min_len_encode_jepa)])

    rl_decode_mses_np = np.array([l[:len(probe_steps_truncated)] for l in rl_decode_mses if len(l) >= len(probe_steps_truncated)])
    jepa_decode_mses_np = np.array([l[:len(probe_steps_truncated)] for l in jepa_decode_mses if len(l) >= len(probe_steps_truncated)])
    rl_encode_mses_np = np.array([l[:len(probe_steps_truncated)] for l in rl_encode_mses if len(l) >= len(probe_steps_truncated)])
    jepa_encode_mses_np = np.array([l[:len(probe_steps_truncated)] for l in jepa_encode_mses if len(l) >= len(probe_steps_truncated)])
else:
    # Fallback if probe_steps couldn't be read (though plotting will be problematic for x-axis)
    rl_decode_mses_np = np.array([l for l in rl_decode_mses if l])
    jepa_decode_mses_np = np.array([l for l in jepa_decode_mses if l])
    rl_encode_mses_np = np.array([l for l in rl_encode_mses if l])
    jepa_encode_mses_np = np.array([l for l in jepa_encode_mses if l])

rl_rewards = np.array(rl_rewards)
jepa_rewards = np.array(jepa_rewards)

# Calculate means and standard deviations
rl_mean = rl_rewards.mean(axis=0)
rl_std = rl_rewards.std(axis=0)
jepa_mean = jepa_rewards.mean(axis=0)
jepa_std = jepa_rewards.std(axis=0)

if rl_decode_mses_np.size > 0:
    rl_decode_mean = rl_decode_mses_np.mean(axis=0)
    rl_decode_std = rl_decode_mses_np.std(axis=0)
else: 
    rl_decode_mean, rl_decode_std = np.array([]), np.array([])

if jepa_decode_mses_np.size > 0:
    jepa_decode_mean = jepa_decode_mses_np.mean(axis=0)
    jepa_decode_std = jepa_decode_mses_np.std(axis=0)
else:
    jepa_decode_mean, jepa_decode_std = np.array([]), np.array([])

if rl_encode_mses_np.size > 0:
    rl_encode_mean = rl_encode_mses_np.mean(axis=0)
    rl_encode_std = rl_encode_mses_np.std(axis=0)
else:
    rl_encode_mean, rl_encode_std = np.array([]), np.array([])

if jepa_encode_mses_np.size > 0:
    jepa_encode_mean = jepa_encode_mses_np.mean(axis=0)
    jepa_encode_std = jepa_encode_mses_np.std(axis=0)
else:
    jepa_encode_mean, jepa_encode_std = np.array([]), np.array([])

# Plot Rewards
plt.figure(figsize=(10, 6))
plt.plot(rl_mean, label='RL', color='blue')
plt.fill_between(range(len(rl_mean)), rl_mean - rl_std, rl_mean + rl_std, color='blue', alpha=0.2)
plt.plot(jepa_mean, label='JEPA+RL', color='green')
plt.fill_between(range(len(jepa_mean)), jepa_mean - jepa_std, jepa_mean + jepa_std, color='green', alpha=0.2)
plt.title('Average Reward vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Reward')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "comparison_plot_rewards.png") # Changed filename
plt.show()

# Plot Decode MSE
if probe_steps and rl_decode_mean.size > 0 and jepa_decode_mean.size > 0:
    plt.figure(figsize=(10, 6))
    plt.plot(probe_steps_truncated, rl_decode_mean, label='RL Decode MSE', color='blue')
    plt.fill_between(probe_steps_truncated, rl_decode_mean - rl_decode_std, rl_decode_mean + rl_decode_std, color='blue', alpha=0.2)
    plt.plot(probe_steps_truncated, jepa_decode_mean, label='JEPA+RL Decode MSE', color='green')
    plt.fill_between(probe_steps_truncated, jepa_decode_mean - jepa_decode_std, jepa_decode_mean + jepa_decode_std, color='green', alpha=0.2)
    plt.title('Average Decode MSE vs Global Step')
    plt.xlabel('Global Step')
    plt.ylabel('Decode MSE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_plot_decode_mse.png")
    plt.show()
elif probe_steps:
    print("Skipping Decode MSE plot due to missing data for one or both modes.")

# Plot Encode MSE
if probe_steps and rl_encode_mean.size > 0 and jepa_encode_mean.size > 0:
    plt.figure(figsize=(10, 6))
    plt.plot(probe_steps_truncated, rl_encode_mean, label='RL Encode MSE', color='blue')
    plt.fill_between(probe_steps_truncated, rl_encode_mean - rl_encode_std, rl_encode_mean + rl_encode_std, color='blue', alpha=0.2)
    plt.plot(probe_steps_truncated, jepa_encode_mean, label='JEPA+RL Encode MSE', color='green')
    plt.fill_between(probe_steps_truncated, jepa_encode_mean - jepa_encode_std, jepa_encode_mean + jepa_encode_std, color='green', alpha=0.2)
    plt.title('Average Encode MSE vs Global Step')
    plt.xlabel('Global Step')
    plt.ylabel('Encode MSE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_plot_encode_mse.png")
    plt.show()
elif probe_steps:
    print("Skipping Encode MSE plot due to missing data for one or both modes.") 