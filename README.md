# Planning with Latent Dynamics Model (PLDM)

This project implements a PLDM model for the DotWall environment, using a Joint Embedding Predictive Architecture (JEPA) approach.

## Architecture

- **Encoder**: Vision Transformer (ViT) that encodes observations into latent states
- **Dynamics Model**: MLP that predicts the next latent state given the current state and action
- **Next-Goal Predictor**: MLP that predicts the next goal state given the current latent state

## Requirements

Install dependencies:

```bash
pip install torch torchvision gymnasium numpy matplotlib tqdm tensorboard imageio
```

## Training

To train the PLDM model:

```bash
python -m pldm.train --epochs 50 --episodes_per_epoch 10 --output_dir results
```

### Training Parameters

- `--encoding_dim`: Dimension of encoded state (default: 128)
- `--hidden_dim`: Dimension of hidden layers (default: 256)
- `--epochs`: Number of training epochs (default: 50)
- `--episodes_per_epoch`: Number of episodes per epoch (default: 10)
- `--max_steps_per_episode`: Maximum steps per episode (default: 100)
- `--search_steps`: Number of steps for action search (default: 10)
- `--gamma`: Discount factor (default: 0.99)
- `--lambda_dynamics`: Weight for dynamics loss (default: 1.0)
- `--encoder_lr`: Learning rate for encoder (default: 1e-4)
- `--dynamics_lr`: Learning rate for dynamics model (default: 1e-3)
- `--policy_lr`: Learning rate for policy (default: 1e-3)
- `--device`: Device to run training on (default: cuda if available, otherwise cpu)
- `--output_dir`: Directory to save model and logs (default: output)

## Evaluation

To evaluate a trained model:

```bash
python -m pldm.test --model_path results/best_model.pt --output_dir test_results
```

### Evaluation Parameters

- `--model_path`: Path to trained model (required)
- `--output_dir`: Directory to save test results (default: test_output)
- `--device`: Device to run on (default: cuda if available, otherwise cpu)
- `--num_episodes`: Number of episodes to evaluate (default: 5)
- `--max_steps`: Maximum steps per episode (default: 100)
- `--search_steps`: Number of steps for action search (default: 10)

## Method

The training objective $\mathcal{J}(\theta, \phi, \psi)$ combines:

1. Policy gradient objective for the next-goal predictor
2. Mean squared error loss for the dynamics model (JEPA objective)

$$\mathcal{J}(\theta, \phi, \psi) = \mathbb{E}_{\{\tau_i\}_{i=1...N}} \Bigg[ 
\sum_{t=1}^T \log \pi_\theta(\hat{z}_t \mid E_\phi(s_{<t})) \cdot A_t - \lambda \sum_{t=1}^T\left\| W_\psi(E_\phi(s_{t-1}), a_{t-1}) - E_\phi(s_t) \right\|^2 
\Bigg]$$

where:
- $\pi_\theta$ is the next-goal predictor
- $E_\phi$ is the encoder
- $W_\psi$ is the dynamics model
- $A_t$ is the advantage
- $\lambda$ is a hyperparameter controlling the weight of the two objectives

## Environment

The environment is DotWall, where:
- An agent (dot) must navigate to a target position
- A wall with a door divides the environment
- The agent must find the door to reach the target on the other side 