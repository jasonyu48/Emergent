import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from pathlib import Path

class ViTEncoder(nn.Module):
    """Vision Transformer Encoder for DotWall environment"""
    
    def __init__(
        self,
        img_size=64,
        patch_size=8,
        in_channels=3,
        embedding_dim=128,
        num_heads=4,
        num_layers=4,
        mlp_ratio=4,
        dropout=0.1
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding layer
        self.patch_embedding = nn.Conv2d(
            in_channels, embedding_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # Position embedding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embedding_dim)
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * mlp_ratio,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.ln = nn.LayerNorm(embedding_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding [B, C, H, W] -> [B, D, H/P, W/P]
        x = self.patch_embedding(x)
        
        # Flatten patches [B, D, H/P, W/P] -> [B, H/P*W/P, D]
        x = x.flatten(2).transpose(1, 2)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embedding
        
        # Apply transformer
        x = self.transformer(x)
        
        # Apply layer norm
        x = self.ln(x)
        
        # Return cls token embedding
        return x[:, 0]


class DynamicsModel(nn.Module):
    """MLP Dynamics Model that predicts next encoded state given current encoded state and action"""
    
    def __init__(self, encoding_dim, action_dim=2, hidden_dim=256, num_layers=3):
        super().__init__()
        
        self.encoding_dim = encoding_dim
        self.action_dim = action_dim
        
        # Build MLP layers
        layers = []
        input_dim = encoding_dim + action_dim
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, encoding_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, z_t, a_t):
        """
        Forward pass through the dynamics model
        
        Args:
            z_t: Current latent state, shape [batch_size, encoding_dim]
            a_t: Action, shape [batch_size, action_dim]
            
        Returns:
            Predicted next latent state z_{t+1}
        """
        # Ensure input tensors have proper requires_grad setting
        # z_t typically doesn't need gradients when searching for actions
        # a_t needs gradients when searching for actions
        
        # Concatenate encoded state and action
        # Make sure they're on the same device and have the same dtype
        if a_t.dtype != z_t.dtype:
            a_t = a_t.to(dtype=z_t.dtype)
            
        x = torch.cat([z_t, a_t], dim=-1)
        
        # Apply MLP - this will maintain gradient flow if inputs require grad
        z_next = self.mlp(x)
        
        return z_next


class NextGoalPredictor(nn.Module):
    """MLP that predicts next goal state given current encoded state"""
    
    def __init__(self, encoding_dim, hidden_dim=256, num_layers=3):
        super().__init__()
        
        self.encoding_dim = encoding_dim
        
        # Build MLP layers
        layers = []
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(encoding_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, encoding_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # For stochastic policy, we'll predict mean and log_std
        self.mean = nn.Linear(encoding_dim, encoding_dim)
        self.log_std = nn.Parameter(torch.zeros(1, encoding_dim))
    
    def forward(self, z_t):
        # Apply MLP
        features = self.mlp(z_t)
        
        # Predict mean and use global log_std
        mean = self.mean(features)
        log_std = self.log_std.expand_as(mean)
        
        # Create distribution
        std = log_std.exp()
        dist = Normal(mean, std)
        
        # Sample next goal
        z_next = dist.rsample()
        
        # Compute log probability
        log_prob = dist.log_prob(z_next).sum(dim=-1)
        
        return z_next, log_prob


class PLDMModel(nn.Module):
    """Complete PLDM model with encoder, dynamics model, and next-goal predictor"""
    
    def __init__(
        self,
        img_size=64,
        in_channels=3,
        encoding_dim=128,
        action_dim=2,
        hidden_dim=256
    ):
        super().__init__()
        
        # Create components
        self.encoder = ViTEncoder(
            img_size=img_size,
            in_channels=in_channels,
            embedding_dim=encoding_dim
        )
        
        self.dynamics = DynamicsModel(
            encoding_dim=encoding_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
        
        self.next_goal_predictor = NextGoalPredictor(
            encoding_dim=encoding_dim,
            hidden_dim=hidden_dim
        )
        
        self.encoding_dim = encoding_dim
        self.action_dim = action_dim
    
    def encode(self, s_t):
        """Encode observation using the ViT encoder"""
        return self.encoder(s_t)
    
    def predict_next_goal(self, z_t):
        """Predict next goal given current encoded state"""
        return self.next_goal_predictor(z_t)
    
    def predict_next_state(self, z_t, a_t):
        """Predict next state given current encoded state and action"""
        return self.dynamics(z_t, a_t)
    
    def search_action(self, z_t, z_target, num_steps=10, lr=1, verbose=False, max_step_norm=12.25):
        """Search for the action that leads from z_t to z_target"""
        # Detach inputs but keep dtype and device
        dtype = z_t.dtype
        device = z_t.device
        
        if verbose:
            print(f"Starting action search with {num_steps} steps, lr={lr}")
            print(f"z_t shape: {z_t.shape}, dtype: {dtype}")
            print(f"z_target shape: {z_target.shape}, dtype: {dtype}")
        
        # Initialize action from uniform distribution within the environment's action bounds
        # The default value for max_step_norm in the DotWall environment is 12.25
        action = torch.rand(z_t.shape[0], self.action_dim, device=device, dtype=torch.float32) 
        action = action * 2 * max_step_norm - max_step_norm  # Scale to [-max_step_norm, max_step_norm]
        
        if verbose:
            print(f"Initial action: {action.cpu().numpy()}, norm: {action.norm().item():.4f}")
        
        # We will implement manual optimization without using autograd
        # This avoids gradient flow issues completely
        z_t_float = z_t.detach().to(torch.float32)
        z_target_float = z_target.detach().to(torch.float32)
        
        # Manual optimization loop
        for i in range(num_steps):
            # Forward pass using float32 precision
            with torch.no_grad():  # Explicitly disable gradients
                # Cast action to model dtype for forward pass
                action_cast = action.to(dtype)
                
                # Predict next state
                z_next_pred = self.dynamics(z_t_float.to(dtype), action_cast)
                
                # Compute loss
                z_next_pred_float = z_next_pred.to(torch.float32)
                loss = ((z_next_pred_float - z_target_float) ** 2).mean().item()
                
                # Estimate gradient using finite differences
                # This is a simple but effective approximation for our purpose
                grad = torch.zeros_like(action)
                eps = 1e-4
                
                # Compute gradient for each dimension of the action
                for j in range(self.action_dim):
                    # Perturb action in positive direction
                    action_pos = action.clone()
                    action_pos[:, j] += eps
                    action_pos_cast = action_pos.to(dtype)
                    
                    # Forward pass with perturbed action
                    z_next_pos = self.dynamics(z_t_float.to(dtype), action_pos_cast)
                    loss_pos = ((z_next_pos.to(torch.float32) - z_target_float) ** 2).mean().item()
                    
                    # Compute approximate gradient
                    grad[:, j] = (loss_pos - loss) / eps
                
                # Update action (gradient descent step)
                action = action - lr * grad
                
                # Optional: Print progress
                if verbose:
                    print(f"  Action search step {i}, Loss: {loss:.6f}, Action: {action.cpu().numpy()}")
        
        if verbose:
            print(f"Final action: {action.cpu().numpy()}, Final loss: {loss:.6f}")
            
        # Cast final action to the model's dtype before returning
        return action.to(dtype).detach() 