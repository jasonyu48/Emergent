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
        embedding_dim=512,
        encoding_dim=32,
        num_heads=8,
        num_layers=3,
        mlp_ratio=4,
        dropout=0.1
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.encoding_dim = encoding_dim
        
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
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.ln = nn.LayerNorm(embedding_dim)
        
        # Projection head to reduce dimension from embedding_dim to encoding_dim
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.GELU(),
            nn.Linear(embedding_dim // 4, encoding_dim)
        )
        
        # Final layer norm for encoded representation
        self.final_ln = nn.LayerNorm(encoding_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Use Kaiming initialization for linear layers
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            # Use Kaiming initialization for convolutional layers
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            # Initialize LayerNorm as before
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        # Special initialization for ViT specific parameters
        elif isinstance(m, ViTEncoder):
            # Initialize position embeddings
            nn.init.trunc_normal_(m.pos_embedding, std=0.02)
            # Initialize class token
            nn.init.trunc_normal_(m.cls_token, std=0.02)
    
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
        
        # Get cls token embedding
        cls_embedding = x[:, 0]
        
        # Project from embedding_dim to encoding_dim
        encoding = self.projection(cls_embedding)
        
        # Apply final layer normalization
        encoding = self.final_ln(encoding)
        
        return encoding


class DynamicsModel(nn.Module):
    """MLP Dynamics Model that predicts next encoded state given current encoded state and action"""
    
    def __init__(self, encoding_dim, action_dim=2, hidden_dim=512, num_layers=6):
        super().__init__()
        
        self.encoding_dim = encoding_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection layer
        self.input_proj = nn.Linear(encoding_dim + action_dim, hidden_dim)
        self.input_activation = nn.GELU()
        
        # Build residual blocks
        self.residual_blocks = nn.ModuleList()
        for _ in range((num_layers - 2) // 2):  # Each residual block has 2 linear layers
            block = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU()
            ])
            self.residual_blocks.append(block)
        
        # Output projection followed by optional LayerNorm to stabilise scale
        self.output_proj = nn.Linear(hidden_dim, encoding_dim)
        self.output_ln = nn.LayerNorm(encoding_dim)
        
        # Layer norm for residual connections
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(len(self.residual_blocks))])
        
        # Initialize weights using Kaiming initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, z_t, a_t):
        """
        Forward pass through the dynamics model
        
        Args:
            z_t: Current latent state, shape [batch_size, encoding_dim]
            a_t: Action, shape [batch_size, action_dim]
            
        Returns:
            Predicted next latent state z_{t+1}
        """
        # Ensure input tensors have proper dtype
        if a_t.dtype != z_t.dtype:
            a_t = a_t.to(dtype=z_t.dtype)
            
        # Concatenate encoded state and action
        x = torch.cat([z_t, a_t], dim=-1)
        
        # Apply input projection
        x = self.input_proj(x)
        x = self.input_activation(x)
        
        # Apply residual blocks
        for i, (block, layer_norm) in enumerate(zip(self.residual_blocks, self.layer_norms)):
            # Save input for residual connection
            residual = x
            
            # Apply block layers
            for layer in block:
                x = layer(x)
            
            # Add residual connection and normalize
            x = layer_norm(x + residual)
        
        # Apply output projection and layer normalisation
        z_next = self.output_ln(self.output_proj(x))
        
        return z_next


class NextGoalPredictor(nn.Module):
    """MLP that predicts next goal state given current encoded state"""
    
    def __init__(self, encoding_dim, hidden_dim=512, num_layers=6):
        super().__init__()
        
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection layer
        self.input_proj = nn.Linear(encoding_dim, hidden_dim)
        self.input_activation = nn.GELU()
        
        # Build residual blocks
        self.residual_blocks = nn.ModuleList()
        for _ in range((num_layers - 2) // 2):  # Each residual block has 2 linear layers
            block = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU()
            ])
            self.residual_blocks.append(block)
        
        # Output projection to features
        self.output_proj = nn.Linear(hidden_dim, encoding_dim)
        # Value head for state-value prediction
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Layer norm for residual connections
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(len(self.residual_blocks))])
        
        # For stochastic policy, we'll predict mean only
        self.mean = nn.Linear(encoding_dim, encoding_dim)
        # Use fixed log_std of 0 instead of learnable parameter
        self.log_std = 0.0
        
        # Initialize weights using Kaiming initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, z_t):
        x = self._compute_features(z_t)
        x = self.output_proj(x)
        mean = self.mean(x)
        
        # Create distribution with fixed std of 1.0 (since log(1.0) = 0.0)
        std = torch.ones_like(mean)
        dist = Normal(mean, std)
        
        # Sample next goal
        z_next = dist.rsample()
        
        # Compute log probability
        log_prob = dist.log_prob(z_next).sum(dim=-1)
        
        return z_next, log_prob

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _compute_features(self, z_t):
        """Shared forward over MLP to get intermediate features."""
        x = self.input_activation(self.input_proj(z_t))
        for block, layer_norm in zip(self.residual_blocks, self.layer_norms):
            residual = x
            for layer in block:
                x = layer(x)
            x = layer_norm(x + residual)
        return x
    
    def _get_distribution(self, z_t):
        """Return Normal distribution N(mean(z_t), 1)."""
        x = self._compute_features(z_t)
        x = self.output_proj(x)
        mean = self.mean(x)
        std = torch.ones_like(mean)
        return Normal(mean, std)
    
    # ------------------------------------------------------------------
    #  Value prediction
    # ------------------------------------------------------------------
    def value(self, z_t):
        """Predict state value V(z_t)."""
        features = self._compute_features(z_t)
        value = self.value_head(features).squeeze(-1)  # [B]
        return value
    
    def log_prob(self, z_t, z_sample):
        """Log probability of `z_sample` under the policy p(z_next|z_t)."""
        dist = self._get_distribution(z_t)
        return dist.log_prob(z_sample).sum(dim=-1)


class Decoder(nn.Module):
    """Decoder that reconstructs a 64x64 RGB image from a 32-dim latent.

    Architecture: FC -> 4x4x256 feature map -> series of ConvTranspose2d
    layers doubling spatial resolution until 64x64, followed by 3-channel
    convolution and Sigmoid.
    """

    def __init__(self, encoding_dim: int, img_size: int = 64, out_channels: int = 3):
        super().__init__()

        assert img_size == 64, "Decoder currently supports img_size=64 only"

        self.fc = nn.Sequential(
            nn.Linear(encoding_dim, 4 * 4 * 256),
            nn.GELU(),
        )

        # Upsampling pathway: 4×4 → 8×8 → 16×16 → 32×32 → 64×64
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8×8
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 16×16
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 32×32
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # 64×64
            nn.GELU(),
            nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),  # final pixel values in [0,1]
        )

    def forward(self, z):
        # z: [B, latent_dim]
        x = self.fc(z)                       # [B, 4*4*256]
        x = x.view(-1, 256, 4, 4)           # [B, 256, 4, 4]
        x = self.deconv(x)                  # [B, 3, 64, 64]
        return x


class PLDMModel(nn.Module):
    """Complete PLDM model with encoder, dynamics model, and next-goal predictor"""
    
    def __init__(
        self,
        img_size=64,
        in_channels=3,
        encoding_dim=32,
        action_dim=2,
        hidden_dim=512,
        encoder_embedding=256
    ):
        super().__init__()
        
        # Create components
        self.encoder = ViTEncoder(
            img_size=img_size,
            in_channels=in_channels,
            embedding_dim=encoder_embedding,
            encoding_dim=encoding_dim
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
        
        # Decoder only used during warm-up epochs to prevent encoder collapse
        self.decoder = Decoder(
            encoding_dim=encoding_dim,
            img_size=img_size,
            out_channels=in_channels,
        )
        
        self.img_size = img_size
        
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
    
    def search_action(self, z_t, z_target, verbose=False, max_step_norm=15, num_samples=100):
        """Search for the action that leads from z_t to z_target using parallel sampling"""
        # Keep track of device and dtype
        dtype = z_t.dtype
        device = z_t.device
        batch_size = z_t.shape[0]

        assert batch_size == 1, "Batch size must be 1 for parallel action search"
        
        if verbose:
            print(f"Starting parallel action search with {num_samples} samples")
            print(f"z_t shape: {z_t.shape}, dtype: {dtype}")
            print(f"z_target shape: {z_target.shape}, dtype: {dtype}")
        
        # Ensure inputs are detached to avoid gradient tracking
        z_t = z_t.detach()
        z_target = z_target.detach()

        # ------------------------------------------------------------------
        # Quadrant‑based sampling instead of full‑space sampling
        # ------------------------------------------------------------------
        # We first pick one of the four quadrants of the 2‑D action space and
        # then sample **at most** 100 actions uniformly inside that quadrant.
        # This adds structured randomness that has proven useful for training
        # the dynamics model while keeping the search budget low.
        # ------------------------------------------------------------------

        # Pick a random quadrant (0:(+,+), 1:(+,-), 2:(-,+), 3:(-,-))
        quadrant = torch.randint(0, 4, (1,), device=device).item()
        sign_x = 1.0 if quadrant in (0, 1) else -1.0  # Q0 & Q1 have +x
        sign_y = 1.0 if quadrant in (0, 2) else -1.0  # Q0 & Q2 have +y

        # Sample actions uniformly in [0, max_step_norm] then assign signs
        sampled_actions = (
            torch.rand(batch_size, num_samples, self.action_dim,
                       device=device, dtype=dtype) * max_step_norm
        )
        sampled_actions[..., 0] *= sign_x
        sampled_actions[..., 1] *= sign_y

        # Expand z_t / z_target to match the sampled actions
        expanded_z_t = z_t.unsqueeze(1).expand(-1, num_samples, -1)
        expanded_z_target = z_target.unsqueeze(1).expand(-1, num_samples, -1)

        # Reshape expanded states to 2‑D [batch_size*num_samples, encoding_dim]
        expanded_z_t = expanded_z_t.reshape(-1, z_t.shape[-1])
        expanded_z_target = expanded_z_target.reshape(-1, z_target.shape[-1])

        # Reshape actions to 2‑D [batch_size*num_samples, action_dim]
        flat_actions = sampled_actions.reshape(-1, self.action_dim)

        # Forward pass through dynamics model to get predicted next states
        with torch.no_grad():
            # Predict next states for all sampled actions in a single forward pass
            z_next_pred = self.dynamics(expanded_z_t, flat_actions)
            
            # Compute squared error loss between predicted next states and target state
            # Shape: [batch_size * num_samples, encoding_dim]
            squared_errors = (z_next_pred - expanded_z_target) ** 2
            
            # Average across encoding dimensions to get scalar loss per action
            # Shape: [batch_size * num_samples]
            losses = squared_errors.mean(dim=-1)
            
            # Reshape losses to [batch_size, num_samples] to find best action per batch item
            losses = losses.reshape(batch_size, num_samples)
            
            # Get indices of best actions (lowest loss) for each item in the batch
            best_action_indices = torch.argmin(losses, dim=1)
            
            # Get the corresponding best actions
            best_actions = torch.stack([
                sampled_actions[i, best_action_indices[i]] for i in range(batch_size)
            ])
            
            if verbose:
                best_losses = torch.stack([
                    losses[i, best_action_indices[i]] for i in range(batch_size)
                ])
                print(f"Best action losses: {best_losses.cpu().numpy()}")
                print(f"Best actions: {best_actions.cpu().numpy()}")
        
        return best_actions
    
    def to(self, *args, **kwargs):
        """
        Moves and/or casts the parameters and buffers.
        
        This method has the same functionality as PyTorch's nn.Module.to() method,
        but ensures all parameters and buffers of the model are properly converted
        to the same dtype, which is important for mixed precision training.
        """
        # Call the parent class's to() method to handle the actual conversion
        device_or_dtype = args[0] if args else kwargs.get('device', None) or kwargs.get('dtype', None)
        
        # If we're converting to BFloat16, we need to be extra careful
        if device_or_dtype == torch.bfloat16 or kwargs.get('dtype') == torch.bfloat16:
            print("Converting model to BFloat16 with special handling")
            
            # First convert the entire model structure
            model = super().to(*args, **kwargs)
            
            # Explicitly convert all parameters and buffers in submodules
            for module in model.modules():
                for param_name, param in module._parameters.items():
                    if param is not None:
                        module._parameters[param_name] = param.to(torch.bfloat16)
                
                for buffer_name, buffer in module._buffers.items():
                    if buffer is not None:
                        module._buffers[buffer_name] = buffer.to(torch.bfloat16)
            
            # Specifically check that encoder's conv layers have BF16 bias
            for module in model.encoder.modules():
                if isinstance(module, nn.Conv2d):
                    if module.bias is not None:
                        # Double-check the bias tensor
                        if module.bias.dtype != torch.bfloat16:
                            print(f"Converting Conv2d bias from {module.bias.dtype} to BFloat16")
                            module.bias = nn.Parameter(module.bias.to(torch.bfloat16))
            
            return model
        else:
            # For other dtypes, use the standard method
            return super().to(*args, **kwargs)
    
    def print_parameter_count(self):
        """Prints the number of parameters for each component of the model"""
        # Count encoder parameters
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        
        # Count dynamics model parameters
        dynamics_params = sum(p.numel() for p in self.dynamics.parameters())
        
        # Count next goal predictor parameters
        predictor_params = sum(p.numel() for p in self.next_goal_predictor.parameters())
        
        # Count decoder parameters
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        
        # Count total parameters
        total_params = sum(p.numel() for p in self.parameters())
        
        # Print results
        print(f"\nPLDM Model Parameter Counts:")
        print(f"Encoder: {encoder_params:,} parameters")
        print(f"Dynamics Model: {dynamics_params:,} parameters")
        print(f"Next Goal Predictor: {predictor_params:,} parameters")
        print(f"Decoder: {decoder_params:,} parameters")
        print(f"Total: {total_params:,} parameters")
        
        # Print percentage breakdown
        print(f"\nPercentage Breakdown:")
        print(f"Encoder: {encoder_params/total_params*100:.1f}%")
        print(f"Dynamics Model: {dynamics_params/total_params*100:.1f}%")
        print(f"Next Goal Predictor: {predictor_params/total_params*100:.1f}%")
        print(f"Decoder: {decoder_params/total_params*100:.1f}%")
        
        return {
            "encoder": encoder_params,
            "dynamics": dynamics_params,
            "predictor": predictor_params,
            "decoder": decoder_params,
            "total": total_params
        }

    def decode(self, z):
        """Reconstruct observation from latent."""
        return self.decoder(z) 