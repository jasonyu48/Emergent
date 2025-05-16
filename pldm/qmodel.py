import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from pathlib import Path
import math

DEFAULT_ENCODING_DIM = 256   # default dimension of the discrete latent vocabulary
DEFAULT_NUM_ACTIONS  = 32   # default number of discrete actions


# ---------------------------------------------------------------------
#  Vision Transformer (default)  --- produces latents of shape [B, DEFAULT_ENCODING_DIM]
# ---------------------------------------------------------------------
class ViTEncoder(nn.Module):
    """Vision Transformer Encoder for DotWall environment"""
    def __init__(
        self,
        img_size      = 64,
        patch_size    = 8,
        in_channels   = 3,
        embedding_dim = 512, # Internal embedding dim of ViT
        output_encoding_dim = DEFAULT_ENCODING_DIM, # Dimension of the output latent
        temperature   : float = 1.0, # Not used in forward
    ):
        super().__init__()
        self.temperature   = temperature
        self.img_size      = img_size
        self.patch_size    = patch_size
        self.embedding_dim = embedding_dim # ViT's own embedding dim
        self.output_encoding_dim = output_encoding_dim

        self.num_patches = (img_size // patch_size) ** 2         # 8×8 = 64

        # Patch-embedding
        self.patch_embedding = nn.Conv2d(
            in_channels, embedding_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # Positional & class token
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embedding_dim))
        self.cls_token     = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = embedding_dim,
            nhead           = 8,
            dim_feedforward = embedding_dim * 4,
            dropout         = 0.1,
            batch_first     = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.ln          = nn.LayerNorm(embedding_dim)

        # Projection to output_encoding_dim
        self.projection  = nn.Linear(embedding_dim, output_encoding_dim)

        self.apply(self._init_weights)

    # ------------------------------------------------------------
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=1e-3)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=1e-3)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, ViTEncoder): # Changed from ViTEncoder
            nn.init.trunc_normal_(m.pos_embedding, std=0.02)
            nn.init.trunc_normal_(m.cls_token, std=0.02)

    # ------------------------------------------------------------
    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embedding(x)                 # [B,D,H',W']
        x = x.flatten(2).transpose(1, 2)            # [B,N,D]

        cls = self.cls_token.expand(B, -1, -1)      # [B,1,D]
        x   = torch.cat([cls, x], dim=1) + self.pos_embedding

        x   = self.transformer(x)
        x   = self.ln(x)
        cls = x[:, 0]                               # [B,D]

        encoding = self.projection(cls)             # [B, output_encoding_dim]
        return encoding



# ---------------------------------------------------------------------
#  Lightweight CNN alternative encoder (rarely used)
# ---------------------------------------------------------------------
class CNNEncoder(nn.Module):
    def __init__(self, img_size, in_channels, output_encoding_dim = DEFAULT_ENCODING_DIM,
                 embedding_dim = 256, # internal embedding, not used by current simple CNN
                 temperature: float = 1.0): # Not used in forward
        super().__init__()
        self.temperature = temperature # Not used
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32,          32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32,          32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        flat_dim        = (img_size // 8) ** 2 * 32
        self.projection = nn.Linear(flat_dim, output_encoding_dim)

    def forward(self, x):
        B = x.size(0)
        out   = self.conv(x).view(B, -1)
        encoding = self.projection(out) # [B, output_encoding_dim]
        return encoding



# ---------------------------------------------------------------------
#  Latent Dynamics Model  P(z_{t+1}|z_t,a_t) (predicts next encoding)
# ---------------------------------------------------------------------
class DynamicsModel(nn.Module):
    def __init__(self,
                 encoding_dim = DEFAULT_ENCODING_DIM, # z_t dimension
                 action_dim   = 2,                    # a_t dimension
                 temperature  : float = 1.0):       # Not used in forward
        super().__init__()
        self.temperature  = temperature # Not used
        
        # Hidden dim is 4 times encoding_dim as per user request
        actual_hidden_dim = 4 * encoding_dim

        self.mlp = nn.Sequential(
            nn.Linear(encoding_dim + action_dim, actual_hidden_dim),
            nn.GELU(),
            nn.Linear(actual_hidden_dim, encoding_dim) # Predicts next encoding
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=1e-3)
        elif isinstance(m, nn.LayerNorm): # Should not be used now, but keep for safety
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, z_t, a_t): # z_t: [B, encoding_dim], a_t: [B, action_dim]
        if a_t.dtype != z_t.dtype: # Ensure consistent dtype for concatenation
            a_t = a_t.to(dtype=z_t.dtype)

        x = torch.cat([z_t, a_t], dim=-1)
        predicted_next_z = self.mlp(x) # [B, encoding_dim]
        return predicted_next_z



# ---------------------------------------------------------------------
#  Policy-Value Network: p(action_idx | z_t) + Value(z_t)
# ---------------------------------------------------------------------
class PolicyValueNetwork(nn.Module):
    def __init__(self,
                 encoding_dim = DEFAULT_ENCODING_DIM, # z_t dimension
                 num_actions  = DEFAULT_NUM_ACTIONS,  # Number of discrete actions
                 policy_temperature  : float = 1.0):
        super().__init__()
        self.policy_temperature = policy_temperature
        
        # Hidden dim for policy MLP is 4 times encoding_dim
        policy_hidden_dim = 4 * encoding_dim

        # Policy MLP: predicts logits for discrete actions
        self.policy_mlp = nn.Sequential(
            nn.Linear(encoding_dim, policy_hidden_dim),
            nn.GELU(),
            nn.Linear(policy_hidden_dim, num_actions) # Outputs logits for num_actions
        )

        # Independent value network (structure from original NextGoalPredictor)
        # Hidden dim for value MLP can be different, using original scaling logic for now
        value_hidden_dim = 512 # Original hidden_dim for value MLP context
        if encoding_dim * 2 > value_hidden_dim: # Heuristic, can be adjusted
            value_hidden_dim = encoding_dim * 2

        self.value_mlp = nn.Sequential(
            nn.Linear(encoding_dim, value_hidden_dim), nn.GELU(), nn.LayerNorm(value_hidden_dim),
            nn.Linear(value_hidden_dim, value_hidden_dim//4), nn.GELU(), nn.LayerNorm(value_hidden_dim//4),
            nn.Linear(value_hidden_dim//4, 1) # Outputs scalar value
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=1e-3)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    
    def _policy_features(self, z_t):
        # The policy_mlp itself is the feature extractor + projector
        return self.policy_mlp(z_t)

    def get_action_distribution(self, z_t):
        """Returns a Categorical distribution over actions given current encoding z_t."""
        action_logits = self._policy_features(z_t) # [B, num_actions]
        return Categorical(logits = action_logits / self.policy_temperature)

    def forward(self, z_t, sample=True):
        """
        Predicts action index, its one-hot representation, and log probability.
        This method is for direct use by PLDMModel's get_action_and_log_prob.
        """
        dist = self.get_action_distribution(z_t)
        if sample:
            action_idx = dist.sample()  # [B]
        else:
            action_idx = torch.argmax(dist.logits, dim=-1) # [B]
            
        log_prob = dist.log_prob(action_idx) # [B]
        # The one-hot action is not directly returned here, PLDMModel will create if needed by trainer
        return action_idx, log_prob

    # utilities for training
    def get_numerical_stable_softmax_probs(self, z_t):
        """Returns numerically stable softmax probabilities for actions."""
        action_logits = self._policy_features(z_t)
        return F.softmax(action_logits / self.policy_temperature, dim=-1)

    def value(self, z_t):
        """Computes state-value V(z_t)."""
        return self.value_mlp(z_t).squeeze(-1) # [B]

    def log_prob_of_action(self, z_t, action_idx_or_one_hot):
        """Computes log probability of a given action (index or one-hot) for z_t."""
        dist = self.get_action_distribution(z_t)
        if action_idx_or_one_hot.dim() == z_t.dim() and action_idx_or_one_hot.shape[-1] != 1: # Likely one-hot
             action_indices = action_idx_or_one_hot.argmax(dim=-1)
        else: # Likely indices
            action_indices = action_idx_or_one_hot.squeeze(-1) if action_idx_or_one_hot.dim() > 1 else action_idx_or_one_hot

        return dist.log_prob(action_indices)



# ---------------------------------------------------------------------
#  Simple Decoder (used only for warm-up reconstruction loss)
# ---------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, encoding_dim:int = DEFAULT_ENCODING_DIM, img_size:int = 64, out_channels:int = 3):
        super().__init__()
        assert img_size == 64 # Current decoder is hardcoded for 64x64
        self.fc = nn.Sequential(nn.Linear(encoding_dim, 4*4*256), nn.GELU())
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1), nn.GELU(),  # 8x8
            nn.ConvTranspose2d(128,64, kernel_size=4,stride=2,padding=1), nn.GELU(),  # 16x16
            nn.ConvTranspose2d(64, 32, kernel_size=4,stride=2,padding=1), nn.GELU(),  # 32x32
            nn.ConvTranspose2d(32, 16, kernel_size=4,stride=2,padding=1), nn.GELU(),  # 64x64
            nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z): # z: [B, encoding_dim]
        x = self.fc(z).view(-1,256,4,4)
        return self.deconv(x)



# ---------------------------------------------------------------------
#  Full PLDM model wrapper
# ---------------------------------------------------------------------
class PLDMModel(nn.Module):
    def __init__(self,
                 img_size          = 64,
                 in_channels       = 3,
                 # Model architecture parameters, to be set from args
                 encoding_dim      = DEFAULT_ENCODING_DIM,
                 num_actions       = DEFAULT_NUM_ACTIONS,
                 # Other specific dimensions
                 action_dim_continuous = 2, # Dimension of continuous action for environment
                 dynamics_hidden_dim_multiplier = 4, # For DynamicsModel
                 policy_hidden_dim_multiplier = 4, # For PolicyValueNetwork policy MLP
                 encoder_embedding_dim = 256, # ViT internal, or CNN internal if adapted
                 encoder_type      = "vit",
                 # Temperatures
                 encoder_temp      : float = 1.0, # Not used by current encoders
                 dynamics_temp     : float = 1.0, # Not used by current dynamics
                 policy_temp       : float = 1.0,
                 # Misc
                 max_step_norm     : float = 15.0): # For action grid generation
        super().__init__()
        self.encoding_dim = encoding_dim
        self.num_actions = num_actions
        self.policy_temperature = policy_temp
        self.max_step_norm = max_step_norm

        # Action grid for mapping discrete action indices to continuous actions
        # This grid defines the set of possible continuous actions.
        grid_size = int(math.ceil(math.sqrt(self.num_actions)))
        coords = torch.linspace(-self.max_step_norm, self.max_step_norm, steps=grid_size)
        xg, yg = torch.meshgrid(coords, coords, indexing='xy')
        flat_actions = torch.stack([xg.flatten(), yg.flatten()], dim=1)
        # Ensure the grid has at least num_actions, then take the first num_actions
        if flat_actions.shape[0] < self.num_actions:
            # This case should ideally not happen if num_actions is reasonable for a grid
            # If it does, pad with zeros or raise error. For now, take what's available.
            # Or, more robustly, ensure grid_size^2 >= num_actions
            # Example: if num_actions is 7, grid_size=3 (9 actions), then take first 7.
            # if num_actions is 10, grid_size=4 (16 actions), then take first 10.
             pass # flat_actions might be larger, we slice it.
        self.register_buffer('action_grid', flat_actions[:self.num_actions]) # [num_actions, action_dim_continuous]


        # Encoder choice
        if encoder_type.lower() == "cnn":
            self.encoder = CNNEncoder(img_size, in_channels,
                                      output_encoding_dim=self.encoding_dim,
                                      embedding_dim=encoder_embedding_dim, # Pass along
                                      temperature=encoder_temp)
        else: # Default to ViT
            self.encoder = ViTEncoder(img_size, patch_size=8, # patch_size hardcoded for ViT
                                      in_channels=in_channels,
                                      embedding_dim=encoder_embedding_dim, # ViT's internal embedding
                                      output_encoding_dim=self.encoding_dim,
                                      temperature=encoder_temp)

        self.dynamics_model = DynamicsModel(encoding_dim=self.encoding_dim,
                                            action_dim=action_dim_continuous, # Dynamics uses continuous actions
                                            temperature=dynamics_temp)
        
        self.policy_value_network = PolicyValueNetwork(encoding_dim=self.encoding_dim,
                                                       num_actions=self.num_actions,
                                                       policy_temperature=self.policy_temperature)
        
        self.decoder = Decoder(encoding_dim=self.encoding_dim,
                               img_size=img_size,
                               in_channels=in_channels)

    # ------------------------------------------------------------
    def encode(self, s_t):
        """Encodes observation s_t to a latent vector z_t."""
        return self.encoder(s_t)

    def get_action_and_log_prob(self, z_t, sample=True):
        """
        Given current encoding z_t, gets a discrete action_idx from policy,
        maps it to a continuous action, and returns continuous_action, log_prob_of_idx, action_idx.
        """
        # Get action index and its log probability from the policy network
        action_idx, log_prob = self.policy_value_network(z_t, sample=sample) # action_idx: [B], log_prob: [B]
        
        # Map discrete action index to continuous action using the predefined grid
        # Ensure action_idx is LongTensor for indexing
        continuous_action = self.action_grid[action_idx.long()] # [B, action_dim_continuous]
        
        return continuous_action, log_prob, action_idx

    def predict_next_latent(self, z_t, continuous_action_t):
        """Predicts the next latent state z_{t+1} given z_t and continuous_action_t."""
        return self.dynamics_model(z_t, continuous_action_t)

    # --- Methods for training, passed through to PolicyValueNetwork ---
    def get_action_distribution_probs(self, z_t):
        return self.policy_value_network.get_numerical_stable_softmax_probs(z_t)

    def get_value_prediction(self, z_t):
        return self.policy_value_network.value(z_t)

    def get_log_prob_of_action(self, z_t, action_idx_or_one_hot):
        return self.policy_value_network.log_prob_of_action(z_t, action_idx_or_one_hot)
    # ------------------------------------------------------------

    def print_parameter_count(self):
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        dyn_params = sum(p.numel() for p in self.dynamics_model.parameters())
        pv_net_params = sum(p.numel() for p in self.policy_value_network.parameters())
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        tot_params = enc_params + dyn_params + pv_net_params + dec_params
        
        print("\nPLDM Model Parameter Counts:")
        print(f"  Encoder             : {enc_params:,}")
        print(f"  Dynamics Model      : {dyn_params:,}")
        print(f"  Policy-Value Net.   : {pv_net_params:,}")
        print(f"  Decoder             : {dec_params:,}")
        print(f"  TOTAL               : {tot_params:,}\n")

    # ------------------------------------------------------------
    def decode(self, z):
        """Decodes latent vector z back to an observation (image)."""
        return self.decoder(z)