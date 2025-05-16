import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from pathlib import Path
import math

NUM_CODES   = 256   # size of the discrete latent vocabulary
NUM_ACTIONS = 32   # number of discrete actions for “RL” search-mode


# ---------------------------------------------------------------------
#  Vision Transformer (default)  --- produces logits over NUM_CODES
# ---------------------------------------------------------------------
class ViTEncoder(nn.Module):
    """Vision Transformer Encoder for DotWall environment"""
    def __init__(
        self,
        img_size      = 64,
        patch_size    = 8,
        in_channels   = 3,
        embedding_dim = 512,
        encoding_dim  = 32,      # not used – kept for API symmetry
        temperature   : float = 1.0,
    ):
        super().__init__()
        self.temperature   = temperature
        self.img_size      = img_size
        self.patch_size    = patch_size
        self.embedding_dim = embedding_dim

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

        # Projection to discrete-code logits
        self.projection  = nn.Linear(embedding_dim, NUM_CODES)

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
        elif isinstance(m, ViTEncoder):
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

        logits = self.projection(cls)               # [B,NUM_CODES]
        return logits                               # caller may apply softmax



# ---------------------------------------------------------------------
#  Lightweight CNN alternative encoder (rarely used)
# ---------------------------------------------------------------------
class CNNEncoder(nn.Module):
    def __init__(self, img_size, in_channels, encoding_dim,
                 embedding_dim, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32,          32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32,          32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        flat_dim        = (img_size // 8) ** 2 * 32
        self.projection = nn.Linear(flat_dim, encoding_dim)

    def forward(self, x):
        B = x.size(0)
        out   = self.conv(x).view(B, -1)
        logits = self.projection(out)
        return F.softmax(logits / self.temperature, dim=-1)      # returns prob-vector



# ---------------------------------------------------------------------
#  Latent Dynamics Model  P(z_{t+1}|z_t,a_t)   (predicts logits)
# ---------------------------------------------------------------------
class DynamicsModel(nn.Module):
    def __init__(self,
                 encoding_dim = NUM_CODES,
                 action_dim   = 2,
                 hidden_dim   = 512,
                 num_layers   = 6,
                 temperature  : float = 1.0):
        super().__init__()
        self.temperature  = temperature
        self.input_proj   = nn.Linear(encoding_dim + action_dim, hidden_dim)
        self.input_act    = nn.GELU()

        self.res_blocks   = nn.ModuleList()
        self.layer_norms  = nn.ModuleList()
        for _ in range((num_layers - 2)//2):
            self.res_blocks.append(nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            ]))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.output_proj = nn.Linear(hidden_dim, NUM_CODES)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=1e-3)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    # ------------------------------------------------------------
    def forward(self, z_t, a_t):
        if a_t.dtype != z_t.dtype:
            a_t = a_t.to(dtype=z_t.dtype)

        x = self.input_act(self.input_proj(torch.cat([z_t, a_t], dim=-1)))

        for block, ln in zip(self.res_blocks, self.layer_norms):
            residual = x
            for layer in block:
                x = layer(x)
            x = ln(x + residual)

        logits = self.output_proj(x)
        return logits                                    # raw logits (no softmax)



# ---------------------------------------------------------------------
#  Next-goal Predictor  p(z_next|z_t)  +  value head
# ---------------------------------------------------------------------
class NextGoalPredictor(nn.Module):
    def __init__(self,
                 encoding_dim = NUM_CODES,
                 hidden_dim   = 512,
                 num_layers   = 6,
                 temperature  : float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.input_proj  = nn.Linear(encoding_dim, hidden_dim)
        self.input_act   = nn.GELU()

        self.res_blocks  = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range((num_layers - 2)//2):
            self.res_blocks.append(nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            ]))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.output_proj = nn.Linear(hidden_dim, NUM_ACTIONS)

        # independent value network
        self.value_mlp = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//4), nn.GELU(), nn.LayerNorm(hidden_dim//4),
            nn.Linear(hidden_dim//4, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=1e-3)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    # ------------------------------------------------------------
    def _features(self, z_t):
        x = self.input_act(self.input_proj(z_t))
        for block, ln in zip(self.res_blocks, self.layer_norms):
            residual = x
            for layer in block:
                x = layer(x)
            x = ln(x + residual)
        return x

    # ------------------------------------------------------------
    def forward(self, z_t):
        x      = self._features(z_t)
        logits = self.output_proj(x)                           # [B, NUM_ACTIONS]
        dist   = Categorical(logits = logits / self.temperature)
        idx    = dist.sample()                                 # one-hot index
        z_next = F.one_hot(idx, num_classes=NUM_ACTIONS).float()
        log_p  = dist.log_prob(idx)
        return z_next, log_p

    # utilities
    def get_numerical_stable_distribution(self, z_t):
        logits = self.output_proj(self._features(z_t))
        return F.softmax(logits / self.temperature, dim=-1)

    def value(self, z_t):
        #return self.value_mlp(z_t).squeeze(-1)
        raw_value = self.value_mlp(z_t).squeeze(-1)
        return 50.0 * torch.tanh(raw_value / 50.0)


    def log_prob(self, z_t, z_sample):
        dist = Categorical(logits=self.output_proj(self._features(z_t)) / self.temperature)
        idx  = z_sample.argmax(dim=-1)
        return dist.log_prob(idx)



# ---------------------------------------------------------------------
#  Simple Decoder (used only for warm-up reconstruction loss)
# ---------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, encoding_dim:int, img_size:int = 64, out_channels:int = 3):
        super().__init__()
        assert img_size == 64
        self.fc = nn.Sequential(nn.Linear(encoding_dim, 4*4*256), nn.GELU())
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1), nn.GELU(),  # 8
            nn.ConvTranspose2d(128,64, kernel_size=4,stride=2,padding=1), nn.GELU(),  # 16
            nn.ConvTranspose2d(64, 32, kernel_size=4,stride=2,padding=1), nn.GELU(),  # 32
            nn.ConvTranspose2d(32, 16, kernel_size=4,stride=2,padding=1), nn.GELU(),  # 64
            nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z).view(-1,256,4,4)
        return self.deconv(x)



# ---------------------------------------------------------------------
#  Full PLDM model wrapper
# ---------------------------------------------------------------------
class PLDMModel(nn.Module):
    def __init__(self,
                 img_size          = 64,
                 in_channels       = 3,
                 encoding_dim      = NUM_CODES,
                 action_dim        = 2,
                 hidden_dim        = 512,
                 encoder_embedding = 256,
                 encoder_type      = "vit",
                 temperature       : float = 1.0,
                 next_goal_temp    : float = None,
                 search_mode       : str   = 'pldm',
                 max_step_norm     : float = 15.0):
        super().__init__()
        self.temperature     = temperature
        self.next_goal_temp  = next_goal_temp if next_goal_temp is not None else temperature
        self.search_mode     = search_mode.lower()
        self.max_step_norm   = max_step_norm

        # RL-grid for “rl” search mode
        if self.search_mode == 'rl':
            n          = NUM_ACTIONS
            grid       = int(math.ceil(math.sqrt(n)))
            coords     = torch.linspace(-max_step_norm, max_step_norm, steps=grid)
            xg, yg     = torch.meshgrid(coords, coords, indexing='xy')
            flat       = torch.stack([xg.flatten(), yg.flatten()], dim=1)
            self.register_buffer('rl_actions', flat[:n])
        else:
            self.rl_actions = None

        # encoder choice
        if encoder_type.lower() == "cnn":
            self.encoder = CNNEncoder(img_size, in_channels,
                                      encoding_dim, encoder_embedding, temperature)
        else:
            self.encoder = ViTEncoder(img_size, 8, in_channels,
                                      encoder_embedding, encoding_dim, temperature)

        self.dynamics            = DynamicsModel(encoding_dim, action_dim,
                                                 hidden_dim, temperature=temperature)
        self.next_goal_predictor = NextGoalPredictor(encoding_dim, hidden_dim,
                                                     temperature=self.next_goal_temp)
        self.decoder             = Decoder(encoding_dim, img_size, in_channels)

    # ------------------------------------------------------------
    def encode(self, s_t):
        return self.encoder(s_t)

    def predict_next_goal(self, z_t):
        return self.next_goal_predictor(z_t)

    def predict_next_state(self, z_t, a_t):
        return self.dynamics(z_t, a_t)

    # ------------------------------------------------------------
    def search_action(self, z_t, z_target, verbose=False,
                      max_step_norm=15, num_samples=100, use_quadrant=True):
        if self.search_mode == 'rl':
            idxs = z_target.argmax(dim=1)
            return self.rl_actions[idxs]

        device      = z_t.device
        batch_size  = z_t.shape[0]
        assert batch_size == 1, "search_action assumes batch_size=1"

        # ----- sample candidate actions -----
        if use_quadrant:
            quadrant = torch.randint(0,4,(1,),device=device).item()
            sx = 1. if quadrant in (0,1) else -1.
            sy = 1. if quadrant in (0,2) else -1.
            sampled = torch.rand(batch_size, num_samples, 2, device=device) * max_step_norm
            sampled[...,0] *= sx; sampled[...,1] *= sy
        else:
            sampled = torch.rand(batch_size, num_samples, 2, device=device)*2*max_step_norm - max_step_norm

        # ----- evaluate -----
        ez  = z_t.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, z_t.size(-1))
        et  = z_target.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, z_t.size(-1))
        act = sampled.reshape(-1,2)

        with torch.no_grad():
            z_pred = self.dynamics(ez, act)
            losses = ((z_pred - et)**2).mean(dim=-1).reshape(batch_size, num_samples)
            best   = torch.argmin(losses, dim=1)
            best_a = torch.stack([sampled[i,best[i]] for i in range(batch_size)])

        if verbose:
            print(f"Best action losses: {losses[0,best[0]].item():.4f}  |  action: {best_a[0].cpu().numpy()}")
        return best_a

    # ------------------------------------------------------------
    def print_parameter_count(self):
        enc = sum(p.numel() for p in self.encoder.parameters())
        dyn = sum(p.numel() for p in self.dynamics.parameters())
        pol = sum(p.numel() for p in self.next_goal_predictor.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        tot = enc + dyn + pol + dec
        print(f"\nPLDM Model Parameter Counts:")
        print(f"  Encoder         : {enc:,}")
        print(f"  Dynamics Model  : {dyn:,}")
        print(f"  Next-Goal Pred. : {pol:,}")
        print(f"  Decoder         : {dec:,}")
        print(f"  TOTAL           : {tot:,}\n")

    # ------------------------------------------------------------
    def decode(self, z):
        return self.decoder(z)