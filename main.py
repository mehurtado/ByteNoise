import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import time
import os
import glob
from functools import partial
from einops import rearrange, reduce # Using einops for clearer tensor manipulations

# --- Configuration ---
# Data Loading Config
DATA_DIR = "./dataset/USE" # Directory containing .txt files
TRAIN_BATCH_SIZE = 16       # Diffusion models can be memory intensive
TRAIN_SEQ_LEN = 128         # Sequence length for training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model Config
VOCAB_SIZE = 256            # Fixed for byte-level tokenization
BYTE_EMBED_DIM = 128        # Dimension for byte embeddings
MODEL_DIM = 192             # Base dimension for U-Net channels (must be multiple of UNET_DIM_HEAD for attention)

# U-Net Specific Config
UNET_DIM_MULTS = (1, 2, 3, 4) # Multipliers for channel dimensions in U-Net
UNET_TIME_EMB_DIM = MODEL_DIM * 4 # Dimension for time embedding dimension relative to model_dim
UNET_RESNET_GROUPS = 8      # Number of groups for GroupNorm in ResNet blocks
UNET_ATTENTION_HEADS = 6
UNET_ATTENTION_DIM_HEAD = MODEL_DIM // UNET_ATTENTION_HEADS # Ensure MODEL_DIM is divisible

# Diffusion Config
NUM_DIFFUSION_TIMESTEPS = 1000
BETA_SCHEDULE = "cosine"    # "linear" or "cosine"
BETA_START = 0.0001         # For linear schedule
BETA_END = 0.02             # For linear schedule

# Training Config
INITIAL_LEARNING_RATE = 5e-5 # Adjusted learning rate
NUM_EPOCHS = 20 # Diffusion models often require more epochs
LOG_INTERVAL = 50 # Log loss every N effective optimizer steps
SAVE_INTERVAL_EPOCHS = 1 # Save model periodically
MODEL_SAVE_PATH = "./byte_diffusion_model.pth"
GRADIENT_ACCUMULATION_STEPS = 16 # Set to 1 to disable, or >1 to enable
COSINE_LR_T_MAX = NUM_EPOCHS * ( -1 ) # Placeholder, will be set based on dataloader length
COSINE_LR_ETA_MIN = 1e-6    # Minimum learning rate for cosine annealing
TEST_INTERVAL_OPTIMIZER_STEPS = 10000 # Generate a sample every N effective optimizer steps


# --- Helper Functions & Modules ---

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    """Extracts values from a for given timesteps t, reshaping to x_shape."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps, beta_start=BETA_START, beta_end=BETA_END):
    return torch.linspace(beta_start, beta_end, timesteps, device=DEVICE)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672 by Nichol and Dhariwal.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, device=DEVICE)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal Positional Embedding for Timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time): # time: (batch_size,)
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1: # zero pad if dim is odd
            embeddings = F.pad(embeddings, (0,1))
        return embeddings

class ByteEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
    def forward(self, x):
        # x: (batch_size, seq_len)
        return self.embedding(x) # Output: (batch_size, seq_len, embed_dim)

# --- U-Net Components ---

class ResnetBlock1D(nn.Module):
    def __init__(self, dim_in, dim_out, *, time_emb_dim=None, groups=UNET_RESNET_GROUPS, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = nn.Sequential(
            nn.Conv1d(dim_in, dim_out, kernel_size=3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU()
        )
        self.dropout = nn.Dropout(dropout)
        self.block2 = nn.Sequential(
            nn.Conv1d(dim_out, dim_out, kernel_size=3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU()
        )
        self.res_conv = nn.Conv1d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, time_emb=None): # x: (batch, channels, seq_len)
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1') # Reshape for broadcasting
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            h = h * (scale + 1) + shift
        
        h = self.dropout(h)
        h = self.block2(h)
        return h + self.res_conv(x)

class Attention1D(nn.Module):
    def __init__(self, dim, heads=UNET_ATTENTION_HEADS, dim_head=UNET_ATTENTION_DIM_HEAD, groups=UNET_RESNET_GROUPS):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = nn.GroupNorm(groups, dim)
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x): # x: (batch, channels, seq_len)
        b, c, n = x.shape
        x_norm = self.norm(x)
        qkv = self.to_qkv(x_norm).chunk(3, dim=1) # (b, h*d, n), (b, h*d, n), (b, h*d, n)
        
        # Rearrange for multi-head attention: b (h d) n -> b h n d
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h n d', h=self.heads), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        
        # Rearrange back: b h n d -> b (h d) n
        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out) + x # Add residual connection

class Downsample1D(nn.Module):
    def __init__(self, dim_in, dim_out=None):
        super().__init__()
        self.conv = nn.Conv1d(dim_in, default(dim_out, dim_in), kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample1D(nn.Module):
    def __init__(self, dim_in, dim_out=None):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(dim_in, default(dim_out, dim_in), kernel_size=3, padding=1)
        )
    def forward(self, x):
        return self.conv(x)

class UNet1D(nn.Module):
    def __init__(
        self,
        model_dim, # Base dimension
        out_dim=None, # Output dimension for noise prediction (should be BYTE_EMBED_DIM)
        dim_mults=UNET_DIM_MULTS,
        in_embed_dim=BYTE_EMBED_DIM, # Dimension of input byte embeddings
        time_emb_dim_mult=4, # Multiplier for time embedding dimension relative to model_dim
        resnet_groups=UNET_RESNET_GROUPS,
        attn_heads=UNET_ATTENTION_HEADS,
        attn_dim_head=UNET_ATTENTION_DIM_HEAD,
        dropout=0.1
    ):
        super().__init__()
        self.in_embed_dim = in_embed_dim
        self.out_dim = default(out_dim, in_embed_dim) # Predict noise of same dim as input embeddings

        # --- Time Embedding ---
        time_dim = model_dim * time_emb_dim_mult
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(model_dim), # Initial time embedding uses model_dim
            nn.Linear(model_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # --- Initial Convolution ---
        # Input x will be (batch, seq_len, in_embed_dim), permute to (batch, in_embed_dim, seq_len)
        self.init_conv = nn.Conv1d(in_embed_dim, model_dim, kernel_size=7, padding=3)

        dims = [model_dim, *map(lambda m: model_dim * m, dim_mults)]
        in_out_dims = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out_dims)

        block_klass = partial(ResnetBlock1D, groups=resnet_groups, time_emb_dim=time_dim, dropout=dropout)
        attn_klass = partial(Attention1D, heads=attn_heads, dim_head=attn_dim_head, groups=resnet_groups)

        # --- Downsampling Path ---
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out_dims):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in), # Two ResNet blocks per level
                block_klass(dim_in, dim_in),
                attn_klass(dim_in),
                Downsample1D(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1)
            ]))

        # --- Bottleneck ---
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = attn_klass(mid_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim)

        # --- Upsampling Path ---
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out_dims)):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out), # Skip connection doubles channels
                block_klass(dim_out + dim_in, dim_out), # (dim_out from prev upsample + dim_in from skip)
                attn_klass(dim_out),
                Upsample1D(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding=1)
            ]))
        
        # --- Final Convolution ---
        self.final_res_block = block_klass(model_dim * 2, model_dim) # model_dim from init_conv skip + model_dim from last upsample
        self.final_conv = nn.Conv1d(model_dim, self.out_dim, 1)


    def forward(self, x_embed, time): # x_embed: (batch, seq_len, in_embed_dim), time: (batch,)
        # Permute input for Conv1d: (batch, in_embed_dim, seq_len)
        x = rearrange(x_embed, 'b n c -> b c n')

        t_emb = self.time_mlp(time)

        h = self.init_conv(x)
        skip_connections = [h.clone()] # Store initial conv output for final skip

        # Downsampling
        for resnet_block1, resnet_block2, attention, downsample in self.downs:
            h = resnet_block1(h, t_emb)
            skip_connections.append(h.clone())
            h = resnet_block2(h, t_emb)
            h = attention(h)
            skip_connections.append(h.clone())
            h = downsample(h)

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Upsampling
        for resnet_block1, resnet_block2, attention, upsample in self.ups:
            h = torch.cat((h, skip_connections.pop()), dim=1) # Add skip from attention
            h = resnet_block1(h, t_emb)
            h = torch.cat((h, skip_connections.pop()), dim=1) # Add skip from resnet_block1
            h = resnet_block2(h, t_emb)
            h = attention(h)
            h = upsample(h)
        
        h = torch.cat((h, skip_connections.pop()), dim=1) # Final skip from init_conv
        h = self.final_res_block(h, t_emb)
        out = self.final_conv(h) # (batch, out_dim, seq_len)

        # Permute back to (batch, seq_len, out_dim)
        return rearrange(out, 'b c n -> b n c')


# --- Gaussian Diffusion Process ---
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_model, # The U-Net model
        *,
        seq_len,
        num_timesteps=NUM_DIFFUSION_TIMESTEPS,
        beta_schedule_fn=BETA_SCHEDULE, # 'linear' or 'cosine'
        loss_type='l2', # 'l1' or 'l2'
        clip_denoised=True
    ):
        super().__init__()
        self.denoise_model = denoise_model
        self.seq_len = seq_len
        self.num_timesteps = num_timesteps
        self.clip_denoised = clip_denoised
        self.loss_type = loss_type

        if beta_schedule_fn == 'linear':
            betas = linear_beta_schedule(num_timesteps)
        elif beta_schedule_fn == 'cosine':
            betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule_fn}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # Add 1.0 at the beginning

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        # Clamp log to avoid log(0)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x0 from x_t and predicted noise (epsilon)."""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start_embed, x_t_embed, t):
        """Compute the mean and variance of q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t_embed.shape) * x_start_embed +
            extract(self.posterior_mean_coef2, t, x_t_embed.shape) * x_t_embed
        )
        posterior_variance = extract(self.posterior_variance, t, x_t_embed.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t_embed.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.no_grad()
    def p_sample(self, x_t_embed, t, t_index): # t is tensor, t_index is scalar
        """Denoise one step from x_t to x_{t-1}."""
        betas_t = extract(self.betas, t, x_t_embed.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t_embed.shape)
        # Corrected: alpha_t = 1 - beta_t
        sqrt_recip_alphas_t = extract(torch.sqrt(1.0 / (1.0 - self.betas)), t, x_t_embed.shape)


        # Equation 11 in DDPM: model predicts noise
        predicted_noise = self.denoise_model(x_t_embed, t)
        model_mean = sqrt_recip_alphas_t * (x_t_embed - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x_t_embed.shape)
            noise = torch.randn_like(x_t_embed)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        """Generate samples by denoising from T to 0."""
        batch_size = shape[0]
        # Start with random noise
        x_t_embed = torch.randn(shape, device=DEVICE) # shape: (batch, seq_len, embed_dim)

        for i in reversed(range(0, self.num_timesteps)):
            current_t_tensor = torch.full((batch_size,), i, device=DEVICE, dtype=torch.long)
            x_t_embed = self.p_sample(x_t_embed, current_t_tensor, i)
        
        # x_t_embed is now approximately x_0_embed
        if self.clip_denoised: # Optional: clip to [-1, 1] if embeddings are normalized
             x_t_embed = torch.clamp(x_t_embed, -1., 1.)
        return x_t_embed


    def q_sample(self, x_start_embed, t, noise=None):
        """Sample x_t from x_start using the reparameterization trick (forward diffusion)."""
        noise = default(noise, lambda: torch.randn_like(x_start_embed))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start_embed.shape) * x_start_embed +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_embed.shape) * noise
        )

    def p_losses(self, x_start_embed, t, noise=None): # x_start_embed: (batch, seq_len, embed_dim)
        """Calculate loss: model predicts noise."""
        noise = default(noise, lambda: torch.randn_like(x_start_embed))

        x_t_embed = self.q_sample(x_start_embed, t, noise=noise)
        predicted_noise = self.denoise_model(x_t_embed, t)

        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        else:
            raise ValueError(f'unknown loss type {self.loss_type}')
        return loss

    def forward(self, x_start_embed): # x_start_embed: (batch, seq_len, embed_dim)
        """Training forward pass: returns loss."""
        batch_size = x_start_embed.shape[0]
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_start_embed.device).long()
        return self.p_losses(x_start_embed, t)


# --- Data Loading ---
def load_text_data(data_dir):
    text = ""
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in directory: {data_dir}")
    print(f"Found {len(txt_files)} .txt files in {data_dir}")
    for filepath in txt_files:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text += f.read()
        except Exception as e:
            print(f"Warning: Could not read file {filepath}: {e}")
    print(f"Total characters loaded: {len(text):,}")
    return text

class ByteTokenizer:
    def __init__(self):
        self.vocab_size = VOCAB_SIZE
        print(f"Vocabulary size: {self.vocab_size} (fixed for byte-level)")

    def encode(self, string: str) -> list[int]:
        return list(string.encode('utf-8', errors='replace'))

    def decode(self, byte_list: list[int]) -> str:
        processed_byte_list = [b if 0 <= b <= 255 else ord('?') for b in byte_list]
        return bytes(processed_byte_list).decode('utf-8', errors='replace')

class TextDataset(Dataset):
    def __init__(self, tokenized_text_bytes: list[int], seq_len: int, pad_token_id=0):
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        
        # Ensure data is at least seq_len long by padding if necessary
        if len(tokenized_text_bytes) < seq_len:
            padding_needed = seq_len - len(tokenized_text_bytes)
            tokenized_text_bytes.extend([self.pad_token_id] * padding_needed)
            print(f"Warning: Initial data was shorter than seq_len. Padded to {seq_len} bytes.")

        self.data = torch.tensor(tokenized_text_bytes, dtype=torch.long)
        
        # Calculate number of sequences, ensuring we can form at least one full sequence
        if self.data.size(0) >= self.seq_len:
            self.num_sequences = self.data.size(0) // self.seq_len
        else: # Should not happen due to padding above, but as a safeguard
            self.num_sequences = 0 
            print(f"Error: Text length ({self.data.size(0)}) is less than seq_len ({self.seq_len}) even after padding. No sequences will be generated.")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len
        return self.data[start_idx:end_idx] # Returns (seq_len,) tensor of byte IDs


# --- Main Model Combining Embedder, U-Net, Diffusion, and Logits Head ---
class ByteDiffusionLM(nn.Module):
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        byte_embed_dim=BYTE_EMBED_DIM,
        model_dim=MODEL_DIM,
        unet_dim_mults=UNET_DIM_MULTS,
        seq_len=TRAIN_SEQ_LEN,
        num_diffusion_timesteps=NUM_DIFFUSION_TIMESTEPS,
        beta_schedule=BETA_SCHEDULE,
        loss_type='l2'
    ):
        super().__init__()
        self.byte_embedder = ByteEmbedding(vocab_size, byte_embed_dim)
        
        self.denoise_unet = UNet1D(
            model_dim=model_dim,
            out_dim=byte_embed_dim, # U-Net predicts noise of the same dim as embeddings
            in_embed_dim=byte_embed_dim,
            dim_mults=unet_dim_mults
            # Other U-Net params like attn_heads, resnet_groups are taken from global config or defaults
        )
        
        self.gaussian_diffusion = GaussianDiffusion(
            denoise_model=self.denoise_unet,
            seq_len=seq_len,
            num_timesteps=num_diffusion_timesteps,
            beta_schedule_fn=beta_schedule,
            loss_type=loss_type
        )
        
        # To project final denoised embeddings (x0_embed) back to byte logits
        self.to_logits = nn.Linear(byte_embed_dim, vocab_size)

    def forward(self, x_bytes): # x_bytes: (batch, seq_len) integer tokens
        """Training forward pass: returns loss."""
        x_start_embed = self.byte_embedder(x_bytes) # (batch, seq_len, embed_dim)
        loss = self.gaussian_diffusion(x_start_embed) # Diffusion process handles noising and loss
        return loss

    @torch.no_grad()
    def sample(self, batch_size=1):
        """Generate samples by denoising from T to 0."""
        self.eval()
        # Shape for initial noise: (batch_size, seq_len, byte_embed_dim)
        shape = (batch_size, self.gaussian_diffusion.seq_len, self.byte_embedder.embedding.embedding_dim)
        
        # p_sample_loop returns the predicted x0_embed
        x0_embed_predicted = self.gaussian_diffusion.p_sample_loop(shape)
        
        # Project embeddings to logits
        logits = self.to_logits(x0_embed_predicted) # (batch, seq_len, vocab_size)
        sampled_byte_ids = torch.argmax(logits, dim=-1) # (batch, seq_len)
        # self.train() # Keep in eval mode if called during eval; switch back in training loop
        return sampled_byte_ids


# --- Main Execution Block ---
if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    # Initialize tokenizer here so it can be used in the test_model_mid_epoch function
    tokenizer = ByteTokenizer()

    print("\n--- Loading and Tokenizing Data (Byte-Level) ---")
    try:
        full_text = load_text_data(DATA_DIR)
        if not full_text:
            raise ValueError("Loaded text is empty. Check data directory and file contents.")

        tokenized_data_bytes = tokenizer.encode(full_text)

        if not tokenized_data_bytes:
             raise ValueError("Tokenized data is empty after encoding.")

        train_dataset = TextDataset(tokenized_data_bytes, TRAIN_SEQ_LEN)
        if len(train_dataset) == 0:
            print(f"Dataset creation resulted in 0 sequences. Text length: {len(tokenized_data_bytes)}, TRAIN_SEQ_LEN: {TRAIN_SEQ_LEN}")
            print("Exiting. Please check your data or TRAIN_SEQ_LEN.")
            exit()
        
        num_workers_val = 4 if DEVICE == 'cuda' else 0 
        train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=num_workers_val, pin_memory=True if DEVICE=='cuda' else False)
        print(f"Dataset created with {len(train_dataset)} sequences of {TRAIN_SEQ_LEN} bytes.")
        print(f"DataLoader created with {len(train_loader)} batches per epoch.")
        
        optimizer_steps_per_epoch = len(train_loader) // GRADIENT_ACCUMULATION_STEPS
        if len(train_loader) % GRADIENT_ACCUMULATION_STEPS != 0:
             optimizer_steps_per_epoch +=1 
        COSINE_LR_T_MAX = optimizer_steps_per_epoch * NUM_EPOCHS
        print(f"CosineAnnealingLR T_max set to: {COSINE_LR_T_MAX} (optimizer steps)")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error during data loading: {e}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during data setup: {e}")
        exit()

    print("\n--- Initializing Byte Diffusion Language Model ---")
    model = ByteDiffusionLM(
        seq_len=TRAIN_SEQ_LEN
    ).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=COSINE_LR_T_MAX, eta_min=COSINE_LR_ETA_MIN)
    
    start_epoch = 0
    completed_optimizer_steps_total = 0 # To track total optimizer steps for mid-epoch testing

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Resuming training from checkpoint: {MODEL_SAVE_PATH}")
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict']) 
        start_epoch = checkpoint['epoch'] + 1
        completed_optimizer_steps_total = checkpoint.get('completed_optimizer_steps_total', 0) # Load if exists
        print(f"Resumed from epoch {start_epoch}. Optimizer and Scheduler states loaded. Completed optimizer steps: {completed_optimizer_steps_total}")


    print(f"\n--- Starting Training ({NUM_EPOCHS} epochs) ---")
    
    total_start_time = time.time() 

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_start_time = time.time()
        epoch_total_loss = 0
        # optimizer.zero_grad() # Moved inside the loop before accumulation check

        current_epoch_optimizer_steps = 0

        for batch_idx, x_bytes in enumerate(train_loader):
            if batch_idx == 0: # Zero gradients at the start of an epoch's batches
                optimizer.zero_grad()

            x_bytes = x_bytes.to(DEVICE)
            
            loss = model(x_bytes)
            loss = loss / GRADIENT_ACCUMULATION_STEPS 
            loss.backward()
            
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
                optimizer.step()
                optimizer.zero_grad() 
                scheduler.step() 
                
                completed_optimizer_steps_total += 1
                current_epoch_optimizer_steps +=1

                # --- Mid-Epoch Testing ---
                if TEST_INTERVAL_OPTIMIZER_STEPS > 0 and completed_optimizer_steps_total % TEST_INTERVAL_OPTIMIZER_STEPS == 0:
                    print(f"\n--- Testing model at Epoch {epoch+1}, Optimizer Step {completed_optimizer_steps_total} ---")
                    model.eval()
                    with torch.no_grad():
                        generated_byte_sequence_tensor = model.sample(batch_size=1)
                        generated_bytes_list = generated_byte_sequence_tensor[0].cpu().tolist()
                        generated_text = tokenizer.decode(generated_bytes_list)
                        print(f"Sample: \"{generated_text[:100]}...\"") # Print a snippet
                    model.train()
                    print("--- End Test Sample ---\n")


            epoch_total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS 
            
            # Logging based on effective optimizer steps
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                if current_epoch_optimizer_steps % LOG_INTERVAL == 0 or (batch_idx + 1) == len(train_loader) : # Check if it's a logging step
                    current_loss_for_log = loss.item() * GRADIENT_ACCUMULATION_STEPS 
                    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Opt_Step_Epoch [{current_epoch_optimizer_steps}/{optimizer_steps_per_epoch}], TotalOptStep [{completed_optimizer_steps_total}], LR: {scheduler.get_last_lr()[0]:.2e}, Loss: {current_loss_for_log:.4f}")
        
        avg_epoch_loss = epoch_total_loss / (len(train_loader) / GRADIENT_ACCUMULATION_STEPS) # Avg loss per optimizer step
        epoch_end_time = time.time()
        print(f"--- Epoch {epoch+1} Finished. Avg Loss: {avg_epoch_loss:.4f} (Took {epoch_end_time - epoch_start_time:.2f} seconds) ---")

        if (epoch + 1) % SAVE_INTERVAL_EPOCHS == 0 or (epoch + 1) == NUM_EPOCHS:
            print(f"Saving model checkpoint at epoch {epoch+1}...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), 
                'loss': avg_epoch_loss,
                'completed_optimizer_steps_total': completed_optimizer_steps_total # Save total optimizer steps
            }, MODEL_SAVE_PATH)
            print(f"Model checkpoint saved to {MODEL_SAVE_PATH}")

    total_training_time = time.time() - total_start_time
    print(f"--- Training Finished (Total time: {total_training_time:.2f} seconds) ---")

    print("\n--- Running Final Generation Example (Byte-Level Diffusion) ---")
    model.eval()
    print(f"Generating {TRAIN_SEQ_LEN} bytes from noise...")
    
    num_samples_to_generate = 3
    for i in range(num_samples_to_generate):
        generated_byte_sequence_tensor = model.sample(batch_size=1) 
        generated_bytes_list = generated_byte_sequence_tensor[0].cpu().tolist()
        
        generated_text = tokenizer.decode(generated_bytes_list)
        print(f"\n--- Generated Sample {i+1} ---")
        print(generated_text)
        print(f"Raw bytes: {generated_bytes_list[:30]}...") 
        print("--- End Sample ---")

