import math
import numpy as np
import torch
from torch import nn
from typing import Dict, Tuple, Optional, Union
from einops import rearrange, repeat

def modulate(x, shift, scale):
    # For THWBD format: x shape is (T, H, W, B, D)
    # shift, scale shape: (B, D)
    # We need to reshape shift/scale to (1, 1, 1, B, D) to broadcast
    if x.ndim == 5:  # THWBD format
        return x * (1 + scale.unsqueeze(0).unsqueeze(0).unsqueeze(0)) + shift.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:  # Standard (S, B, D) format
        return x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected tensor dimensions: {x.ndim}")

# ===================== RMS NORMALIZATION =====================
class RMSNorm(nn.Module):
    """
    Pure PyTorch implementation of RMSNorm to replace te.pytorch.RMSNorm
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Cast to float32 for stability (matches transformer_engine behavior)
        input_dtype = x.dtype
        x = x.float()
        
        # Compute RMS normalization
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        
        # Apply weight and cast back
        return (x_normed * self.weight).to(input_dtype)

def get_normalization_pure_torch(name: str, channels: int):
    """Pure PyTorch replacement for transformer_engine's get_normalization"""
    if name == "I":
        return nn.Identity()
    elif name == "R":
        return RMSNorm(channels, eps=1e-6)
    else:
        raise ValueError(f"Normalization {name} not found")

# ===================== SINCOS POS EMB (for extra_per_block_abs_pos_emb) =====================
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    EXACTLY matching official implementation
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    # CRITICAL: Order is [sin, cos] not [cos, sin]!
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

def normalize(x: torch.Tensor, dim: Optional[list] = None, eps: float = 0) -> torch.Tensor:
    """
    EXACTLY matching official normalize function
    """
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    # CRITICAL: This specific formula for adding epsilon
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

class SinCosPosEmbAxis(nn.Module):
    """
    Fixed implementation matching official for extra_per_block_abs_pos_emb
    """
    def __init__(
        self,
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        **kwargs
    ):
        super().__init__()
        dim = model_channels
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t

        # Generate embeddings with extrapolation
        emb_h = get_1d_sincos_pos_embed_from_grid(dim_h, np.arange(len_h) * 1.0 / h_extrapolation_ratio)
        emb_w = get_1d_sincos_pos_embed_from_grid(dim_w, np.arange(len_w) * 1.0 / w_extrapolation_ratio)
        emb_t = get_1d_sincos_pos_embed_from_grid(dim_t, np.arange(len_t) * 1.0 / t_extrapolation_ratio)

        self.register_buffer("pos_emb_h", torch.from_numpy(emb_h).float(), persistent=False)
        self.register_buffer("pos_emb_w", torch.from_numpy(emb_w).float(), persistent=False)
        self.register_buffer("pos_emb_t", torch.from_numpy(emb_t).float(), persistent=False)

    def forward(self, x_patches: torch.Tensor, fps=None) -> torch.Tensor:
        B, T, H, W, C = x_patches.shape
        device = x_patches.device
        dtype = x_patches.dtype
        
        # Move buffers to correct device if needed
        if self.pos_emb_h.device != device:
            self.pos_emb_h = self.pos_emb_h.to(device)
            self.pos_emb_w = self.pos_emb_w.to(device)
            self.pos_emb_t = self.pos_emb_t.to(device)
        
        emb_h_H = self.pos_emb_h[:H]
        emb_w_W = self.pos_emb_w[:W]
        emb_t_T = self.pos_emb_t[:T]
        
        # Concatenate in the EXACT order as official
        emb = torch.cat(
            [
                repeat(emb_t_T, "t d-> b t h w d", b=B, h=H, w=W),
                repeat(emb_h_H, "h d-> b t h w d", b=B, t=T, w=W),
                repeat(emb_w_W, "w d-> b t h w d", b=B, t=T, h=H),
            ],
            dim=-1
        ).to(dtype)
        
        # CRITICAL: Apply normalization exactly as official
        return normalize(emb, dim=-1, eps=1e-6)

# ===================== ROPE IMPLEMENTATION =====================
def apply_rotary_pos_emb_pure_torch(x: torch.Tensor, freqs: torch.Tensor, tensor_format: str = "sbhd", fused: bool = True):
    """
    Apply rotary position embeddings to input tensor.
    
    CRITICAL: The official transformer_engine version expects raw frequencies,
    not pre-computed sin/cos values. This is a key difference!
    
    Args:
        x: Input tensor in format specified by tensor_format
        freqs: Raw frequency values (not sin/cos), shape should broadcast with x
        tensor_format: Format string for input tensor
        fused: Whether to use fused implementation
    """
    # Ensure freqs are on the same device and dtype as x
    device = x.device
    dtype = x.dtype
    
    if tensor_format == "sbhd":
        # x shape: (S, B, H, D)
        # freqs shape: (S, 1, 1, D) 
        seq_len, batch, heads, dim = x.shape
        
        # freqs contains raw frequencies, need to compute sin/cos
        freqs = freqs.squeeze(1).squeeze(1)  # (S, D)
        
        # Expand frequencies to match x dimensions
        freqs = freqs.unsqueeze(1).unsqueeze(1).expand(seq_len, batch, heads, dim)
        
        # Split x into two halves for rotation
        x_r, x_i = x.chunk(2, dim=-1)
        
        # Apply rotation using angle (frequency * position is already in freqs)
        cos_freqs = freqs[..., :dim//2].cos().to(dtype)
        sin_freqs = freqs[..., :dim//2].sin().to(dtype)
        
        # Apply rotation formula: (x_r + i*x_i) * (cos + i*sin)
        out_r = x_r * cos_freqs - x_i * sin_freqs
        out_i = x_r * sin_freqs + x_i * cos_freqs
        
        # Concatenate back
        return torch.cat([out_r, out_i], dim=-1)
    else:
        raise NotImplementedError(f"Tensor format {tensor_format} not implemented")

class CleanRoPE3D(nn.Module):
    def __init__(self, head_dim: int, 
                 h_extrapolation_ratio: float = 1.0,
                 w_extrapolation_ratio: float = 1.0,
                 t_extrapolation_ratio: float = 1.0,
                 **kwargs):
        super().__init__()
        self.head_dim = head_dim
        # Keep buffer for checkpoint compatibility
        self.register_buffer("seq", torch.arange(128, dtype=torch.float))
        
        # Split dimensions EXACTLY like official
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h  
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t
        
        self.dim_h = dim_h
        self.dim_w = dim_w
        self.dim_t = dim_t
        
        # CRITICAL: Create frequency ranges - no cuda() here, will move to device later
        dim_spatial_range = torch.arange(0, dim_h, 2)[:dim_h//2].float() / dim_h
        dim_temporal_range = torch.arange(0, dim_t, 2)[:dim_t//2].float() / dim_t
        
        self.register_buffer("dim_spatial_range", dim_spatial_range, persistent=False)
        self.register_buffer("dim_temporal_range", dim_temporal_range, persistent=False)
        
        # Calculate NTK factors EXACTLY like official
        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2)) if dim_h > 2 else h_extrapolation_ratio
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2)) if dim_w > 2 else w_extrapolation_ratio
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2)) if dim_t > 2 else t_extrapolation_ratio

    def forward(self, x_patches: torch.Tensor, fps=None):
        B, T_p, H_p, W_p, D_model = x_patches.shape
        device = x_patches.device
        dtype = x_patches.dtype
        
        # Move buffers to correct device if needed
        if self.dim_spatial_range.device != device:
            self.dim_spatial_range = self.dim_spatial_range.to(device)
            self.dim_temporal_range = self.dim_temporal_range.to(device)
            self.seq = self.seq.to(device)
        
        # Compute theta values with NTK scaling
        h_theta = 10000.0 * self.h_ntk_factor
        w_theta = 10000.0 * self.w_ntk_factor  
        t_theta = 10000.0 * self.t_ntk_factor
        
        # Compute frequencies
        h_spatial_freqs = 1.0 / (h_theta ** self.dim_spatial_range)
        w_spatial_freqs = 1.0 / (w_theta ** self.dim_spatial_range)
        temporal_freqs = 1.0 / (t_theta ** self.dim_temporal_range)
        
        # Create position sequences
        seq_t = self.seq[:T_p] if T_p <= len(self.seq) else torch.arange(T_p, device=device, dtype=torch.float)
        seq_h = self.seq[:H_p] if H_p <= len(self.seq) else torch.arange(H_p, device=device, dtype=torch.float)
        seq_w = self.seq[:W_p] if W_p <= len(self.seq) else torch.arange(W_p, device=device, dtype=torch.float)
        
        # Compute outer products for half embeddings (frequencies only, not sin/cos yet)
        half_emb_t = torch.outer(seq_t, temporal_freqs)
        half_emb_h = torch.outer(seq_h, h_spatial_freqs)
        half_emb_w = torch.outer(seq_w, w_spatial_freqs)
        
        # CRITICAL: Concatenate in [t, h, w] pattern, then repeat ENTIRE pattern
        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H_p, w=W_p),
                repeat(half_emb_h, "h d -> t h w d", t=T_p, w=W_p),
                repeat(half_emb_w, "w d -> t h w d", t=T_p, h=H_p),
            ] * 2,  # Repeat the ENTIRE concatenated list
            dim=-1,
        )
        
        # Reshape to expected format
        return rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").to(dtype)

# ===================== ATTENTION IMPLEMENTATION =====================
class Attention(nn.Module):
    """
    Pure PyTorch implementation matching the official transformer_engine-based Attention
    """
    def __init__(
        self,
        query_dim: int,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        qkv_bias: bool = False,
        out_bias: bool = False,
        qkv_norm: str = "RRI",  # R=RMSNorm for q,k; I=Identity for v
        qkv_norm_mode: str = "per_head",
        backend: str = "torch",
        qkv_format: str = "sbhd",
    ) -> None:
        super().__init__()

        self.is_selfattn = context_dim is None
        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head
        self.qkv_norm_mode = qkv_norm_mode
        self.qkv_format = qkv_format
        self.backend = backend

        if self.qkv_norm_mode == "per_head":
            norm_dim = dim_head
        else:
            raise ValueError(f"Normalization mode {self.qkv_norm_mode} not found")

        # Create Q, K, V projections with normalization
        self.to_q = nn.Sequential(
            nn.Linear(query_dim, inner_dim, bias=qkv_bias),
            get_normalization_pure_torch(qkv_norm[0], norm_dim),
        )
        self.to_k = nn.Sequential(
            nn.Linear(context_dim, inner_dim, bias=qkv_bias),
            get_normalization_pure_torch(qkv_norm[1], norm_dim),
        )
        self.to_v = nn.Sequential(
            nn.Linear(context_dim, inner_dim, bias=qkv_bias),
            get_normalization_pure_torch(qkv_norm[2], norm_dim),
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, bias=out_bias),
            nn.Dropout(dropout),
        )

    def cal_qkv(self, x, context=None, mask=None, rope_emb=None, **kwargs):
        """
        Calculate Q, K, V with per-head normalization
        Now expects 3D tensors only - 5D handling is done in VideoAttn
        """
        # Apply linear projections
        q = self.to_q[0](x)
        
        # Handle context for cross-attention
        if context is None:
            context = x
            
        k = self.to_k[0](context)
        v = self.to_v[0](context)
        
        # Reshape for per-head normalization
        q, k, v = map(
            lambda t: rearrange(t, "s b (n c) -> s b n c", n=self.heads, c=self.dim_head),
            (q, k, v),
        )
        
        # Apply per-head normalization
        q = self.to_q[1](q)
        k = self.to_k[1](k)
        v = self.to_v[1](v)
        
        # Apply RoPE only for self-attention
        if self.is_selfattn and rope_emb is not None:
            q = apply_rotary_pos_emb_pure_torch(q, rope_emb, tensor_format=self.qkv_format, fused=True)
            k = apply_rotary_pos_emb_pure_torch(k, rope_emb, tensor_format=self.qkv_format, fused=True)
            
        return q, k, v

    def cal_attn(self, q, k, v, mask=None):
        """Calculate attention - handles both THWBD and standard formats"""
        # q, k, v are (S, B, H, D) where S could be T*H*W or M
        S_q, B, H, D = q.shape
        S_kv = k.shape[0]
        
        # Convert to (B, H, S, D) for PyTorch's scaled_dot_product_attention
        q = q.permute(1, 2, 0, 3)  # (B, H, S_q, D)
        k = k.permute(1, 2, 0, 3)  # (B, H, S_kv, D)
        v = v.permute(1, 2, 0, 3)  # (B, H, S_kv, D)
        
        # Apply scaled dot-product attention
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=0.0,
            is_causal=False
        )
        
        # Convert back to (S, B, H, D)
        out = out.permute(2, 0, 1, 3)  # (S_q, B, H, D)
        
        # Flatten heads and apply output projection
        out = rearrange(out, "s b h d -> s b (h d)")
        return self.to_out(out)

    def forward(self, x, context=None, mask=None, rope_emb=None, **kwargs):
        """
        Forward pass matching the official Attention interface
        """
        q, k, v = self.cal_qkv(x, context, mask, rope_emb=rope_emb, **kwargs)
        return self.cal_attn(q, k, v, mask)

# ===================== VIDEO ATTENTION WRAPPER =====================
class VideoAttn(nn.Module):
    def __init__(self, x_dim: int, context_dim: Optional[int], num_heads: int, 
                 bias: bool = False, **kwargs):
        super().__init__()
        self.attn = Attention(
            query_dim=x_dim, 
            context_dim=context_dim, 
            heads=num_heads,
            dim_head=x_dim // num_heads, 
            qkv_bias=bias, 
            out_bias=bias,
            qkv_norm="RRI",
            qkv_norm_mode="per_head",
            backend="torch",
            qkv_format="sbhd"
        )
    
    def forward(self, x, context=None, rope_emb_L_1_1_D=None, **kwargs):
        # Store original shape if 5D
        if x.ndim == 5:
            T, H, W, B, D = x.shape
            # Flatten spatial-temporal dimensions
            x_flat = rearrange(x, "t h w b d -> (t h w) b d")
            
            # Flatten context if it's also 5D
            if context is not None and context.ndim == 5:
                context = rearrange(context, "t h w b d -> (t h w) b d")
            elif context is not None and context.ndim == 3:
                # Context is already in (M, B, D) format for cross-attention
                pass
            
            # Apply attention
            out_flat = self.attn(x_flat, context=context, rope_emb=rope_emb_L_1_1_D, **kwargs)
            
            # Reshape back to 5D
            out = rearrange(out_flat, "(t h w) b d -> t h w b d", t=T, h=H, w=W)
            return out
        else:
            # Standard 3D processing
            return self.attn(x, context=context, rope_emb=rope_emb_L_1_1_D, **kwargs)

# ===================== FEEDFORWARD NETWORK =====================
class OfficialGPT2FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False, **kwargs):
        super().__init__()
        self.layer1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.activation = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GPT-2 style Feed Forward network.
        Automatically handles both 3D and 5D tensors
        """
        # Store original shape if 5D
        if x.ndim == 5:
            T, H, W, B, D = x.shape
            x = rearrange(x, "t h w b d -> (t h w) b d")
            is_5d = True
        else:
            is_5d = False
            
        # Process through MLP
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        
        # Reshape back if needed
        if is_5d:
            x = rearrange(x, "(t h w) b d -> t h w b d", t=T, h=H, w=W)
            
        return x

# ===================== TIMESTEP EMBEDDING =====================
class CleanTimesteps(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps):
        # Implementation from original blocks.py
        in_dtype = timesteps.dtype
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return emb.to(in_dtype)

class CleanTimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_adaln_lora: bool, adaln_lora_dim: int = 256):
        super().__init__()
        self.use_adaln_lora = use_adaln_lora
        self.linear_1 = nn.Linear(in_channels, out_channels, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_channels, 3 * out_channels, bias=False)
        else:
            self.linear_2 = nn.Linear(out_channels, out_channels, bias=True)

    def forward(self, sample: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        processed_emb = self.linear_1(sample)
        processed_emb = self.activation(processed_emb)
        processed_emb = self.linear_2(processed_emb)

        if self.use_adaln_lora:
            adaln_lora_emb = processed_emb
            main_emb = sample
        else:
            main_emb = processed_emb
            adaln_lora_emb = None

        return main_emb, adaln_lora_emb

# ===================== PATCH EMBEDDING =====================
class CleanPatchEmbed(nn.Module):
    def __init__(self, spatial_patch_size: int, temporal_patch_size: int, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        
        patch_dim = in_channels * (spatial_patch_size ** 2) * temporal_patch_size
        
        self.proj = nn.ModuleDict({
            '1': nn.Linear(patch_dim, out_channels, bias=bias)
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        
        assert H % self.spatial_patch_size == 0
        assert W % self.spatial_patch_size == 0
        assert T % self.temporal_patch_size == 0

        patches = rearrange(
            x, 'b c (t r) (h m) (w n) -> b t h w (c r m n)', 
            r=self.temporal_patch_size, 
            m=self.spatial_patch_size, 
            n=self.spatial_patch_size
        )
        
        embedded_patches = self.proj['1'](patches)
        
        return embedded_patches

# ===================== DIT BUILDING BLOCKS =====================
class OfficialDITBuildingBlock(nn.Module):
    def __init__(self, block_type: str, x_dim: int, context_dim: Optional[int], num_heads: int,
                  mlp_ratio: float = 4.0, bias: bool = False, **kwargs):
        super().__init__()
        block_type = block_type.lower()
        self.block_type = block_type
        self.use_adaln_lora = kwargs.get('use_adaln_lora', False)
        adaln_lora_dim = kwargs.get('adaln_lora_dim', 256)

        if block_type in ("fa", "full_attn"):
            self.block = VideoAttn(x_dim, None, num_heads, bias=bias, **kwargs)
        elif block_type in ("ca", "cross_attn"):
            self.block = VideoAttn(x_dim, context_dim, num_heads, bias=bias, **kwargs)
        elif block_type in ("mlp", "ff"):
            self.block = OfficialGPT2FeedForward(x_dim, int(x_dim*mlp_ratio), bias=bias, **kwargs)
        
        self.norm_state = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.n_adaln_chunks = 3
        if self.use_adaln_lora:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * x_dim, bias=False)
            )
        else:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, self.n_adaln_chunks*x_dim, bias=False))

    def forward(
        self,
        x: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
        crossattn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.use_adaln_lora and adaln_lora_B_3D is not None:
            modulation = self.adaLN_modulation(emb_B_D) + adaln_lora_B_3D
        else:
            modulation = self.adaLN_modulation(emb_B_D)
        
        shift, scale, gate = modulation.chunk(self.n_adaln_chunks, dim=1)
        x_modulated = modulate(self.norm_state(x), shift, scale)
        
        # Process through the block
        if self.block_type in ["mlp", "ff"]:
            block_output = self.block(x_modulated)
        elif self.block_type in ["ca", "cross_attn"]:
            block_output = self.block(x_modulated, context=crossattn_emb, rope_emb_L_1_1_D=None)
        else:  # self-attention ("fa")
            block_output = self.block(x_modulated, context=None, rope_emb_L_1_1_D=rope_emb_L_1_1_D)

        # Apply gate and residual
        if x.ndim == 5:
            # gate is (B, D), need to broadcast to (T, H, W, B, D)
            gate_expanded = gate.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            return x + gate_expanded * block_output
        else:
            return x + gate.unsqueeze(0) * block_output

class OfficialGeneralDITTransformerBlock(nn.Module):
    def __init__(self, x_dim: int, context_dim: int, num_heads: int, block_config: str,
                  mlp_ratio: float = 4.0, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList()
        for block_type in block_config.split("-"):
            self.blocks.append(OfficialDITBuildingBlock(
                block_type.strip(), x_dim, context_dim, num_heads, mlp_ratio, **kwargs
            ))
            
    def forward(
        self,
        x: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Add extra positional embedding once at the beginning
        if extra_per_block_pos_emb is not None:
            x = x + extra_per_block_pos_emb
            
        for block in self.blocks:
            x = block(
                x,
                emb_B_D,
                crossattn_emb,
                adaln_lora_B_3D=adaln_lora_B_3D,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                extra_per_block_pos_emb=None,  # Don't pass it to individual blocks
                crossattn_mask=crossattn_mask
            )
        return x

# ===================== FINAL LAYER =====================
class OfficialFinalLayer(nn.Module):
    def __init__(self, hidden_size, spatial_patch_size, temporal_patch_size, out_channels, **kwargs):
        super().__init__()
        self.use_adaln_lora = kwargs.get('use_adaln_lora', False)
        self.hidden_size = hidden_size
        adaln_lora_dim = kwargs.get('adaln_lora_dim', 256)
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, spatial_patch_size**2 * temporal_patch_size * out_channels, bias=False)
        n_adaln_chunks = 2
        if self.use_adaln_lora:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, n_adaln_chunks * hidden_size, bias=False)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, n_adaln_chunks * hidden_size, bias=False)
            )
            
    def forward(self, x_BT_HW_D, emb_B_D, adaln_lora_B_3D: Optional[torch.Tensor] = None):
        input_dtype = x_BT_HW_D.dtype
        
        if self.use_adaln_lora:
            adaln_lora_chunk = adaln_lora_B_3D[:, : 2 * self.hidden_size]
            modulation = self.adaLN_modulation(emb_B_D) + adaln_lora_chunk
        else:
            modulation = self.adaLN_modulation(emb_B_D)
        
        shift, scale = modulation.chunk(2, dim=1)

        B = emb_B_D.shape[0]
        T = x_BT_HW_D.shape[0] // B
        shift_BT_D = repeat(shift, "b d -> (b t) d", t=T)
        scale_BT_D = repeat(scale, "b d -> (b t) d", t=T)
        
        x_modulated = self.norm_final(x_BT_HW_D) * (1 + scale_BT_D.unsqueeze(1)) + shift_BT_D.unsqueeze(1)
        
        return self.linear(x_modulated.to(input_dtype))

# ===================== MAIN MODEL =====================
class CleanGeneralDIT(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        model_channels = kwargs['model_channels']
        num_blocks = kwargs['num_blocks']
        num_heads = kwargs['num_heads']
        in_channels = kwargs['in_channels']
        out_channels = kwargs['out_channels']
        crossattn_emb_channels = kwargs['crossattn_emb_channels']
        block_config = kwargs['block_config']
        mlp_ratio = kwargs['mlp_ratio']
        affline_emb_norm = kwargs.get('affline_emb_norm', True)
        self.additional_concat_ch = kwargs.get('additional_concat_ch', 0)
        self._patch_embed_bias = getattr(self, '_patch_embed_bias', True)
        self.concat_padding_mask = kwargs.get('concat_padding_mask', True)
        
        # Extract extrapolation ratios
        self.rope_h_extrapolation_ratio = kwargs.get('rope_h_extrapolation_ratio', 1.0)
        self.rope_w_extrapolation_ratio = kwargs.get('rope_w_extrapolation_ratio', 1.0)
        self.rope_t_extrapolation_ratio = kwargs.get('rope_t_extrapolation_ratio', 1.0)
        
        # Extra positional embedding settings
        self.extra_per_block_abs_pos_emb = kwargs.get('extra_per_block_abs_pos_emb', False)
        self.extra_per_block_abs_pos_emb_type = kwargs.get('extra_per_block_abs_pos_emb_type', 'sincos')
        self.extra_h_extrapolation_ratio = kwargs.get('extra_h_extrapolation_ratio', 1.0)
        self.extra_w_extrapolation_ratio = kwargs.get('extra_w_extrapolation_ratio', 1.0)
        self.extra_t_extrapolation_ratio = kwargs.get('extra_t_extrapolation_ratio', 1.0)

        self.patch_spatial = kwargs['patch_spatial']
        self.patch_temporal = kwargs['patch_temporal']
        
        # Get max dimensions for positional embeddings
        self.max_img_h = kwargs.get('max_img_h', 1024)
        self.max_img_w = kwargs.get('max_img_w', 1024)
        self.max_frames = kwargs.get('max_frames', 128)
        
        in_ch = in_channels + self.additional_concat_ch + (1 if kwargs.get('concat_padding_mask', True) else 0)
        self.x_embedder = CleanPatchEmbed(
            self.patch_spatial, self.patch_temporal, in_ch, model_channels, bias=self._patch_embed_bias
        )
        
        self.t_embedder = nn.Sequential(
            CleanTimesteps(model_channels),
            CleanTimestepEmbedding(model_channels, model_channels, 
                                  use_adaln_lora=kwargs.get('use_adaln_lora', False), 
                                  adaln_lora_dim=kwargs.get('adaln_lora_dim', 256))
        )
        
        # Initialize RoPE with extrapolation ratios
        self.pos_embedder = CleanRoPE3D(
            head_dim=model_channels // num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio
        )
        
        # Initialize extra positional embeddings if needed
        if self.extra_per_block_abs_pos_emb:
            if self.extra_per_block_abs_pos_emb_type.lower() == 'sincos':
                self.extra_pos_embedder = SinCosPosEmbAxis(
                    model_channels=model_channels,
                    len_h=self.max_img_h // self.patch_spatial,
                    len_w=self.max_img_w // self.patch_spatial,
                    len_t=self.max_frames // self.patch_temporal,
                    h_extrapolation_ratio=self.extra_h_extrapolation_ratio,
                    w_extrapolation_ratio=self.extra_w_extrapolation_ratio,
                    t_extrapolation_ratio=self.extra_t_extrapolation_ratio
                )
        
        self.blocks = nn.ModuleDict()
        
        block_kwargs = kwargs.copy()
        for key in ['x_dim', 'context_dim', 'num_heads', 'block_config', 'mlp_ratio']:
            block_kwargs.pop(key, None)
            
        for i in range(num_blocks):
            self.blocks[f"block{i}"] = OfficialGeneralDITTransformerBlock(
                x_dim=model_channels, context_dim=crossattn_emb_channels, num_heads=num_heads,
                block_config=block_config, mlp_ratio=mlp_ratio, **block_kwargs
            )
        
        final_layer_kwargs = kwargs.copy()
        explicit_keys = ['hidden_size', 'spatial_patch_size', 'temporal_patch_size', 'out_channels']
        for key in explicit_keys:
            final_layer_kwargs.pop(key, None)

        self.final_layer = OfficialFinalLayer(
            hidden_size=model_channels, 
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal, 
            out_channels=out_channels, 
            **final_layer_kwargs
        )

        # Use our improved RMSNorm for affine embedding normalization
        if affline_emb_norm:
            self.affline_norm = RMSNorm(model_channels)
        else:
            self.affline_norm = nn.Identity()

    def forward(self, x, timesteps, crossattn_emb, latent_condition, **kwargs):
        """
        Base forward pass for the General DiT.
        """
        # 1. Prepare Timestep Embeddings
        timesteps = timesteps.to(x.dtype)
        t_emb, adaln_lora_emb = self.t_embedder(timesteps.flatten())
        affline_emb = self.affline_norm(t_emb)
        
        # Ensure embeddings maintain input dtype
        affline_emb = affline_emb.to(x.dtype)
        if adaln_lora_emb is not None:
            adaln_lora_emb = adaln_lora_emb.to(x.dtype)

        # 2. Concatenate input `x` with the condition and padding mask if needed.
        tensors_to_cat = [x, latent_condition]
        
        if self.concat_padding_mask:
            padding_mask = torch.ones(x.shape[0], 1, *x.shape[2:], device=x.device, dtype=x.dtype)
            tensors_to_cat.append(padding_mask)

        x_conditioned = torch.cat(tensors_to_cat, dim=1)
        
        # 3. Patch Embeddings
        x_patches = self.x_embedder(x_conditioned)
        B, T_p, H_p, W_p, D = x_patches.shape

        # 4. Positional Embeddings (RoPE)
        rope_emb = self.pos_embedder(x_patches) 
        
        # 4b. Extra positional embeddings if enabled
        extra_pos_emb = None
        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_patches)
            # Convert to THWBD format
            extra_pos_emb = rearrange(extra_pos_emb, "B T H W D -> T H W B D")
        
        # 5. Main Transformer Blocks - use THWBD format!
        x_rearranged = rearrange(x_patches, "B T H W D -> T H W B D")
        
        # Cross-attention context also needs rearranging
        if crossattn_emb.ndim == 3:  # (B, M, D)
            crossattn_emb_rearranged = rearrange(crossattn_emb, "B M D -> M B D")
        else:
            crossattn_emb_rearranged = crossattn_emb

        for i in range(len(self.blocks)):
            block = self.blocks[f"block{i}"]
            x_rearranged = block(
                x=x_rearranged,
                emb_B_D=affline_emb,
                crossattn_emb=crossattn_emb_rearranged,
                adaln_lora_B_3D=adaln_lora_emb,
                rope_emb_L_1_1_D=rope_emb,
                extra_per_block_pos_emb=extra_pos_emb
            )
        
        # 6. Final Layer - convert back from THWBD
        x_final = rearrange(x_rearranged, "T H W B D -> B T H W D")
        x_final_rearranged = rearrange(x_final, "B T H W D -> (B T) (H W) D")
        
        # Ensure dtype consistency before final layer
        x_final_rearranged = x_final_rearranged.to(x.dtype)
        output = self.final_layer(x_final_rearranged, affline_emb, adaln_lora_B_3D=adaln_lora_emb)
        
        # 7. Unpatchify
        output_unpatched = rearrange(
            output,
            "(B T) (H W) (ph pw pt C) -> B C (T pt) (H ph) (W pw)",
            ph=self.patch_spatial,
            pw=self.patch_spatial,
            pt=self.patch_temporal,
            H=H_p, W=W_p, B=B, T=T_p
        )
        
        return output_unpatched

# ===================== DIFFUSION RENDERER VARIANT =====================
class CleanDiffusionRendererGeneralDIT(CleanGeneralDIT):
    def __init__(self, additional_concat_ch: int = 16, use_context_embedding: bool = True, **kwargs):
        self.use_context_embedding = use_context_embedding
        self._patch_embed_bias = False
        kwargs['use_adaln_lora'] = True
        kwargs['adaln_lora_dim'] = 256
        super().__init__(additional_concat_ch=additional_concat_ch, **kwargs)
        if self.use_context_embedding:
            self.context_embedding = nn.Embedding(num_embeddings=16, embedding_dim=kwargs["crossattn_emb_channels"])
            # Initialize with fixed seed like official implementation
            rng_state = torch.get_rng_state()
            torch.manual_seed(42)
            torch.nn.init.uniform_(self.context_embedding.weight, -0.3, 0.3)
            torch.set_rng_state(rng_state)
    
    def forward(self, x, timesteps, latent_condition, context_index, **kwargs):
        
        # 1. Prepare Cross-Attention Embeddings from context_index
        if self.use_context_embedding:
            crossattn_emb = self.context_embedding(context_index.long())
            if crossattn_emb.ndim == 2:
                crossattn_emb = crossattn_emb.unsqueeze(1)
        else:
            B = x.shape[0]
            # Create a dummy tensor if not using context embedding
            crossattn_emb_channels = self.blocks['block0'].blocks[1].block.attn.to_k[0].in_features
            crossattn_emb = torch.zeros(B, 1, crossattn_emb_channels, device=x.device, dtype=x.dtype)

        # 2. Call the parent class's forward method with all the prepared inputs
        return super().forward(
            x=x,
            timesteps=timesteps,
            crossattn_emb=crossattn_emb,
            latent_condition=latent_condition,
            **kwargs
        )