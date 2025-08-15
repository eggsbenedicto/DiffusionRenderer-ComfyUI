import math
import torch
from torch import nn
from typing import Dict, Tuple, Optional, Union
from einops import rearrange, repeat

def modulate(x, shift, scale):
    # x shape: (S, B, D) where S is sequence length
    # shift, scale shape: (B, D)
    # We need to reshape shift/scale to (1, B, D) to broadcast over the sequence dim
    return x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)

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

# ===================== ROPE IMPLEMENTATION =====================
def apply_rotary_pos_emb_pure_torch(x: torch.Tensor, rope_emb: torch.Tensor, tensor_format: str = "sbhd", fused: bool = True) -> torch.Tensor:
    """
    Pure PyTorch implementation matching transformer_engine's apply_rotary_pos_emb behavior
    
    Args:
        x: Input tensor
        rope_emb: RoPE embedding frequencies (raw frequencies, not sin/cos) 
        tensor_format: Format of input tensor ("sbhd" = seq, batch, heads, dim)
        fused: Whether to use fused implementation (always True for our case)
    """
    if tensor_format == "sbhd":
        # x shape: (S, B, H, D)
        # rope_emb shape: (S, 1, 1, D)
        seq_len, batch, heads, dim = x.shape
        
        # Remove singleton dimensions from rope_emb
        freqs = rope_emb.squeeze(1).squeeze(1)  # (S, D)
        
        # From transformer_engine issue #552, the core rotation formula is:
        # t = (t * freqs.cos().to(t.dtype)) + (_rotate_half(t) * freqs.sin().to(t.dtype))
        # This matches the optimization mentioned in the issue
        
        def rotate_half(x):
            """Rotates half the hidden dims of the input."""
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
        
        # Expand freqs to match x dimensions: (S, 1, 1, D) -> (S, B, H, D)
        freqs_expanded = freqs.unsqueeze(1).unsqueeze(1).expand_as(x)
        
        # Apply the rotation formula from transformer_engine
        # This is the exact formula mentioned in transformer_engine issue #552
        cos_freqs = freqs_expanded.cos().to(x.dtype)
        sin_freqs = freqs_expanded.sin().to(x.dtype)
        
        rotated = (x * cos_freqs) + (rotate_half(x) * sin_freqs)
        
        return rotated
    else:
        raise NotImplementedError(f"Tensor format {tensor_format} not implemented")

class CleanRoPE3D(nn.Module):
    def __init__(self, head_dim: int, **kwargs):
        super().__init__()
        self.head_dim = head_dim
        # Use a longer sequence buffer to match official implementation
        self.register_buffer("seq", torch.arange(max(512, head_dim), dtype=torch.float))
        
        # Split dimensions EXACTLY like official VideoRopePosition3DEmb
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h  
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"RoPE head_dim setup failed: {dim} != {dim_h} + {dim_w} + {dim_t}"

        # Store dimension info
        self.dim_h = dim_h
        self.dim_w = dim_w
        self.dim_t = dim_t
        
        # Pre-compute frequency ranges EXACTLY like official implementation
        self.register_buffer("dim_spatial_range", 
                           torch.arange(0, dim_h, 2)[:dim_h//2].float() / dim_h,
                           persistent=False)
        self.register_buffer("dim_temporal_range",
                           torch.arange(0, dim_t, 2)[:dim_t//2].float() / dim_t,
                           persistent=False)
        
        # Default extrapolation factors - EXACTLY matching official VideoRopePosition3DEmb
        self.h_ntk_factor = 1.0  
        self.w_ntk_factor = 1.0
        self.t_ntk_factor = 2.0  # Official uses 2.0 for temporal

    def forward(self, x_patches: torch.Tensor):
        """
        Generate RoPE embeddings EXACTLY matching official VideoRopePosition3DEmb.generate_embeddings
        """
        B, T_p, H_p, W_p, D_model = x_patches.shape
        device = x_patches.device
        
        # Compute theta values with NTK scaling - EXACTLY like official
        h_theta = 10000.0 * self.h_ntk_factor
        w_theta = 10000.0 * self.w_ntk_factor  
        t_theta = 10000.0 * self.t_ntk_factor
        
        # Compute frequencies - EXACTLY like official
        h_spatial_freqs = 1.0 / (h_theta ** self.dim_spatial_range.to(device))
        w_spatial_freqs = 1.0 / (w_theta ** self.dim_spatial_range.to(device))
        temporal_freqs = 1.0 / (t_theta ** self.dim_temporal_range.to(device))
        
        # Create position sequences - EXACTLY like official
        seq_t = self.seq[:T_p].to(device)
        seq_h = self.seq[:H_p].to(device) 
        seq_w = self.seq[:W_p].to(device)
        
        # Compute outer products - EXACTLY like official
        half_emb_t = torch.outer(seq_t, temporal_freqs)
        half_emb_h = torch.outer(seq_h, h_spatial_freqs)
        half_emb_w = torch.outer(seq_w, w_spatial_freqs)
        
        # THIS IS THE CRITICAL PART: concatenate in [t, h, w] * 2 pattern
        # EXACTLY matching official VideoRopePosition3DEmb.generate_embeddings
        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H_p, w=W_p),
                repeat(half_emb_h, "h d -> t h w d", t=T_p, w=W_p),  
                repeat(half_emb_w, "w d -> t h w d", t=T_p, h=H_p),
            ] * 2,  # Repeat the entire pattern twice - this is the key difference!
            dim=-1,
        )
        
        # Reshape to expected format - EXACTLY like official
        final_rope_emb = rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d")
        
        return final_rope_emb.to(x_patches.dtype)

# ===================== ATTENTION IMPLEMENTATION =====================
class PytorchDotProductAttention(nn.Module):
    """
    Pure PyTorch implementation to replace transformer_engine's DotProductAttention
    """
    def __init__(self, heads, dim_head, num_gqa_groups=None, attention_dropout=0.0, 
                 qkv_format="sbhd", attn_mask_type="no_mask", **kwargs):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.num_gqa_groups = num_gqa_groups or heads
        self.attention_dropout = attention_dropout
        self.qkv_format = qkv_format
        self.attn_mask_type = attn_mask_type
        
        if attention_dropout > 0:
            self.dropout = nn.Dropout(attention_dropout)
        else:
            self.dropout = None
    
    def forward(self, q, k, v, core_attention_bias_type=None, core_attention_bias=None):
        """
        Forward pass matching transformer_engine's DotProductAttention interface
        """
        if self.qkv_format == "sbhd":
            # Convert from (seq, batch, heads, dim) to (batch, heads, seq, dim)
            q = q.permute(1, 2, 0, 3)  # (B, H, S, D)
            k = k.permute(1, 2, 0, 3)  # (B, H, S, D)
            v = v.permute(1, 2, 0, 3)  # (B, H, S, D)
        
        # Apply scaled dot-product attention
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False
        )
        
        if self.qkv_format == "sbhd":
            # Convert back to (seq, batch, heads, dim)
            out = out.permute(2, 0, 1, 3)
        
        return out

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
        qkv_norm: str = "RRI",  # R=RMSNorm for q,k; I=Identity for v (matches official)
        qkv_norm_mode: str = "per_head",
        backend: str = "torch",  # We're always using torch
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
            raise ValueError(f"Normalization mode {self.qkv_norm_mode} not found, only support 'per_head'")

        # Create Q, K, V projections with normalization (matching official pattern)
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

        # Use our pure PyTorch attention implementation
        self.attn_op = PytorchDotProductAttention(
            heads=heads,
            dim_head=dim_head,
            num_gqa_groups=heads,
            attention_dropout=0.0,
            qkv_format=qkv_format,
        )

    def cal_qkv(self, x, context=None, mask=None, rope_emb=None, **kwargs):
        """
        Calculate Q, K, V with per-head normalization - EXACTLY matching official Attention.cal_qkv
        """
        if self.qkv_norm_mode == "per_head":
            q = self.to_q[0](x)
            context = x if context is None else context
            k = self.to_k[0](context)
            v = self.to_v[0](context)
            
            # Reshape for per-head normalization - CRITICAL STEP from official implementation
            # This matches: "map(lambda t: rearrange(t, "b ... (n c) -> b ... n c", n=self.heads, c=self.dim_head)"
            q, k, v = map(
                lambda t: rearrange(t, "s b (n c) -> s b n c", n=self.heads, c=self.dim_head),
                (q, k, v),
            )
        else:
            raise ValueError(f"Normalization mode {self.qkv_norm_mode} not found")

        # Apply per-head normalization - EXACTLY like official
        q = self.to_q[1](q)
        k = self.to_k[1](k)
        v = self.to_v[1](v)
        
        # Apply RoPE only for self-attention - EXACTLY like official
        if self.is_selfattn and rope_emb is not None:
            q = apply_rotary_pos_emb_pure_torch(q, rope_emb, tensor_format=self.qkv_format, fused=True)
            k = apply_rotary_pos_emb_pure_torch(k, rope_emb, tensor_format=self.qkv_format, fused=True)
            
        return q, k, v

    def cal_attn(self, q, k, v, mask=None):
        """Calculate attention using pure PyTorch - matching transformer_engine interface"""
        if self.backend == "torch":
            # Use our PyTorch attention implementation that mimics transformer_engine behavior
            out = self.attn_op(q, k, v)
            return self.to_out(out)
        else:
            raise ValueError(f"Backend {self.backend} not found")

    def forward(self, x, context=None, mask=None, rope_emb=None, **kwargs):
        """
        Forward pass matching the official Attention interface exactly
        """
        q, k, v = self.cal_qkv(x, context, mask, rope_emb=rope_emb, **kwargs)
        return self.cal_attn(q, k, v, mask)

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
    def __init__(self, in_channels: int, out_channels: int, use_adaln_lora: bool, adaln_lora_dim: int):
        super().__init__()
        self.use_adaln_lora = use_adaln_lora
        self.linear_1 = nn.Linear(in_channels, out_channels, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        if use_adaln_lora:
            # The output here is split into chunks for each block later on.
            self.linear_2 = nn.Linear(out_channels, 3 * out_channels, bias=False)
        else:
            self.linear_2 = nn.Linear(out_channels, out_channels, bias=True)

    def forward(self, sample: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Processes the sinusoidal timestep embedding.
        Args:
            sample: The sinusoidal embedding from the CleanTimesteps module.
        Returns:
            A tuple of (emb_B_D, adaln_lora_B_3D).
        """
        # Pass through the linear layers
        processed_emb = self.linear_1(sample)
        processed_emb = self.activation(processed_emb)
        processed_emb = self.linear_2(processed_emb)

        if self.use_adaln_lora:
            # The fully processed embedding is the LoRA component
            adaln_lora_emb = processed_emb
            # The main embedding for AdaLN is the original sinusoidal input
            main_emb = sample
        else:
            # If not using LoRA, the main embedding is the processed one
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
        
        # Create an nn.ModuleDict to hold the linear layer under the key '1'.
        # This will create the parameter name "proj.1.weight", exactly what the checkpoint has.
        self.proj = nn.ModuleDict({
            '1': nn.Linear(patch_dim, out_channels, bias=bias)
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Takes a 5D video tensor and embeds it into a sequence of patch tokens.
        Args:
            x: Input tensor of shape (B, C, T, H, W).
        Returns:
            A tensor of shape (B, T_p, H_p, W_p, D_model) where D_model is out_channels.
        """
        # B: Batch size, C: Channels, T: Time, H: Height, W: Width
        B, C, T, H, W = x.shape
        
        # Assert that dimensions are divisible by patch sizes, as in the original code.
        # This is a good sanity check.
        assert H % self.spatial_patch_size == 0, f"Input height {H} is not divisible by patch size {self.spatial_patch_size}"
        assert W % self.spatial_patch_size == 0, f"Input width {W} is not divisible by patch size {self.spatial_patch_size}"
        assert T % self.temporal_patch_size == 0, f"Input temporal dim {T} is not divisible by patch size {self.temporal_patch_size}"

        # Use einops.rearrange to slice the input into patches and flatten each patch.
        # 'm' and 'n' are spatial_patch_size, 'r' is temporal_patch_size.
        # The output of rearrange is (B, T_patches, H_patches, W_patches, C*r*m*n)
        patches = rearrange(
            x, 'b c (t r) (h m) (w n) -> b t h w (c r m n)', 
            r=self.temporal_patch_size, 
            m=self.spatial_patch_size, 
            n=self.spatial_patch_size
        )
        
        # Apply the linear projection to the last dimension (the flattened patch).
        embedded_patches = self.proj['1'](patches)
        
        return embedded_patches

# ===================== VIDEO ATTENTION WRAPPER =====================
class VideoAttn(nn.Module):
    def __init__(self, x_dim: int, context_dim: Optional[int], num_heads: int, bias: bool = False, **kwargs):
        super().__init__()
        self.attn = Attention(
            query_dim=x_dim, 
            context_dim=context_dim, 
            heads=num_heads,
            dim_head=x_dim // num_heads, 
            qkv_bias=bias, 
            out_bias=bias,
            qkv_norm="RRI",  # R=RMSNorm for q,k; I=Identity for v
            qkv_norm_mode="per_head",
            backend="torch",
            qkv_format="sbhd"
        )
    
    def forward(self, x, context=None, rope_emb_L_1_1_D=None, **kwargs):
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
        """
        # Pass through the first linear layer
        x = self.layer1(x)
        
        # Apply the GELU activation function
        x = self.activation(x)
        
        # Pass through the second linear layer
        x = self.layer2(x)
        
        return x

# ===================== DIT BUILDING BLOCKS =====================
class OfficialDITBuildingBlock(nn.Module):
    def __init__( self, block_type: str, x_dim: int, context_dim: Optional[int], num_heads: int,
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
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.use_adaln_lora and adaln_lora_B_3D is not None:
            modulation = self.adaLN_modulation(emb_B_D) + adaln_lora_B_3D
        else:
            modulation = self.adaLN_modulation(emb_B_D)
        
        shift, scale, gate = modulation.chunk(self.n_adaln_chunks, dim=1)
        x_modulated = modulate(self.norm_state(x), shift, scale)

        # Use a float32 autocast context for the attention and MLP blocks
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            if self.block_type in ["mlp", "ff"]:
                block_output = self.block(x_modulated)
            elif self.block_type in ["ca", "cross_attn"]:
                block_output = self.block(x_modulated, context=crossattn_emb, rope_emb_L_1_1_D=rope_emb_L_1_1_D)
            else: # self-attention ("fa")
                block_output = self.block(x_modulated, context=None, rope_emb_L_1_1_D=rope_emb_L_1_1_D)

        return x + gate.unsqueeze(0) * block_output

class OfficialGeneralDITTransformerBlock(nn.Module):
    def __init__( self, x_dim: int, context_dim: int, num_heads: int, block_config: str,
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
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None # EXPLICIT ARGUMENT
    ) -> torch.Tensor:
        # Pass the rope_emb explicitly to each building block
        for block in self.blocks:
            x = block(
                x,
                emb_B_D,
                crossattn_emb,
                adaln_lora_B_3D=adaln_lora_B_3D,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D
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
        # 1. Calculate modulation params
        if self.use_adaln_lora:
            # The original code takes a slice of the lora embedding
            adaln_lora_chunk = adaln_lora_B_3D[:, : 2 * self.hidden_size]
            modulation = self.adaLN_modulation(emb_B_D) + adaln_lora_chunk
        else:
            modulation = self.adaLN_modulation(emb_B_D)
        
        shift, scale = modulation.chunk(2, dim=1)

        # 2. Apply AdaLN
        # x is (B*T, H*W, D). shift/scale are (B, D).
        # We need to repeat shift/scale to match the batch+time dim.
        B = emb_B_D.shape[0]
        T = x_BT_HW_D.shape[0] // B
        shift_BT_D = repeat(shift, "b d -> (b t) d", t=T)
        scale_BT_D = repeat(scale, "b d -> (b t) d", t=T)
        
        # modulate for this shape: x is (BT, S, D), shift/scale are (BT, D)
        x_modulated = self.norm_final(x_BT_HW_D) * (1 + scale_BT_D.unsqueeze(1)) + shift_BT_D.unsqueeze(1)
        
        # 3. Final linear projection
        return self.linear(x_modulated)

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
        self._patch_embed_bias = getattr(self, '_patch_embed_bias', True)

        self.patch_spatial = kwargs['patch_spatial']
        self.patch_temporal = kwargs['patch_temporal']
        
        in_ch = in_channels + self.additional_concat_ch + (1 if kwargs.get('concat_padding_mask', True) else 0)
        self.x_embedder = CleanPatchEmbed(
            self.patch_spatial, self.patch_temporal, in_ch, model_channels, bias=self._patch_embed_bias
        )
        
        self.t_embedder = nn.Sequential(
            CleanTimesteps(model_channels),
            CleanTimestepEmbedding(model_channels, model_channels, use_adaln_lora=kwargs.get('use_adaln_lora', False), adaln_lora_dim=kwargs.get('adaln_lora_dim', 256))
        )
        self.pos_embedder = CleanRoPE3D(head_dim=model_channels // num_heads)
        
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
        affline_emb_norm = kwargs.get('affline_emb_norm', True)
        if affline_emb_norm:
            self.affline_norm = RMSNorm(model_channels)
        else:
            self.affline_norm = nn.Identity()

    def forward(self, x, timesteps, crossattn_emb, latent_condition, **kwargs):
        """
        Base forward pass for the General DiT.
        'x' is the noised latent.
        'latent_condition' is the encoded condition map(s).
        'crossattn_emb' is for text or other guidance (like context_index).
        """
        # 1. Prepare Timestep Embeddings
        timesteps = timesteps.to(x.dtype)
        t_emb, adaln_lora_emb = self.t_embedder(timesteps.flatten())
        affline_emb = self.affline_norm(t_emb)

        # 2. Concatenate input `x` with the condition and padding mask if needed.
        tensors_to_cat = [x, latent_condition]
        
        if self.concat_padding_mask:
            padding_mask = torch.ones(x.shape[0], 1, *x.shape[2:], device=x.device, dtype=x.dtype)
            tensors_to_cat.append(padding_mask)

        x_conditioned = torch.cat(tensors_to_cat, dim=1)
        
        # 3. Patch Embeddings
        x_patches = self.x_embedder(x_conditioned)
        B, T_p, H_p, W_p, D = x_patches.shape

        # 4. Positional Embeddings (RoPE) - now using improved implementation
        rope_emb = self.pos_embedder(x_patches) 
        
        # 5. Main Transformer Blocks
        x_rearranged = rearrange(x_patches, "B T H W D -> (T H W) B D")
        
        # Cross-attention context also needs rearranging if it's not already
        if crossattn_emb.ndim == 3: # (B, M, D)
             crossattn_emb_rearranged = rearrange(crossattn_emb, "B M D -> M B D")
        else: # Already rearranged
             crossattn_emb_rearranged = crossattn_emb

        for i in range(len(self.blocks)):
             block = self.blocks[f"block{i}"]
             x_rearranged = block(
                 x=x_rearranged,
                 emb_B_D=affline_emb,
                 crossattn_emb=crossattn_emb_rearranged,
                 adaln_lora_B_3D=adaln_lora_emb,
                 rope_emb_L_1_1_D=rope_emb
             )
        
        # 6. Final Layer
        x_final = rearrange(x_rearranged, "(T H W) B D -> B T H W D", T=T_p, H=H_p, W=W_p)
        x_final_rearranged = rearrange(x_final, "B T H W D -> (B T) (H W) D")
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
    
    def forward(self, x, timesteps, latent_condition, context_index, **kwargs):
        
        # 1. Prepare Cross-Attention Embeddings from context_index
        if self.use_context_embedding:
            crossattn_emb = self.context_embedding(context_index.long())
            if crossattn_emb.ndim == 2:
                crossattn_emb = crossattn_emb.unsqueeze(1)
        else:
            B = x.shape[0]
            # Create a dummy tensor if not using context embedding (e.g., for forward renderer)
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