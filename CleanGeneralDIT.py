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

def rotate_half(x):
    """Rotates the last dimension of a tensor by half."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, rope_emb):
    """
    Applies rotary positional embedding to the input tensor.
    x shape: (S, B, H, D_head)
    rope_emb shape: (S, 1, 1, D_head) - S is sequence length (T*H*W)
    """
    # The RoPE tensor from CleanRoPE3D already has sin/cos components prepared
    cos_emb = rope_emb[..., :x.shape[-1] // 2].repeat(1, 1, 1, 2)
    sin_emb = rope_emb[..., x.shape[-1] // 2:].repeat(1, 1, 1, 2)
    return (x * cos_emb) + (rotate_half(x) * sin_emb)

class RMSNormPlaceholder(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        # Compute the square root of the mean of squares of the input tensor x along the last dimension.
        # Add eps for numerical stability.
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Normalize the input and then scale it by the learnable weight.
        return self._norm(x.float()).to(x.dtype) * self.weight

class CleanRoPE3D(nn.Module):
    def __init__(self, head_dim: int, **kwargs):
        super().__init__()
        self.head_dim = head_dim
        self.register_parameter("seq", nn.Parameter(torch.zeros(128)))
        
        # Split dimensions for T, H, W as in the original code
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"RoPE head_dim setup failed."

        # Pre-compute frequency ranges (thetas)
        self.register_buffer("h_freqs", self._compute_freqs(dim_h), persistent=False)
        self.register_buffer("w_freqs", self._compute_freqs(dim_w), persistent=False)
        self.register_buffer("t_freqs", self._compute_freqs(dim_t), persistent=False)
        
    def _compute_freqs(self, dim):
        # Same logic as original VideoRopePosition3DEmb
        theta = 10000.0
        dim_range = torch.arange(0, dim, 2).float() / dim
        freqs = 1.0 / (theta**dim_range)
        return freqs

    def forward(self, x_patches: torch.Tensor):
        """
        Calculates the RoPE tensor based on the input patches' dimensions.
        Args:
            x_patches: The input tensor after patch embedding. Shape: (B, T_p, H_p, W_p, D_model).
        Returns:
            A tensor containing the combined sin/cos embeddings ready for application in attention.
            Shape: (T_p * H_p * W_p, 1, 1, D_head).
        """
        B, T_p, H_p, W_p, D_model = x_patches.shape
        device = x_patches.device

        # Create position indices [0, 1, 2, ...] for each dimension
        t_pos = torch.arange(T_p, device=device, dtype=torch.float32)
        h_pos = torch.arange(H_p, device=device, dtype=torch.float32)
        w_pos = torch.arange(W_p, device=device, dtype=torch.float32)

        # Calculate position * frequency for each dimension
        # torch.outer(a, b) creates a matrix of a_i * b_j
        t_pos_freqs = torch.outer(t_pos, self.t_freqs.to(device)) # (T_p, dim_t/2)
        h_pos_freqs = torch.outer(h_pos, self.h_freqs.to(device)) # (H_p, dim_h/2)
        w_pos_freqs = torch.outer(w_pos, self.w_freqs.to(device)) # (W_p, dim_w/2)
        
        # Now, we create the full embedding tensor by broadcasting and concatenating
        # We need a tensor of shape (T_p, H_p, W_p, D_head)
        
        # Expand each pos_freqs tensor to the full 4D shape
        t_emb = repeat(t_pos_freqs, "t d -> t h w d", h=H_p, w=W_p)
        h_emb = repeat(h_pos_freqs, "h d -> t h w d", t=T_p, w=W_p)
        w_emb = repeat(w_pos_freqs, "w d -> t h w d", t=T_p, h=H_p)
        
        # Concatenate the half-dimension embeddings to form the full head dimension
        # Shape of concatenated tensor: (T_p, H_p, W_p, D_head/2)
        pos_freqs_4d = torch.cat([t_emb, h_emb, w_emb], dim=-1)
        
        # Create the final RoPE tensor with both sin and cos values
        # This is what will be used to rotate Q and K
        # Shape: (T_p, H_p, W_p, D_head)
        rope_emb_4d = torch.cat((pos_freqs_4d, pos_freqs_4d), dim=-1)
        
        # Reshape for application within the attention mechanism
        # The attention op expects sequence length as the first dimension
        # Shape: (T_p * H_p * W_p, 1, 1, D_head)
        return rearrange(rope_emb_4d, "t h w d -> (t h w) 1 1 d")

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

class OfficialAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, qkv_bias=False, out_bias=False, **kwargs):
        super().__init__()
        self.is_selfattn = context_dim is None
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim
        
        self.to_q = nn.Sequential(nn.Linear(query_dim, inner_dim, bias=qkv_bias), RMSNormPlaceholder(dim_head))
        self.to_k = nn.Sequential(nn.Linear(context_dim, inner_dim, bias=qkv_bias), RMSNormPlaceholder(dim_head))
        self.to_v = nn.Sequential(nn.Linear(context_dim, inner_dim, bias=qkv_bias), nn.Identity())
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim, bias=out_bias), nn.Dropout(0.0))

    def forward(self, x, context=None, rope_emb_L_1_1_D=None, **kwargs):
        # 1. Project to Q, K, V
        q_proj = self.to_q[0](x)
        context = x if context is None else context
        k_proj = self.to_k[0](context)
        v_proj = self.to_v[0](context)
        
        # 2. Split into heads
        # Input x is (S, B, D), so projections are (S, B, D_inner)
        q = rearrange(q_proj, "s b (h d) -> s b h d", h=self.heads)
        k = rearrange(k_proj, "s b (h d) -> s b h d", h=self.heads)
        v = rearrange(v_proj, "s b (h d) -> s b h d", h=self.heads)
        
        # 3. Normalize heads
        q = self.to_q[1](q)
        k = self.to_k[1](k)
        # self.to_v[1] is an nn.Identity, so it does nothing.

        # 4. Apply RoPE if it's a self-attention block and embeddings are provided
        if self.is_selfattn and rope_emb_L_1_1_D is not None:
            q = apply_rotary_pos_emb(q, rope_emb_L_1_1_D)
            k = apply_rotary_pos_emb(k, rope_emb_L_1_1_D)
        
        # 5. Attention calculation using PyTorch's optimized backend
        # Reshape for native attention: (S, B, H, D) -> (B, H, S, D)
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)
        
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        # 6. Combine heads and project out
        # Reshape back: (B, H, S, D) -> (S, B, H, D) -> (S, B, D_inner)
        out = out.permute(2, 0, 1, 3).contiguous()
        out = rearrange(out, "s b h d -> s b (h d)")
        
        return self.to_out(out)

class VideoAttn(nn.Module):
    def __init__(self, x_dim: int, context_dim: Optional[int], num_heads: int, bias: bool = False, **kwargs):
        super().__init__()
        self.attn = OfficialAttention(
            query_dim=x_dim, context_dim=context_dim, heads=num_heads,
            dim_head=x_dim // num_heads, qkv_bias=bias, out_bias=bias
        )
    def forward(self, x, context=None, **kwargs):
        # This wrapper just calls the underlying attention module
        return self.attn(x, context=context, **kwargs)

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

class OfficialDITBuildingBlock(nn.Module):
    def __init__( self, block_type: str, x_dim: int, context_dim: Optional[int], num_heads: int,
                  mlp_ratio: float = 4.0, bias: bool = False, **kwargs):
        super().__init__()
        block_type = block_type.lower()
        self.block_type = block_type # Store for use in forward
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

    # IMPLEMENTED FORWARD PASS
    def forward(
        self,
        x: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        # 1. Calculate modulation params
        if self.use_adaln_lora:
            # Note: The original adaln_lora_B_3D is (B, 3*D). We need to select the right chunk for this block.
            # However, the code seems to add the whole thing. Let's replicate that for now.
            # A cleaner implementation might pass chunks, but we must match the original.
            # Official code: (self.adaLN_modulation(emb_B_D) + adaln_lora_B_3D)
            # This implies adaln_lora_B_3D is shaped specifically for *this block*. Let's assume t_embedder provides that.
            modulation = self.adaLN_modulation(emb_B_D) + adaln_lora_B_3D
        else:
            modulation = self.adaLN_modulation(emb_B_D)
        
        shift, scale, gate = modulation.chunk(self.n_adaln_chunks, dim=1)

        # 2. Apply AdaLN
        # The input x is expected to be in (S, B, D) format from the main forward pass.
        # shift, scale, gate are (B, D). We need to broadcast.
        x_modulated = modulate(self.norm_state(x), shift, scale)
        
        # 3. Pass through the actual block
        if self.block_type in ["mlp", "ff"]:
            # MLP doesn't use cross-attention or RoPE
            block_output = self.block(x_modulated)
        else: # Attention blocks
            block_output = self.block(
                x_modulated,
                context=crossattn_emb,
                **kwargs # Pass rope_emb_L_1_1_D through here
            )

        # 4. Apply gating and residual connection
        # gate is (B, D), needs broadcasting to (S, B, D)
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
        **kwargs # Accept other args like rope_emb
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(
                x,
                emb_B_D,
                crossattn_emb,
                **kwargs
            )
        return x

class OfficialFinalLayer(nn.Module):
    def __init__(self, hidden_size, spatial_patch_size, temporal_patch_size, out_channels, **kwargs):
        super().__init__()
        use_adaln_lora = kwargs.get('use_adaln_lora', False)
        adaln_lora_dim = kwargs.get('adaln_lora_dim', 256)
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, spatial_patch_size**2 * temporal_patch_size * out_channels, bias=False)
        n_adaln_chunks = 2
        if use_adaln_lora:
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

        # THE FINAL FIX: The original code uses RMSNorm, which has no bias.
        # We replace the incorrect LayerNorm with our RMSNormPlaceholder.
        if affline_emb_norm:
            self.affline_norm = RMSNormPlaceholder(model_channels)
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
        t_emb, adaln_lora_emb = self.t_embedder(timesteps.flatten())
        affline_emb = self.affline_norm(t_emb)

        # 2. Concatenate input `x` with the condition
        x_conditioned = torch.cat([x, latent_condition], dim=1)
        
        # 3. Patch Embeddings
        x_patches = self.x_embedder(x_conditioned)
        B, T_p, H_p, W_p, D = x_patches.shape

        # 4. Positional Embeddings (RoPE)
        # Your placeholder returns None, which is fine for now. A real implementation would generate a tensor here.
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
            "(B T) (H W) (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial,
            p2=self.patch_temporal,
            H=H_p, W=W_p, B=B, T=T_p
        )
        
        return output_unpatched

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
                **kwargs)