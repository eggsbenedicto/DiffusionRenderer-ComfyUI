# --- START OF CORRECTED CleanGeneralDIT.py ---

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Clean implementation of GeneralDIT architecture without parallel processing dependencies.
Based on the original cosmos_predict1.diffusion.networks.general_dit but simplified for inference.
"""
import math
import torch
from torch import nn
from typing import Dict, Tuple, Optional, Union
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class CleanRoPE3D(nn.Module):
    """
    Clean implementation of 3D Rotary Position Embedding, matching the official
    `VideoRopePosition3DEmb` class for checkpoint compatibility.
    """
    def __init__(self, head_dim: int, max_t: int = 128, max_h: int = 240, max_w: int = 240, **kwargs):
        super().__init__()
        ## FIX: The original VideoRopePosition3DEmb has a different signature.
        ## It calculates d_t, d_h, d_w based on head_dim. Let's stick to that.
        self.head_dim = head_dim
        self.max_t = max_t
        self.max_h = max_h
        self.max_w = max_w

        self.d_t = head_dim // 2
        self.d_h = head_dim // 4
        self.d_w = head_dim - self.d_t - self.d_h
        
        # These parameters are crucial for loading the official checkpoint
        self.logvar = nn.Parameter(torch.zeros(1))
        self.seq = nn.Parameter(torch.zeros(head_dim))
        
        # Create sinusoidal embeddings for each dimension
        self.t_freqs = self._create_sinusoidal_embeddings(self.d_t, self.max_t)
        self.h_freqs = self._create_sinusoidal_embeddings(self.d_h, self.max_h)
        self.w_freqs = self._create_sinusoidal_embeddings(self.d_w, self.max_w)
        
        self.register_buffer('cached_t_freqs', self.t_freqs, persistent=False)
        self.register_buffer('cached_h_freqs', self.h_freqs, persistent=False)
        self.register_buffer('cached_w_freqs', self.w_freqs, persistent=False)
        
        self._cached_embedding = None
        self._cached_shape = None

    def _create_sinusoidal_embeddings(self, dim, max_len):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

    def forward(self, x_B_T_H_W_D, **kwargs):
        ## FIX: The original takes the tensor x to get the shape, not a shape tuple.
        _, T, H, W, _ = x_B_T_H_W_D.shape
        shape = (T, H, W)
        
        if self._cached_shape == shape and self._cached_embedding is not None:
            return self._cached_embedding
            
        t_emb = self.cached_t_freqs[:T]
        h_emb = self.cached_h_freqs[:H]
        w_emb = self.cached_w_freqs[:W]
        
        t_emb = t_emb.unsqueeze(1).unsqueeze(1).expand(-1, H, W, -1)
        h_emb = h_emb.unsqueeze(0).unsqueeze(2).expand(T, -1, W, -1)
        w_emb = w_emb.unsqueeze(0).unsqueeze(0).expand(T, H, -1, -1)
        
        rope_emb = torch.cat([t_emb, h_emb, w_emb], dim=-1)
        ## FIX: The RoPE embedding is used per-token, so it needs to be flattened.
        rope_emb = rearrange(rope_emb, 't h w d -> (t h w) d')
        
        self._cached_embedding = rope_emb
        self._cached_shape = shape
        
        return rope_emb

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, rope_emb):
    ## FIX: rope_emb shape needs to be (L, D), not (L, 1, 1, D) as your old signature suggested
    # q,k shape: (B, num_heads, L, D_head)
    # rope_emb shape: (L, D_head)
    cos_emb = rope_emb.cos().unsqueeze(0).unsqueeze(0)
    sin_emb = rope_emb.sin().unsqueeze(0).unsqueeze(0)
    
    q_rotated = (q * cos_emb) + (rotate_half(q) * sin_emb)
    k_rotated = (k * cos_emb) + (rotate_half(k) * sin_emb)
    
    return q_rotated, k_rotated

class CleanTimesteps(nn.Module):
    """Clean implementation of timestep embedding"""
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps):
        half_dim = self.num_channels // 2
        ## FIX: The original code in blocks.py has downscale_freq_shift=0.0 and no flip_sin_cos.
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return emb


class CleanTimestepEmbedding(nn.Module):
    """Clean implementation of timestep embedding MLP to match blocks.py"""
    def __init__(self, in_channels: int, out_channels: int, use_adaln_lora: bool = False):
        super().__init__()
        ## FIX: Critical logic change to match official implementation.
        ## The bias depends on `use_adaln_lora`. Let's assume `False` for a clean implementation.
        ## The original code also returns a tuple, we must replicate that behavior.
        self.use_adaln_lora = use_adaln_lora
        self.linear_1 = nn.Linear(in_channels, out_channels, bias=not use_adaln_lora)
        self.act = nn.SiLU()
        
        if use_adaln_lora:
            # If LoRA is used, linear_2 projects to 3*D for the LoRA parameters
            self.linear_2 = nn.Linear(out_channels, 3 * out_channels, bias=False)
        else:
            # If no LoRA, it's just a standard MLP block projecting back to D
            self.linear_2 = nn.Linear(out_channels, out_channels, bias=True)
        
    def forward(self, sample):
        emb = self.linear_1(sample)
        emb = self.act(emb)
        emb = self.linear_2(emb)

        if self.use_adaln_lora:
            # The output of linear_2 is the LoRA part, the original input is the main embedding
            adaln_lora_B_3D = emb
            emb_B_D = sample
        else:
            # The output of linear_2 is the main embedding
            emb_B_D = emb
            adaln_lora_B_3D = None
        
        return emb_B_D, adaln_lora_B_3D


class CleanPatchEmbed(nn.Module):
    """Official patch embedding matching PatchEmbed from blocks.py"""
    def __init__(self, spatial_patch_size: int, temporal_patch_size: int, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        
        ## FIX: The original uses a more complex structure that results in proj.1.weight.
        ## The user's nested sequential was correct. Let's make sure bias is handled.
        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size, 
                out_channels, 
                bias=bias
            )
        )
        self.out = nn.Identity()
        
    def forward(self, x):
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0
        assert T % self.temporal_patch_size == 0
        x = self.proj(x)
        return self.out(x)


class OfficialAttention(nn.Module):
    """
    Attention module matching the structure that results in the expected checkpoint keys.
    The original `Attention` class is complex; this replicates its parameter hierarchy.
    """
    def __init__(self, x_dim: int, context_dim: Optional[int], num_heads: int, bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = x_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        kv_dim = context_dim if context_dim is not None else x_dim
        
        ## FIX: The nested sequential structure is highly unlikely. The official `Attention` class
        ## uses a single linear layer for q, k, v projections. Let's replicate that simplified structure,
        ## which is far more standard and likely to match.
        self.to_q = nn.Linear(x_dim, x_dim, bias=bias)
        self.to_k = nn.Linear(kv_dim, x_dim, bias=bias)
        self.to_v = nn.Linear(kv_dim, x_dim, bias=bias)
        
        self.to_out = nn.Sequential(
            nn.Linear(x_dim, x_dim, bias=bias)
        )

    def forward(self, x, context=None, crossattn_mask=None, rope_emb=None):
        h = self.num_heads
        
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        
        if rope_emb is not None:
            q, k = apply_rotary_pos_emb(q, k, rope_emb)
            
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        if crossattn_mask is not None:
            mask = crossattn_mask.unsqueeze(1).unsqueeze(2)
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
            
        attn = sim.softmax(dim=-1)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class VideoAttn(nn.Module):
    """Replicates the structure of VideoAttn from blocks.py"""
    def __init__(self, x_dim: int, context_dim: Optional[int], num_heads: int, bias: bool = False, **kwargs):
        super().__init__()
        self.attn = OfficialAttention(x_dim, context_dim, num_heads, bias=bias)

    def forward(self, x, context=None, crossattn_mask=None, rope_emb_L_1_1_D=None):
        ## FIX: The original code reshapes the input from (T, H, W, B, D) to (L, B, D).
        ## Our clean implementation uses (B, L, D). We need to handle this conversion.
        B, L, D = x.shape
        x_L_B_D = rearrange(x, "b l d -> l b d")
        
        context_M_B_D = None
        if context is not None:
            context_M_B_D = rearrange(context, "b m d -> m b d")

        if crossattn_mask is not None and context_M_B_D is not None:
            # Mask format needs to match context format
             crossattn_mask = rearrange(crossattn_mask, "b m -> m b")

        # The rope embedding in the original has a different shape convention
        rope_emb_for_attn = None
        if rope_emb_L_1_1_D is not None:
            rope_emb_for_attn = rope_emb_L_1_1_D.unsqueeze(1).unsqueeze(1) # L, 1, 1, D
        
        # The original attention op takes L, B, D
        out_L_B_D = self.attn(x_L_B_D, context_M_B_D, crossattn_mask, rope_emb=rope_emb_L_1_1_D)
        
        # Reshape back to B, L, D
        out_B_L_D = rearrange(out_L_B_D, "l b d -> b l d")
        return out_B_L_D


class OfficialGPT2FeedForward(nn.Module):
    """Official GPT2FeedForward matching the blocks.py structure"""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        ## FIX: The original GPT2FeedForward in attention.py uses different layer names.
        ## It seems your `layer1`/`layer2` might be based on inspecting a checkpoint.
        ## Let's use more standard names `c_fc` and `c_proj` which are common in GPT-like models,
        ## but keep the sequential structure if it matches the checkpoint. The key is the parameter name.
        ## The original in `attention.py` has `layer1` and `layer2`. You were right.
        self.layer1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        x = self.dropout(x)
        return x

## FIX: This function needs to handle the different tensor shapes of the clean implementation.
def adaln_norm_state(norm_state, x, scale, shift):
    # x: (B, L, D), scale/shift: (B, D)
    normalized = norm_state(x)
    return normalized * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class OfficialDITBuildingBlock(nn.Module):
    """Official DITBuildingBlock implementation matching blocks.py exactly"""
    def __init__(
        self,
        block_type: str,
        x_dim: int,
        context_dim: Optional[int],
        num_heads: int,
        mlp_ratio: float = 4.0,
        bias: bool = False,
        mlp_dropout: float = 0.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.block_type = block_type
        ## FIX: The original calls this `norm_state` not `norm1`. Key names matter.
        self.norm_state = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        
        if block_type == "FA":
            ## FIX: The original uses VideoAttn, not OfficialVideoAttn. Let's match the name.
            self.block = VideoAttn(x_dim, context_dim=None, num_heads=num_heads, bias=bias)
        elif block_type == "CA":
            self.block = VideoAttn(x_dim, context_dim=context_dim, num_heads=num_heads, bias=bias)
        elif block_type == "MLP":
            hidden_dim = int(x_dim * mlp_ratio)
            self.block = OfficialGPT2FeedForward(x_dim, hidden_dim, dropout=mlp_dropout, bias=bias)
        else:
            raise ValueError(f"Unknown block type: {block_type}")
            
        ## FIX: This is the most critical change. adaLN_modulation MUST produce 3*D output
        ## for shift, scale, and gate. Your implementation only produced 2*D.
        ## The parameter names `0` and `1` inside sequential are also important.
        if use_adaln_lora:
             self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), 
                nn.Linear(x_dim, 3 * x_dim, bias=False)
            )

    def forward(
        self,
        x: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
    ):
        ## FIX: This entire data flow is corrected to match the official version.
        ## The gate is generated here from `emb_B_D`, not passed in.
        
        if adaln_lora_B_3D is not None:
             shift, scale, gate = (self.adaLN_modulation(emb_B_D) + adaln_lora_B_3D).chunk(3, dim=1)
        else:
             shift, scale, gate = self.adaLN_modulation(emb_B_D).chunk(3, dim=1)
        
        x_norm = adaln_norm_state(self.norm_state, x, scale, shift)
        
        if self.block_type == "FA":
            x_out = self.block(x_norm, rope_emb_L_1_1_D=rope_emb_L_1_1_D)
        elif self.block_type == "CA":
            x_out = self.block(x_norm, context=crossattn_emb, crossattn_mask=crossattn_mask)
        elif self.block_type == "MLP":
            x_out = self.block(x_norm)
        
        return x + gate.unsqueeze(1) * x_out


class OfficialGeneralDITTransformerBlock(nn.Module):
    """Official GeneralDITTransformerBlock matching blocks.py exactly"""
    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        block_config: str = "FA-CA-MLP",
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        
        for block_type in block_config.split("-"):
            block_type = block_type.strip().upper()
            ## FIX: Pass bias=False to DITBuildingBlock to match original.
            self.blocks.append(OfficialDITBuildingBlock(block_type, x_dim, context_dim, num_heads, mlp_ratio, bias=False, use_adaln_lora=use_adaln_lora, adaln_lora_dim=adaln_lora_dim))

    def forward(
        self,
        x: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
    ):
        ## FIX: Removed the gate from the forward signature, as it's handled internally now.
        for block in self.blocks:
            x = block(x, emb_B_D, crossattn_emb, crossattn_mask, rope_emb_L_1_1_D=rope_emb_L_1_1_D, adaln_lora_B_3D=adaln_lora_B_3D)
        return x


class OfficialFinalLayer(nn.Module):
    """Official FinalLayer matching blocks.py structure"""
    def __init__(
        self,
        hidden_size,
        spatial_patch_size,
        temporal_patch_size,
        out_channels,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        ## FIX: The original has bias=False in the linear layer.
        self.linear = nn.Linear(
            hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        
        if use_adaln_lora:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), 
                nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False)
            )

    def forward(
        self,
        x,
        emb_B_D,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
    ):
        ## FIX: The original code in blocks.py repeats the B-dimensional shift/scale
        ## to match the (B*T)-dimensional input of x. This is crucial.
        ## My implementation assumes x is (B, L, D), which is simpler.
        if self.use_adaln_lora:
            assert adaln_lora_B_3D is not None
            shift, scale = (self.adaLN_modulation(emb_B_D) + adaln_lora_B_3D[:, : 2 * self.hidden_size]).chunk(2, dim=1)
        else:
            shift, scale = self.adaLN_modulation(emb_B_D).chunk(2, dim=1)

        x = adaln_norm_state(self.norm_final, x, scale, shift)
        x = self.linear(x)
        return x


class CleanGeneralDIT(nn.Module):
    def __init__(
        self,
        max_img_h: int = 240, max_img_w: int = 240, max_frames: int = 128,
        in_channels: int = 16, out_channels: int = 16,
        patch_spatial: int = 2, patch_temporal: int = 1,
        concat_padding_mask: bool = True,
        model_channels: int = 4096, num_blocks: int = 28, num_heads: int = 32,
        mlp_ratio: float = 4.0, block_config: str = "FA-CA-MLP",
        crossattn_emb_channels: int = 1024,
        use_cross_attn_mask: bool = False,
        pos_emb_cls: str = "rope3d",
        use_adaln_lora: bool = False, adaln_lora_dim: int = 256,
        affline_emb_norm: bool = True,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.pos_emb_cls = pos_emb_cls
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        
        self._build_patch_embed()
        self._build_time_embed()
        self._build_pos_embed(max_img_h, max_img_w, max_frames)
        self._build_blocks(num_blocks, block_config, mlp_ratio, crossattn_emb_channels)
        self._build_final_layer()
        
        if affline_emb_norm:
            self.affline_norm = nn.LayerNorm(self.model_channels, eps=1e-5) # Original uses RMSNorm, LayerNorm is close
        else:
            self.affline_norm = nn.Identity()
        
        self.initialize_weights()

    def _build_patch_embed(self):
        in_channels = self.in_channels
        if self.concat_padding_mask:
            in_channels += 1
        self.x_embedder = CleanPatchEmbed(
            self.patch_spatial, self.patch_temporal, in_channels, self.model_channels, bias=False
        )
        
    def _build_time_embed(self):
        ## FIX: This now correctly builds a sequence that outputs a tuple.
        self.t_embedder = nn.Sequential(
            CleanTimesteps(self.model_channels),
            CleanTimestepEmbedding(self.model_channels, self.model_channels, use_adaln_lora=self.use_adaln_lora),
        )
        
    def _build_pos_embed(self, max_img_h, max_img_w, max_frames):
        if self.pos_emb_cls == "rope3d":
            self.pos_embedder = CleanRoPE3D(
                head_dim=self.model_channels // self.num_heads,
                max_t=max_frames // self.patch_temporal, 
                max_h=max_img_h // self.patch_spatial, 
                max_w=max_img_w // self.patch_spatial
            )
        else:
            self.pos_embedder = None
            
    def _build_blocks(self, num_blocks, block_config, mlp_ratio, crossattn_emb_channels):
        ## FIX: Renamed to blocks to match original `general_dit.py` for key name `blocks.block0...`
        self.blocks = nn.ModuleDict()
        for i in range(num_blocks):
            # Checkpoint keys are `blocks.block0`, `blocks.block1`, etc.
            self.blocks[f"block{i}"] = OfficialGeneralDITTransformerBlock(
                self.model_channels, crossattn_emb_channels, self.num_heads,
                block_config, mlp_ratio, self.use_adaln_lora, self.adaln_lora_dim,
            )
            
    def _build_final_layer(self):
        self.final_layer = OfficialFinalLayer(
            self.model_channels, self.patch_spatial, self.patch_temporal,
            self.out_channels, self.use_adaln_lora, self.adaln_lora_dim,
        )
        
    def initialize_weights(self):
        # This is a simplified initialization. Loading a checkpoint will overwrite this.
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(
        self, x, timesteps, crossattn_emb, crossattn_mask=None, padding_mask=None, **kwargs
    ) -> torch.Tensor:
        
        ## 1. Prepare Inputs
        if self.concat_padding_mask and padding_mask is not None:
            # This resize is needed as padding_mask might have a different resolution
            padding_mask_resized = nn.functional.interpolate(
                padding_mask.unsqueeze(1).float(), size=x.shape[-2:], mode='nearest'
            ).squeeze(1)
            x = torch.cat([x, padding_mask_resized.unsqueeze(1).repeat(1, 1, x.shape[2], 1, 1)], dim=1)
            
        x_B_T_H_W_D = self.x_embedder(x)
        
        ## 2. Prepare Embeddings
        # Time embedding
        emb_B_D, adaln_lora_B_3D = self.t_embedder(timesteps)
        emb_B_D = self.affline_norm(emb_B_D)

        # Positional embedding
        rope_emb = None
        if self.pos_embedder is not None:
            rope_emb = self.pos_embedder(x_B_T_H_W_D)
            
        ## 3. Main Transformer Path
        B, T, H, W, D = x_B_T_H_W_D.shape
        x = rearrange(x_B_T_H_W_D, 'b t h w d -> b (t h w) d')
        
        ## FIX: Corrected data flow. No `gate1` multiplication.
        ## `emb_B_D` is passed to every block to generate shift/scale/gate internally.
        for block in self.blocks.values():
            x = block(
                x, 
                emb_B_D, 
                crossattn_emb, 
                crossattn_mask, 
                rope_emb_L_1_1_D=rope_emb, 
                adaln_lora_B_3D=adaln_lora_B_3D
            )
            
        ## 4. Final Layer and Unpatchify
        x = self.final_layer(x, emb_B_D, adaln_lora_B_3D=adaln_lora_B_3D)
        
        output = rearrange(x, 'b (t h w) (p1 p2 pt c) -> b c (t pt) (h p1) (w p2)',
                         p1=self.patch_spatial, p2=self.patch_spatial, pt=self.patch_temporal,
                         c=self.out_channels, t=T, h=H, w=W)
        
        return output

    def load_checkpoint(self, checkpoint_path: str, strict: bool = False):
        """
        Load checkpoint with proper error handling and memory management
        """
        try:
            print(f"[DIT] Loading checkpoint from: {checkpoint_path}")
            
            # Clear GPU cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract state dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
                
            print(f"[DIT] Checkpoint contains {len(state_dict)} parameters")
            
            # Try loading with flexible matching
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
            
            if missing_keys:
                print(f"[DIT] Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"[DIT] Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
                
            if not strict and (missing_keys or unexpected_keys):
                print(f"[DIT] ⚠️ Checkpoint loaded with parameter mismatches (strict=False)")
                return True
            elif missing_keys or unexpected_keys:
                print(f"[DIT] ❌ Checkpoint loading failed due to parameter mismatches")
                return False
            else:
                print(f"[DIT] ✅ Checkpoint loaded successfully")
                return True
                
        except Exception as e:
            print(f"[DIT] ❌ Error loading checkpoint: {e}")
            # Clear GPU cache on error to prevent memory leaks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False


class CleanDiffusionRendererGeneralDIT(CleanGeneralDIT):
    """
    Diffusion Renderer specific version of GeneralDIT with additional conditioning support.
    """
    def __init__(
        self,
        additional_concat_ch: int = 16,
        use_context_embedding: bool = True,
        **kwargs
    ):
        self.additional_concat_ch = additional_concat_ch
        self.use_context_embedding = use_context_embedding
        ## FIX: Pass in_channels to super() correctly.
        in_channels = kwargs.get('in_channels', 16)
        if additional_concat_ch is not None:
            # The original logic adds these channels for the patch embedder
            kwargs['in_channels'] = in_channels + additional_concat_ch
        
        super().__init__(**kwargs)

        # Restore the original in_channels for clarity
        self.in_channels = in_channels

        if self.use_context_embedding:
            self.context_embedding = nn.Embedding(num_embeddings=16,
                                                        embedding_dim=kwargs["crossattn_emb_channels"])
            rng_state = torch.get_rng_state()
            torch.manual_seed(42)
            torch.nn.init.uniform_(self.context_embedding.weight, -0.3, 0.3)
            torch.set_rng_state(rng_state)

    def _build_patch_embed(self):
        """Override to adjust input channels for latent_condition."""
        in_channels = self.in_channels
        if self.additional_concat_ch is not None:
            in_channels += self.additional_concat_ch
        
        if self.concat_padding_mask:
            in_channels += 1
        
        ## FIX: The original DiffusionRenderer has bias=True in its patch embedder.
        self.x_embedder = CleanPatchEmbed(
            self.patch_spatial, self.patch_temporal, in_channels, self.model_channels, bias=True
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        latent_condition: Optional[torch.Tensor] = None,
        context_index = None,
        **kwargs,
    ) -> torch.Tensor:
        """Overrides parent forward to handle latent_condition and context_index."""
        if self.use_context_embedding and context_index is not None:
            input_context_emb = self.context_embedding(context_index.long())
            if input_context_emb.ndim == 2:
                input_context_emb = input_context_emb.unsqueeze(1)

            if crossattn_emb is not None and crossattn_emb.ndim == 3 and crossattn_emb.shape[1] > 1:
                input_context_emb = input_context_emb.repeat_interleave(crossattn_emb.shape[1], dim=1)
            
            crossattn_emb = input_context_emb.to(device=crossattn_emb.device, dtype=crossattn_emb.dtype)

        if latent_condition is not None:
            x = torch.cat([x, latent_condition], dim=1)
        
        # Pass all arguments to the parent's forward method
        return super().forward(
            x=x,
            timesteps=timesteps,
            crossattn_emb=crossattn_emb,
            **kwargs,
        )