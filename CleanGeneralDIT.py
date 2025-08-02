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
    def __init__(self, head_dim: int, max_t: int = 128, max_h: int = 240, max_w: int = 240):
        super().__init__()
        self.head_dim = head_dim
        self.max_t = max_t
        self.max_h = max_h
        self.max_w = max_w
        
        # Split head_dim for each axis, ensuring it matches the official distribution
        self.d_t = head_dim // 2
        self.d_h = head_dim // 4
        self.d_w = head_dim - self.d_t - self.d_h
        
        # This parameter is crucial for loading the official checkpoint
        self.logvar = nn.Parameter(torch.zeros(1))
        
        # Add seq parameter to match checkpoint structure - should be 1D [head_dim]
        self.seq = nn.Parameter(torch.zeros(head_dim))
        
        # Create sinusoidal embeddings for each dimension
        self.t_freqs = self._create_sinusoidal_embeddings(self.d_t, self.max_t)
        self.h_freqs = self._create_sinusoidal_embeddings(self.d_h, self.max_h)
        self.w_freqs = self._create_sinusoidal_embeddings(self.d_w, self.max_w)
        
        # Register buffers for device management
        self.register_buffer('cached_t_freqs', self.t_freqs, persistent=False)
        self.register_buffer('cached_h_freqs', self.h_freqs, persistent=False)
        self.register_buffer('cached_w_freqs', self.w_freqs, persistent=False)
        
        # Caching for efficiency
        self._cached_embedding = None
        self._cached_shape = None

    def _create_sinusoidal_embeddings(self, dim, max_len):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

    def forward(self, shape):
        T, H, W = shape
        
        # Return cached embedding if shape matches
        if self._cached_shape == shape and self._cached_embedding is not None:
            return self._cached_embedding
            
        # Get embeddings for the current shape
        t_emb = self.cached_t_freqs[:T]
        h_emb = self.cached_h_freqs[:H]
        w_emb = self.cached_w_freqs[:W]
        
        # Expand dimensions to match the (T, H, W) grid
        t_emb = t_emb.unsqueeze(1).unsqueeze(1).expand(-1, H, W, -1)
        h_emb = h_emb.unsqueeze(0).unsqueeze(2).expand(T, -1, W, -1)
        w_emb = w_emb.unsqueeze(0).unsqueeze(0).expand(T, H, -1, -1)
        
        # Concatenate and rearrange to (L, D) where L = T*H*W
        rope_emb = torch.cat([t_emb, h_emb, w_emb], dim=-1)
        rope_emb = rearrange(rope_emb, 't h w d -> (t h w) d')
        
        # Cache the result
        self._cached_embedding = rope_emb
        self._cached_shape = shape
        
        return rope_emb

from typing import Dict, Tuple, Optional, Union

import torch
import torch.nn as nn
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, rope_emb):
    """Applies Rotary Position Embedding to the query and key tensors."""
    # rope_emb is (L, D_head)
    # q, k are (B, num_heads, L, D_head)
    # Unsqueeze rope_emb for batch and head dimensions for broadcasting
    cos_emb = rope_emb.cos().unsqueeze(0).unsqueeze(0)
    sin_emb = rope_emb.sin().unsqueeze(0).unsqueeze(0)
    
    q_rotated = (q * cos_emb) + (rotate_half(q) * sin_emb)
    k_rotated = (k * cos_emb) + (rotate_half(k) * sin_emb)
    
    return q_rotated, k_rotated


class CleanTimesteps(nn.Module):
    """Clean implementation of timestep embedding"""
    
    def __init__(self, num_channels: int, flip_sin_cos: bool = False, downscale_freq_shift: float = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_cos = flip_sin_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - self.downscale_freq_shift)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        # Concatenate sin and cos
        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        
        if self.flip_sin_cos:
            emb = torch.cat([sin_emb, cos_emb], dim=-1)
        else:
            emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return emb


class CleanTimestepEmbedding(nn.Module):
    """Clean implementation of timestep embedding MLP"""

    def __init__(self, in_channels: int, out_channels: int, use_adaln_lora: bool = False, act_fn: str = "silu"):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, out_channels, bias=True)
        self.act = nn.SiLU() if act_fn == "silu" else nn.GELU()
        # Official implementation outputs out_channels * 3 in the second linear
        self.linear_2 = nn.Linear(out_channels, out_channels * 3, bias=True)
        self.use_adaln_lora = use_adaln_lora
        
    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        # Return the 3*D tensor and None for lora part
        return sample, None


class CleanPatchEmbed(nn.Module):
    """Official patch embedding matching PatchEmbed from blocks.py"""
    
    def __init__(self, spatial_patch_size: int, temporal_patch_size: int, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        
        self.rearrange = Rearrange(
            "b c (t r) (h m) (w n) -> b t h w (c r m n)",
            r=temporal_patch_size,
            m=spatial_patch_size,
            n=spatial_patch_size,
        )
        self.proj = nn.Sequential(
            nn.Identity(),
            nn.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size, 
                out_channels, 
                bias=True
            )
        )
        self.out = nn.Identity()
        
    def forward(self, x):
        """
        x: (B, C, T, H, W)
        output: (B, T', H', W', D) where T'=T//temporal_patch_size, etc.
        """
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0
        assert T % self.temporal_patch_size == 0
        x = self.rearrange(x)
        x = self.proj(x)
        return self.out(x)


class DummyWeight(nn.Module):
    """A dummy module that holds a single weight parameter."""
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim))

    def forward(self, x):
        return x


class ProjLayer(nn.Module):
    """A helper module for projection layers."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=bias)
    def forward(self, x):
        return self.proj(x)


class OfficialVideoAttn(nn.Module):
    """Official VideoAttn implementation matching blocks.py with attn submodule"""
    
    def __init__(self, x_dim: int, context_dim: Optional[int], num_heads: int, bias: bool = True):
        super().__init__()
        # Create nested attn module to match checkpoint structure
        self.attn = OfficialAttention(x_dim, context_dim, num_heads, bias)

    def forward(self, x, context=None, crossattn_mask=None, rope_emb_L_1_1_D=None):
        return self.attn(x, context, crossattn_mask, rope_emb_L_1_1_D)


class OfficialAttention(nn.Module):
    """Nested attention module with the actual attention layers"""
    
    def __init__(self, x_dim: int, context_dim: Optional[int], num_heads: int, bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = x_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Match checkpoint structure exactly
        self.to_q = nn.Sequential(
            nn.Linear(x_dim, x_dim, bias=False),
            nn.Linear(x_dim, x_dim, bias=bias)
        )
        
        kv_dim = context_dim if context_dim is not None else x_dim
        self.to_k = nn.Sequential(
            nn.Linear(kv_dim, x_dim, bias=False),
            nn.Linear(x_dim, x_dim, bias=bias)
        )
        self.to_v = nn.Sequential(
            nn.Linear(kv_dim, x_dim, bias=False)
        )
        
        self.to_out = nn.Sequential(
            nn.Linear(x_dim, x_dim, bias=bias)
        )

    def forward(self, x, context=None, crossattn_mask=None, rope_emb_L_1_1_D=None):
        h = self.num_heads
        
        q = self.to_q(x)
        
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        
        if rope_emb_L_1_1_D is not None:
            q, k = apply_rotary_pos_emb(q, k, rope_emb_L_1_1_D)
            
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        if crossattn_mask is not None:
            mask = crossattn_mask.unsqueeze(1).unsqueeze(2)
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
            
        attn = sim.softmax(dim=-1)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class OfficialGPT2FeedForward(nn.Module):
    """Official GPT2FeedForward matching the blocks.py structure with layer1/layer2"""
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        # Use layer1 and layer2 to match checkpoint structure
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


def adaln_norm_state(norm_state, x, scale, shift):
    """Official adaln_norm_state function from blocks.py"""
    normalized = norm_state(x)
    # x is (B, N, D), scale/shift are (B, D)
    # We need to unsqueeze scale/shift to (B, 1, D) for broadcasting
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
        self.norm1 = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        
        if block_type == "FA":
            self.block = OfficialVideoAttn(x_dim, context_dim=None, num_heads=num_heads, bias=bias)
        elif block_type == "CA":
            self.block = OfficialVideoAttn(x_dim, context_dim=context_dim, num_heads=num_heads, bias=bias)
        elif block_type == "MLP":
            hidden_dim = int(x_dim * mlp_ratio)
            self.block = OfficialGPT2FeedForward(x_dim, hidden_dim, dropout=mlp_dropout, bias=bias)
        else:
            raise ValueError(f"Unknown block type: {block_type}")
            
        # Use Sequential structure to match checkpoint: adaLN_modulation.1.weight, adaLN_modulation.2.weight
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(x_dim, x_dim, bias=False),
            nn.Linear(x_dim, 2 * x_dim, bias=True)
        )

    def forward(
        self,
        x: torch.Tensor,
        emb_B_D: torch.Tensor,
        gate: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
    ):
        shift, scale = self.adaLN_modulation(emb_B_D).chunk(2, dim=1)
        x_norm = adaln_norm_state(self.norm1, x, scale, shift)
        
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
        # Use ModuleList for numeric indexing to match checkpoint
        self.blocks = nn.ModuleList()
        
        # Parse block config: "FA-CA-MLP" -> ["FA", "CA", "MLP"]
        for block_type in block_config.split("-"):
            block_type = block_type.strip().upper()
            if block_type == "FA":
                self.blocks.append(OfficialDITBuildingBlock(block_type, x_dim, context_dim=None, num_heads=num_heads, mlp_ratio=mlp_ratio, use_adaln_lora=use_adaln_lora, adaln_lora_dim=adaln_lora_dim))
            elif block_type == "CA":
                self.blocks.append(OfficialDITBuildingBlock(block_type, x_dim, context_dim=context_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, use_adaln_lora=use_adaln_lora, adaln_lora_dim=adaln_lora_dim))
            elif block_type == "MLP":
                self.blocks.append(OfficialDITBuildingBlock(block_type, x_dim, context_dim=None, num_heads=num_heads, mlp_ratio=mlp_ratio, use_adaln_lora=use_adaln_lora, adaln_lora_dim=adaln_lora_dim))
            else:
                raise ValueError(f"Unknown block type: {block_type}")

    def forward(
        self,
        x: torch.Tensor,
        emb_B_D: torch.Tensor,
        gate: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
    ):
        for block in self.blocks:
            x = block(x, emb_B_D, gate, crossattn_emb, crossattn_mask, rope_emb_L_1_1_D=rope_emb_L_1_1_D, adaln_lora_B_3D=adaln_lora_B_3D)
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
        self.linear = nn.Linear(
            hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=True
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        
        # Match official FinalLayer structure
        if use_adaln_lora:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=True),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), 
                nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=True)
            )

    def forward(
        self,
        x,
        emb_B_D,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
    ):
        shift, scale = self.adaLN_modulation(emb_B_D).chunk(2, dim=1)
        x = adaln_norm_state(self.norm_final, x, scale, shift)
        x = self.linear(x)
        return x


class CleanGeneralDIT(nn.Module):
    """
    Clean implementation of General Diffusion Transformer (DiT) architecture.
    Simplified version of the original GeneralDIT without parallel processing.
    """
    
    def __init__(
        self,
        # Input/output configuration
        max_img_h: int = 240,
        max_img_w: int = 240, 
        max_frames: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        
        # Patch configuration
        patch_spatial: int = 2,
        patch_temporal: int = 1,
        concat_padding_mask: bool = True,
        
        # Architecture configuration
        model_channels: int = 4096,
        num_blocks: int = 28,
        num_heads: int = 32,
        mlp_ratio: float = 4.0,
        block_config: str = "FA-CA-MLP",
        
        # Cross attention
        crossattn_emb_channels: int = 1024,  # FIXED: Match context embedding dimension
        use_cross_attn_mask: bool = False,
        
        # Position embedding
        pos_emb_cls: str = "rope3d",
        
        # AdaLN LoRA
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        
        # Additional features
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
        self.use_cross_attn_mask = use_cross_attn_mask
        self.concat_padding_mask = concat_padding_mask
        self.pos_emb_cls = pos_emb_cls
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.affline_emb_norm = affline_emb_norm
        
        self._build_patch_embed()
        self._build_time_embed()
        self._build_pos_embed()
        self._build_blocks(num_blocks, block_config, mlp_ratio, crossattn_emb_channels)
        self._build_final_layer()
        
        self.initialize_weights()

    def _build_patch_embed(self):
        in_channels = self.in_channels
        if self.concat_padding_mask:
            in_channels += 1
        self.x_embedder = CleanPatchEmbed(
            self.patch_spatial, self.patch_temporal, in_channels, self.model_channels, bias=False
        )
        
    def _build_time_embed(self):
        self.t_embedder = nn.Sequential(
            CleanTimesteps(self.model_channels),
            CleanTimestepEmbedding(self.model_channels, self.model_channels, use_adaln_lora=self.use_adaln_lora),
        )
        
    def _build_pos_embed(self):
        self.pos_embedder = CleanRoPE3D(
            head_dim=self.model_channels // self.num_heads,
            max_t=128, max_h=240, max_w=240
        )
            
    def _build_blocks(self, num_blocks, block_config, mlp_ratio, crossattn_emb_channels):
        self.blocks = nn.ModuleDict()
        for i in range(num_blocks):
            self.blocks[f"block{i}"] = OfficialGeneralDITTransformerBlock(
                self.model_channels,
                crossattn_emb_channels,
                self.num_heads,
                block_config,
                mlp_ratio,
                self.use_adaln_lora,
                self.adaln_lora_dim,
            )
            
    def _build_final_layer(self):
        self.final_layer = OfficialFinalLayer(
            self.model_channels,
            self.patch_spatial,
            self.patch_temporal,
            self.out_channels,
            self.use_adaln_lora,
            self.adaln_lora_dim,
        )
        if self.affline_emb_norm:
            self.affline_norm = nn.LayerNorm(self.model_channels, bias=True)
        else:
            self.affline_norm = nn.Identity()
        
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
        self.apply(_basic_init)
        
        # Initialize timestep embedding
        nn.init.normal_(self.t_embedder[1].linear_1.weight, std=0.02)
        nn.init.constant_(self.t_embedder[1].linear_1.bias, 0)
        nn.init.normal_(self.t_embedder[1].linear_2.weight, std=0.02)
        nn.init.constant_(self.t_embedder[1].linear_2.bias, 0)
        
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks.values():
            for sub_block in block.blocks:
                # Handle both nn.Linear and nn.Sequential cases
                if isinstance(sub_block.adaLN_modulation, nn.Sequential):
                    nn.init.constant_(sub_block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(sub_block.adaLN_modulation[-1].bias, 0)
                else:
                    nn.init.constant_(sub_block.adaLN_modulation.weight, 0)
                    nn.init.constant_(sub_block.adaLN_modulation.bias, 0)

        # Zero-out final layer
        if isinstance(self.final_layer.adaLN_modulation, nn.Sequential):
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        else:
            nn.init.constant_(self.final_layer.adaLN_modulation.weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation.bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
            
    def prepare_inputs(self, x, timesteps, crossattn_emb, padding_mask=None, latent_condition=None):
        if self.concat_padding_mask and padding_mask is not None:
            x = torch.cat([x, padding_mask.unsqueeze(1).repeat(1, 1, x.shape[2], 1, 1)], dim=1)
            
        x = self.x_embedder(x)
        
        # t_embedder returns a tuple (embedding, lora_embedding)
        emb, adaln_lora = self.t_embedder(timesteps)
        
        rope_emb = None
        if self.pos_embedder is not None:
            rope_emb = self.pos_embedder(shape=(x.shape[1], x.shape[2], x.shape[3]))
            
        return x, emb, adaln_lora, rope_emb
        
    def forward(
        self,
        x,
        timesteps,
        crossattn_emb,
        crossattn_mask=None,
        padding_mask=None,
        latent_condition=None,
        scalar_feature=None,
        **kwargs
    ) -> torch.Tensor:
        x, emb, adaln_lora, rope_emb = self.prepare_inputs(x, timesteps, crossattn_emb, padding_mask, latent_condition)
        
        # Split the 3*D embedding into the main embedding, and two gates
        ada_emb, gate1, gate2 = emb.chunk(3, dim=1)
        
        B, T, H, W, D = x.shape
        # Flatten spatial and temporal dimensions into a single sequence dimension
        x = rearrange(x, 'b t h w d -> b (t h w) d')
        
        # Apply first gate
        x = x * gate1.unsqueeze(1)
        
        for block in self.blocks.values():
            x = block(
                x, 
                ada_emb, 
                gate2, # Use the second gate for the residual connections in the blocks
                crossattn_emb, 
                crossattn_mask, 
                rope_emb_L_1_1_D=rope_emb, 
                adaln_lora_B_3D=adaln_lora
            )
            
        # The final layer uses the main embedding to compute its shift and scale
        x = self.final_layer(x, ada_emb, adaln_lora_B_3D=adaln_lora)
        
        # Unpatchify the output
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
        super().__init__(**kwargs)

        # init context embedding
        if self.use_context_embedding:
            self.context_embedding = nn.Embedding(num_embeddings=16,
                                                        embedding_dim=kwargs["crossattn_emb_channels"])
            rng_state = torch.get_rng_state()
            torch.manual_seed(42)
            torch.nn.init.uniform_(self.context_embedding.weight, -0.3, 0.3)
            torch.set_rng_state(rng_state)

    def _build_patch_embed(self):
        """
        Override to adjust input channels for latent_condition.
        """
        in_channels = self.in_channels
        if self.additional_concat_ch is not None:
            in_channels += self.additional_concat_ch
        
        if self.concat_padding_mask:
            in_channels += 1
        
        self.x_embedder = CleanPatchEmbed(
            self.patch_spatial, self.patch_temporal, in_channels, self.model_channels, bias=True
        )

    def prepare_inputs(self, x, timesteps, crossattn_emb, padding_mask=None, latent_condition=None):
        """
        Override to concatenate latent_condition to the input tensor.
        """
        if latent_condition is not None:
            x = torch.cat([x, latent_condition], dim=1)

        if self.concat_padding_mask and padding_mask is not None:
            x = torch.cat([x, padding_mask.unsqueeze(1).repeat(1, 1, x.shape[2], 1, 1)], dim=1)
            
        x = self.x_embedder(x)
        
        # t_embedder returns a tuple (embedding, lora_embedding)
        emb, adaln_lora = self.t_embedder(timesteps)
        
        rope_emb = None
        if self.pos_embedder is not None:
            rope_emb = self.pos_embedder(shape=(x.shape[1], x.shape[2], x.shape[3]))
            
        return x, emb, adaln_lora, rope_emb

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        latent_condition: Optional[torch.Tensor] = None,
        scalar_feature: Optional[torch.Tensor] = None,
        context_index = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Overrides the parent `forward` method. It handles the context_index, and passes all given arguments to the
        parent method and returns its result.
        """
        if self.use_context_embedding and context_index is not None:
            # Overwrite the context embedding
            input_context_emb = self.context_embedding(context_index.long())
            if input_context_emb.ndim == 2:
                input_context_emb = input_context_emb.unsqueeze(1).clone()

            if crossattn_emb is not None:
                input_context_emb = input_context_emb.repeat_interleave(crossattn_emb.shape[1], dim=1)
            
            input_context_emb = input_context_emb.to(device=crossattn_emb.device, dtype=crossattn_emb.dtype)
            crossattn_emb = input_context_emb

        return super().forward(
            x=x,
            timesteps=timesteps,
            crossattn_emb=crossattn_emb,
            crossattn_mask=crossattn_mask,
            padding_mask=padding_mask,
            latent_condition=latent_condition,
            scalar_feature=scalar_feature,
            **kwargs,
        )
