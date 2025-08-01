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

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Union
from einops import rearrange
from einops.layers.torch import Rearrange
import math


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
            emb = torch.cat([cos_emb, sin_emb], dim=-1)
        else:
            emb = torch.cat([sin_emb, cos_emb], dim=-1)

        return emb


class CleanTimestepEmbedding(nn.Module):
    """Clean implementation of timestep embedding MLP"""

    def __init__(self, in_channels: int, out_channels: int, use_adaln_lora: bool = False, act_fn: str = "silu"):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, out_channels)
        self.act = nn.SiLU() if act_fn == "silu" else nn.GELU()
        # Official implementation outputs out_channels * 3 in the second linear
        self.linear_2 = nn.Linear(out_channels, out_channels * 3)
        self.use_adaln_lora = use_adaln_lora
        
    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class CleanPatchEmbed(nn.Module):
    """Official patch embedding matching PatchEmbed from blocks.py"""
    
    def __init__(self, spatial_patch_size: int, temporal_patch_size: int, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        
        # Official structure: sequential with Rearrange + Linear to match proj.1.weight
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
            ),
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
        x = self.proj(x)
        return self.out(x)


class CleanRoPE3D:
    """Simplified 3D RoPE (Rotary Position Embedding) for video"""
    
    def __init__(self, head_dim: int, max_t: int, max_h: int, max_w: int):
        self.head_dim = head_dim
        self.max_t = max_t
        self.max_h = max_h  
        self.max_w = max_w
        
    def get_rope_embeddings(self, t: int, h: int, w: int, device: torch.device):
        """Generate 3D rotary embeddings for given dimensions"""
        # Create position grids
        pos_t = torch.arange(t, device=device).float()
        pos_h = torch.arange(h, device=device).float()
        pos_w = torch.arange(w, device=device).float()
        
        # Create frequency bases
        dim_t = self.head_dim // 6
        dim_h = self.head_dim // 6  
        dim_w = self.head_dim // 6
        
        # Temporal frequencies
        freqs_t = 1.0 / (10000 ** (torch.arange(0, dim_t * 2, 2, device=device).float() / (dim_t * 2)))
        # Spatial frequencies  
        freqs_h = 1.0 / (10000 ** (torch.arange(0, dim_h * 2, 2, device=device).float() / (dim_h * 2)))
        freqs_w = 1.0 / (10000 ** (torch.arange(0, dim_w * 2, 2, device=device).float() / (dim_w * 2)))
        
        # Create embeddings
        t_emb = torch.outer(pos_t, freqs_t)
        h_emb = torch.outer(pos_h, freqs_h)
        w_emb = torch.outer(pos_w, freqs_w)
        
        # Expand to full grid
        rope_emb = torch.zeros(t, h, w, self.head_dim, device=device)
        
        # Fill in the embeddings (simplified version)
        rope_emb[:, :, :, :dim_t*2] = t_emb[:, None, None, :]
        rope_emb[:, :, :, dim_t*2:dim_t*2+dim_h*2] = h_emb[None, :, None, :]
        rope_emb[:, :, :, dim_t*2+dim_h*2:dim_t*2+dim_h*2+dim_w*2] = w_emb[None, None, :, :]
        
        return rope_emb


class OfficialVideoAttn(nn.Module):
    """Official VideoAttn implementation matching blocks.py"""
    
    def __init__(self, x_dim: int, context_dim: Optional[int], num_heads: int, bias: bool = False):
        super().__init__()
        self.x_dim = x_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = x_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Match official VideoAttn structure using simplified attention
        self.to_q = nn.Linear(x_dim, x_dim, bias=bias)
        self.to_k = nn.Linear(context_dim if context_dim else x_dim, x_dim, bias=bias)
        self.to_v = nn.Linear(context_dim if context_dim else x_dim, x_dim, bias=bias)
        self.to_out = nn.Linear(x_dim, x_dim, bias=bias)
        
    def forward(self, x, context=None, crossattn_mask=None, rope_emb_L_1_1_D=None):
        """
        x: (T, H, W, B, D) - official format
        context: (M, B, D) or None for self-attention
        """
        T, H, W, B, D = x.shape
        
        # Reshape to sequence format
        x_seq = rearrange(x, "t h w b d -> (t h w) b d")  # (L, B, D)
        x_seq = rearrange(x_seq, "l b d -> b l d")  # (B, L, D)
        
        if context is not None:
            context = rearrange(context, "m b d -> b m d")  # (B, M, D)
            k_input = v_input = context
        else:
            k_input = v_input = x_seq
            
        # Compute attention
        q = self.to_q(x_seq).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(k_input).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(v_input).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE if provided
        if rope_emb_L_1_1_D is not None and context is None:  # Only for self-attention
            rope_emb = rope_emb_L_1_1_D.squeeze(1).squeeze(1)  # (L, D)
            rope_emb = rope_emb[:q.shape[2]]  # Match sequence length
            q, k = apply_rotary_pos_emb(q, k, rope_emb)
            
        # Standard attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if crossattn_mask is not None:
            attn = attn.masked_fill(crossattn_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
            
        attn = attn.softmax(dim=-1)
        out = attn @ v
        
        out = out.transpose(1, 2).reshape(B, -1, D)
        out = self.to_out(out)
        
        # Convert back to official format
        out = rearrange(out, "b (t h w) d -> t h w b d", t=T, h=H, w=W)
        return out


class OfficialGPT2FeedForward(nn.Module):
    """Official GPT2FeedForward matching the blocks.py structure"""
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.layer1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.layer2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """x: (T, H, W, B, D) - official format"""
        original_shape = x.shape
        x = rearrange(x, "t h w b d -> (t h w b) d")
        
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        
        x = x.reshape(original_shape)
        return x


def adaln_norm_state(norm_state, x, scale, shift):
    """Official adaln_norm_state function from blocks.py"""
    normalized = norm_state(x)
    return normalized * (1 + scale) + shift


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
        block_type = block_type.lower()
        
        self.block_type = block_type
        self.use_adaln_lora = use_adaln_lora
        
        # Create the actual block
        if block_type in ["cross_attn", "ca"]:
            self.block = OfficialVideoAttn(x_dim, context_dim, num_heads, bias=bias)
        elif block_type in ["full_attn", "fa"]:
            self.block = OfficialVideoAttn(x_dim, None, num_heads, bias=bias)
        elif block_type in ["mlp", "ff"]:
            self.block = OfficialGPT2FeedForward(x_dim, int(x_dim * mlp_ratio), dropout=mlp_dropout, bias=bias)
        else:
            raise ValueError(f"Unknown block type: {block_type}")
            
        # AdaLN components
        self.norm_state = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.n_adaln_chunks = 3
        
        if use_adaln_lora:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * x_dim, bias=False),
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), 
                nn.Linear(x_dim, self.n_adaln_chunks * x_dim, bias=False)
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
        """
        x: (T, H, W, B, D) - official format
        emb_B_D: (B, D) - timestep embedding
        crossattn_emb: (M, B, D) - cross attention context
        """
        # AdaLN modulation
        if self.use_adaln_lora and adaln_lora_B_3D is not None:
            shift_B_D, scale_B_D, gate_B_D = (self.adaLN_modulation(emb_B_D) + adaln_lora_B_3D).chunk(
                self.n_adaln_chunks, dim=1
            )
        else:
            shift_B_D, scale_B_D, gate_B_D = self.adaLN_modulation(emb_B_D).chunk(self.n_adaln_chunks, dim=1)

        # Expand to match x dimensions: (T, H, W, B, D)
        shift_1_1_1_B_D = shift_B_D.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        scale_1_1_1_B_D = scale_B_D.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        gate_1_1_1_B_D = gate_B_D.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Apply block with adaptive normalization
        if self.block_type in ["mlp", "ff"]:
            x = x + gate_1_1_1_B_D * self.block(
                adaln_norm_state(self.norm_state, x, scale_1_1_1_B_D, shift_1_1_1_B_D)
            )
        elif self.block_type in ["full_attn", "fa"]:
            x = x + gate_1_1_1_B_D * self.block(
                adaln_norm_state(self.norm_state, x, scale_1_1_1_B_D, shift_1_1_1_B_D),
                context=None,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
            )
        elif self.block_type in ["cross_attn", "ca"]:
            x = x + gate_1_1_1_B_D * self.block(
                adaln_norm_state(self.norm_state, x, scale_1_1_1_B_D, shift_1_1_1_B_D),
                context=crossattn_emb,
                crossattn_mask=crossattn_mask,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
            )
        
        return x


class OfficialGeneralDITTransformerBlock(nn.Module):
    """Official GeneralDITTransformerBlock matching blocks.py exactly"""
    
    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        block_config: str = "ca-fa-mlp",
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        
        # Parse block config: "ca-fa-mlp" -> ["ca", "fa", "mlp"]
        for block_type in block_config.split("-"):
            self.blocks.append(
                OfficialDITBuildingBlock(
                    block_type,
                    x_dim,
                    context_dim,
                    num_heads,
                    mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ):
        if extra_per_block_pos_emb is not None:
            x = x + extra_per_block_pos_emb
            
        for block in self.blocks:
            x = block(
                x,
                emb_B_D,
                crossattn_emb,
                crossattn_mask,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_3D=adaln_lora_B_3D,
            )
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
            hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        
        if use_adaln_lora:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False),
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False)
            )

    def forward(
        self,
        x_BT_HW_D,
        emb_B_D,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
    ):
        if self.use_adaln_lora and adaln_lora_B_3D is not None:
            shift_B_D, scale_B_D = (self.adaLN_modulation(emb_B_D) + adaln_lora_B_3D[:, : 2 * self.hidden_size]).chunk(
                2, dim=1
            )
        else:
            shift_B_D, scale_B_D = self.adaLN_modulation(emb_B_D).chunk(2, dim=1)

        B = emb_B_D.shape[0]
        T = x_BT_HW_D.shape[0] // B
        
        from einops import repeat
        shift_BT_D, scale_BT_D = repeat(shift_B_D, "b d -> (b t) d", t=T), repeat(scale_B_D, "b d -> (b t) d", t=T)
        
        def modulate(x, shift, scale):
            return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            
        x_BT_HW_D = modulate(self.norm_final(x_BT_HW_D), shift_BT_D, scale_BT_D)
        x_BT_HW_D = self.linear(x_BT_HW_D)
        return x_BT_HW_D


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
        
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.model_channels = model_channels
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.concat_padding_mask = concat_padding_mask
        self.use_cross_attn_mask = use_cross_attn_mask
        self.crossattn_emb_channels = crossattn_emb_channels
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        
        # Build components
        self._build_patch_embed()
        self._build_time_embed()
        self._build_pos_embed()
        self._build_blocks()
        self._build_final_layer()
        
        # Normalization
        if affline_emb_norm:
            self.affline_norm = nn.LayerNorm(model_channels)
        else:
            self.affline_norm = nn.Identity()
            
        self.initialize_weights()
        
    def _build_patch_embed(self):
        """Build patch embedding layer"""
        in_ch = self.in_channels
        if self.concat_padding_mask:
            in_ch += 1
            
        self.x_embedder = CleanPatchEmbed(
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            in_channels=in_ch,
            out_channels=self.model_channels,
            bias=False
        )
        
    def _build_time_embed(self):
        """Build timestep embedding"""
        self.t_embedder = nn.Sequential(
            CleanTimesteps(self.model_channels),
            CleanTimestepEmbedding(self.model_channels, self.model_channels, use_adaln_lora=self.use_adaln_lora)
        )
        
    def _build_pos_embed(self):
        """Build position embedding with learnable embeddings to match official"""
        if hasattr(self, 'pos_emb_cls') and self.pos_emb_cls == "rope3d":
            head_dim = self.model_channels // self.num_heads
            self.pos_embedder = CleanRoPE3D(
                head_dim=head_dim,
                max_t=self.max_frames // self.patch_temporal,
                max_h=self.max_img_h // self.patch_spatial,
                max_w=self.max_img_w // self.patch_spatial
            )
        else:
            self.pos_embedder = None
            
        # Add learnable positional embeddings to match official pos_embedder.seq
        max_seq_len = (self.max_frames // self.patch_temporal) * \
                     (self.max_img_h // self.patch_spatial) * \
                     (self.max_img_w // self.patch_spatial)
        self.pos_embedder_seq = nn.Parameter(torch.randn(max_seq_len, self.model_channels) * 0.02)
            
    def _build_blocks(self):
        """Build transformer blocks using official structure"""
        self.blocks = nn.ModuleDict()
        for i in range(self.num_blocks):
            self.blocks[f"block{i}"] = OfficialGeneralDITTransformerBlock(
                x_dim=self.model_channels,
                context_dim=self.crossattn_emb_channels,
                num_heads=self.num_heads,
                block_config="ca-fa-mlp",  # Official config
                mlp_ratio=4.0,
                use_adaln_lora=self.use_adaln_lora,
                adaln_lora_dim=self.adaln_lora_dim,
            )
            
    def _build_final_layer(self):
        """Build final output layer using official structure"""
        self.final_layer = OfficialFinalLayer(
            hidden_size=self.model_channels,
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            out_channels=self.out_channels,
            use_adaln_lora=self.use_adaln_lora,
            adaln_lora_dim=self.adaln_lora_dim
        )
        
    def initialize_weights(self):
        """Initialize model weights to match official implementation"""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
        self.apply(_basic_init)
        
        # Initialize timestep embedding
        nn.init.normal_(self.t_embedder[1].linear_1.weight, std=0.02)
        nn.init.normal_(self.t_embedder[1].linear_2.weight, std=0.02)
        
        # Zero-out adaLN modulation layers in official blocks
        for block_wrapper in self.blocks.values():
            for building_block in block_wrapper.blocks:
                # Zero initialize the final layer of adaLN_modulation
                if hasattr(building_block, 'adaLN_modulation'):
                    if len(building_block.adaLN_modulation) > 1:
                        nn.init.constant_(building_block.adaLN_modulation[-1].weight, 0)
                        if building_block.adaLN_modulation[-1].bias is not None:
                            nn.init.constant_(building_block.adaLN_modulation[-1].bias, 0)
            
    def prepare_inputs(self, x, timesteps, crossattn_emb, padding_mask=None, latent_condition=None):
        """Prepare inputs for the forward pass with official tensor format conversion"""
        B, C, T, H, W = x.shape
        
        # Handle latent conditions by concatenating
        if latent_condition is not None:
            x = torch.cat([x, latent_condition], dim=1)
            
        # Add padding mask if needed
        if self.concat_padding_mask and padding_mask is not None:
            x = torch.cat([x, padding_mask], dim=1)
            
        # Patch embedding
        x = self.x_embedder(x)  # (B, T', H', W', D)
        B, T_p, H_p, W_p, D = x.shape
        
        # Convert to official format: (B, T', H', W', D) -> (T', H', W', B, D)
        x = rearrange(x, "b t h w d -> t h w b d")
        
        # Time embedding  
        timestep_emb = self.t_embedder(timesteps)  # (B, D*3) -> extract first D for embeddings
        if timestep_emb.shape[1] > self.model_channels:
            timestep_emb = timestep_emb[:, :self.model_channels]  # Use first part for basic embedding
        timestep_emb = self.affline_norm(timestep_emb)
        
        # Convert crossattn_emb to official format if provided
        if crossattn_emb is not None:
            crossattn_emb = rearrange(crossattn_emb, "b s d -> s b d")  # (B, S, D) -> (S, B, D)
        
        # Position embedding for RoPE
        rope_emb = None
        if self.pos_embedder is not None:
            rope_emb_3d = self.pos_embedder.get_rope_embeddings(T_p, H_p, W_p, x.device)
            rope_emb = rearrange(rope_emb_3d, "t h w d -> (t h w) 1 1 d")  # (L, 1, 1, D)
            
        return x, timestep_emb, crossattn_emb, rope_emb, (B, C, T, H, W), (T_p, H_p, W_p)
        
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
        """
        Forward pass of the DiT model.
        
        Args:
            x: Input tensor (B, C, T, H, W)
            timesteps: Timestep tensor (B,) 
            crossattn_emb: Cross attention embeddings (B, S, D)
            crossattn_mask: Cross attention mask (B, S)
            padding_mask: Padding mask (B, 1, T, H, W)
            latent_condition: Additional conditioning (B, C_cond, T, H, W)
            
        Returns:
            Output tensor (B, C_out, T, H, W)
        """
        print(f"[DIT] Input x shape: {x.shape} (expected 5D: B,C,T,H,W)")
        print(f"[DIT] timesteps shape: {timesteps.shape}")
        if crossattn_emb is not None:
            print(f"[DIT] crossattn_emb shape: {crossattn_emb.shape}")
        
        # Prepare inputs
        x, timestep_emb, crossattn_emb, rope_emb, orig_shape, patch_shape = self.prepare_inputs(
            x, timesteps, crossattn_emb, padding_mask, latent_condition
        )
        print(f"[DIT] After prepare_inputs, x shape: {x.shape}")
        print(f"[DIT] orig_shape: {orig_shape}, patch_shape: {patch_shape}")
        
        # Process cross attention mask
        if self.use_cross_attn_mask and crossattn_mask is not None:
            crossattn_mask = crossattn_mask[:, None, None, :]  # (B, 1, 1, S)
        else:
            crossattn_mask = None
            
        # Apply transformer blocks with official format
        for i, block in enumerate(self.blocks.values()):
            x = block(x, timestep_emb, crossattn_emb, crossattn_mask, rope_emb)
            if i == 0:  # Log first block output
                print(f"[DIT] After block {i}, x shape: {x.shape} (official format: T,H,W,B,D)")
        
        # Convert back to batch-first for final layer: (T,H,W,B,D) -> (B*T*H*W, D)
        T_p, H_p, W_p, B, D = x.shape
        x = rearrange(x, "t h w b d -> (b t h w) d")
        
        # Final layer
        x = self.final_layer(x, timestep_emb)
        print(f"[DIT] After final_layer, x shape: {x.shape}")
        
        # Reshape back to video format
        B, C_orig, T_orig, H_orig, W_orig = orig_shape
        
        x = rearrange(
            x, 
            "(b t h w) (p1 p2 pt c) -> b c (t pt) (h p1) (w p2)",
            b=B, t=T_p, h=H_p, w=W_p,
            p1=self.patch_spatial, p2=self.patch_spatial, pt=self.patch_temporal, c=self.out_channels
        )
        print(f"[DIT] Final output shape: {x.shape} (expected 5D: B,C,T,H,W)")
        
        return x


class CleanDiffusionRendererGeneralDIT(CleanGeneralDIT):
    """
    Diffusion Renderer specific version of GeneralDIT with additional conditioning support.
    """
    
    def __init__(
        self,
        additional_concat_ch: int = 0,
        use_context_embedding: bool = True,
        **kwargs
    ):
        self.additional_concat_ch = additional_concat_ch
        self.use_context_embedding = use_context_embedding
        
        # Adjust input channels for additional conditioning
        if 'in_channels' in kwargs:
            kwargs['in_channels'] += additional_concat_ch
            
        super().__init__(**kwargs)
        
        # Context embedding for inverse renderer - CRITICAL: Use 1024 not 4096!
        if self.use_context_embedding:
            self.context_embedding = nn.Embedding(
                num_embeddings=16,  # For different G-buffer types
                embedding_dim=1024  # FIXED: Official checkpoint expects 1024, not 4096!
            )
            # Initialize context embeddings
            with torch.no_grad():
                nn.init.uniform_(self.context_embedding.weight, -0.3, 0.3)
                
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor = None,
        context_index: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with context embedding support.
        
        Args:
            x: Input tensor (B, C, T, H, W)
            timesteps: Timestep tensor (B,)
            crossattn_emb: Cross attention embeddings (B, S, D) - optional for forward renderer
            context_index: Context indices for G-buffer type (B, 1) - for inverse renderer
            **kwargs: Additional arguments
            
        Returns:
            Output tensor (B, C_out, T, H, W)
        """
        # Handle context embedding for inverse renderer
        if self.use_context_embedding and context_index is not None:
            # Get context embeddings for the specified G-buffer type
            context_emb = self.context_embedding(context_index)  # (B, 1, D)
            
            # Overwrite crossattn_emb to match official implementation
            crossattn_emb = context_emb
                
        return super().forward(x, timesteps, crossattn_emb, **kwargs)
