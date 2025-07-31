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
    
    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu", use_adaln_lora: bool = False):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU() if act_fn == "silu" else nn.GELU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)
        
    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class CleanPatchEmbed(nn.Module):
    """Clean patch embedding for spatial and temporal patches"""
    
    def __init__(self, spatial_patch_size: int, temporal_patch_size: int, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        
        self.proj = nn.Conv3d(
            in_channels, 
            out_channels,
            kernel_size=(temporal_patch_size, spatial_patch_size, spatial_patch_size),
            stride=(temporal_patch_size, spatial_patch_size, spatial_patch_size),
            bias=bias
        )
        
    def forward(self, x):
        """
        x: (B, C, T, H, W)
        output: (B, T', H', W', D) where T'=T//temporal_patch_size, etc.
        """
        x = self.proj(x)  # (B, D, T', H', W')
        x = x.permute(0, 2, 3, 4, 1)  # (B, T', H', W', D)
        return x


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


class CleanMultiHeadAttention(nn.Module):
    """Simplified multi-head attention without flash attention"""
    
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_out = nn.Linear(dim, dim)
        
    def forward(self, x, rope_emb=None):
        B, L, D = x.shape
        
        q = self.to_q(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)  
        v = self.to_v(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE if provided
        if rope_emb is not None:
            q, k = apply_rotary_pos_emb(q, k, rope_emb)
            
        # Standard attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.to_out(out)
        
        return out


class CleanCrossAttention(nn.Module):
    """Cross attention for conditioning"""
    
    def __init__(self, query_dim: int, context_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)
        
    def forward(self, x, context, mask=None):
        B, L, D = x.shape
        _, S, _ = context.shape
        
        q = self.to_q(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))
            
        attn = attn.softmax(dim=-1)
        
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.to_out(out)
        
        return out


class CleanMLP(nn.Module):
    """Feed-forward MLP"""
    
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class CleanAdaLNModulation(nn.Module):
    """Adaptive Layer Norm modulation"""
    
    def __init__(self, dim: int, num_modulations: int = 6):
        super().__init__()
        self.linear = nn.Linear(dim, num_modulations * dim, bias=True)
        self.num_modulations = num_modulations
        
    def forward(self, x):
        return self.linear(x).chunk(self.num_modulations, dim=-1)


class CleanDITBlock(nn.Module):
    """Single DiT transformer block with self-attention, cross-attention, and MLP"""
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        context_dim: int,
        mlp_ratio: float = 4.0,
        block_config: str = "FA-CA-MLP"
    ):
        super().__init__()
        self.dim = dim
        self.block_config = block_config
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        # Attention layers
        self.self_attn = CleanMultiHeadAttention(dim, num_heads)
        self.cross_attn = CleanCrossAttention(dim, context_dim, num_heads)
        self.mlp = CleanMLP(dim, mlp_ratio)
        
        # AdaLN modulation
        self.adaLN_modulation = CleanAdaLNModulation(dim, num_modulations=6)
        
    def forward(self, x, timestep_emb, context_emb, rope_emb=None, context_mask=None):
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(timestep_emb)
        
        # Self-attention block
        norm_x = self.norm1(x)
        norm_x = norm_x * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out = self.self_attn(norm_x, rope_emb)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # Cross-attention block
        norm_x = self.norm2(x)
        cross_out = self.cross_attn(norm_x, context_emb, context_mask)
        x = x + cross_out
        
        # MLP block
        norm_x = self.norm3(x)
        norm_x = norm_x * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(norm_x)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x


class CleanFinalLayer(nn.Module):
    """Final layer to decode from hidden dim to output channels"""
    
    def __init__(self, hidden_size: int, spatial_patch_size: int, temporal_patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, 
            spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, 
            bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.out_channels = out_channels
        
    def forward(self, x, timestep_emb):
        shift, scale = self.adaLN_modulation(timestep_emb).chunk(2, dim=-1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
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
        crossattn_emb_channels: int = 4096,
        use_cross_attn_mask: bool = False,
        
        # Position embedding
        pos_emb_cls: str = "rope3d",
        
        # AdaLN LoRA
        use_adaln_lora: bool = True,
        adaln_lora_dim: int = 256,
        
        # Additional features
        affline_emb_norm: bool = True,
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
            CleanTimestepEmbedding(self.model_channels, self.model_channels)
        )
        
    def _build_pos_embed(self):
        """Build position embedding (simplified)"""
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
            
    def _build_blocks(self):
        """Build transformer blocks"""
        self.blocks = nn.ModuleDict()
        for i in range(self.num_blocks):
            self.blocks[f"block{i}"] = CleanDITBlock(
                dim=self.model_channels,
                num_heads=self.num_heads,
                context_dim=self.crossattn_emb_channels,
                mlp_ratio=4.0,
                block_config="FA-CA-MLP"
            )
            
    def _build_final_layer(self):
        """Build final output layer"""
        self.final_layer = CleanFinalLayer(
            hidden_size=self.model_channels,
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            out_channels=self.out_channels
        )
        
    def initialize_weights(self):
        """Initialize model weights"""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
        self.apply(_basic_init)
        
        # Initialize timestep embedding
        nn.init.normal_(self.t_embedder[1].linear_1.weight, std=0.02)
        nn.init.normal_(self.t_embedder[1].linear_2.weight, std=0.02)
        
        # Zero-out adaLN modulation layers
        for block in self.blocks.values():
            nn.init.constant_(block.adaLN_modulation.linear.weight, 0)
            nn.init.constant_(block.adaLN_modulation.linear.bias, 0)
            
    def prepare_inputs(self, x, timesteps, crossattn_emb, padding_mask=None, latent_condition=None):
        """Prepare inputs for the forward pass"""
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
        
        # Flatten spatial dimensions
        x = rearrange(x, "B T H W D -> B (T H W) D")
        
        # Time embedding
        timestep_emb = self.t_embedder(timesteps)  # (B, D)
        timestep_emb = self.affline_norm(timestep_emb)
        
        # Position embedding
        rope_emb = None
        if self.pos_embedder is not None:
            rope_emb = self.pos_embedder.get_rope_embeddings(T_p, H_p, W_p, x.device)
            rope_emb = rearrange(rope_emb, "T H W D -> (T H W) D")
            
        return x, timestep_emb, crossattn_emb, rope_emb, (B, C, T, H, W), (T_p, H_p, W_p)
        
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        latent_condition: Optional[torch.Tensor] = None,
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
        # Prepare inputs
        x, timestep_emb, crossattn_emb, rope_emb, orig_shape, patch_shape = self.prepare_inputs(
            x, timesteps, crossattn_emb, padding_mask, latent_condition
        )
        
        # Process cross attention mask
        if self.use_cross_attn_mask and crossattn_mask is not None:
            crossattn_mask = crossattn_mask[:, None, None, :]  # (B, 1, 1, S)
        else:
            crossattn_mask = None
            
        # Apply transformer blocks
        for block in self.blocks.values():
            x = block(x, timestep_emb, crossattn_emb, rope_emb, crossattn_mask)
            
        # Final layer
        x = self.final_layer(x, timestep_emb)
        
        # Reshape back to video format
        B, C_orig, T_orig, H_orig, W_orig = orig_shape
        T_p, H_p, W_p = patch_shape
        
        x = rearrange(
            x, 
            "B (T H W) (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            T=T_p, H=H_p, W=W_p,
            p1=self.patch_spatial, p2=self.patch_spatial, t=self.patch_temporal
        )
        
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
        
        # Context embedding for inverse renderer
        if self.use_context_embedding:
            self.context_embedding = nn.Embedding(
                num_embeddings=16,  # For different G-buffer types
                embedding_dim=kwargs.get('crossattn_emb_channels', 4096)
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
