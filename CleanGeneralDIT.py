# --- START OF DEFINITIVE, KEY-MATCHED CleanGeneralDIT.py ---

import math
import torch
from torch import nn
from typing import Dict, Tuple, Optional, Union
from einops.layers.torch import Rearrange

# Stand-in for transformer_engine's RMSNorm. Has a .weight, which is all we need.
class RMSNormPlaceholder(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x): return x 

class CleanRoPE3D(nn.Module):
    def __init__(self, head_dim: int, **kwargs):
        super().__init__()
        self.logvar = nn.Parameter(torch.zeros(1))
        self.seq = nn.Parameter(torch.zeros(head_dim))
    def forward(self, x): return None

class CleanTimesteps(nn.Module):
    def __init__(self, num_channels: int): super().__init__()
    def forward(self, x): return torch.zeros(x.shape[0], num_channels, device=x.device)

class CleanTimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_adaln_lora: bool = False):
        super().__init__()
        # Official implementation has bias=True when not using lora
        self.linear_1 = nn.Linear(in_channels, out_channels, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        if use_adaln_lora: self.linear_2 = nn.Linear(out_channels, 3*out_channels, bias=False)
        else: self.linear_2 = nn.Linear(out_channels, out_channels, bias=True)
    def forward(self, x): return None, None

class CleanPatchEmbed(nn.Module):
    def __init__(self, spatial_patch_size: int, temporal_patch_size: int, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Identity(), # Placeholder for Rearrange layer
            nn.Linear(in_channels * spatial_patch_size**2 * temporal_patch_size, out_channels, bias=bias)
        )
    def forward(self, x): return self.proj(x)

# This class now perfectly mimics the structure of the official `Attention` class
class OfficialAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, qkv_bias=False, out_bias=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        # Structure from attention.py: nn.Sequential with Linear and Normalization
        # qkv_norm is "RRI", so Q and K get RMSNorm (which has a .weight), V gets Identity.
        self.to_q = nn.Sequential(
            nn.Linear(query_dim, inner_dim, bias=qkv_bias),
            RMSNormPlaceholder(dim_head)
        )
        self.to_k = nn.Sequential(
            nn.Linear(context_dim, inner_dim, bias=qkv_bias),
            RMSNormPlaceholder(dim_head)
        )
        self.to_v = nn.Sequential(
            nn.Linear(context_dim, inner_dim, bias=qkv_bias),
            nn.Identity()
        )
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, bias=out_bias),
            nn.Dropout(0.0)
        )
    def forward(self, x, **kwargs): return x

class VideoAttn(nn.Module):
    def __init__(self, x_dim: int, context_dim: Optional[int], num_heads: int, bias: bool = True, **kwargs):
        super().__init__()
        self.attn = OfficialAttention(
            query_dim=x_dim, context_dim=context_dim, heads=num_heads,
            dim_head=x_dim // num_heads, qkv_bias=bias, out_bias=bias
        )
    def forward(self, x, **kwargs): return x

class OfficialGPT2FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.layer1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.layer2 = nn.Linear(hidden_dim, dim, bias=bias)
    def forward(self, x): return x

class OfficialDITBuildingBlock(nn.Module):
    def __init__(self, block_type: str, x_dim: int, context_dim: Optional[int], num_heads: int,
                 mlp_ratio: float = 4.0, bias: bool = True, mlp_dropout: float = 0.0, **kwargs):
        super().__init__()
        block_type = block_type.lower()
        if block_type in ("fa", "full_attn"):
            self.block = VideoAttn(x_dim, context_dim=None, num_heads=num_heads, bias=bias, **kwargs)
        elif block_type in ("ca", "cross_attn"):
            self.block = VideoAttn(x_dim, context_dim=context_dim, num_heads=num_heads, bias=bias, **kwargs)
        elif block_type in ("mlp", "ff"):
            self.block = OfficialGPT2FeedForward(x_dim, int(x_dim * mlp_ratio), dropout=mlp_dropout, bias=bias)
        self.norm_state = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
    def forward(self, x, **kwargs): return x

class OfficialGeneralDITTransformerBlock(nn.Module):
    def __init__(self, x_dim: int, context_dim: int, num_heads: int, block_config: str,
                 mlp_ratio: float = 4.0, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList()
        for block_type in block_config.split("-"):
            self.blocks.append(OfficialDITBuildingBlock(
                block_type.strip(), x_dim, context_dim, num_heads, mlp_ratio, **kwargs
            ))
    def forward(self, x, **kwargs): return x

class OfficialFinalLayer(nn.Module):
    def __init__(self, hidden_size, spatial_patch_size, temporal_patch_size, out_channels, **kwargs):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, spatial_patch_size**2 * temporal_patch_size * out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))
    def forward(self, x, **kwargs): return x

class CleanGeneralDIT(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # Deconstruct kwargs to pass to modules
        model_channels = kwargs['model_channels']
        num_blocks = kwargs['num_blocks']
        num_heads = kwargs['num_heads']
        in_channels = kwargs['in_channels']
        out_channels = kwargs['out_channels']
        patch_spatial = kwargs['patch_spatial']
        patch_temporal = kwargs['patch_temporal']
        crossattn_emb_channels = kwargs['crossattn_emb_channels']
        block_config = kwargs['block_config']
        mlp_ratio = kwargs['mlp_ratio']
        affline_emb_norm = kwargs.get('affline_emb_norm', True)
        concat_padding_mask = kwargs.get('concat_padding_mask', True)
        self.additional_concat_ch = kwargs.get('additional_concat_ch', 0)
        self._patch_embed_bias = getattr(self, '_patch_embed_bias', False)

        # Build modules
        in_ch = in_channels + self.additional_concat_ch + (1 if concat_padding_mask else 0)
        self.x_embedder = CleanPatchEmbed(
            patch_spatial, patch_temporal, in_ch, model_channels, bias=self._patch_embed_bias
        )
        self.t_embedder = nn.Sequential(
            CleanTimesteps(model_channels),
            CleanTimestepEmbedding(model_channels, model_channels)
        )
        self.pos_embedder = CleanRoPE3D(head_dim=model_channels // num_heads)
        
        self.blocks = nn.ModuleDict()
        
        # This part is already correct from the previous fix
        block_kwargs = kwargs.copy()
        for key in ['x_dim', 'context_dim', 'num_heads', 'block_config', 'mlp_ratio']:
            block_kwargs.pop(key, None)
            
        for i in range(num_blocks):
            self.blocks[f"block{i}"] = OfficialGeneralDITTransformerBlock(
                x_dim=model_channels, context_dim=crossattn_emb_channels, num_heads=num_heads,
                block_config=block_config, mlp_ratio=mlp_ratio, **block_kwargs
            )

        # Create a clean copy of kwargs for the final layer, removing explicit arguments.
        final_layer_kwargs = kwargs.copy()
        for key in ['hidden_size', 'spatial_patch_size', 'temporal_patch_size', 'out_channels']:
            final_layer_kwargs.pop(key, None)

        self.final_layer = OfficialFinalLayer(
            hidden_size=model_channels, 
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal, 
            out_channels=out_channels, 
            **final_layer_kwargs # Use the cleaned kwargs
        )

        self.affline_norm = nn.LayerNorm(model_channels, eps=1e-5, elementwise_affine=True) if affline_emb_norm else nn.Identity()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("This is a placeholder model for loading weights.")

class CleanDiffusionRendererGeneralDIT(CleanGeneralDIT):
    def __init__(self, additional_concat_ch: int = 16, use_context_embedding: bool = True, **kwargs):
        self.use_context_embedding = use_context_embedding
        self._patch_embed_bias = True
        super().__init__(additional_concat_ch=additional_concat_ch, **kwargs)
        if self.use_context_embedding:
            self.context_embedding = nn.Embedding(num_embeddings=16, embedding_dim=kwargs["crossattn_emb_channels"])