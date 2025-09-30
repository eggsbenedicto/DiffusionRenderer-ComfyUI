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
Clean configuration system for Diffusion Renderer models.
This replaces the complex hydra/attrs-based config system with simple Python dictionaries.
"""

from typing import Dict, List, Any, Optional


class CleanDiffusionRendererConfig:
    """Base configuration for diffusion renderer models."""
    
    def __init__(self):
        # Base model parameters
        self.sigma_data = 0.5
        self.precision = "bfloat16"
        self.input_data_key = "video"
        
        # Default latent shape (will be overridden by specific configs)
        self.latent_shape = [16, 8, 88, 160]  # [C, T, H, W]
        
        # Default condition parameters
        self.condition_keys = ["rgb"]
        self.condition_drop_rate = 0.0
        self.append_condition_mask = False
        
        # Model architecture defaults
        self.model_channels = 4096
        self.num_blocks = 28
        self.num_heads = 32


def get_network_config() -> Dict[str, Any]:
    """Get FADITV2_7B network architecture configuration."""
    return {
        # Core architecture
        "model_channels": 4096,
        "num_blocks": 28,
        "num_heads": 32,
        'head_dim': 128,
        'mlp_ratio': 4.0,
        'context_dim': 1024,
        'adaln_lora_dim': 256,

        'time_embed_dim': 4096,  
        'max_time_embed_period': 10000,

        # Input/Output channels
        "in_channels": 16,
        "out_channels": 16,
        
        
        # Patch embedding
        "patch_spatial": 2,
        "patch_temporal": 1,
        
        # Dimensions
        "max_img_h": 240,
        "max_img_w": 240,
        "max_frames": 128,
        
        # Architecture details
        "block_config": "FA-CA-MLP",
        "concat_padding_mask": True,
        "block_x_format": "THWBD",
        
        # Positional embedding
        "pos_emb_cls": "rope3d",
        "pos_emb_learnable": False,
        "pos_emb_interpolation": "crop",
        
        # RoPE extrapolation ratios
        "rope_h_extrapolation_ratio": 1.0,
        "rope_w_extrapolation_ratio": 1.0,
        "rope_t_extrapolation_ratio": 2.0,
        
        # Additional features
        "affline_emb_norm": True,
        "use_adaln_lora": True,
        "adaln_lora_dim": 256,
        "extra_per_block_abs_pos_emb": True,
        "extra_per_block_abs_pos_emb_type": "sincos",
        "extra_h_extrapolation_ratio": 1.0,
        "extra_w_extrapolation_ratio": 1.0,
        "extra_t_extrapolation_ratio": 1.0,
        
        # Cross attention - CRITICAL: Must match checkpoint context embedding dimension
        "crossattn_emb_channels": 1024,  # Official checkpoint expects 1024, not 4096!
    }


def get_scheduler_config() -> Dict[str, Any]:
    """Get EDM Euler scheduler configuration."""
    return {
        "type": "EDMEulerScheduler",
        "sigma_max": 80.0,
        "sigma_min": 0.02,
        "sigma_data": 0.5,
        "num_train_timesteps": 1000,
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
    }


def get_vae_config(num_frames: int = 57) -> Dict[str, Any]:
    """Get VAE/Tokenizer configuration."""
    return {
        "pixel_chunk_duration": num_frames,
        "latent_channels": 16,
        "spatial_compression_ratio": 8,
        "temporal_compression_ratio": 8,
    }


def get_inverse_renderer_config(
    height: int = 704, 
    width: int = 1280, 
    num_frames: int = 57
) -> Dict[str, Any]:
    """
    Configuration for Inverse Renderer (RGB -> other maps).
    
    Args:
        height: Input image height
        width: Input image width  
        num_frames: Number of video frames
    """
    # Calculate latent dimensions
    latent_h = height // 8  # 88 for 704
    latent_w = width // 8   # 160 for 1280
    latent_t = num_frames // 8 + 1  # 8 for 57 frames
    
    base_config = CleanDiffusionRendererConfig()
    network_config = get_network_config()
    
    config = {
        # Model parameters
        "sigma_data": base_config.sigma_data,
        "precision": base_config.precision,
        "input_data_key": base_config.input_data_key,
        
        # Latent shape [C, T, H, W]
        "latent_shape": [16, latent_t, latent_h, latent_w],
        
        # Condition handling - Inverse takes RGB input
        "condition_keys": ["rgb"],
        "condition_drop_rate": 0.1,
        "append_condition_mask": False,
        
        # Network architecture
        "net": {
            **network_config,
            "additional_concat_ch": 16,  # RGB condition channels
            "use_context_embedding": True,
            "crossattn_emb_channels": 1024,  # FIXED: Must match checkpoint expectation!
        },
        
        # Scheduler
        "scheduler": get_scheduler_config(),
        
        # VAE/Tokenizer
        "vae": get_vae_config(num_frames),
        
        # Inference parameters
        "guidance": 2.0,
        "num_steps": 20,
        "height": height,
        "width": width,
        "num_video_frames": num_frames,
    }
    
    return config


def get_forward_renderer_config(
    height: int = 704,
    width: int = 1280, 
    num_frames: int = 57
) -> Dict[str, Any]:
    """
    Configuration for Forward Renderer (maps -> RGB).
    
    Args:
        height: Output image height
        width: Output image width
        num_frames: Number of video frames
    """
    # Calculate latent dimensions
    latent_h = height // 8  # 88 for 704
    latent_w = width // 8   # 160 for 1280
    latent_t = num_frames // 8 + 1  # 8 for 57 frames
    
    base_config = CleanDiffusionRendererConfig()
    network_config = get_network_config()
    
    config = {
        # Model parameters
        "sigma_data": base_config.sigma_data,
        "precision": base_config.precision,
        "input_data_key": base_config.input_data_key,
        
        # Latent shape [C, T, H, W]  
        "latent_shape": [16, latent_t, latent_h, latent_w],
        
        # Condition handling - Forward takes all G-buffer maps + environment
        "condition_keys": [
            "basecolor", "normal", "metallic", "roughness", "depth",
            "env_ldr", "env_log", "env_nrm"
        ],
        "condition_drop_rate": 0.05,
        "append_condition_mask": False,
        
        # Network architecture
        "net": {
            **network_config,
            "additional_concat_ch": 16,  
            "use_context_embedding": True,
            "crossattn_emb_channels": 1024,  # FIXED: Must match checkpoint expectation!
        },
        
        # Scheduler
        "scheduler": get_scheduler_config(),
        
        # VAE/Tokenizer
        "vae": get_vae_config(num_frames),
        
        # Inference parameters
        "guidance": 2.0,
        "num_steps": 20,
        "height": height,
        "width": width,
        "num_video_frames": num_frames,
    }
    
    return config


def get_config_by_model_type(
    model_type: str,
    height: int = 704,
    width: int = 1280,
    num_frames: int = 57
) -> Dict[str, Any]:
    """
    Get configuration by model type.
    
    Args:
        model_type: "inverse" or "forward"
        height: Image height (can be inferred from input)
        width: Image width (can be inferred from input)
        num_frames: Number of video frames (can be inferred from input)
    """
    if model_type.lower() == "inverse":
        return get_inverse_renderer_config(height, width, num_frames)
    elif model_type.lower() == "forward":
        return get_forward_renderer_config(height, width, num_frames)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'inverse' or 'forward'")


def get_config_from_tensor_shape(model_type, tensor_shape):
    print(f"[Config] Generating config from tensor_shape: {tensor_shape}")
    
    # --- THIS IS THE FIX ---
    if len(tensor_shape) == 5:
        B, C, T, H, W = tensor_shape
        print(f"[Config] 5D tensor detected: B={B}, C={C}, T={T}, H={H}, W={W}")
    else:
        raise ValueError(f"Expected a 5D tensor shape, but got {len(tensor_shape)} dimensions.")

    # Now, use the correctly parsed dimensions
    if model_type == 'inverse':
        config = get_inverse_renderer_config(height=H, width=W, num_frames=T)
    elif model_type == 'forward':
        config = get_forward_renderer_config(height=H, width=W, num_frames=T)
    else:
        raise ValueError(f"Unknown model type for config generation: {model_type}")

    # The rest of the logic inside get_..._config should correctly use these values
    # to calculate the latent dimensions.
    
    # Example latent calculation from get_inverse_renderer_config:
    # latent_t = 1 if num_frames == 1 else (num_frames - 1) // 8 + 1
    # latent_h = H // 8
    # latent_w = W // 8
    # print(f"[Config] Calculated latent dimensions: T={latent_t}, H={latent_h}, W={latent_w}")
    # config['latent_shape'] = [16, latent_t, latent_h, latent_w]
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary for completeness and correctness.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    print(f"[Config] Validating configuration...")
    
    required_keys = [
        "sigma_data", "precision", "input_data_key", "latent_shape",
        "condition_keys", "net", "scheduler", "vae"
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate latent shape format [C, T, H, W]
    latent_shape = config["latent_shape"]
    if not isinstance(latent_shape, list) or len(latent_shape) != 4:
        raise ValueError(f"Invalid latent_shape: {latent_shape}. Expected [C, T, H, W] format.")
    
    C, T, H, W = latent_shape
    print(f"[Config] Latent shape validation: C={C}, T={T}, H={H}, W={W}")
    
    # Validate network config
    net_config = config["net"]
    required_net_keys = ["model_channels", "num_blocks", "num_heads", "in_channels", "out_channels"]
    for key in required_net_keys:
        if key not in net_config:
            raise ValueError(f"Missing required net config key: {key}")
    
    print(f"[Config] ✅ Configuration validated successfully")
    print(f"[Config]   Model type: {config.get('model_type', 'unknown')}")
    print(f"[Config]   Latent shape: {latent_shape}")
    print(f"[Config]   Condition keys: {config['condition_keys']}")
    print(f"[Config]   Network: {net_config['model_channels']}ch, {net_config['num_blocks']}blocks, {net_config['num_heads']}heads")


# Configuration presets for common use cases
PRESET_CONFIGS = {
    "inverse_1024x1024": get_inverse_renderer_config(1024, 1024, 1),
    "forward_1024x1024": get_forward_renderer_config(1024, 1024, 1),
    "inverse_704x1280_video": get_inverse_renderer_config(704, 1280, 57),
    "forward_704x1280_video": get_forward_renderer_config(704, 1280, 57),
}


def get_preset_config(preset_name: str) -> Dict[str, Any]:
    """Get a preset configuration by name."""
    if preset_name not in PRESET_CONFIGS:
        available = list(PRESET_CONFIGS.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
    
    return PRESET_CONFIGS[preset_name].copy()
