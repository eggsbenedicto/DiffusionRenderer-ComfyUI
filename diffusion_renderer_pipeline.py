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

import numpy as np
from typing import Any, Optional, Tuple, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# Optional imports for checkpoint loading
try:
    from safetensors.torch import load_file as load_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

# Import our clean model implementation
from .model_diffusion_renderer import CleanDiffusionRendererModel
from .diffusion_renderer_config import get_config_by_model_type, get_config_from_tensor_shape, validate_config


class CleanDiffusionRendererPipeline:
    def __init__(self, checkpoint_dir: str, checkpoint_name: str, 
                 model_type: str = "inverse",  # "inverse" or "forward"
                 vae_instance = None,  # Pre-loaded CleanVAE instance
                 model_instance = None,  # Pre-loaded CleanDiffusionRendererModel instance
                 guidance: float = 2.0, num_steps: int = 20, 
                 height: int = 1024, width: int = 1024, 
                 num_video_frames: int = 1, seed: int = 42):
        """
        Clean Diffusion Renderer Pipeline with dynamic configuration
        
        Args:
            checkpoint_dir: Directory containing model checkpoints
            checkpoint_name: Name of checkpoint file (.pt or .safetensors)
            model_type: "inverse" (RGB->maps) or "forward" (maps->RGB)
            vae_instance: Pre-loaded CleanVAE instance (or None for mock VAE)
            model_instance: Pre-loaded CleanDiffusionRendererModel instance (or None to load dynamically)
            guidance: Guidance scale for generation
            num_steps: Number of denoising steps
            height: Default image height (will be overridden by input)
            width: Default image width (will be overridden by input)
            num_video_frames: Default number of frames (will be overridden by input)
            seed: Random seed
        """
        
        # Store initialization parameters to match original interface
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.model_type = model_type.lower() if model_type else None  # Handle None model_type
        self.vae_instance = vae_instance  # Store pre-loaded VAE instance
        self.pre_loaded_model_instance = model_instance  # Store pre-loaded model instance
        
        # Runtime parameters that can be modified by ComfyUI nodes
        self.guidance = guidance
        self.num_steps = num_steps
        
        # Default dimensions (these will be overridden by actual input dimensions)
        self.default_height = height
        self.default_width = width
        self.default_num_video_frames = num_video_frames
        self.seed = seed
        
        # Device/precision settings
        self.device = torch.device('cuda')
        self.dtype = torch.bfloat16
        
        # Configuration will be generated dynamically based on input
        self.config = None
        self.model = None
        
        # Model caching for performance
        self._config_cache = {}  # Cache configs by their hash/key
        self._model_cache = {}   # Cache models by config hash
        
        model_type_str = model_type if model_type else "dynamic (set by inference nodes)"
        print(f"Initialized {model_type_str} renderer pipeline")
        print(f"  Checkpoint: {checkpoint_name}")
        print(f"  Default dimensions: {width}x{height}, frames={num_video_frames}")
        print(f"  Pre-loaded model: {'✅ Yes' if model_instance else '❌ No (will load dynamically)'}")
        print(f"  Note: Actual dimensions will be inferred from input tensors")
    
    def set_model_type(self, model_type: str):
        """Set model type and force model reload if type changes"""
        new_model_type = model_type.lower()
        if self.model_type != new_model_type:
            print(f"Switching pipeline from {self.model_type} to {new_model_type}")
            self.model_type = new_model_type
            # Force model reload by clearing config and model
            self.config = None
            self.model = None
        elif self.model_type is None:
            # First time setting model type
            print(f"Setting pipeline model type to {new_model_type}")
            self.model_type = new_model_type
    
    def _ensure_model_loaded(self, input_tensor_shape: tuple):
        """
        Ensure model is loaded with correct configuration for the given input shape.
        Uses smart caching to avoid unnecessary reloads - only reloads when config actually changes.
        """
        # Generate config from input tensor shape
        new_config = get_config_from_tensor_shape(self.model_type, input_tensor_shape)
        
        # Add model_type to config for logging
        new_config['model_type'] = self.model_type
        
        # Create a config hash for caching
        config_hash = self._get_config_hash(new_config)
        
        # Check if we can reuse the existing model (config unchanged)
        if self.config is not None and self._get_config_hash(self.config) == config_hash:
            # Configuration unchanged - reuse existing model
            print(f"✅ Reusing cached model for shape {input_tensor_shape} (config unchanged)")
            return self.model
        
        # Check if we have this config cached
        if config_hash in self._model_cache:
            print(f"✅ Loading model from cache for shape {input_tensor_shape}")
            self.config = new_config
            self.model = self._model_cache[config_hash]
            return self.model
        
        # Need to create/load model with new configuration
        print(f"🔄 Loading/creating model for shape {input_tensor_shape}")
        self.config = new_config
        validate_config(self.config)
        
        # Use pre-loaded model if available, otherwise load dynamically
        if self.pre_loaded_model_instance is not None:
            print("✅ Using pre-loaded model instance")
            self.model = self._configure_pre_loaded_model(self.pre_loaded_model_instance, self.config)
        else:
            print("❌ No pre-loaded model - loading dynamically (slower)")
            self.model = self._load_model_with_config()
        
        # Cache the configured model
        self._model_cache[config_hash] = self.model
        print(f"📦 Cached model for config hash: {config_hash[:8]}...")
        
        return self.model
    
    def _get_config_hash(self, config):
        """Generate a hash for the config to enable smart caching"""
        import hashlib
        import json
        
        # Create a deterministic string representation of the config
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _configure_pre_loaded_model(self, model_instance, config):
        """
        Configure a pre-loaded model instance with new config without reloading weights.
        This is much faster than loading from disk.
        """
        print(f"🔧 Reconfiguring pre-loaded model:")
        print(f"  - New latent shape: {config['latent_shape']}")
        print(f"  - New condition keys: {config['condition_keys']}")
        print(f"  - Model type: {config.get('model_type', 'unknown')}")
        
        # Update the model's configuration
        model_instance.config = config
        
        # Update condition-related attributes
        model_instance.condition_keys = config.get('condition_keys', ["image", "depth", "normal", "basecolor", "roughness", "metallic"])
        model_instance.condition_drop_rate = config.get('condition_drop_rate', 0.0)
        model_instance.append_condition_mask = config.get('append_condition_mask', True)
        model_instance.input_data_key = config.get('input_data_key', "video")
        
        # Use the pre-loaded VAE instance if we have one
        if self.vae_instance:
            print("✅ Using pre-loaded VAE instance")
            model_instance.vae = self.vae_instance
        else:
            print("ℹ️  Using model's existing VAE")
        
        # Ensure model is on correct device
        model_instance = model_instance.to(self.device)
        
        print("✅ Pre-loaded model reconfigured successfully")
        return model_instance
    
    def _move_to_device(self, data_batch):
        """Replace misc.to() with simple device/dtype conversion"""
        result = {}
        for key, value in data_batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(device=self.device, dtype=self.dtype)
            else:
                result[key] = value
        return result
    
    def _load_model_with_config(self):
        """
        Load model using current configuration (fallback when no pre-loaded model available).
        This is slower than using a pre-loaded model but still supports dynamic configs.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)
        print(f"⚠️  Loading model dynamically (slower than pre-loaded):")
        print(f"  Latent shape: {self.config['latent_shape']}")
        print(f"  Condition keys: {self.config['condition_keys']}")
        print(f"  Model channels: {self.config['net']['model_channels']}")
        
        # Create the model instance first
        model = CleanDiffusionRendererModel(self.config)
        
        # Load checkpoint if it exists
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        model.load_checkpoint(checkpoint_path, strict=True)
        
        # Move to device
        model = model.to(self.device)
        
        # Use pre-loaded VAE instance if provided
        if self.vae_instance:
            print("✅ Using pre-loaded VAE instance")
            # The model should accept our CleanVAE instance
            model.vae = self.vae_instance
        else:
            print("Using mock VAE (no VAE instance provided)")
        
        return model

    def generate_video(self, data_batch: Dict[str, torch.Tensor], 
                      normalize_normal: bool = False, seed: int = None) -> np.ndarray:
        """Generate video with dynamic configuration based on input tensor shapes"""
        
        # Use the seed passed in, or fall back to instance seed
        effective_seed = seed if seed is not None else self.seed
        
        # 1. Data prep (replace misc.to)
        data_batch = self._move_to_device(data_batch)
        
        # 2. Find video tensor and infer dimensions
        possible_shape_keys = ['rgb', 'image', 'basecolor', 'normal', 'depth', 'roughness', 'metallic']
        
        for key in possible_shape_keys:
            if key in data_batch:
                video_tensor = data_batch[key]
                break
        
        # We no longer need to check for the key 'video' specifically, as the noise
        # will be generated from scratch by the model.
        if video_tensor is None:
            raise ValueError(f"No suitable input tensor for shape inference found in data_batch. Looked for {possible_shape_keys}")
        
        print(f"[Pipeline] Found input tensor for shape inference: {video_tensor.shape} (key='{key}')")
        
        # 3. Ensure model is loaded with correct config for this input shape
        model = self._ensure_model_loaded(video_tensor.shape)
        
        # 4. Calculate state shape using dynamic config
        config_latent_shape = self.config['latent_shape']  # [C, T, H, W]
        C, expected_T, expected_H, expected_W = config_latent_shape
        
        # Verify tensor matches expected dimensions (or auto-adjust)
        B, input_C, input_T, input_H, input_W = video_tensor.shape
        F = (input_T - 1) // 8 + 1  # Temporal compression
        H = input_H // 8  # Spatial compression  
        W = input_W // 8  # Spatial compression
        state_shape = [C, F, H, W]
        
        print(f"[Pipeline] Input shape: {video_tensor.shape}")
        print(f"[Pipeline] State shape: {state_shape}")
        print(f"[Pipeline] Expected latent shape from config: {config_latent_shape}")
        
        # 4. Core diffusion sampling - use runtime guidance and num_steps
        sample = self.model.generate_samples_from_batch(
            data_batch,
            guidance=self.guidance,  # This gets set by ComfyUI nodes
            state_shape=state_shape,
            num_steps=self.num_steps,  # This gets set by ComfyUI nodes
            is_negative_prompt=False,
            seed=effective_seed,
        )
        print(f"[Pipeline] Diffusion sample output shape: {sample.shape}")
        
        # 5. VAE decode
        video = self.model.decode(sample)
        print(f"[Pipeline] After VAE decode shape: {video.shape}")
        
        # 6. Post-processing (exact logic from original)
        if normalize_normal:
            norm = torch.norm(video, dim=1, p=2, keepdim=True)
            video_normalized = video / norm.clamp(min=1e-12)
            norm_threshold_upper = 0.4
            norm_threshold_lower = 0.2
            blend_ratio = torch.clip(
                (norm - norm_threshold_lower) / (norm_threshold_upper - norm_threshold_lower),
                0, 1
            )
            video = video_normalized * blend_ratio + video * (1 - blend_ratio)
            print(f"[Pipeline] After normal normalization shape: {video.shape}")

        # 7. Convert to numpy (exact logic from original)
        video = (1.0 + video).clamp(0, 2) / 2
        print(f"[Pipeline] Before permute/convert, video shape: {video.shape}")
        video = (video[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy()
        print(f"[Pipeline] Final numpy output shape: {video.shape}")
        
        return video