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
Minimal wrapper around diffusers AutoencoderKLCosmos for video support.
Handles 4D tensor interface (T, C, H, W) and leverages diffusers' native 5D support.
"""

import torch
import torch.nn.functional as F
from typing import Tuple
from diffusers import AutoencoderKLCosmos

class CleanVAE:
    """
    Minimal wrapper around diffusers AutoencoderKLCosmos for video support.
    Handles 4D tensor interface (T, C, H, W) while leveraging diffusers' native 5D functionality.
    """
    
    def __init__(self, vae_model: AutoencoderKLCosmos, temporal_compression_ratio: int = 8):
        """
        Initialize with a pre-loaded diffusers VAE model.
        
        Args:
            vae_model: Already-loaded AutoencoderKLCosmos from diffusers
            temporal_compression_ratio: Temporal compression factor (default 8)
        """
        if vae_model is None:
            raise ValueError("vae_model is required")
            
        self.model = vae_model
        self.config = vae_model.config
        self.temporal_compression_ratio = temporal_compression_ratio
        
        # Essential properties for compatibility with existing pipeline code
        self.latent_ch = self.config.latent_channels
        self.spatial_compression_ratio = self.config.spatial_compression_ratio
        
        print(f"CleanVAE initialized:")
        print(f"  - Latent channels: {self.latent_ch}")
        print(f"  - Spatial compression: {self.spatial_compression_ratio}x")
        print(f"  - Temporal compression: {self.temporal_compression_ratio}x")
    
    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        """Calculate number of latent frames from pixel frames"""
        if num_pixel_frames == 1:
            return 1  # Single frame doesn't get temporal compression
        
        # For video: temporal compression with +1 offset
        return (num_pixel_frames - 1) // self.temporal_compression_ratio + 1
        
    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        """Calculate number of pixel frames from latent frames"""
        if num_latent_frames == 1:
            return 1  # Single frame case
            
        # Reverse the temporal compression calculation
        return (num_latent_frames - 1) * self.temporal_compression_ratio + 1
    
    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode video/image from pixel space to latent space.
        
        Args:
            state: (T, C, H, W) - temporal, channel, height, width
            
        Returns:
            latent: (latent_ch, T_compressed, latent_H, latent_W)
        """
        print(f"[VAE] encode input shape: {state.shape} (expected 4D: T,C,H,W)")
        if state.ndim != 4:
            raise ValueError(f"Expected 4D input (T, C, H, W), got {state.ndim}D with shape {state.shape}")
        
        original_dtype = state.dtype
        T, C, H, W = state.shape
        
        print(f"[VAE] encode: {state.shape} -> ", end="")
        
        # Calculate target shape with temporal compression
        if T == 1:
            latent_T = 1
        else:
            latent_T = self.get_latent_num_frames(T)
        
        latent_H = H // self.spatial_compression_ratio
        latent_W = W // self.spatial_compression_ratio
        target_shape = (self.latent_ch, latent_T, latent_H, latent_W)
        print(f"{target_shape}")
        
        # Convert to 5D format for diffusers: (T, C, H, W) -> (1, C, T, H, W)
        state_5d = state.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (1, C, T, H, W)
        print(f"[VAE] converted to 5D for diffusers: {state_5d.shape}")
        
        # Use diffusers' native 5D encode method
        encoded = self.model.encode(state_5d)
        latent_5d = encoded.latent_dist.sample()  # (1, latent_ch, T_compressed, latent_H, latent_W)
        print(f"[VAE] diffusers encode output (5D): {latent_5d.shape}")
        
        # Convert back to 4D format: (1, latent_ch, T_compressed, latent_H, latent_W) -> (latent_ch, T_compressed, latent_H, latent_W)
        latent_4d = latent_5d.squeeze(0)
        print(f"[VAE] final output (4D): {latent_4d.shape}")
        
        return latent_4d.to(original_dtype)
    
    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode from latent space to pixel space.
        
        Args:
            latent: (latent_ch, T_compressed, latent_H, latent_W)
            
        Returns:
            decoded: (T, C, H, W)
        """
        print(f"[VAE] decode input shape: {latent.shape} (expected 4D: latent_ch,T,H,W)")
        if latent.ndim != 4:
            raise ValueError(f"Expected 4D input (latent_ch, T, latent_H, latent_W), got {latent.ndim}D with shape {latent.shape}")
        
        original_dtype = latent.dtype
        C, latent_T, latent_H, latent_W = latent.shape
        
        print(f"[VAE] decode: {latent.shape} -> ", end="")
        
        # Calculate output dimensions with temporal decompression
        if latent_T == 1:
            pixel_T = 1
        else:
            pixel_T = self.get_pixel_num_frames(latent_T)
            
        pixel_H = latent_H * self.spatial_compression_ratio
        pixel_W = latent_W * self.spatial_compression_ratio
        target_shape = (pixel_T, 3, pixel_H, pixel_W)
        print(f"{target_shape}")
        
        # Convert to 5D format for diffusers: (latent_ch, T, latent_H, latent_W) -> (1, latent_ch, T, latent_H, latent_W)
        latent_5d = latent.unsqueeze(0)  # (1, latent_ch, T, latent_H, latent_W)
        print(f"[VAE] converted to 5D for diffusers: {latent_5d.shape}")
        
        # Use diffusers' native 5D decode method
        decoded_5d = self.model.decode(latent_5d)
        pixel_5d = decoded_5d.sample  # (1, 3, T, pixel_H, pixel_W)
        print(f"[VAE] diffusers decode output (5D): {pixel_5d.shape}")
        
        # Convert back to 4D format: (1, 3, T, pixel_H, pixel_W) -> (T, 3, pixel_H, pixel_W)
        pixel_4d = pixel_5d.squeeze(0).permute(1, 0, 2, 3)  # (T, 3, pixel_H, pixel_W)
        print(f"[VAE] final output (4D): {pixel_4d.shape}")
        
        return pixel_4d.to(original_dtype)
    
    # Compatibility properties for existing pipeline code
    @property
    def spatial_resolution(self) -> Tuple[int, int]:
        """Get the configured spatial resolution (H, W)."""
        return (self.config.resolution, self.config.resolution)
    
    @property
    def channel(self) -> int:
        """Get number of latent channels"""
        return self.latent_ch
    
    def to(self, device):
        """Move VAE to device"""
        self.model = self.model.to(device)
        return self
    
    def reset_dtype(self, dtype: torch.dtype):
        """Reset the dtype for the VAE (for compatibility)"""
        # Note: diffusers handles dtype internally
        pass
