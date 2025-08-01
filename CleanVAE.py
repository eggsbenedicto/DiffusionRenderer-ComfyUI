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
Clean implementation of Video VAE/Tokenizer using diffusers AutoencoderKLCosmos.
Based on the official cosmos_predict1.diffusion.module.pretrained_vae implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
from einops import rearrange
import os
import math

from diffusers import AutoencoderKLCosmos


# Hardcoded path to the trusted VAE configuration file.
# This ensures we always load the correct model architecture.

script_directory = os.path.dirname(os.path.abspath(__file__))
VAE_CONFIG_PATH = os.path.join(script_directory, "VAE_config.json")

class CleanVAE:
    """
    Clean implementation of Video VAE using diffusers AutoencoderKLCosmos.
    Handles spatial and temporal compression for video diffusion models.
    """
    
    def __init__(self, vae_model: 'AutoencoderKLCosmos', temporal_compression_ratio: int = 8):
        """
        Initialize the VAE with a loaded diffusers model.
        
        Args:
            vae_model: An already-loaded AutoencoderKLCosmos model from diffusers
            temporal_compression_ratio: Temporal compression factor (default 8)
        """
        if vae_model is None:
            raise ValueError("vae_model is required. Load VAE model externally and pass it to this constructor.")
            
        self.model = vae_model
        self.config = vae_model.config
        
        # Core parameters from diffusers config
        self.latent_ch = self.config.latent_channels
        self.spatial_compression_ratio = self.config.spatial_compression_ratio
        self.temporal_compression_ratio = temporal_compression_ratio  # Not in diffusers config
        
        # Calculate spatial resolution from config
        # The config has a 'resolution' field which is the target resolution
        self.spatial_resolution = (self.config.resolution, self.config.resolution)
        
        # Batch processing limits (for memory management)
        self.max_enc_batch_size = 8
        self.max_dec_batch_size = 4
        
        # Device and dtype settings
        self.device = next(self.model.parameters()).device if hasattr(self.model, 'parameters') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.bfloat16
        
        print(f"CleanVAE initialized with diffusers model:")
        print(f"  Model class: {self.config._class_name}")
        print(f"  Latent channels: {self.latent_ch}")
        print(f"  Spatial resolution: {self.spatial_resolution[1]}x{self.spatial_resolution[0]}")
        print(f"  Spatial compression: {self.spatial_compression_ratio}x")
        print(f"  Temporal compression: {self.temporal_compression_ratio}x")
        print(f"  Scaling factor: {self.config.scaling_factor}")
        
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
        
    def transform_encode_state_shape(self, state: torch.Tensor) -> torch.Tensor:
        """
        Transform input state for chunked encoding.
        For now, this is a pass-through, but in full implementation
        this would handle temporal chunking for long videos.
        """
        return state
        
    def transform_decode_state_shape(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Transform latent for chunked decoding.
        For now, this is a pass-through, but in full implementation
        this would handle temporal chunking for long videos.
        """
        return latent
        
    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode pixel space video to latent space.
        
        Args:
            state: Input tensor (B, C, T, H, W) in range [0, 1] or [-1, 1]
            
        Returns:
            Latent tensor (B, latent_ch, latent_T, latent_H, latent_W)
        """
        original_dtype = state.dtype
        B, C, T, H, W = state.shape

        # Validate and conform spatial resolution
        expected_resolution = self.spatial_resolution
        if (H, W) != expected_resolution:
            print(f"Warning: Input resolution ({W}x{H}) mismatches VAE config ({expected_resolution[1]}x{expected_resolution[0]}). Resizing input.")
            # Reshape for interpolation (B*T, C, H, W)
            state_reshaped = state.view(-1, C, H, W)
            
            state_resized = F.interpolate(
                state_reshaped,
                size=expected_resolution,
                mode='bilinear',
                align_corners=False
            )
            
            # Reshape back to original 5D tensor
            state = state_resized.view(B, C, T, *expected_resolution)
            B, C, T, H, W = state.shape # Update dimensions
        
        print(f"VAE encode: {state.shape} -> ", end="")
        
        # Handle single frame vs video differently
        if T == 1:
            # Single frame: no temporal compression
            latent_T = 1
            latent_H = H // self.spatial_compression_ratio
            latent_W = W // self.spatial_compression_ratio
        else:
            # Video: apply temporal compression
            latent_T = self.get_latent_num_frames(T)
            latent_H = H // self.spatial_compression_ratio
            latent_W = W // self.spatial_compression_ratio
            
        target_shape = (B, self.latent_ch, latent_T, latent_H, latent_W)
        print(f"{target_shape}")
        
        # Use diffusers VAE model
        state = state.to(device=self.device, dtype=self.dtype)
        
        # Process each frame through the VAE encoder
        # Note: diffusers VAE expects 4D input (B, C, H, W)
        encoded_frames = []
        for t in range(T):
            frame = state[:, :, t, :, :]  # (B, C, H, W)
            
            # Use diffusers encode method
            encoded = self.model.encode(frame)
            latent_dist = encoded.latent_dist
            latent_frame = latent_dist.sample()  # Sample from the distribution
            
            # Apply scaling factor
            latent_frame = latent_frame * self.config.scaling_factor
            
            encoded_frames.append(latent_frame)
        
        # Stack frames back into 5D tensor
        if T == 1:
            encoded_state = encoded_frames[0].unsqueeze(2)  # Add temporal dimension
        else:
            # For video, apply temporal compression
            encoded_state = torch.stack(encoded_frames, dim=2)  # (B, C, T, H, W)
            
            # Apply temporal compression if needed
            if T != latent_T:
                # Simple temporal downsampling for now
                # In a full implementation, this would use learned temporal compression
                encoded_state = F.interpolate(
                    encoded_state.permute(0, 1, 3, 4, 2),  # (B, C, H, W, T)
                    size=latent_T,
                    mode='linear',
                    align_corners=False
                ).permute(0, 1, 4, 2, 3)  # Back to (B, C, T, H, W)
        
        return encoded_state.to(original_dtype)
            
    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent space representation to pixel space video.
        
        Args:
            latent: Latent tensor (B, latent_ch, latent_T, latent_H, latent_W)
            
        Returns:
            Pixel tensor (B, 3, pixel_T, pixel_H, pixel_W) in range [-1, 1]
        """
        original_dtype = latent.dtype
        B, C, latent_T, latent_H, latent_W = latent.shape
        
        print(f"VAE decode: {latent.shape} -> ", end="")
        
        # Calculate output dimensions
        if latent_T == 1:
            # Single frame: no temporal decompression
            pixel_T = 1
        else:
            # Video: apply temporal decompression
            pixel_T = self.get_pixel_num_frames(latent_T)
            
        pixel_H = latent_H * self.spatial_compression_ratio
        pixel_W = latent_W * self.spatial_compression_ratio
        target_shape = (B, 3, pixel_T, pixel_H, pixel_W)
        print(f"{target_shape}")
        
        # Use diffusers VAE model
        latent = latent.to(device=self.device, dtype=self.dtype)
        
        # Apply temporal decompression if needed
        
        # Process each frame through the VAE decoder
        # Note: diffusers VAE expects 4D input (B, C, H, W)
        decoded_frames = []
        for t in range(pixel_T):
            latent_frame = latent[:, :, t, :, :]  # (B, C, H, W)
            
            # Remove scaling factor before decoding
            latent_frame = latent_frame / self.config.scaling_factor
            
            # Use diffusers decode method
            decoded_frame = self.model.decode(latent_frame).sample
            
            decoded_frames.append(decoded_frame)
        
        # Stack frames back into 5D tensor
        if pixel_T == 1:
            decoded_state = decoded_frames[0].unsqueeze(2)  # Add temporal dimension
        else:
            decoded_state = torch.stack(decoded_frames, dim=2)  # (B, C, T, H, W)
        
        return decoded_state.to(original_dtype)
        return self.temporal_compression_ratio
        
    @property
    def spatial_resolution(self) -> Tuple[int, int]:
        """Get the configured spatial resolution (H, W)."""
        return self.spatial_resolution

    @property
    def channel(self) -> int:
        """Get number of latent channels"""
        return self.latent_ch

    def reset_dtype(self, dtype: torch.dtype):
        """Reset the dtype for the VAE"""
        self.dtype = dtype
        
    def to(self, device: torch.device):
        """Move VAE to specified device"""
        self.device = device
        self.model = self.model.to(device)
        return self


# Clean, simple interface - no backward compatibility cruft needed
