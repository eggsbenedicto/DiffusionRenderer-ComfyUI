import torch
from typing import Tuple
from diffusers import AutoencoderKLCosmos

class CleanVAE:
    def __init__(self, model_path: str):
        print(f"[CleanVAE] Loading AutoencoderKLCosmos model from path: {model_path}")
        self.model = AutoencoderKLCosmos.from_pretrained(model_path, torch_dtype=torch.float32)
        
        if self.model is None:
            raise ValueError(f"Failed to load VAE model from {model_path}")

        self.config = self.model.config
        
        # FIX: Use underscore version consistently
        self._spatial_compression_factor = self.model.config.spatial_compression_ratio
        self._temporal_compression_factor = 1  # 1 for images, 8 for videos
        self.latent_ch = self.config.latent_channels
        
        # Always use float32 for quality
        self.dtype = torch.float32
        
        print(f"CleanVAE initialized successfully:")
        print(f"  - Latent channels: {self.latent_ch}")
        print(f"  - Spatial compression: {self._spatial_compression_factor}x")
        print(f"  - Temporal compression: {self._temporal_compression_factor}x")
        print(f"  - Precision: float32 (for quality)")

    # REMOVED the class-level self._spatial_compression_factor lines!
    
    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        """
        Calculate number of latent frames from pixel frames
        For images (T=1), return 1
        For videos, apply temporal compression
        """
        if num_pixel_frames == 1:
            return 1
        # Apply temporal compression for videos
        return (num_pixel_frames - 1) // self._temporal_compression_factor + 1
    
    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        """
        Calculate number of pixel frames from latent frames (inverse)
        """
        if num_latent_frames == 1:
            return 1
        return (num_latent_frames - 1) * self._temporal_compression_factor + 1
    
    @torch.no_grad()
    def encode(self, state_5d: torch.Tensor) -> torch.Tensor:
        """ Encodes a 5D tensor, always using float32 internally """
        if state_5d.ndim != 5:
            raise ValueError(f"CleanVAE expects a 5D input (B, C, T, H, W), but got {state_5d.shape}")
        
        # Remember input dtype
        input_dtype = state_5d.dtype
        
        # Ensure float32 for encoding
        if state_5d.dtype != torch.float32:
            state_5d = state_5d.to(torch.float32)
        
        encoded = self.model.encode(state_5d)
        latent = encoded.latent_dist.sample()
        
        # Return in original dtype if needed by downstream model
        if input_dtype != torch.float32:
            return latent.to(input_dtype)
        return latent
    
    @torch.no_grad()
    def decode(self, latent_5d: torch.Tensor) -> torch.Tensor:
        """ Decodes a 5D latent tensor, always using float32 internally """
        if latent_5d.ndim != 5:
            raise ValueError(f"CleanVAE expects a 5D latent (B, C, T, H, W), but got {latent_5d.shape}")

        # Ensure float32 for decoding
        if latent_5d.dtype != torch.float32:
            latent_5d = latent_5d.to(torch.float32)
            
        decoded = self.model.decode(latent_5d)
        
        # Keep output in float32 for quality
        return decoded.sample
    
    def to(self, device):
        self.model = self.model.to(device)
        return self
    
    def reset_dtype(self, dtype: torch.dtype):
        # Override to always stay in float32, which we consider best for quality
        print(f"[CleanVAE] Request to change dtype to {dtype}, but staying in float32 for quality")

    @property
    def spatial_compression_factor(self):
        """Spatial downsampling factor (8x for 1024→128)"""
        return self._spatial_compression_factor
    
    @property
    def temporal_compression_factor(self):
        """Temporal compression factor (1x for images, 8x for videos)"""
        return self._temporal_compression_factor