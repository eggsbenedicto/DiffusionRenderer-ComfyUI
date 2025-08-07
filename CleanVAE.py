import torch
from typing import Tuple
from diffusers import AutoencoderKLCosmos

import torch
from typing import Tuple
from diffusers import AutoencoderKLCosmos

class CleanVAE:
    """
    A pure pass-through wrapper for the diffusers AutoencoderKLCosmos.
    It accepts and returns 5D tensors (B, C, T, H, W) as required by the underlying 3D model.
    """
    
    def __init__(self, model_path: str):
        print(f"[CleanVAE] Loading AutoencoderKLCosmos model from path: {model_path}")
        self.model = AutoencoderKLCosmos.from_pretrained(model_path)
        
        if self.model is None:
            raise ValueError(f"Failed to load VAE model from {model_path}")

        self.config = self.model.config
        
        self.spatial_compression_factor = self.model.config.spatial_compression_ratio
        self.latent_ch = self.config.latent_channels
        
        print(f"CleanVAE (Image VAE) initialized successfully:")
        print(f"  - Latent channels: {self.latent_ch}")
        print(f"  - Spatial compression: {self.spatial_compression_factor}x")

    @property
    def temporal_compression_factor(self):
        return 1

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        return 1
        
    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        return 1
    
    @torch.no_grad()
    def encode(self, state_5d: torch.Tensor) -> torch.Tensor:
        """ Encodes a 5D (B, C, T, H, W) tensor. """
        if state_5d.ndim != 5:
            raise ValueError(f"CleanVAE (Image VAE) expects a 5D input (B, C, T, H, W), but got {state_5d.shape}")
        
        encoded = self.model.encode(state_5d)
        return encoded.latent_dist.sample()
    
    @torch.no_grad()
    def decode(self, latent_5d: torch.Tensor) -> torch.Tensor:
        """ Decodes a 5D (B, C, T, H, W) latent tensor. """
        if latent_5d.ndim != 5:
            raise ValueError(f"CleanVAE (Image VAE) expects a 5D latent (B, C, T, H, W), but got {latent_5d.shape}")

        decoded = self.model.decode(latent_5d)
        return decoded.sample
    
    def to(self, device):
        self.model = self.model.to(device)
        return self
    
    def reset_dtype(self, dtype: torch.dtype):
        self.model.to(dtype)