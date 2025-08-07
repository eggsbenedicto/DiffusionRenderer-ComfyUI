import torch
from typing import Tuple
from diffusers import AutoencoderKLCosmos

class CleanVAE:
    """
    A pure 4D wrapper for the diffusers AutoencoderKLCosmos for single-frame encoding/decoding.
    It accepts and returns standard 4D (B, C, H, W) tensors.
    The JointImageVideoTokenizer is responsible for managing the time dimension.
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

    # Property for compatibility with the Joint Tokenizer interface
    @property
    def temporal_compression_factor(self):
        return 1 # An image VAE has no temporal compression.

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        return 1
        
    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        return 1
    
    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """ Encodes a 4D (B,C,H,W) tensor. """
        if state.ndim != 4:
            raise ValueError(f"CleanVAE (Image VAE) expects a 4D input (B, C, H, W), but got {state.shape}")
        
        # Pass directly to the diffusers model
        encoded = self.model.encode(state)
        return encoded.latent_dist.sample()
    
    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """ Decodes a 4D (B,C,H,W) latent tensor. """
        if latent.ndim != 4:
            raise ValueError(f"CleanVAE (Image VAE) expects a 4D latent (B, C, H, W), but got {latent.shape}")

        # Pass directly to the diffusers model
        decoded = self.model.decode(latent)
        return decoded.sample
    
    def to(self, device):
        self.model = self.model.to(device)
        return self
    
    def reset_dtype(self, dtype: torch.dtype):
        self.model.to(dtype)