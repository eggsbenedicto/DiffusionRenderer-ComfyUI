import torch
from typing import Tuple
from diffusers import AutoencoderKLCosmos

class CleanVAE:
    """
    Wrapper for the standard diffusers AutoencoderKLCosmos, designed for single-frame encoding.
    It now handles its own model loading using `from_pretrained`.
    """
    
    def __init__(self, model_path: str, temporal_compression_ratio: int = 8):
        """
        Initialize and load the diffusers VAE model from a specified path.
        
        Args:
            model_path (str): Path to the directory containing the diffusers model 
                              (e.g., the '.../Cosmos-1.0-Tokenizer-CV8x8x8/vae' subfolder).
            temporal_compression_ratio (int): The *video* VAE's compression ratio. This is
                                              only used for property compatibility in the
                                              JointImageVideoTokenizer. This VAE itself
                                              does not perform temporal compression.
        """
        print(f"[CleanVAE] Loading AutoencoderKLCosmos model from path: {model_path}")
        # Use the standard, robust diffusers loading method.
        self.model = AutoencoderKLCosmos.from_pretrained(model_path)
        
        if self.model is None:
            raise ValueError(f"Failed to load VAE model from {model_path}")

        self.config = self.model.config
        
        # This property is for compatibility with the Video Tokenizer interface.
        # This image VAE itself has a temporal compression of 1.
        self.temporal_compression_ratio = temporal_compression_ratio
        
        # Essential properties for compatibility with the rest of the pipeline
        self.latent_ch = self.config.latent_channels
        self.spatial_compression_ratio = self.config.scaling_factor # Diffusers calls it scaling_factor
        
        print(f"CleanVAE (Image VAE) initialized successfully:")
        print(f"  - Latent channels: {self.latent_ch}")
        print(f"  - Spatial compression: {self.spatial_compression_ratio}x")

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        """For a single image, the number of latent frames is always 1."""
        if num_pixel_frames != 1:
            # This VAE should only be used for single frames.
            print(f"Warning: CleanVAE (Image VAE) received input with {num_pixel_frames} frames. It is designed for single frames only.")
        return 1
        
    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        """For a single latent, the number of pixel frames is always 1."""
        if num_latent_frames != 1:
            print(f"Warning: CleanVAE (Image VAE) received latent with {num_latent_frames} frames. It is designed for single frames only.")
        return 1
    
    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode a single image from pixel space to latent space.
        
        Args:
            state: (T, C, H, W) where T must be 1.
            
        Returns:
            latent: (latent_ch, 1, latent_H, latent_W)
        """
        if state.ndim != 4 or state.shape[0] != 1:
            raise ValueError(f"CleanVAE (Image VAE) expects a 4D input with T=1 (shape [1, C, H, W]), but got {state.shape}")
        
        # The diffusers VAE expects (B, C, H, W). Since T=1, we can just squeeze it.
        state_bchw = state.squeeze(0)
        
        # Add a batch dimension for the model
        state_bchw = state_bchw.unsqueeze(0)
        
        # Use diffusers' native encode method
        encoded = self.model.encode(state_bchw)
        latent_b_chw = encoded.latent_dist.sample()
        
        # Add the temporal dimension back for consistency with the video VAE interface
        latent_4d = latent_b_chw.squeeze(0).unsqueeze(1) # (latent_ch, 1, H, W)
        
        return latent_4d.to(state.dtype)
    
    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode from latent space to pixel space for a single image.
        
        Args:
            latent: (latent_ch, T, latent_H, latent_W) where T must be 1.
            
        Returns:
            decoded: (1, C, H, W)
        """
        if latent.ndim != 4 or latent.shape[1] != 1:
            raise ValueError(f"CleanVAE (Image VAE) expects a 4D latent with T=1 (shape [C, 1, H, W]), but got {latent.shape}")
        
        # The diffusers VAE expects (B, C, H, W).
        latent_bchw = latent.permute(1, 0, 2, 3) # (1, latent_ch, H, W)
        
        decoded_5d = self.model.decode(latent_bchw)
        pixel_bchw = decoded_5d.sample
        
        # Add the temporal dimension back
        pixel_4d = pixel_bchw.unsqueeze(0).permute(1, 0, 2, 3) # (1, C, H, W)
        
        return pixel_4d.to(latent.dtype)
    
    # Compatibility properties and methods
    @property
    def spatial_resolution(self) -> Tuple[int, int]:
        return (self.config.sample_size, self.config.sample_size)
    
    @property
    def channel(self) -> int:
        return self.latent_ch
    
    def to(self, device):
        self.model = self.model.to(device)
        return self
    
    def reset_dtype(self, dtype: torch.dtype):
        self.model.to(dtype)