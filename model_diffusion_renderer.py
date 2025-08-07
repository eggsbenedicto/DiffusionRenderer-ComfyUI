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
Clean implementation of DiffusionRendererModel without parallel processing dependencies.
Extracts core diffusion sampling logic while removing training/distributed infrastructure.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Union, Optional, Any
from torch import Tensor
from .CleanVAE import CleanVAE
from .CleanGeneralDIT import CleanDiffusionRendererGeneralDIT

class FourierFeaturesPlaceholder(nn.Module):
    def __init__(self, num_channels, **kwargs):
        super().__init__()
        # These buffers will be overwritten by the state_dict
        self.register_buffer("freqs", torch.randn(num_channels))
        self.register_buffer("phases", torch.randn(num_channels))
    def forward(self, x): return x

class CleanEDMEulerScheduler:
    """Clean implementation of EDM Euler scheduler with proper sigma scheduling"""
    
    def __init__(self, sigma_max: float = 80.0, sigma_min: float = 0.02, sigma_data: float = 0.5):
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data
        self.init_noise_sigma = sigma_max
        self.sigmas = None
        self.timesteps = None
        self.current_step = 0
        
    def set_timesteps(self, num_steps: int, device: Optional[torch.device] = None):
        """Set the timesteps for the diffusion process with proper sigma schedule"""
        # Create log-linear schedule from sigma_max to sigma_min
        # Add sigma_min=0 at the end for final denoising
        sigmas = torch.logspace(
            np.log10(self.sigma_max), 
            np.log10(self.sigma_min), 
            num_steps,
            device=device
        )
        # Append 0 for final step
        zero_tensor = torch.tensor([0.0], device=device)
        self.sigmas = torch.cat([sigmas, zero_tensor])
        self.timesteps = self.sigmas[:-1]  # Don't include the final 0 in timesteps
        self.current_step = 0
        
    def scale_model_input(self, sample: Tensor, timestep: Tensor) -> Tensor:
        """Scale the model input according to EDM formulation"""
        sigma = timestep
        c_in = 1 / torch.sqrt(sigma**2 + self.sigma_data**2)
        return sample * c_in
        
    def step(self, model_output: Tensor, timestep: Tensor, sample: Tensor, use_heun: bool = False):
        """Single denoising step using Euler method with proper EDM formulation"""
        sigma = timestep
        
        # Safety check for scheduler state
        if self.sigmas is None or self.current_step >= len(self.sigmas) - 1:
            raise RuntimeError("Scheduler not properly initialized or step counter out of bounds")
        
        # Get current and next sigma values
        sigma_curr = self.sigmas[self.current_step]
        sigma_next = self.sigmas[self.current_step + 1]
        
        # EDM denoising formulation
        c_out = sigma_curr * self.sigma_data / torch.sqrt(sigma_curr**2 + self.sigma_data**2)
        c_skip = self.sigma_data**2 / (sigma_curr**2 + self.sigma_data**2)
        
        # Compute denoised sample (this is the predicted x_0)
        denoised = c_skip * sample + c_out * model_output
        
        # Euler step: x_{t+1} = x_t + (sigma_next - sigma_curr) * d/dt
        # For EDM, the derivative is (denoised - sample) / sigma_curr
        if sigma_next == 0:
            # Final step - return denoised directly
            prev_sample = denoised
        else:
            # Euler step
            derivative = (denoised - sample) / sigma_curr
            prev_sample = sample + (sigma_next - sigma_curr) * derivative
            
            # Optional: Heun's method (2nd order) for better accuracy
            if use_heun and self.current_step < len(self.sigmas) - 2:
                # Second evaluation at the predicted point
                prev_sample_scaled = self.scale_model_input(prev_sample, sigma_next)
                
                # Note: In a real implementation, we would call the model again here
                # For now, we'll skip the second evaluation to avoid recursive calls
                pass
        
        # Advance step counter
        self.current_step += 1
        
        class StepResult:
            def __init__(self, prev_sample):
                self.prev_sample = prev_sample
                
        return StepResult(prev_sample)


class CleanCondition:
    """Simple condition container with to_dict() method"""
    
    def __init__(self, **kwargs):
        self.data = kwargs
        
    def to_dict(self):
        return self.data


class CleanConditioner:
    """Simplified conditioner without complex dependencies"""
    
    def __init__(self):
        pass
        
    def get_condition_uncondition(self, data_batch: Dict) -> Tuple[CleanCondition, CleanCondition]:
        """Get condition and uncondition from data batch with proper CFG support"""
        condition_data = {}
        uncondition_data = {}
        
        # Copy relevant conditioning information
        for key in ['latent_condition', 'context_index']:
            if key in data_batch:
                condition_data[key] = data_batch[key]
                # For uncondition, zero out conditioning signals for proper CFG
                if key == 'latent_condition':
                    # Zero out latent conditions for unconditional branch - this is key to CFG!
                    uncondition_data[key] = torch.zeros_like(data_batch[key])
                elif key == 'context_index':
                    # Zero out context indices for unconditional branch
                    uncondition_data[key] = torch.zeros_like(data_batch[key])
        
        return CleanCondition(**condition_data), CleanCondition(**uncondition_data)


# CleanVAE is now imported from the separate CleanVAE.py file


class CleanDiffusionRendererModel(nn.Module):
    """Clean implementation of DiffusionRendererModel as a PyTorch module"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        
        # Use provided config or create default
        if config is None:
            config = self._get_default_config()
        
        self.config = config
        
        # Extract configuration parameters
        net_config = config.get('net', {})
        scheduler_config = config.get('scheduler', {})
        vae_config = config.get('vae', {})
        
        # Non-learnable components (stay as attributes)
        self.scheduler = CleanEDMEulerScheduler(
            sigma_max=scheduler_config.get('sigma_max', 80.0),
            sigma_min=scheduler_config.get('sigma_min', 0.02),
            sigma_data=scheduler_config.get('sigma_data', 0.5)
        )
        self.conditioner = CleanConditioner()
        
        # Learnable components (registered as submodules automatically)
        self.net = CleanDiffusionRendererGeneralDIT(**net_config)
        
        # Initialize with mock VAE - the pipeline will replace this with a real VAE if available
        self.vae = self._create_mock_vae(vae_config)

        self.logvar = torch.nn.Sequential(
            FourierFeaturesPlaceholder(num_channels=128),
            torch.nn.Linear(128, 1, bias=False)
        )

        # Configuration from config dict
        # For inverse rendering: condition only on the input image (like official implementation)
        # For forward rendering: condition on all G-buffer inputs
        model_type = config.get('model_type', 'inverse')
        if model_type == 'inverse':
            # Inverse rendering: RGB image -> G-buffers (one at a time via context_index)
            self.condition_keys = config.get('condition_keys', ["image"])
        else:
            # Forward rendering: G-buffers -> RGB image
            self.condition_keys = config.get('condition_keys', ["depth", "normal", "basecolor", "roughness", "metallic"])
        
        self.condition_drop_rate = config.get('condition_drop_rate', 0.0)
        self.append_condition_mask = config.get('append_condition_mask', True)
        self.input_data_key = config.get('input_data_key', "video")  # Will be set dynamically
        
        # Create mock tokenizer for interface compatibility
        self.tokenizer = self._create_tokenizer()
        
        # Set to eval mode by default (inference model)
        self.eval()
        
        # Log successful initialization
        print(f"CleanDiffusionRendererModel initialized with:")
        print(f"  Model type: {config.get('model_type', 'unknown')}")
        print(f"  Network: CleanDiffusionRendererGeneralDIT")
        print(f"  Model channels: {net_config.get('model_channels', 'unknown')}")
        print(f"  Num blocks: {net_config.get('num_blocks', 'unknown')}")
        print(f"  Additional concat channels: {net_config.get('additional_concat_ch', 'unknown')}")
        print(f"  Use context embedding: {net_config.get('use_context_embedding', 'unknown')}")
        print(f"  VAE: CleanVAE with {self.vae.latent_ch} latent channels")
        print(f"  VAE compression: {self.vae.spatial_compression_factor}x spatial, {self.vae.temporal_compression_factor}x temporal")
        
    def _get_default_config(self):
        """Get default configuration for backward compatibility"""
        from diffusion_renderer_config import get_inverse_renderer_config
        return get_inverse_renderer_config(height=1024, width=1024, num_frames=1)
    
    def _create_mock_vae(self, vae_config):
        """Create a mock VAE for initialization - will be replaced by pipeline if real VAE is available"""
        class MockVAE:
            def __init__(self, config):
                # Extract values from config or use defaults
                self.latent_ch = config.get('latent_ch', 16)
                self.spatial_compression_factor = config.get('spatial_compression_factor', 8)
                self.temporal_compression_factor = config.get('temporal_compression_factor', 8)
                
            def encode(self, x):
                # Mock encode - just downsample
                B, C, T, H, W = x.shape
                latent_H = H // self.spatial_compression_factor
                latent_W = W // self.spatial_compression_factor
                latent_T = self.get_latent_num_frames(T)
                return torch.zeros(B, self.latent_ch, latent_T, latent_H, latent_W, 
                                 device=x.device, dtype=x.dtype)
                
            def decode(self, x):
                # Mock decode - just upsample
                B, C, T, H, W = x.shape
                pixel_H = H * self.spatial_compression_factor
                pixel_W = W * self.spatial_compression_factor
                pixel_T = self.get_pixel_num_frames(T)
                return torch.zeros(B, 3, pixel_T, pixel_H, pixel_W,
                                 device=x.device, dtype=x.dtype)
                                 
            def get_latent_num_frames(self, num_pixel_frames):
                if num_pixel_frames == 1:
                    return 1
                return (num_pixel_frames - 1) // self.temporal_compression_factor + 1
                
            def get_pixel_num_frames(self, num_latent_frames):
                if num_latent_frames == 1:
                    return 1
                return (num_latent_frames - 1) * self.temporal_compression_factor + 1
                
        return MockVAE(vae_config)
        
    def _create_tokenizer(self):
        """Create tokenizer interface for compatibility using real VAE properties"""
        class MockTokenizer:
            def __init__(self, vae):
                self.channel = vae.latent_ch  # Use real VAE latent channels
                self.spatial_compression_factor = vae.spatial_compression_factor  # Use real VAE compression
                self.temporal_compression_factor = vae.temporal_compression_factor  # Use real VAE temporal compression
        return MockTokenizer(self.vae)
    
    def _get_tensor_kwargs(self):
        """Dynamically get device and dtype from model parameters"""
        try:
            # Get device and dtype from first parameter
            param = next(self.parameters())
            return {"device": param.device, "dtype": param.dtype}
        except StopIteration:
            # Fallback if no parameters (shouldn't happen with real model)
            return {"device": torch.device("cuda"), "dtype": torch.bfloat16}
    
    def forward(self, *args, **kwargs):
        """Forward pass - delegates to generate_samples_from_batch for compatibility"""
        # For compatibility with nn.Module interface
        # In practice, users should call generate_samples_from_batch directly
        if len(args) == 1 and isinstance(args[0], dict):
            return self.generate_samples_from_batch(args[0], **kwargs)
        else:
            raise NotImplementedError("Use generate_samples_from_batch() for diffusion sampling")
    
    def load_checkpoint(self, checkpoint_path: str, strict: bool = True):
        """Load checkpoint using standard PyTorch methods"""
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        try:
            # Handle different checkpoint formats
            if checkpoint_path.endswith('.safetensors'):
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(checkpoint_path)
                    print("Loaded .safetensors checkpoint")
                except ImportError:
                    raise ImportError("safetensors not available. Install with: pip install safetensors")
            else:
                # Load .pt checkpoint
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                
                # Handle nested checkpoint structure
                if isinstance(checkpoint, dict):
                    if "model" in checkpoint:
                        state_dict = checkpoint["model"]
                        print("Found nested 'model' key in checkpoint")
                        
                        # Handle EMA weights if present
                        if "ema" in checkpoint and checkpoint["ema"] is not None:
                            print("Found EMA weights, merging with model weights")
                            ema_state_dict = checkpoint["ema"]
                            # Convert EMA buffer names: "-" → "."
                            ema_state_dict = {k.replace("-", "."): v for k, v in ema_state_dict.items()}
                            state_dict.update(ema_state_dict)
                    else:
                        state_dict = checkpoint
                        print("Using checkpoint as state_dict directly")
                else:
                    state_dict = checkpoint
            
            # Load state dict with error handling
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=True)
            
            # Report loading results
            print(f"Checkpoint loaded successfully:")
            print(f"  - {len(missing_keys)} missing keys")
            print(f"  - {len(unexpected_keys)} unexpected keys")
            
            if missing_keys and len(missing_keys) < 10:  # Only show if not too many
                print(f"  Missing keys: {missing_keys}")
            if unexpected_keys and len(unexpected_keys) < 10:  # Only show if not too many  
                print(f"  Unexpected keys: {unexpected_keys}")
                
            return self
            
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            if not strict:
                print("Continuing with mock model (checkpoint loading failed)")
                return self
            else:
                raise
        
    def encode(self, x: Tensor) -> Tensor:
        """
        VAE encode wrapper.
        This is now a clean pass-through. The CleanVAE class is responsible
        for all required dimension handling and permutation.
        """
        print(f"[Model] encode input shape, passing to VAE: {x.shape}")

        # The input `x` from the pipeline is 5D: (B, C, T, H, W).
        # We need to process each item in the batch individually for the VAE wrapper.
        if x.ndim != 5:
            raise ValueError(f"Model encode expects a 5D tensor (B,C,T,H,W), but got {x.ndim}D.")

        # The CleanVAE expects a 4D tensor (T, C, H, W).
        # We need to iterate over the batch and correctly shape the input for it.
        # B, C, T, H, W -> permute to -> B, T, C, H, W -> iterate over B
        
        x_permuted = x.permute(0, 2, 1, 3, 4) # (B, T, C, H, W)
        
        encoded_list = [self.vae.encode(item) for item in x_permuted]
        
        # Stack the results back into a single tensor.
        # The VAE output is (latent_ch, T_compressed, H_latent, W_latent).
        # Stacking creates (B, latent_ch, T_compressed, H_latent, W_latent).
        latent = torch.stack(encoded_list, dim=0)

        print(f"[Model] VAE encode final output shape: {latent.shape}")
        return latent

    def decode(self, x: Tensor) -> Tensor:
        """
        VAE decode wrapper. This also becomes a clean pass-through.
        """
        print(f"[Model] decode input latent shape, passing to VAE: {x.shape}")
        
        # Input `x` from diffusion is 5D: (B, latent_ch, T_comp, H_l, W_l).
        # The CleanVAE expects a 4D tensor (latent_ch, T_comp, H_l, W_l).
        if x.ndim != 5:
             raise ValueError(f"Model decode expects a 5D latent (B,C,T,H,W), but got {x.ndim}D.")

        decoded_list = [self.vae.decode(item) for item in x]

        # Stack the results. VAE output is (T, C, H, W).
        # Stacking creates (B, T, C, H, W).
        decoded_stacked = torch.stack(decoded_list, dim=0)

        # Permute back to the pipeline's expected (B, C, T, H, W) format.
        decoded = decoded_stacked.permute(0, 2, 1, 3, 4)

        print(f"[Model] VAE decode final output shape: {decoded.shape}")
        return decoded
        
    def prepare_diffusion_renderer_latent_conditions(
        self, 
        data_batch: Dict[str, Tensor],
        condition_keys: list = None,
        condition_drop_rate: float = 0,
        append_condition_mask: bool = True,
        dtype: torch.dtype = None,
        device: torch.device = None,
        latent_shape: Union[Tuple[int, int, int, int, int], torch.Size] = None,
        mode: str = "inference",
    ) -> Tensor:
        """Prepare latent conditions (extracted from original)"""
        
        if condition_keys is None:
            condition_keys = self.condition_keys

        #official latent shape: latent_shape = (B, 16, T // 8 + 1, H // 8, W // 8)
        #we're trying a dynamic latent shape
            
        if latent_shape is None:
            # Get shape from config if available
            if hasattr(self, 'config') and 'latent_shape' in self.config:
                config_shape = self.config['latent_shape']  # [C, T, H, W]
                # Convert to batch format: find first available condition to get batch size
                B = 1
                for key in condition_keys:
                    if key in data_batch:
                        B = data_batch[key].shape[0]
                        break
                latent_shape = (B, config_shape[0], config_shape[1], config_shape[2], config_shape[3])
            else:
                # Fallback: Get shape from first available condition using real VAE properties
                for key in condition_keys:
                    if key in data_batch:
                        B, C, T, H, W = data_batch[key].shape
                        latent_C = self.vae.latent_ch
                        latent_T = self.vae.get_latent_num_frames(T)
                        latent_H = H // self.vae.spatial_compression_factor
                        latent_W = W // self.vae.spatial_compression_factor
                        latent_shape = (B, latent_C, latent_T, latent_H, latent_W)
                        break
                    
        if append_condition_mask:
            latent_mask_shape = (latent_shape[0], 1, latent_shape[2], latent_shape[3], latent_shape[4])
            
        if dtype is None:
            dtype = next(iter(data_batch.values())).dtype
        if device is None:
            device = next(iter(data_batch.values())).device
            
        latent_condition_list = []
        
        for cond_key in condition_keys:
            is_condition_dropped = condition_drop_rate > 0 and np.random.rand() < condition_drop_rate
            is_condition_skipped = cond_key not in data_batch
            
            if is_condition_dropped or is_condition_skipped:
                # Dropped or skipped condition
                condition_state = torch.zeros(latent_shape, dtype=dtype, device=device)
                latent_condition_list.append(condition_state)
                if append_condition_mask:
                    condition_mask = torch.zeros(latent_mask_shape, dtype=dtype, device=device)
                    latent_condition_list.append(condition_mask)
            else:
                # Valid condition
                condition_state = data_batch[cond_key].to(device=device, dtype=dtype)
                condition_state = self.encode(condition_state).contiguous()
                latent_condition_list.append(condition_state)
                if append_condition_mask:
                    condition_mask = torch.ones(latent_mask_shape, dtype=dtype, device=device)
                    latent_condition_list.append(condition_mask)
                    
        return torch.cat(latent_condition_list, dim=1)
        
    def _get_conditions(self, data_batch: Dict, is_negative_prompt: bool = False):
        """Get conditions from data batch (extracted from original)"""
        
        # Determine input data key
        self.input_data_key = "video"
        if "video" not in data_batch:
            # Find the first available key as in the original pipeline
            for key in ['rgb', 'basecolor', 'normal', 'depth', 'roughness', 'metallic', 'image']:
                if key in data_batch:
                    self.input_data_key = key
                    break
                    
        raw_state = data_batch[self.input_data_key]
        
        with torch.no_grad():
            latent_condition = self.prepare_diffusion_renderer_latent_conditions(
                data_batch,
                condition_keys=self.condition_keys,
                condition_drop_rate=0,
                append_condition_mask=self.append_condition_mask,
                dtype=raw_state.dtype,
                device=raw_state.device,
                latent_shape=None,
                mode="inference",
            )
            
        data_batch["latent_condition"] = latent_condition
        
        condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)
        
        return condition, uncondition
        
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 0.0,
        seed: int = 1000,
        state_shape: Tuple = None,
        n_sample: int = 1,
        is_negative_prompt: bool = False,
        num_steps: int = 15,
    ) -> Tensor:
        """Generate samples from a data batch using diffusion sampling (exact logic from original)"""
        
        with torch.no_grad():
            # Set random seed
            torch.manual_seed(seed)
            
            # Get conditions
            condition, uncondition = self._get_conditions(data_batch, is_negative_prompt)
            
            # Set timesteps and reset step counter (important for multiple calls)
            tensor_kwargs = self._get_tensor_kwargs()
            self.scheduler.set_timesteps(num_steps, device=tensor_kwargs["device"])
            
            # Get dynamic tensor kwargs based on model's current device/dtype
            # (moved after scheduler setup to ensure consistency)
            
            # Initialize noise
            xt = torch.randn(size=(n_sample,) + tuple(state_shape), **tensor_kwargs) * self.scheduler.init_noise_sigma
            print(f"[Model] Initialized noise shape: {xt.shape} with state_shape: {state_shape}")
            
            # Denoising loop with proper EDM Euler integration
            for i, t in enumerate(self.scheduler.timesteps):
                xt = xt.to(**tensor_kwargs)
                xt_scaled = self.scheduler.scale_model_input(xt, timestep=t)
                print(f"[Model] Step {i+1}/{num_steps}: xt_scaled shape: {xt_scaled.shape}")
                
                # Predict the noise residual  
                t = t.to(**tensor_kwargs)
                
                print(f"Step {i+1}/{num_steps}: sigma={float(t):.4f}")
                
                # Use official parameter passing - clean and correct
                net_output_cond = self.net(x=xt_scaled, timesteps=t, **condition.to_dict())
                print(f"[Model] net_output_cond shape: {net_output_cond.shape}")
                net_output = net_output_cond
                
                if guidance > 0:
                    net_output_uncond = self.net(x=xt_scaled, timesteps=t, **uncondition.to_dict())
                    net_output = net_output_cond + guidance * (net_output_cond - net_output_uncond)
                    print(f"[Model] guided net_output shape: {net_output.shape}")
                    
                # Compute the previous noisy sample x_t -> x_t-1 using proper Euler integration
                xt = self.scheduler.step(net_output, t, xt).prev_sample
                print(f"[Model] After scheduler step, xt shape: {xt.shape}")
                
            samples = xt
            print(f"[Model] Final samples shape: {samples.shape}")
            
            return samples
