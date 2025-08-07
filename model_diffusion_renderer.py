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
    """
    Clean implementation of DiffusionRendererModel.
    This version is adapted to work DIRECTLY with the official
    JointImageVideoTokenizer.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        
        if config is None:
            # You'll need to create this function or a default config dict
            from .diffusion_renderer_config import get_inverse_renderer_config
            config = get_inverse_renderer_config()
        
        self.config = config
        net_config = config.get('net', {})
        scheduler_config = config.get('scheduler', {})
        
        self.scheduler = CleanEDMEulerScheduler(**scheduler_config)
        self.conditioner = CleanConditioner()
        self.net = CleanDiffusionRendererGeneralDIT(**net_config)
        
        # The VAE will be assigned by the pipeline.
        self.vae = None

        self.logvar = torch.nn.Sequential(
            FourierFeaturesPlaceholder(num_channels=128),
            torch.nn.Linear(128, 1, bias=False)
        )
        
        model_type = config.get('model_type', 'inverse')
        if model_type == 'inverse':
            self.condition_keys = config.get('condition_keys', ["image", "rgb"]) # Added 'rgb' for compatibility
        else:
            self.condition_keys = config.get('condition_keys', ["depth", "normal", "basecolor", "roughness", "metallic"])
        
        self.condition_drop_rate = config.get('condition_drop_rate', 0.0)
        self.append_condition_mask = config.get('append_condition_mask', True)
        self.input_data_key = config.get('input_data_key', "video")
        
        self.tokenizer = None
        self.eval()

    # ... (_get_default_config, _get_tensor_kwargs, forward, load_checkpoint can remain the same) ...

    def encode(self, x: Tensor) -> Tensor:
        """
        VAE encode pass-through. The official JointImageVideoTokenizer handles batching and T=1 logic.
        """
        print(f"[Model] VAE encode input: {x.shape}")
        if self.vae is None: raise RuntimeError("VAE not initialized in model.")
        
        # The official VAE expects (B, C, T, H, W) directly. No looping needed.
        if x.ndim != 5:
            raise ValueError(f"Model encode expects a 5D tensor (B,C,T,H,W), but got {x.ndim}D.")
            
        encoded = self.vae.encode(x)
        print(f"[Model] VAE encode output: {encoded.shape}")
        return encoded
        
    def decode(self, x: Tensor) -> Tensor:
        """
        VAE decode pass-through. The official JointImageVideoTokenizer handles everything.
        """
        print(f"[Model] VAE decode input: {x.shape}")
        if self.vae is None: raise RuntimeError("VAE not initialized in model.")

        # The official VAE expects (B, latent_C, latent_T, latent_H, latent_W) directly.
        if x.ndim != 5:
             raise ValueError(f"Model decode expects a 5D latent (B,C,T,H,W), but got {x.ndim}D.")
             
        decoded = self.vae.decode(x)
        print(f"[Model] VAE decode output: {decoded.shape}")
        # The official VAE decode returns (B, C, T, H, W)
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
        """Prepare latent conditions using the official VAE's properties."""
        if self.vae is None: raise RuntimeError("VAE not initialized in model.")
        if condition_keys is None:
            condition_keys = self.condition_keys

        # Determine latent shape dynamically using the VAE's own methods
        if latent_shape is None:
            found_shape = False
            for key in condition_keys:
                if key in data_batch:
                    B, C, T, H, W = data_batch[key].shape
                    latent_C = self.vae.latent_ch
                    latent_T = self.vae.get_latent_num_frames(T)
                    latent_H = H // self.vae.spatial_compression_factor
                    latent_W = W // self.vae.spatial_compression_factor
                    latent_shape = (B, latent_C, latent_T, latent_H, latent_W)
                    found_shape = True
                    break
            if not found_shape:
                 raise ValueError(f"Could not determine latent shape. None of condition_keys {condition_keys} found in data_batch.")
                    
        if append_condition_mask:
            latent_mask_shape = (latent_shape[0], 1, latent_shape[2], latent_shape[3], latent_shape[4])
            
        if dtype is None:
            dtype = next(iter(data_batch.values())).dtype
        if device is None:
            device = next(iter(data_batch.values())).device
            
        latent_condition_list = []
        
        for cond_key in condition_keys:
            # Handle both 'rgb' and 'image' as potential keys for the same condition
            actual_key = cond_key if cond_key in data_batch else ('rgb' if 'rgb' in data_batch and cond_key == 'image' else None)
            
            is_condition_dropped = condition_drop_rate > 0 and np.random.rand() < condition_drop_rate
            is_condition_skipped = actual_key is None
            
            if is_condition_dropped or is_condition_skipped:
                condition_state = torch.zeros(latent_shape, dtype=dtype, device=device)
                latent_condition_list.append(condition_state)
                if append_condition_mask:
                    condition_mask = torch.zeros(latent_mask_shape, dtype=dtype, device=device)
                    latent_condition_list.append(condition_mask)
            else:
                condition_state = data_batch[actual_key].to(device=device, dtype=dtype)
                condition_state = self.encode(condition_state).contiguous()
                latent_condition_list.append(condition_state)
                if append_condition_mask:
                    condition_mask = torch.ones(latent_mask_shape, dtype=dtype, device=device)
                    latent_condition_list.append(condition_mask)
                    
        return torch.cat(latent_condition_list, dim=1)
        
    def _get_conditions(self, data_batch: Dict, is_negative_prompt: bool = False):
        """Get conditions from data batch (extracted from original)"""
        
        self.input_data_key = "video"
        # Find the first available key
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
        """Generate samples from a data batch using diffusion sampling"""
        
        with torch.no_grad():
            torch.manual_seed(seed)
            condition, uncondition = self._get_conditions(data_batch, is_negative_prompt)
            
            tensor_kwargs = self._get_tensor_kwargs()
            self.scheduler.set_timesteps(num_steps, device=tensor_kwargs["device"])
            
            xt = torch.randn(size=(n_sample,) + tuple(state_shape), **tensor_kwargs) * self.scheduler.init_noise_sigma
            print(f"[Model] Initialized noise shape: {xt.shape} with state_shape: {state_shape}")
            
            for i, t in enumerate(self.scheduler.timesteps):
                xt = xt.to(**tensor_kwargs)
                xt_scaled = self.scheduler.scale_model_input(xt, timestep=t)
                
                t = t.to(**tensor_kwargs)
                print(f"Step {i+1}/{num_steps}: sigma={float(t):.4f}")
                
                net_output_cond = self.net(x=xt_scaled, timesteps=t, **condition.to_dict())
                net_output = net_output_cond
                
                if guidance > 0:
                    net_output_uncond = self.net(x=xt_scaled, timesteps=t, **uncondition.to_dict())
                    net_output = net_output_cond + guidance * (net_output_cond - net_output_uncond)
                    
                xt = self.scheduler.step(net_output, t, xt).prev_sample
                
            samples = xt
            print(f"[Model] Final samples shape: {samples.shape}")
            
            return samples
