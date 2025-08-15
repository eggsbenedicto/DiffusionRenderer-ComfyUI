import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Union, Optional, Any
from torch import Tensor
from .CleanGeneralDIT import CleanDiffusionRendererGeneralDIT
from .diffusion_renderer_config import get_inverse_renderer_config

class FourierFeaturesPlaceholder(nn.Module):
    def __init__(self, num_channels, **kwargs):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels))
        self.register_buffer("phases", torch.randn(num_channels))
    def forward(self, x): return x

class CleanEDMEulerScheduler:
    def __init__(self, sigma_max=80.0, sigma_min=0.02, sigma_data=0.5, **kwargs):
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data
        self.sigmas, self.timesteps, self.current_step = None, None, 0
        
    def set_timesteps(self, num_steps, device=None):
        # Timesteps should be calculated in float32
        sigmas = torch.logspace(np.log10(self.sigma_max), np.log10(self.sigma_min), num_steps, device=device, dtype=torch.float32)
        self.sigmas = torch.cat([sigmas, torch.tensor([0.0], device=device, dtype=torch.float32)])
        self.timesteps = self.sigmas[:-1]
        self.current_step = 0
        
    def scale_model_input(self, sample, timestep):
        # --- FIX: Perform calculations in float32 for numerical stability ---
        # Store original dtype
        orig_dtype = sample.dtype
        
        # Convert inputs to float32
        sample_fp32 = sample.to(torch.float32)
        timestep_fp32 = timestep.to(torch.float32)
        
        # Calculate c_in using float32
        c_in = 1 / torch.sqrt(timestep_fp32**2 + self.sigma_data**2)
        scaled_sample = sample_fp32 * c_in
        
        # Cast back to the original dtype before returning
        return scaled_sample.to(orig_dtype)
        
    def step(self, model_output, timestep, sample):
        if self.sigmas is None or self.current_step >= len(self.sigmas) - 1:
            raise RuntimeError("Scheduler not initialized or timesteps exhausted")
        
        # --- FIX: Perform all intermediate calculations in float32 ---
        # Store original dtype
        orig_dtype = sample.dtype
        
        # Convert all inputs to float32
        model_output_fp32 = model_output.to(torch.float32)
        timestep_fp32 = timestep.to(torch.float32)
        sample_fp32 = sample.to(torch.float32)
        
        # Define current and next sigma in float32
        sigma = timestep_fp32
        sigma_next = self.sigmas[self.current_step + 1]

        # Calculate preconditioning constants in float32
        sigma_data = self.sigma_data
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = (sigma * sigma_data) / torch.sqrt(sigma**2 + sigma_data**2)

        # Predict the denoised sample in float32
        denoised = c_skip * sample_fp32 + c_out * model_output_fp32
        
        # Perform the Euler step in float32
        derivative = (sample_fp32 - denoised) / sigma
        dt = sigma_next - sigma
        prev_sample = sample_fp32 + derivative * dt
        
        self.current_step += 1
        
        # Return the result in the expected class format, cast back to the original dtype
        class StepResult:
            def __init__(self, prev_sample): self.prev_sample = prev_sample
            
        return StepResult(prev_sample.to(orig_dtype))

class CleanCondition:
    def __init__(self, **kwargs): self.data = kwargs
    def to_dict(self): return self.data

class CleanConditioner:
    def get_condition_uncondition(self, data_batch: Dict) -> Tuple[CleanCondition, CleanCondition]:
        condition_data, uncondition_data = {}, {}
        for key in ['latent_condition', 'context_index']:
            if key in data_batch:
                condition_data[key] = data_batch[key]
                if key == 'latent_condition' or key == 'context_index':
                    uncondition_data[key] = torch.zeros_like(data_batch[key])
        return CleanCondition(**condition_data), CleanCondition(**uncondition_data)

class CleanDiffusionRendererModel(nn.Module):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        
        if config is None: config = get_inverse_renderer_config()
        
        self.config = config
        net_config = config.get('net', {})
        scheduler_config = config.get('scheduler', {})

        scheduler_config.pop('prediction_type', None)
        
        self.scheduler = CleanEDMEulerScheduler(**scheduler_config)
        self.conditioner = CleanConditioner()
        self.net = CleanDiffusionRendererGeneralDIT(**net_config)
        self.vae = None
        self.logvar = torch.nn.Sequential(
            FourierFeaturesPlaceholder(num_channels=128),
            torch.nn.Linear(128, 1, bias=False)
        )
        
        model_type = config.get('model_type', 'inverse')
        if model_type == 'inverse':
            self.condition_keys = config.get('condition_keys', ["image", "rgb"])
        else:
            self.condition_keys = config.get('condition_keys', ["depth", "normal", "basecolor", "roughness", "metallic"])
        
        self.condition_drop_rate = config.get('condition_drop_rate', 0.0)
        self.append_condition_mask = config.get('append_condition_mask', True)
        self.input_data_key = config.get('input_data_key', "video")
        self.tokenizer = None
        self.eval()

    def _get_tensor_kwargs(self):
        try:
            param = next(self.parameters())
            return {"device": param.device, "dtype": param.dtype}
        except StopIteration:
            return {"device": torch.device("cuda"), "dtype": torch.bfloat16}
    
    def encode(self, x: Tensor) -> Tensor:
        """
        VAE encode pass-through, now with correct EDM scaling.
        """
        if self.vae is None: raise RuntimeError("VAE not initialized in model.")
        if x.ndim != 5: raise ValueError(f"Model encode expects a 5D tensor (B,C,T,H,W), but got {x.ndim}D.")
        
        # Encode and then scale by sigma_data, as done in the official implementation.
        return self.vae.encode(x) * self.scheduler.sigma_data
        
    def decode(self, x: Tensor) -> Tensor:
        """
        VAE decode pass-through, now with correct EDM scaling.
        """
        if self.vae is None: raise RuntimeError("VAE not initialized in model.")
        if x.ndim != 5: raise ValueError(f"Model decode expects a 5D latent (B,C,T,H,W), but got {x.ndim}D.")
        
        # Scale by 1/sigma_data before decoding, as done in the official implementation.
        return self.vae.decode(x / self.scheduler.sigma_data)
        
    def prepare_diffusion_renderer_latent_conditions(
        self, data_batch: Dict[str, Tensor], condition_keys: list = None, **kwargs
    ) -> Tensor:
        if self.vae is None: raise RuntimeError("VAE not initialized in model.")
        if condition_keys is None: condition_keys = self.condition_keys

        latent_shape = None
        for key in condition_keys:
            if key in data_batch:
                B, C, T, H, W = data_batch[key].shape
                latent_C = self.vae.latent_ch
                latent_T = self.vae.get_latent_num_frames(T)
                latent_H = H // self.vae.spatial_compression_factor
                latent_W = W // self.vae.spatial_compression_factor
                latent_shape = (B, latent_C, latent_T, latent_H, latent_W)
                break
        if latent_shape is None:
             raise ValueError(f"Could not determine latent shape from keys {condition_keys}.")
                    
        latent_condition_list = []
        append_condition_mask = self.append_condition_mask
        
        for cond_key in condition_keys:
            actual_key = cond_key if cond_key in data_batch else ('rgb' if 'rgb' in data_batch and cond_key == 'image' else None)
            
            if actual_key is None:
                condition_state = torch.zeros(latent_shape, dtype=data_batch[self.input_data_key].dtype, device=data_batch[self.input_data_key].device)
                latent_condition_list.append(condition_state)
                if append_condition_mask:
                    mask_shape = (latent_shape[0], 1, *latent_shape[2:])
                    latent_condition_list.append(torch.zeros(mask_shape, dtype=condition_state.dtype, device=condition_state.device))
            else:
                condition_state = data_batch[actual_key]
                condition_state = self.encode(condition_state).contiguous()
                latent_condition_list.append(condition_state)
                if append_condition_mask:
                    mask_shape = (latent_shape[0], 1, *latent_shape[2:])
                    latent_condition_list.append(torch.ones(mask_shape, dtype=condition_state.dtype, device=condition_state.device))
                    
        return torch.cat(latent_condition_list, dim=1)
        
    def _get_conditions(self, data_batch: Dict, is_negative_prompt: bool = False):
        for key in ['rgb', 'basecolor', 'normal', 'depth', 'roughness', 'metallic', 'image']:
            if key in data_batch:
                self.input_data_key = key
                break
        
        with torch.no_grad():
            latent_condition = self.prepare_diffusion_renderer_latent_conditions(
                data_batch, self.condition_keys)
        data_batch["latent_condition"] = latent_condition
        return self.conditioner.get_condition_uncondition(data_batch)
        
    def generate_samples_from_batch(
        self, data_batch: Dict, guidance: float = 0.0, seed: int = 1000,
        state_shape: Tuple = None, num_steps: int = 15, **kwargs
    ) -> Tensor:
        with torch.no_grad():
            torch.manual_seed(seed)
            condition, uncondition = self._get_conditions(data_batch)
            
            tensor_kwargs = self._get_tensor_kwargs()
            self.scheduler.set_timesteps(num_steps, device=tensor_kwargs["device"])
            
            xt = torch.randn(size=(1, *state_shape), **tensor_kwargs) * self.scheduler.sigmas[0] # Use initial sigma
            
            for i, t in enumerate(self.scheduler.timesteps):
                xt_scaled = self.scheduler.scale_model_input(xt, timestep=t)
                
                net_output_cond = self.net(x=xt_scaled, timesteps=t, **condition.to_dict())
                net_output = net_output_cond
                
                if guidance > 0:
                    net_output_uncond = self.net(x=xt_scaled, timesteps=t, **uncondition.to_dict())
                    net_output = net_output_cond + guidance * (net_output_cond - net_output_uncond)
                    
                xt = self.scheduler.step(net_output, t, xt).prev_sample
            return xt