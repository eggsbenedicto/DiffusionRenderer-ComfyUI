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
    """
    An intelligent EDM-style scheduler that can initialize its sigma schedule
    from a beta-based (DDPM/LDM) configuration.
    """
    def __init__(self, sigma_max: float = 80.0, sigma_min: float = 0.02, sigma_data: float = 0.5,
                 beta_start: float = 0.00085, beta_end: float = 0.012, beta_schedule: str = "scaled_linear",
                 num_train_timesteps: int = 1000, **kwargs):
        # Store all parameters, including beta-related ones.
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.num_train_timesteps = num_train_timesteps
        
        # This will hold the final sigmas for sampling.
        self.sigmas = None
        self.timesteps = None
        self.current_step = 0
        
    def set_timesteps(self, num_steps: int, device: Optional[torch.device] = None):
        """
        Calculates the sigma schedule. If beta schedule is 'scaled_linear',
        it computes sigmas from betas. Otherwise, it uses a log-linear schedule.
        """
        if self.beta_schedule == "scaled_linear":
            # Generate betas
            betas = torch.linspace(self.beta_start**0.5, self.beta_end**0.5, self.num_train_timesteps, dtype=torch.float32) ** 2
            
            # Calculate alphas and cumulative alphas
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            
            # Calculate sigmas from alphas_cumprod (EDM formula)
            sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
            
            # Invert and select a subset for inference
            sigmas = torch.flip(sigmas, [0])
            log_sigmas = torch.log(sigmas)
            
            # Select 'num_steps' sigmas from the full training schedule
            indices = torch.linspace(0, self.num_train_timesteps - 1, num_steps, device=device).long()
            final_sigmas = sigmas[indices].to(device=device, dtype=torch.float32)

        else: # Fallback to the original log-linear schedule
            final_sigmas = torch.logspace(np.log10(self.sigma_max), np.log10(self.sigma_min), num_steps, device=device)
        
        # Add sigma=0 for the final denoising step
        self.sigmas = torch.cat([final_sigmas, torch.tensor([0.0], device=device)])
        self.timesteps = self.sigmas[:-1]
        self.current_step = 0
        
    def scale_model_input(self, sample: Tensor, timestep: Tensor) -> Tensor:
        sigma = timestep
        c_in = 1 / torch.sqrt(sigma**2 + self.sigma_data**2)
        return sample * c_in
        
    def step(self, model_output: Tensor, timestep: Tensor, sample: Tensor, use_heun: bool = False):
        if self.sigmas is None or self.current_step >= len(self.sigmas) - 1:
            raise RuntimeError("Scheduler not properly initialized or step counter out of bounds")
        
        sigma_curr = self.sigmas[self.current_step]
        sigma_next = self.sigmas[self.current_step + 1]
        
        c_out = sigma_curr * self.sigma_data / torch.sqrt(sigma_curr**2 + self.sigma_data**2)
        c_skip = self.sigma_data**2 / (sigma_curr**2 + self.sigma_data**2)
        
        denoised = c_skip * sample + c_out * model_output
        
        if sigma_next == 0:
            prev_sample = denoised
        else:
            derivative = (denoised - sample) / sigma_curr
            prev_sample = sample + (sigma_next - sigma_curr) * derivative
        
        self.current_step += 1
        
        class StepResult:
            def __init__(self, prev_sample): self.prev_sample = prev_sample
        return StepResult(prev_sample)

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
        
        # The new scheduler now accepts all keys, so no popping is needed.
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
        if self.vae is None: raise RuntimeError("VAE not initialized in model.")
        if x.ndim != 5: raise ValueError(f"Model encode expects a 5D tensor (B,C,T,H,W), but got {x.ndim}D.")
        return self.vae.encode(x)
        
    def decode(self, x: Tensor) -> Tensor:
        if self.vae is None: raise RuntimeError("VAE not initialized in model.")
        if x.ndim != 5: raise ValueError(f"Model decode expects a 5D latent (B,C,T,H,W), but got {x.ndim}D.")
        return self.vae.decode(x)
        
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