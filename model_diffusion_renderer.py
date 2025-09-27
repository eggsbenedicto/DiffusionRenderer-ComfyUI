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
        self.sigmas = None
        self.timesteps = None
        self.init_noise_sigma = sigma_max  # CRITICAL: Set this!
        self.prediction_type = kwargs.get('prediction_type', 'v_prediction')
        
    def set_timesteps(self, num_steps, device=None):
        # Create sigma schedule
        sigmas = torch.linspace(np.log(self.sigma_max), np.log(self.sigma_min), num_steps, device=device)
        sigmas = torch.exp(sigmas)
        
        # Append zero for final step
        self.sigmas = torch.cat([sigmas, torch.tensor([0.0], device=device)])
        
        # For EDM, timesteps ARE the sigmas (not indices)
        self.timesteps = self.sigmas[:-1]
        
        # Set initial noise sigma
        self.init_noise_sigma = self.sigmas[0]
        
    def scale_model_input(self, sample, timestep):
        """
        EDM scaling: x_scaled = x / sqrt(sigma^2 + sigma_data^2)
        """
        sigma = timestep  # In EDM, timestep IS the sigma value
        c_in = 1.0 / torch.sqrt(sigma**2 + self.sigma_data**2)
        return sample * c_in
        
    def step(self, model_output, timestep, sample):
        """
        EDM Euler step with v-prediction support exactly matching diffusers
        """
        sigma = timestep
    
        # Get next sigma
        indices = (self.timesteps == sigma).nonzero(as_tuple=True)[0]
        if len(indices) == 0:
            raise ValueError(f"Timestep {sigma} not found in schedule")
        idx = indices[0].item()
        sigma_next = self.sigmas[idx + 1]
    
        if self.prediction_type == "v_prediction":
            # CRITICAL: For v-prediction, c_out is NEGATIVE!
            sigma_data = self.sigma_data
            c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
            c_out = -sigma * sigma_data / ((sigma**2 + sigma_data**2) ** 0.5)
        else:
            # Epsilon prediction
            c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
            c_out = sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)
    
        # Apply preconditioning
        denoised = c_skip * sample + c_out * model_output
    
        # Euler step
        d = (sample - denoised) / sigma
        dt = sigma_next - sigma
        sample_next = sample + d * dt
    
        class StepOutput:
            def __init__(self, prev_sample):
                self.prev_sample = prev_sample
            
        return StepOutput(sample_next)

class CleanCondition:
    def __init__(self, **kwargs): 
        self.data = kwargs
    
    def to_dict(self): 
        return self.data

class CleanConditioner:
    def get_condition_uncondition(self, data_batch: Dict) -> Tuple[CleanCondition, CleanCondition]:
        condition_data = {}
        uncondition_data = {}
        
        # Pass through latent_condition and context_index
        for key in ['latent_condition', 'context_index']:
            if key in data_batch:
                condition_data[key] = data_batch[key]
                # For uncondition, keep context_index but zero out latent_condition
                if key == 'latent_condition':
                    uncondition_data[key] = torch.zeros_like(data_batch[key])
                elif key == 'context_index':
                    # Keep context_index for uncondition too - we still need to know which output to generate
                    uncondition_data[key] = data_batch[key]
                    
        return CleanCondition(**condition_data), CleanCondition(**uncondition_data)

class CleanDiffusionRendererModel(nn.Module):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        
        if config is None: 
            config = get_inverse_renderer_config()
        
        self.config = config
        net_config = config.get('net', {})
        scheduler_config = config.get('scheduler', {})

        # Remove prediction_type if present (not used in EDM)
        #scheduler_config.pop('prediction_type', None)
        
        self.scheduler = CleanEDMEulerScheduler(**scheduler_config)
        self.conditioner = CleanConditioner()
        self.net = CleanDiffusionRendererGeneralDIT(**net_config)
        self.vae = None
        self.logvar = torch.nn.Sequential(
            FourierFeaturesPlaceholder(num_channels=128),
            torch.nn.Linear(128, 1, bias=False)
        )
        
        # Get condition keys from config directly
        self.condition_keys = config.get('condition_keys', ["rgb"])
        self.condition_drop_rate = config.get('condition_drop_rate', 0.0)
        self.append_condition_mask = config.get('append_condition_mask', True)
        self.input_data_key = config.get('input_data_key', "video")
        self.tokenizer = None
        self.sigma_data = config.get('sigma_data', 0.5)
        self.eval()

    def _get_tensor_kwargs(self):
        try:
            param = next(self.parameters())
            return {"device": param.device, "dtype": param.dtype}
        except StopIteration:
            return {"device": torch.device("cuda"), "dtype": torch.bfloat16}
    
    def encode(self, x: Tensor) -> Tensor:
        """Encode input to latent space using float32 for accuracy"""
        # Store original dtype
        orig_dtype = x.dtype
        
        # Convert to float32 for VAE
        x_fp32 = x.to(dtype=torch.float32)
        
        # Encode in float32
        with torch.cuda.amp.autocast(enabled=False):  # Disable autocast
            encoded = self.vae.encode(x_fp32)
        
        # Convert back to original dtype for diffusion model
        return encoded.to(dtype=orig_dtype)

    def decode(self, x: Tensor) -> Tensor:
        """Decode from latent space using float32 for accuracy"""
        # Store original dtype
        orig_dtype = x.dtype
        
        # Convert to float32 for VAE
        x_fp32 = x.to(dtype=torch.float32)
        
        # Decode in float32
        with torch.cuda.amp.autocast(enabled=False):  # Disable autocast
            decoded = self.vae.decode(x_fp32)
        
        # Keep output in float32 for quality (convert later if needed)
        return decoded
        
    def prepare_diffusion_renderer_latent_conditions(
        self, data_batch: Dict[str, Tensor], 
        condition_keys: list = None,
        condition_drop_rate: float = None,
        append_condition_mask: bool = None,
        **kwargs
    ) -> Tensor:
        """
        Prepare latent conditions for diffusion renderer.
        For inverse renderer: encodes RGB and appends mask
        For forward renderer: encodes all G-buffers and environment maps with masks
        """
        if self.vae is None: 
            raise RuntimeError("VAE not initialized in model.")
        
        if condition_keys is None: 
            condition_keys = self.condition_keys
        if condition_drop_rate is None:
            condition_drop_rate = self.condition_drop_rate
        if append_condition_mask is None:
            append_condition_mask = self.append_condition_mask

        # Determine latent shape from first available condition
        latent_shape = None
        for key in condition_keys:
            # Check for the key directly or common aliases
            check_keys = [key]
            if key == "rgb":
                check_keys.append("image")  # Also check for 'image' if looking for 'rgb'
            
            for check_key in check_keys:
                if check_key in data_batch:
                    B, C, T, H, W = data_batch[check_key].shape
                    latent_C = self.vae.latent_ch
                    latent_T = self.vae.get_latent_num_frames(T)
                    latent_H = H // self.vae.spatial_compression_factor
                    latent_W = W // self.vae.spatial_compression_factor
                    latent_shape = (B, latent_C, latent_T, latent_H, latent_W)
                    break
            if latent_shape is not None:
                break
                
        if latent_shape is None:
            raise ValueError(f"Could not determine latent shape from keys {condition_keys}. Available keys: {list(data_batch.keys())}")
                    
        latent_condition_list = []
        device = data_batch[self.input_data_key].device
        dtype = data_batch[self.input_data_key].dtype
        
        for cond_key in condition_keys:
            # For inverse renderer, handle 'rgb' key properly
            if cond_key == "rgb" and cond_key not in data_batch and "image" in data_batch:
                actual_key = "image"
            elif cond_key in data_batch:
                actual_key = cond_key
            else:
                actual_key = None
            
            # Check if we should drop this condition (training only)
            is_dropped = condition_drop_rate > 0 and np.random.rand() < condition_drop_rate
            
            if actual_key is None or is_dropped:
                # Add zero condition
                condition_state = torch.zeros(latent_shape, dtype=dtype, device=device)
                latent_condition_list.append(condition_state)
                if append_condition_mask:
                    mask_shape = (latent_shape[0], 1, *latent_shape[2:])
                    latent_condition_list.append(torch.zeros(mask_shape, dtype=dtype, device=device))
            else:
                # Encode actual condition
                condition_state = data_batch[actual_key].to(device=device, dtype=dtype)
                condition_state = self.encode(condition_state).contiguous()
                latent_condition_list.append(condition_state)
                if append_condition_mask:
                    mask_shape = (latent_shape[0], 1, *latent_shape[2:])
                    latent_condition_list.append(torch.ones(mask_shape, dtype=dtype, device=device))
                    
        return torch.cat(latent_condition_list, dim=1)
        
    def _get_conditions(self, data_batch: Dict, is_negative_prompt: bool = False):
        """
        Get condition and uncondition tensors for the diffusion process.
        This prepares the latent conditions and passes through context_index.
        """
        # Find the input data key (what we're conditioning on)
        for key in ['rgb', 'basecolor', 'normal', 'depth', 'roughness', 'metallic', 'image', 'video']:
            if key in data_batch:
                # Don't override input_data_key if it's already set correctly
                if key in ['rgb', 'image', 'video']:
                    # These are valid input keys
                    pass
                break
        
        # Prepare latent conditions (encoded RGB/maps + masks)
        with torch.no_grad():
            latent_condition = self.prepare_diffusion_renderer_latent_conditions(
                data_batch, 
                condition_keys=self.condition_keys,
                condition_drop_rate=0.0,  # No dropping during inference
                append_condition_mask=self.append_condition_mask
            )
        
        data_batch["latent_condition"] = latent_condition
        
        # The conditioner will pass through both latent_condition and context_index
        return self.conditioner.get_condition_uncondition(data_batch)
        
    def generate_samples_from_batch(
        self, data_batch: Dict, 
        guidance: float = 0.0, 
        seed: int = 1000,
        state_shape: Tuple = None, 
        num_steps: int = 15, 
        is_negative_prompt: bool = False,
        **kwargs
    ) -> Tensor:
        """
        Generate samples using the diffusion process.
        
        Args:
            data_batch: Dictionary containing input conditions and context_index
            guidance: Classifier-free guidance scale (0 = no guidance)
            seed: Random seed for noise initialization
            state_shape: Shape of the latent state [C, T, H, W]
            num_steps: Number of diffusion steps
            is_negative_prompt: Whether to use negative prompting
        
        Returns:
            Generated latent samples
        """
        with torch.no_grad():
            torch.manual_seed(seed)
            
            # Get conditions (this includes both latent_condition and context_index)
            condition, uncondition = self._get_conditions(data_batch, is_negative_prompt)
            
            tensor_kwargs = self._get_tensor_kwargs()
            self.scheduler.set_timesteps(num_steps, device=tensor_kwargs["device"])

            print(f"[Debug] Scheduler timesteps: {self.scheduler.timesteps[:3]}...")
            print(f"[Debug] Init noise sigma: {self.scheduler.init_noise_sigma}")
            
            # Initialize noise
            xt = torch.randn(size=(1, *state_shape), **tensor_kwargs) * self.scheduler.init_noise_sigma
            
            print(f"[Debug] Initial noise: mean={xt.mean():.3f}, std={xt.std():.3f}")

            # Diffusion loop
            for i, t in enumerate(self.scheduler.timesteps):
                xt_scaled = self.scheduler.scale_model_input(xt, timestep=t)

                if i < 3:
                    print(f"[Debug] Step {i}: sigma={t:.3f}, xt_scaled range=[{xt_scaled.min():.2f}, {xt_scaled.max():.2f}]")
                
                # Model prediction with conditions
                # The condition dict contains both latent_condition and context_index
                net_output_cond = self.net(
                    x=xt_scaled, 
                    timesteps=t, 
                    **condition.to_dict()  # Unpacks latent_condition and context_index
                )

                if i < 3:
                    print(f"[Debug] Step {i}: net_output range=[{net_output_cond.min():.2f}, {net_output_cond.max():.2f}]")

                # Apply guidance if requested
                if guidance > 0:
                    net_output_uncond = self.net(
                        x=xt_scaled, 
                        timesteps=t, 
                        **uncondition.to_dict()
                    )
                    net_output = net_output_cond + guidance * (net_output_cond - net_output_uncond)
                else:
                    net_output = net_output_cond
                    
                # Scheduler step
                xt = self.scheduler.step(net_output, t, xt).prev_sample

                if i < 3:
                    print(f"[Debug] Step {i}: after step xt range=[{xt.min():.2f}, {xt.max():.2f}]")

            return xt