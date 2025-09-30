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

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Union, Optional, Any, List
from torch import Tensor
from diffusers import EDMEulerScheduler  # Import official scheduler from diffusers
from .CleanGeneralDIT import CleanDiffusionRendererGeneralDIT
from .diffusion_renderer_config import get_inverse_renderer_config, get_forward_renderer_config


class FourierFeaturesPlaceholder(nn.Module):
    """Placeholder for Fourier features module."""
    def __init__(self, num_channels, **kwargs):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels))
        self.register_buffer("phases", torch.randn(num_channels))
    
    def forward(self, x): 
        return x


class CleanCondition:
    """Container for diffusion model conditions."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self):
        """Convert condition to dictionary for unpacking."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def to(self, device=None, dtype=None):
        """Move all tensors to specified device/dtype."""
        new_kwargs = {}
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                if device is not None:
                    value = value.to(device)
                if dtype is not None:
                    value = value.to(dtype)
            new_kwargs[key] = value
        return CleanCondition(**new_kwargs)


class CleanConditioner(nn.Module):
    """Conditioner that handles condition/uncondition pairs for diffusion."""
    def __init__(self):
        super().__init__()
    
    def get_condition_uncondition(self, data_batch):
        """Extract condition and uncondition from data batch.
        
        This handles the latent_condition and context_index from the data batch
        and returns them as condition objects.
        """
        # Extract the prepared latent condition
        latent_condition = data_batch.get('latent_condition')
        context_index = data_batch.get('context_index')
        
        # Create condition with both latent and context
        condition_dict = {}
        if latent_condition is not None:
            condition_dict['latent_condition'] = latent_condition
        if context_index is not None:
            condition_dict['context_index'] = context_index
        
        # Create uncondition (zeros for latent, keep context)
        uncondition_dict = {}
        if latent_condition is not None:
            uncondition_dict['latent_condition'] = torch.zeros_like(latent_condition)
        if context_index is not None:
            uncondition_dict['context_index'] = context_index
        
        return CleanCondition(**condition_dict), CleanCondition(**uncondition_dict)
    
    def get_condition_with_negative_prompt(self, data_batch):
        """Same as get_condition_uncondition for this implementation."""
        return self.get_condition_uncondition(data_batch)


class CleanDiffusionRendererModel(nn.Module):
    """Clean Diffusion Renderer Model using diffusers EDMEulerScheduler.
    
    This model handles both inverse rendering (RGB -> maps) and forward rendering (maps -> RGB)
    using the official diffusers library scheduler for compatibility with model_t2w.py.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        
        if config is None: 
            config = get_inverse_renderer_config()
        
        self.config = config
        net_config = config.get('net', {})
        self.scheduler_config = config.get('scheduler', {})
        
        # Store sigma_data for later use
        self.sigma_data = config.get('sigma_data', 0.5)
        
        # Initialize scheduler as None - will be created lazily
        self._scheduler = None
        
        # Initialize model components
        self.conditioner = CleanConditioner()
        self.net = CleanDiffusionRendererGeneralDIT(**net_config)
        self.vae = None
        self.tokenizer = None
        
        # Logvar module for variance estimation
        self.logvar = torch.nn.Sequential(
            FourierFeaturesPlaceholder(num_channels=128),
            torch.nn.Linear(128, 1, bias=False)
        )
        
        # Configuration from config
        self.condition_keys = config.get('condition_keys', ["rgb"])
        self.condition_drop_rate = config.get('condition_drop_rate', 0.0)
        self.append_condition_mask = config.get('append_condition_mask', True)
        self.input_data_key = config.get('input_data_key', "video")
        
        # Precision settings
        self.precision = config.get('precision', 'float32')
        self.dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16
        }
        
        # State shape for latent dimensions
        self.state_shape = config.get('latent_shape', [16, 8, 88, 160])
    
    def _create_scheduler(self, device=None):
        """Create scheduler with proper device handling."""
        try:
            # Try to create the scheduler normally
            scheduler = EDMEulerScheduler(
                sigma_max=self.scheduler_config.get('sigma_max', 80.0),
                sigma_min=self.scheduler_config.get('sigma_min', 0.02),
                sigma_data=self.sigma_data
            )
        except (NotImplementedError, RuntimeError) as e:
            # If we hit meta tensor issues, create a minimal scheduler
            # and set the configuration manually
            import warnings
            warnings.warn(f"Standard scheduler initialization failed: {e}. Using fallback initialization.")
            
            # Create scheduler with minimal init
            scheduler = EDMEulerScheduler.__new__(EDMEulerScheduler)
            
            # Set config attributes manually
            scheduler.config = {
                'sigma_max': self.scheduler_config.get('sigma_max', 80.0),
                'sigma_min': self.scheduler_config.get('sigma_min', 0.02),
                'sigma_data': self.sigma_data,
                'num_train_timesteps': 1000,
                'prediction_type': 'epsilon',
                'interpolation_type': 'linear',
                'use_karras_sigmas': False,
                'timestep_spacing': 'leading',
                'steps_offset': 0
            }
            
            # Set required attributes
            scheduler.sigma_max = scheduler.config['sigma_max']
            scheduler.sigma_min = scheduler.config['sigma_min'] 
            scheduler.sigma_data = scheduler.config['sigma_data']
            scheduler.num_train_timesteps = scheduler.config['num_train_timesteps']
            scheduler.init_noise_sigma = scheduler.sigma_max
            scheduler.timesteps = None
            scheduler.sigmas = None
            scheduler.num_inference_steps = None
            
            # Set is_scale_input_called flag
            scheduler.is_scale_input_called = False
            
        return scheduler
    
    @property
    def scheduler(self):
        """Lazy initialization of scheduler to avoid meta tensor issues."""
        if self._scheduler is None:
            self._scheduler = self._create_scheduler()
        return self._scheduler
    
    @scheduler.setter
    def scheduler(self, value):
        """Allow setting scheduler externally if needed."""
        self._scheduler = value
    
    def _get_tensor_kwargs(self) -> Dict[str, Any]:
        """Get tensor creation kwargs."""
        return {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'dtype': self.dtype_map.get(self.precision, torch.float32)
        }
    
    def prepare_latent_condition(
        self,
        data_batch: Dict[str, Tensor],
        condition_keys: Optional[List[str]] = None,
        condition_drop_rate: Optional[float] = None,
        append_condition_mask: Optional[bool] = None,
        mode: str = "train"
    ) -> Tensor:
        """Prepare latent conditions from data batch.
        
        Args:
            data_batch: Dictionary containing condition tensors
            condition_keys: Keys to extract from data_batch
            condition_drop_rate: Dropout rate for conditions during training
            append_condition_mask: Whether to append binary mask channel
            mode: "train" or "inference"
        
        Returns:
            Concatenated latent condition tensor
        """
        if condition_keys is None:
            condition_keys = self.condition_keys
        if condition_drop_rate is None:
            condition_drop_rate = self.condition_drop_rate if mode == "train" else 0.0
        if append_condition_mask is None:
            append_condition_mask = self.append_condition_mask
        
        # Get the first available condition to determine shape
        first_cond = None
        for key in condition_keys:
            if key in data_batch:
                first_cond = data_batch[key]
                break
        
        if first_cond is None:
            raise ValueError(f"No condition found in data_batch for keys: {condition_keys}")
        
        B, C, T, H, W = first_cond.shape
        device = first_cond.device
        dtype = first_cond.dtype
        
        # Calculate latent shape (with temporal and spatial compression)
        # For VAE with spatial_compression_factor=8 and temporal=1 for images
        latent_t = T // 8 + 1 if T > 1 else 1  # Temporal compression for videos
        latent_h = H // 8      # Spatial compression
        latent_w = W // 8      # Spatial compression
        latent_shape = (B, 16, latent_t, latent_h, latent_w)  # 16 latent channels
        
        latent_condition_list = []
        
        for cond_key in condition_keys:
            # Check if condition should be dropped (training) or is missing
            is_condition_dropped = mode == "train" and condition_drop_rate > 0 and np.random.rand() < condition_drop_rate
            is_condition_skipped = cond_key not in data_batch
            
            if is_condition_dropped or is_condition_skipped:
                # Create zero tensor for dropped/missing condition
                condition_state = torch.zeros(latent_shape, dtype=dtype, device=device)
                latent_condition_list.append(condition_state)
                
                # Add zero mask if needed
                if append_condition_mask:
                    condition_mask = torch.zeros((B, 1, latent_t, latent_h, latent_w), dtype=dtype, device=device)
                    latent_condition_list.append(condition_mask)
            else:
                # Valid condition - encode it through VAE
                condition_state = data_batch[cond_key].to(device=device, dtype=dtype)
                
                # CRITICAL: Use the encode method which properly uses VAE
                with torch.no_grad():
                    condition_state = self.encode(condition_state).contiguous()
                
                # The encoded state should already be the right shape from VAE
                latent_condition_list.append(condition_state)
                
                # Add ones mask if needed to indicate valid condition
                if append_condition_mask:
                    condition_mask = torch.ones((B, 1, latent_t, latent_h, latent_w), dtype=dtype, device=device)
                    latent_condition_list.append(condition_mask)
        
        # Concatenate all conditions
        if latent_condition_list:
            latent_condition = torch.cat(latent_condition_list, dim=1)
        else:
            latent_condition = torch.zeros(latent_shape, device=device, dtype=dtype)
        
        return latent_condition
    
    def prepare_diffusion_renderer_latent_conditions(
        self,
        data_batch: Dict[str, Tensor],
        condition_keys: Optional[List[str]] = None,
        condition_drop_rate: Optional[float] = None,
        append_condition_mask: Optional[bool] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        latent_shape: Optional[Tuple[int, ...]] = None,
        mode: str = "train"
    ) -> Tensor:
        """Prepare diffusion renderer latent conditions.
        
        This is an alias for prepare_latent_condition with additional parameters
        for compatibility with the original implementation.
        
        Args:
            data_batch: Dictionary containing condition tensors
            condition_keys: Keys to extract from data_batch
            condition_drop_rate: Dropout rate for conditions during training
            append_condition_mask: Whether to append binary mask channel
            dtype: Target dtype for conditions
            device: Target device for conditions
            latent_shape: Expected latent shape (used for validation)
            mode: "train" or "inference"
        
        Returns:
            Concatenated latent condition tensor
        """
        # Prepare conditions using the main method
        latent_condition = self.prepare_latent_condition(
            data_batch=data_batch,
            condition_keys=condition_keys,
            condition_drop_rate=condition_drop_rate,
            append_condition_mask=append_condition_mask,
            mode=mode
        )
        
        # Move to specified device/dtype if provided
        if device is not None:
            latent_condition = latent_condition.to(device)
        if dtype is not None:
            latent_condition = latent_condition.to(dtype)
        
        return latent_condition
    
    def _get_conditions(
        self,
        data_batch: Dict[str, Tensor],
        is_negative_prompt: bool = False
    ) -> Tuple[CleanCondition, CleanCondition]:
        """Prepare conditions and unconditions for classifier-free guidance.
        
        Args:
            data_batch: Input data batch
            is_negative_prompt: Whether to use negative prompts
        
        Returns:
            Tuple of (condition, uncondition) CleanCondition objects
        """
        # Find input data key if not already set in data_batch
        if self.input_data_key not in data_batch:
            for key in ['video', 'rgb', 'basecolor', 'normal', 'depth', 'roughness', 'metallic', 'image']:
                if key in data_batch:
                    data_batch[self.input_data_key] = data_batch[key]
                    break
        
        # Get raw state for dtype and device info
        raw_state = data_batch.get(self.input_data_key)
        if raw_state is not None:
            dtype = raw_state.dtype
            device = raw_state.device
        else:
            dtype = torch.float32
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Prepare latent condition using the renderer-specific method
        with torch.no_grad():
            latent_condition = self.prepare_diffusion_renderer_latent_conditions(
                data_batch,
                condition_keys=self.condition_keys,
                condition_drop_rate=0,  # No dropout during inference
                append_condition_mask=self.append_condition_mask,
                dtype=dtype,
                device=device,
                latent_shape=None,
                mode="inference"
            )
        
        # Store latent condition in data_batch for conditioner
        data_batch["latent_condition"] = latent_condition
        
        # Use the conditioner to get condition/uncondition pair
        # This properly handles both latent_condition and context_index
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)
        
        return condition, uncondition
    
    @torch.no_grad()
    def sample(
        self,
        data_batch: Dict[str, Tensor],
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Optional[Tuple[int, ...]] = None,
        n_sample: int = 1,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        **kwargs
    ) -> Tensor:
        """Generate samples using EDM Euler sampling with classifier-free guidance.
        
        This method implements the complete diffusion sampling loop using the
        official diffusers EDMEulerScheduler for compatibility with model_t2w.py.
        
        Args:
            data_batch: Input data containing conditions
            guidance: Classifier-free guidance scale
            seed: Random seed for reproducibility
            state_shape: Shape of latent state tensor
            n_sample: Number of samples to generate
            is_negative_prompt: Whether to use negative prompts
            num_steps: Number of denoising steps
            **kwargs: Additional arguments
        
        Returns:
            Generated samples tensor
        """
        # Set random seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Use provided state shape or default
        if state_shape is None:
            state_shape = self.state_shape
        
        # Get device and dtype
        tensor_kwargs = self._get_tensor_kwargs()
        
        # Ensure scheduler is properly initialized
        # If scheduler was created with meta tensors, recreate it
        if self._scheduler is not None:
            try:
                # Test if scheduler has proper tensors
                if hasattr(self._scheduler, 'sigmas') and self._scheduler.sigmas is not None:
                    _ = self._scheduler.sigmas.device
            except (RuntimeError, NotImplementedError):
                # Scheduler has meta tensors, recreate it
                self._scheduler = None
        
        # Prepare conditions for classifier-free guidance
        condition, uncondition = self._get_conditions(data_batch, is_negative_prompt)
        
        # Move conditions to correct device/dtype
        condition = condition.to(
            device=tensor_kwargs['device'],
            dtype=tensor_kwargs['dtype']
        )
        uncondition = uncondition.to(
            device=tensor_kwargs['device'],
            dtype=tensor_kwargs['dtype']
        )
        
        # Set timesteps for the scheduler
        self.scheduler.set_timesteps(num_steps, device=tensor_kwargs['device'])
        
        # Initialize noise with correct sigma scaling
        xt = torch.randn(
            size=(n_sample, *state_shape),
            **tensor_kwargs
        ) * self.scheduler.init_noise_sigma
        
        # Denoising loop
        for i, t in enumerate(self.scheduler.timesteps):
            # Ensure timestep is on correct device
            t = t.to(tensor_kwargs['device'])
            
            # Scale model input according to EDM formulation
            xt_scaled = self.scheduler.scale_model_input(xt, timestep=t)
            
            # Predict noise with condition
            net_output_cond = self.net(
                x=xt_scaled,
                timesteps=t,
                **condition.to_dict()
            )
            
            # Apply classifier-free guidance if requested
            if guidance > 0:
                # Predict noise without condition
                net_output_uncond = self.net(
                    x=xt_scaled,
                    timesteps=t,
                    **uncondition.to_dict()
                )
                # Combine predictions with guidance
                net_output = net_output_cond + guidance * (net_output_cond - net_output_uncond)
            else:
                net_output = net_output_cond
            
            # Perform denoising step
            xt = self.scheduler.step(net_output, t, xt).prev_sample
        
        # Return final denoised samples
        return xt
    
    def generate_samples_from_batch(
        self,
        data_batch: Dict[str, Tensor],
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Optional[Tuple[int, ...]] = None,
        n_sample: int = 1,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        **kwargs
    ) -> Tensor:
        """Generate samples from a data batch using diffusion sampling.
        
        This function generates samples from either image or video data batches using diffusion sampling.
        It handles both conditional and unconditional generation with classifier-free guidance.
        
        This method is a wrapper around the sample() method for compatibility with existing code.
        
        Args:
            data_batch (dict): Raw data batch from the training data loader
            guidance (float, optional): Classifier-free guidance weight. Defaults to 1.5.
            seed (int, optional): Random seed for reproducibility. Defaults to 1.
            state_shape (tuple | None, optional): Shape of the state tensor. Uses self.state_shape if None. Defaults to None.
            n_sample (int | None, optional): Number of samples to generate. Defaults to 1.
            is_negative_prompt (bool, optional): Whether to use negative prompt for unconditional generation. Defaults to False.
            num_steps (int, optional): Number of diffusion sampling steps. Defaults to 35.
        
        Returns:
            Tensor: Generated samples after diffusion sampling
        """
        # Call the sample method with the same parameters
        return self.sample(
            data_batch=data_batch,
            guidance=guidance,
            seed=seed,
            state_shape=state_shape,
            n_sample=n_sample,
            is_negative_prompt=is_negative_prompt,
            num_steps=num_steps,
            **kwargs
        )
    
    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """Encode input state into latent representation using VAE.
        
        Args:
            state: Input tensor to encode [B, C, T, H, W]
        
        Returns:
            Encoded latent representation scaled by sigma_data
        """
        if self.vae is not None:
            # Check if it's our CleanVAE or has an encode method
            if hasattr(self.vae, 'encode'):
                # VAE expects float32 for best quality
                orig_dtype = state.dtype
                state_fp32 = state.to(torch.float32)
                
                # Encode using VAE
                encoded = self.vae.encode(state_fp32)
                
                # Convert back to original dtype and scale by sigma_data
                encoded = encoded.to(orig_dtype)
                return encoded * self.sigma_data
            else:
                raise ValueError("VAE doesn't have encode method")
        elif self.tokenizer is not None and hasattr(self.tokenizer, 'encode'):
            return self.tokenizer.encode(state) * self.sigma_data
        else:
            # This shouldn't happen in production
            raise ValueError("No VAE or tokenizer available for encoding")
    
    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to pixel space using VAE.
        
        Args:
            latent: Latent tensor to decode [B, C, T, H, W]
        
        Returns:
            Decoded tensor in pixel space
        """
        # Unscale by sigma_data first
        latent = latent / self.sigma_data
        
        if self.vae is not None:
            # Check if it's our CleanVAE or has a decode method
            if hasattr(self.vae, 'decode'):
                # VAE expects float32 for best quality
                orig_dtype = latent.dtype
                latent_fp32 = latent.to(torch.float32)
                
                # Decode using VAE
                decoded = self.vae.decode(latent_fp32)
                
                # Output stays in float32 for quality (convert later if needed in pipeline)
                return decoded
            else:
                raise ValueError("VAE doesn't have decode method")
        elif self.tokenizer is not None and hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(latent)
        else:
            # This shouldn't happen in production
            raise ValueError("No VAE or tokenizer available for decoding")
    
    def load_checkpoint(self, checkpoint_path: str, strict: bool = True):
        """Load model checkpoint from file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            strict: Whether to enforce strict state dict loading
        """
        import os
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"[Model] Loading checkpoint from: {checkpoint_path}")
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load state dict
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
        
        if missing_keys:
            print(f"[Model] Warning: Missing keys in checkpoint: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"[Model] Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")
        
        print(f"[Model] ✅ Checkpoint loaded successfully")
    
    @torch.no_grad()
    def sample_with_debug(
        self,
        data_batch: Dict[str, Tensor],
        guidance: float = 1.5,
        num_steps: int = 35,
        **kwargs
    ) -> Tensor:
        """Sample with debug output to verify scheduler behavior.
        
        This method includes debug prints to help verify the scheduler is working
        correctly with the diffusers library implementation.
        """
        print("[Debug] Using diffusers EDMEulerScheduler")
        print(f"[Debug] Scheduler config: sigma_max={self.scheduler.config.sigma_max}, "
              f"sigma_min={self.scheduler.config.sigma_min}, sigma_data={self.scheduler.config.sigma_data}")
        
        # Get tensor kwargs
        tensor_kwargs = self._get_tensor_kwargs()
        
        # Set timesteps
        self.scheduler.set_timesteps(num_steps, device=tensor_kwargs['device'])
        print(f"[Debug] Timesteps shape: {self.scheduler.timesteps.shape}")
        print(f"[Debug] First 3 timesteps: {self.scheduler.timesteps[:3].tolist()}")
        print(f"[Debug] Init noise sigma: {self.scheduler.init_noise_sigma}")
        
        # Prepare conditions
        condition, uncondition = self._get_conditions(data_batch, False)
        condition = condition.to(**tensor_kwargs)
        uncondition = uncondition.to(**tensor_kwargs)
        
        # Initialize noise
        state_shape = self.state_shape
        xt = torch.randn(size=(1, *state_shape), **tensor_kwargs) * self.scheduler.init_noise_sigma
        print(f"[Debug] Initial noise stats: mean={xt.mean().item():.3f}, std={xt.std().item():.3f}")
        
        # Sampling loop with debug
        for i, t in enumerate(self.scheduler.timesteps):
            t = t.to(tensor_kwargs['device'])
            xt_scaled = self.scheduler.scale_model_input(xt, timestep=t)
            
            if i < 3:
                print(f"[Debug] Step {i}: t={t.item():.3f}, "
                      f"xt_scaled range=[{xt_scaled.min().item():.2f}, {xt_scaled.max().item():.2f}]")
            
            # Model prediction
            net_output_cond = self.net(x=xt_scaled, timesteps=t, **condition.to_dict())
            
            if guidance > 0:
                net_output_uncond = self.net(x=xt_scaled, timesteps=t, **uncondition.to_dict())
                net_output = net_output_cond + guidance * (net_output_cond - net_output_uncond)
            else:
                net_output = net_output_cond
            
            if i < 3:
                print(f"[Debug] Step {i}: net_output range=[{net_output.min().item():.2f}, "
                      f"{net_output.max().item():.2f}]")
            
            # Scheduler step
            xt = self.scheduler.step(net_output, t, xt).prev_sample
            
            if i < 3:
                print(f"[Debug] Step {i}: after step xt range=[{xt.min().item():.2f}, "
                      f"{xt.max().item():.2f}]")
        
        print("[Debug] Sampling complete!")
        return xt
    
    def forward(
        self,
        data_batch: Dict[str, Tensor],
        mode: str = "train"
    ) -> Dict[str, Tensor]:
        """Forward pass for training or inference.
        
        Args:
            data_batch: Input data batch
            mode: "train" or "inference"
        
        Returns:
            Dictionary with model outputs
        """
        if mode == "inference":
            samples = self.sample(data_batch)
            return {'samples': samples}
        else:
            # Training forward pass would go here
            raise NotImplementedError("Training forward pass not implemented in this clean version")
    
    def set_vae(self, vae):
        """Set the VAE model for encoding/decoding."""
        self.vae = vae
    
    def set_tokenizer(self, tokenizer):
        """Set the tokenizer model."""
        self.tokenizer = tokenizer
    
    def to(self, *args, **kwargs):
        """Override to() method to ensure scheduler is properly initialized on device move."""
        # Call parent to() method
        result = super().to(*args, **kwargs)
        
        # If scheduler was already created, recreate it to avoid device issues
        if self._scheduler is not None:
            self._scheduler = None
        
        return result
    
    def cuda(self, device=None):
        """Override cuda() method to ensure proper initialization."""
        result = super().cuda(device)
        # Force scheduler recreation on next use
        if self._scheduler is not None:
            self._scheduler = None
        return result
    
    def cpu(self):
        """Override cpu() method to ensure proper initialization."""
        result = super().cpu()
        # Force scheduler recreation on next use
        if self._scheduler is not None:
            self._scheduler = None
        return result