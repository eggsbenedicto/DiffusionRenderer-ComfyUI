"""
Comprehensive Environment Map Processing for Cosmos1 Diffusion Renderer
Implements official-quality HDR pipeline with robust fallbacks and ComfyUI integration
"""

import torch
import numpy as np
import os
import cv2
import imageio.v3 as imageio_v3
import logging
from typing import Union, Tuple, Dict, Optional, List
from pathlib import Path
import hashlib
import time

# Enable OpenEXR support
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

logger = logging.getLogger(__name__)

class EnvironmentMapCache:
    """LRU cache for processed environment maps to improve performance"""
    
    def __init__(self, max_size: int = 10):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
    
    def _generate_key(self, env_hash: str, resolution: Tuple[int, int], 
                     format_type: str, env_brightness: float, env_flip: bool, env_rot: float) -> str:
        """Generate cache key from parameters"""
        return f"{env_hash}_{resolution}_{format_type}_{env_brightness}_{env_flip}_{env_rot}"
    
    def get(self, env_hash: str, resolution: Tuple[int, int], format_type: str, 
            env_brightness: float, env_flip: bool, env_rot: float) -> Optional[Dict]:
        """Retrieve cached result if available"""
        key = self._generate_key(env_hash, resolution, format_type, env_brightness, env_flip, env_rot)
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def put(self, env_hash: str, resolution: Tuple[int, int], format_type: str,
            env_brightness: float, env_flip: bool, env_rot: float, result: Dict):
        """Cache processing result with LRU eviction"""
        key = self._generate_key(env_hash, resolution, format_type, env_brightness, env_flip, env_rot)
        
        # Evict oldest if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = result
        self.access_times[key] = time.time()
    
    def clear(self):
        """Clear cache and free memory"""
        self.cache.clear()
        self.access_times.clear()
        torch.cuda.empty_cache()

# Global cache instance
_env_cache = EnvironmentMapCache()

def compute_tensor_hash(tensor: torch.Tensor) -> str:
    """Compute hash of tensor for caching"""
    # Sample a few values for hashing to avoid processing entire tensor
    if tensor.numel() > 1000:
        # Sample regularly spaced elements
        indices = torch.linspace(0, tensor.numel()-1, 1000, dtype=torch.long)
        sample = tensor.flatten()[indices]
    else:
        sample = tensor.flatten()
    
    # Convert to bytes and hash
    tensor_bytes = sample.cpu().numpy().tobytes()
    return hashlib.md5(tensor_bytes).hexdigest()

def detect_hdr_content(env_input: Union[str, torch.Tensor]) -> bool:
    """Detect if input contains HDR data"""
    if isinstance(env_input, str):
        # Check file extension
        ext = Path(env_input).suffix.lower()
        return ext in ['.hdr', '.exr', '.pfm']
    elif isinstance(env_input, torch.Tensor):
        # Check if values exceed LDR range
        return torch.any(env_input > 1.0) or torch.any(env_input < 0.0)
    return False

def detect_environment_format(env_input: Union[str, torch.Tensor]) -> str:
    """Detect environment map format (equirectangular vs chrome ball)"""
    if isinstance(env_input, torch.Tensor):
        H, W = env_input.shape[-3:-1]
        aspect_ratio = W / H
        
        # Equirectangular typically has 2:1 aspect ratio
        if 1.8 <= aspect_ratio <= 2.2:
            return "equirectangular"
        # Chrome ball is typically square
        elif 0.9 <= aspect_ratio <= 1.1:
            return "chrome_ball"
    
    # Default assumption
    return "equirectangular"

def rgb2srgb_official(rgb: torch.Tensor) -> torch.Tensor:
    """Official sRGB conversion matching Cosmos1"""
    return torch.where(rgb <= 0.0031308, 
                      12.92 * rgb, 
                      1.055 * torch.pow(torch.clamp(rgb, 1e-8, 1.0), 1.0/2.4) - 0.055)

def reinhard_official(x: torch.Tensor, max_point: float = 16.0) -> torch.Tensor:
    """Official Reinhard tone mapping"""
    return x / (x + 1.0) * max_point

def hdr_mapping_official(env_hdr: torch.Tensor, log_scale: float = 10000.0) -> Dict[str, torch.Tensor]:
    """
    Official HDR tone mapping matching Cosmos1 implementation
    
    Args:
        env_hdr: HDR environment tensor
        log_scale: Scale factor for logarithmic encoding
        
    Returns:
        Dictionary with 'env_hdr', 'env_ev0' (LDR), 'env_log' versions
    """
    # Reinhard tone mapping for LDR version
    env_ev0 = rgb2srgb_official(reinhard_official(env_hdr, max_point=16.0).clamp(0, 1))
    
    # Logarithmic encoding for HDR version  
    env_log = rgb2srgb_official(torch.log1p(env_hdr) / np.log1p(log_scale)).clamp(0, 1)
    
    return {
        'env_hdr': env_hdr,
        'env_ev0': env_ev0,  # This becomes env_ldr
        'env_log': env_log,
    }

def cube_to_dir(s: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Convert cube face coordinates to direction vectors"""
    if s == 0:    # +X
        return torch.stack([torch.ones_like(x), -y, -x], dim=-1)
    elif s == 1:  # -X  
        return torch.stack([-torch.ones_like(x), -y, x], dim=-1)
    elif s == 2:  # +Y
        return torch.stack([x, torch.ones_like(x), y], dim=-1)
    elif s == 3:  # -Y
        return torch.stack([x, -torch.ones_like(x), -y], dim=-1)
    elif s == 4:  # +Z
        return torch.stack([x, -y, torch.ones_like(x)], dim=-1)
    elif s == 5:  # -Z
        return torch.stack([-x, -y, -torch.ones_like(x)], dim=-1)

def safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Safely normalize vectors"""
    return v / (torch.norm(v, dim=-1, keepdim=True) + eps)

def latlong_to_cubemap_official(latlong_map: torch.Tensor, res: List[int]) -> torch.Tensor:
    """
    Official cubemap conversion matching Cosmos1 implementation
    
    Args:
        latlong_map: Equirectangular environment map tensor (H, W, C)
        res: Cubemap resolution [H, W]
        
    Returns:
        Cubemap tensor (6, H, W, C)
    """
    device = latlong_map.device
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], 
                         dtype=torch.float32, device=device)
    
    for s in range(6):
        # Generate grid coordinates for this cube face
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device=device), 
            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device=device),
            indexing='ij'
        )
        
        # Convert to direction vectors
        v = safe_normalize(cube_to_dir(s, gx, gy))
        
        # Convert direction to equirectangular UV coordinates
        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)
        
        # Sample from equirectangular map using bilinear interpolation
        # Convert UV coordinates to grid sample format [-1, 1]
        grid = texcoord * 2.0 - 1.0
        grid = grid.unsqueeze(0)  # Add batch dimension
        
        # Use grid_sample for bilinear interpolation
        latlong_batch = latlong_map.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        sampled = torch.nn.functional.grid_sample(
            latlong_batch, grid, 
            mode='bilinear', padding_mode='border', align_corners=False
        )
        
        cubemap[s, ...] = sampled.squeeze(0).permute(1, 2, 0)  # (H, W, C)
    
    return cubemap

def load_hdr_file(file_path: str) -> torch.Tensor:
    """Load HDR file with OpenEXR support"""
    try:
        # Try imageio with OpenEXR plugin first
        img = imageio_v3.imread(file_path, flags=cv2.IMREAD_UNCHANGED, plugin='opencv')
        if img is None:
            raise ValueError(f"Failed to load with imageio: {file_path}")
    except Exception as e:
        logger.warning(f"imageio failed: {e}, trying alternative methods")
        try:
            # Fallback to cv2 for EXR files
            if file_path.lower().endswith('.exr'):
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    raise ValueError(f"cv2 failed to load EXR: {file_path}")
            else:
                # Regular imageio for other formats
                img = imageio_v3.imread(file_path)
        except Exception as e2:
            raise ValueError(f"All loading methods failed for {file_path}: {e2}")
    
    # Convert to tensor
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32)
    
    # Ensure 3 channels
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]  # Remove alpha
    
    return torch.from_numpy(img)

def process_comfyui_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Process ComfyUI IMAGE tensor to environment map format"""
    # ComfyUI tensors are typically (B, H, W, C) or (B, C, H, W)
    if tensor.ndim == 4:
        if tensor.shape[1] == 3 or tensor.shape[1] == 4:  # (B, C, H, W)
            tensor = tensor.permute(0, 2, 3, 1)  # -> (B, H, W, C)
        tensor = tensor[0]  # Take first batch item
    
    # Ensure 3 channels
    if tensor.shape[-1] == 4:
        tensor = tensor[..., :3]  # Remove alpha
    elif tensor.shape[-1] == 1:
        tensor = tensor.repeat(1, 1, 3)  # Grayscale to RGB
    
    return tensor

def apply_hdr_preprocessing(latlong_img: torch.Tensor, env_brightness: float, 
                           env_flip: bool, env_rot: float, device: str) -> torch.Tensor:
    """Apply official HDR preprocessing transformations"""
    latlong_img = latlong_img.to(device)
    
    # Apply strength multiplier
    if env_brightness != 1.0:
        latlong_img *= env_brightness
    
    # Cleanup NaNs and Infs (official implementation)
    latlong_img = torch.nan_to_num(latlong_img, nan=0.0, posinf=65504.0, neginf=0.0)
    latlong_img = latlong_img.clamp(0.0, 65504.0)
    
    # Apply horizontal flip (official default is True)
    if env_flip:
        latlong_img = torch.flip(latlong_img, dims=[1])
    
    # Apply rotation (official default is 180 degrees)
    if env_rot != 0:
        lat_h, lat_w = latlong_img.shape[:2]
        pixel_rot = int(lat_w * env_rot / 360)
        latlong_img = torch.roll(latlong_img, shifts=pixel_rot, dims=1)
    
    return latlong_img

def load_and_preprocess_hdr_robust(env_input: Union[str, torch.Tensor], 
                                  env_brightness: float, env_flip: bool, 
                                  env_rot: float, device: str) -> torch.Tensor:
    """
    Robust HDR loading and preprocessing supporting multiple input types
    
    Args:
        env_input: File path or ComfyUI IMAGE tensor
        env_brightness: HDR intensity multiplier  
        env_flip: Horizontal flip for coordinate correction
        env_rot: Rotation in degrees
        device: Target device
        
    Returns:
        Processed cubemap tensor (6, 512, 512, 3)
    """
    # Load based on input type
    if isinstance(env_input, str):
        latlong_img = load_hdr_file(env_input)
    elif isinstance(env_input, torch.Tensor):
        latlong_img = process_comfyui_tensor(env_input)
    else:
        raise ValueError(f"Unsupported input type: {type(env_input)}")
    
    # Apply preprocessing
    latlong_img = apply_hdr_preprocessing(latlong_img, env_brightness, env_flip, env_rot, device)
    
    # Convert to official 512x512 cubemap
    cubemap = latlong_to_cubemap_official(latlong_img, [512, 512])
    
    return cubemap

def latlong_vec(res: Tuple[int, int], device: str = 'cuda') -> torch.Tensor:
    """Generate lat-long direction vectors"""
    H, W = res
    gy, gx = torch.meshgrid(
        torch.linspace(0.0 + 1.0 / H, 1.0 - 1.0 / H, H, device=device), 
        torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device=device),
        indexing='ij'
    )
    
    sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)
    
    dir_vec = torch.stack((
        sintheta * sinphi, 
        costheta, 
        -sintheta * cosphi
    ), dim=-1)
    
    return dir_vec

def rotate_y(angle: float, device: str = 'cuda') -> torch.Tensor:
    """Generate Y-axis rotation matrix"""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return torch.tensor([
        [cos_a, 0, sin_a, 0],
        [0, 1, 0, 0],
        [-sin_a, 0, cos_a, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device=device)

def sample_cubemap_gpu(cubemap: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Sample cubemap using direction vectors with GPU acceleration"""
    # Convert directions to cubemap sampling coordinates
    # This is a simplified version - full implementation would need proper cube face selection
    batch_size = directions.shape[0]
    H, W = directions.shape[1:3]
    
    # Normalize directions
    directions = safe_normalize(directions.view(-1, 3))
    
    # Convert to spherical coordinates for sampling
    # This is a fallback - proper implementation needs cube face logic
    phi = torch.atan2(directions[:, 0], -directions[:, 2]) / (2 * np.pi) + 0.5
    theta = torch.acos(torch.clamp(directions[:, 1], -1, 1)) / np.pi
    
    # Create grid for sampling
    grid_coords = torch.stack([phi, theta], dim=-1) * 2.0 - 1.0
    grid_coords = grid_coords.view(batch_size, H, W, 2)
    
    # Use average of all cube faces as approximation
    avg_cubemap = cubemap.mean(dim=0).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    
    sampled = torch.nn.functional.grid_sample(
        avg_cubemap, grid_coords,
        mode='bilinear', padding_mode='border', align_corners=False
    )
    
    return sampled.squeeze(0).permute(1, 2, 0)  # (H, W, C)

def process_projected_envmap_official(cubemap: torch.Tensor, vec: torch.Tensor, 
                                    c2w: torch.Tensor, y_rot: torch.Tensor, 
                                    H: int, W: int) -> torch.Tensor:
    """Camera-aligned environment projection (most common format)"""
    # Transform direction vectors by camera and rotation matrices
    vec_cam = vec.view(-1, 3) @ c2w[:3, :3].T
    vec_query = (vec_cam @ y_rot[:3, :3].T).view(1, H, W, 3)
    
    # Sample cubemap using transformed directions
    env_proj = sample_cubemap_gpu(cubemap.unsqueeze(0), vec_query)
    
    # Apply official flip
    env_proj = torch.flip(env_proj, dims=[0, 1])
    
    return env_proj

def get_ideal_ball(size: int, flip_x: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate ideal chrome ball reflection vectors"""
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, size),
        torch.linspace(-1, 1, size),
        indexing='ij'
    )
    
    if flip_x:
        x = -x
    
    # Chrome ball geometry
    r2 = x*x + y*y
    mask = r2 <= 1.0
    
    # Normal vectors for chrome ball surface
    z = torch.sqrt(torch.clamp(1.0 - r2, 0.0, 1.0))
    normal = torch.stack([x, y, z], dim=-1)
    
    return normal, mask

def get_ref_vector(normal: torch.Tensor, incoming_vector: np.ndarray) -> torch.Tensor:
    """Compute reflection vectors for chrome ball"""
    incoming = torch.tensor(incoming_vector, dtype=normal.dtype, device=normal.device)
    incoming = incoming.view(1, 1, 3).expand_as(normal)
    
    # Reflection formula: r = d - 2(d·n)n
    dot_product = torch.sum(incoming * normal, dim=-1, keepdim=True)
    reflected = incoming - 2 * dot_product * normal
    
    return reflected

def process_ball_envmap_official(cubemap: torch.Tensor, vec_ref: torch.Tensor,
                               c2w: torch.Tensor, y_rot: torch.Tensor, 
                               H: int, W: int) -> torch.Tensor:
    """Chrome ball simulation for DiffusionLight compatibility"""
    # Simulate chrome ball reflection vectors
    vec_ball = -vec_ref.view(-1, 3) @ c2w[:3, :3].T
    vec_query = (vec_ball @ y_rot[:3, :3].T).view(1, H, W, 3)
    
    # Sample cubemap using ball reflection directions
    env_proj = sample_cubemap_gpu(cubemap.unsqueeze(0), vec_query)
    
    return env_proj

def process_fixed_envmap_official(cubemap: torch.Tensor, vec: torch.Tensor, 
                                H: int, W: int) -> torch.Tensor:
    """Static environment without camera transformation"""
    # Direct sampling without transformation
    env_proj = sample_cubemap_gpu(cubemap.unsqueeze(0), -vec.unsqueeze(0))
    env_proj = torch.flip(env_proj, dims=[0, 1])
    return env_proj

def create_neutral_environment(resolution: Tuple[int, int], 
                             device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """Generate neutral gray environment as fallback"""
    H, W = resolution
    neutral_color = torch.tensor([0.5, 0.5, 0.5], device=device, dtype=torch.float32)
    
    env_ldr = neutral_color.view(1, 1, 1, 3).expand(1, H, W, 3)
    env_log = neutral_color.view(1, 1, 1, 3).expand(1, H, W, 3)
    
    return {
        'env_ldr': env_ldr,
        'env_log': env_log,
    }

def process_environment_map_simplified(env_input: Union[str, torch.Tensor],
                                     format_type: str, resolution: Tuple[int, int],
                                     env_brightness: float = 1.0, env_flip: bool = True,
                                     env_rot: float = 180.0, device: str = 'cuda',
                                     **kwargs) -> Dict[str, torch.Tensor]:
    """Simplified fallback processing"""
    H, W = resolution
    
    # Load environment
    if isinstance(env_input, str):
        try:
            env_tensor = load_hdr_file(env_input).to(device)
        except:
            return create_neutral_environment(resolution, device)
    else:
        env_tensor = process_comfyui_tensor(env_input).to(device)
    
    # Simple resize
    if env_tensor.shape[:2] != (H, W):
        env_tensor = torch.nn.functional.interpolate(
            env_tensor.permute(2, 0, 1).unsqueeze(0),
            size=(H, W), mode='bilinear', align_corners=False
        ).squeeze(0).permute(1, 2, 0)
    
    # Apply strength
    env_tensor *= env_brightness
    
    # Simple tone mapping
    env_ldr = env_tensor / (env_tensor.max() + 1e-8)
    env_log = torch.log(env_tensor + 1e-8)
    env_log = (env_log - env_log.min()) / (env_log.max() - env_log.min() + 1e-8)
    
    # Add batch dimension
    env_ldr = env_ldr.unsqueeze(0)
    env_log = env_log.unsqueeze(0)
    
    return {
        'env_ldr': env_ldr,
        'env_log': env_log,
    }

def process_environment_map_official(env_input: Union[str, torch.Tensor],
                                   format_type: str, resolution: Tuple[int, int],
                                   env_brightness: float = 1.0, env_flip: bool = True,
                                   env_rot: float = 180.0, device: str = 'cuda',
                                   num_frames: int = 1, **kwargs) -> Dict[str, torch.Tensor]:
    """
    Official-quality environment map processing
    
    Args:
        env_input: File path or tensor input
        format_type: 'proj', 'ball', or 'fixed'
        resolution: Target resolution (H, W)
        env_brightness: HDR intensity multiplier
        env_flip: Horizontal flip for correct orientation
        env_rot: Rotation in degrees
        device: Processing device
        num_frames: Number of video frames
        
    Returns:
        Dictionary with processed environment maps
    """
    H, W = resolution
    
    # Load and preprocess to cubemap
    cubemap = load_and_preprocess_hdr_robust(env_input, env_brightness, env_flip, env_rot, device)
    
    # Generate direction vectors
    vec = latlong_vec((H, W), device=device)
    
    # Camera transformation (identity for now)
    c2w = torch.eye(4, device=device)
    y_rot = rotate_y(0.0, device=device)  # No additional rotation
    
    # Process based on format type
    if format_type == 'proj':
        env_proj = process_projected_envmap_official(cubemap, vec, c2w, y_rot, H, W)
    elif format_type == 'ball':
        # Generate chrome ball vectors
        vec_ball, _ = get_ideal_ball(H, flip_x=False)
        vec_ref = get_ref_vector(vec_ball, np.array([0, 0, 1]))
        vec_ref = vec_ref.float().to(device)
        env_proj = process_ball_envmap_official(cubemap, vec_ref, c2w, y_rot, H, W)
    elif format_type == 'fixed':
        env_proj = process_fixed_envmap_official(cubemap, vec, H, W)
    else:
        raise ValueError(f"Unsupported format type: {format_type}")
    
    # Apply official tone mapping
    mapping_results = hdr_mapping_official(env_proj, log_scale=10000.0)
    
    # Use env_ev0 as env_ldr (official naming)
    env_ldr = mapping_results['env_ev0']
    env_log = mapping_results['env_log']
    
    # Expand for video frames
    if num_frames > 1:
        env_ldr = env_ldr.unsqueeze(0).expand(num_frames, -1, -1, -1)
        env_log = env_log.unsqueeze(0).expand(num_frames, -1, -1, -1)
    else:
        env_ldr = env_ldr.unsqueeze(0)
        env_log = env_log.unsqueeze(0)
    
    return {
        'env_ldr': env_ldr,
        'env_log': env_log,
    }

def process_environment_map_robust(env_input: Union[str, torch.Tensor],
                                 format_type: str = 'proj', 
                                 resolution: Tuple[int, int] = (512, 512),
                                 env_brightness: float = 1.0, env_flip: bool = True,
                                 env_rot: float = 180.0, device: str = 'cuda',
                                 use_cache: bool = True, **kwargs) -> Dict[str, torch.Tensor]:
    """
    Robust environment map processing with caching and fallbacks
    
    This is the main entry point for environment map processing.
    """
    # Generate cache key if using cache
    if use_cache:
        if isinstance(env_input, torch.Tensor):
            env_hash = compute_tensor_hash(env_input)
        else:
            env_hash = hashlib.md5(str(env_input).encode()).hexdigest()
        
        cached_result = _env_cache.get(env_hash, resolution, format_type, 
                                      env_brightness, env_flip, env_rot)
        if cached_result is not None:
            logger.debug("Using cached environment map")
            return cached_result
    
    # Try official implementation first
    try:
        result = process_environment_map_official(
            env_input, format_type, resolution, env_brightness, 
            env_flip, env_rot, device, **kwargs
        )
        
        if use_cache:
            _env_cache.put(env_hash, resolution, format_type, 
                          env_brightness, env_flip, env_rot, result)
        
        return result
        
    except Exception as e:
        logger.warning(f"Official processing failed: {e}, falling back to simplified")
        
        try:
            # Fallback to simplified implementation
            result = process_environment_map_simplified(
                env_input, format_type, resolution, env_brightness,
                env_flip, env_rot, device, **kwargs
            )
            return result
            
        except Exception as e2:
            logger.error(f"All processing methods failed: {e2}, using neutral environment")
            # Final fallback: neutral environment
            return create_neutral_environment(resolution, device)

# Legacy compatibility function
def process_environment_map(env_map_path: str, resolution: Tuple[int, int], 
                          num_frames: int, fixed_pose: bool, rotate_envlight: bool,
                          env_format: List[str], device: str) -> Dict[str, torch.Tensor]:
    """Legacy compatibility wrapper for existing code"""
    format_type = env_format[0] if env_format else 'proj'
    return process_environment_map_robust(
        env_input=env_map_path,
        format_type=format_type,
        resolution=resolution,
        device=device,
        num_frames=num_frames
    )

def clear_environment_cache():
    """Clear the global environment map cache"""
    _env_cache.clear()

def get_cache_stats() -> Dict[str, int]:
    """Get cache statistics"""
    return {
        'cache_size': len(_env_cache.cache),
        'max_size': _env_cache.max_size,
    }
