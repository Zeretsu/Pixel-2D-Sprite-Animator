"""
Motion Blur & Frame Interpolation

Creates smooth "pixel motion blur" like Owlboy and Hyper Light Drifter.

Techniques:
1. Subframe blending - Generate intermediate frames, blend together
2. Trail/ghost rendering - Render faded copies at previous positions  
3. Velocity-based blur - Blur pixels based on their motion vectors
4. Accumulation buffer - Accumulate multiple frames with decay

All operations work in linear color space for correct blending.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# Constants
# =============================================================================

GAMMA = 2.2
INV_GAMMA = 1.0 / GAMMA


# =============================================================================
# Color Space Conversion
# =============================================================================

def to_linear(srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB (0-255) to linear color space (0-1)"""
    return (srgb / 255.0) ** GAMMA


def to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear (0-1) to sRGB (0-255)"""
    return (np.clip(linear, 0, 1) ** INV_GAMMA * 255).astype(np.uint8)


def to_linear_premul(pixels: np.ndarray) -> np.ndarray:
    """Convert RGBA to linear premultiplied alpha"""
    linear = np.zeros(pixels.shape, dtype=np.float32)
    
    # Convert RGB to linear
    linear[..., :3] = (pixels[..., :3] / 255.0) ** GAMMA
    
    # Alpha stays linear
    alpha = pixels[..., 3:4] / 255.0
    linear[..., 3] = alpha[..., 0]
    
    # Premultiply RGB by alpha
    linear[..., :3] *= alpha
    
    return linear


def from_linear_premul(linear_premul: np.ndarray) -> np.ndarray:
    """Convert linear premultiplied back to sRGB RGBA"""
    result = np.zeros(linear_premul.shape, dtype=np.uint8)
    
    alpha = linear_premul[..., 3:4]
    alpha_safe = np.where(alpha > 1e-10, alpha, 1.0)
    
    # Unpremultiply
    rgb_linear = linear_premul[..., :3] / alpha_safe
    
    # Convert to sRGB
    rgb_linear = np.clip(rgb_linear, 0, 1)
    result[..., :3] = (rgb_linear ** INV_GAMMA * 255).astype(np.uint8)
    result[..., 3] = (np.clip(linear_premul[..., 3], 0, 1) * 255).astype(np.uint8)
    
    return result


# =============================================================================
# Motion Blur Modes
# =============================================================================

class BlurMode(Enum):
    """Motion blur blending modes"""
    AVERAGE = "average"           # Simple average of all samples
    WEIGHTED = "weighted"         # More weight to recent frames
    EXPONENTIAL = "exponential"   # Exponential decay (most natural)
    FRONT_WEIGHTED = "front"      # More weight to destination
    BACK_WEIGHTED = "back"        # More weight to source (trailing)


# =============================================================================
# Core Motion Blur Functions
# =============================================================================

def motion_blur(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    samples: int = 4,
    mode: BlurMode = BlurMode.EXPONENTIAL,
    intensity: float = 1.0,
    interpolate_fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = None
) -> np.ndarray:
    """
    Create motion blur between two frames.
    
    Generates intermediate subframes and blends them together,
    creating smooth pixel motion blur like Owlboy/Hyper Light Drifter.
    
    Args:
        frame_a: Source frame (RGBA uint8)
        frame_b: Destination frame (RGBA uint8)
        samples: Number of intermediate samples (4-8 recommended)
        mode: Blending mode for samples
        intensity: Blur strength (0-1, affects weight distribution)
        interpolate_fn: Custom interpolation function(a, b, t) -> frame
                       If None, uses linear pixel interpolation
    
    Returns:
        Motion-blurred frame (RGBA uint8)
    
    Example:
        blurred = motion_blur(frame1, frame2, samples=4)
        blurred = motion_blur(frame1, frame2, mode=BlurMode.EXPONENTIAL)
    """
    if samples < 2:
        return frame_b.copy()
    
    # Convert to linear premultiplied for correct blending
    linear_a = to_linear_premul(frame_a)
    linear_b = to_linear_premul(frame_b)
    
    # Generate sample weights based on mode
    weights = _generate_weights(samples, mode, intensity)
    
    # Accumulate blended samples
    h, w = frame_a.shape[:2]
    accumulated = np.zeros((h, w, 4), dtype=np.float32)
    
    for i in range(samples):
        t = i / (samples - 1)  # 0 to 1
        
        if interpolate_fn:
            # Custom interpolation (e.g., for position-based blur)
            sample = interpolate_fn(frame_a, frame_b, t)
            sample_linear = to_linear_premul(sample)
        else:
            # Linear interpolation in linear color space
            sample_linear = linear_a * (1 - t) + linear_b * t
        
        accumulated += sample_linear * weights[i]
    
    # Normalize by total weight
    total_weight = np.sum(weights)
    if total_weight > 0:
        accumulated /= total_weight
    
    return from_linear_premul(accumulated)


def _generate_weights(samples: int, mode: BlurMode, intensity: float) -> np.ndarray:
    """Generate sample weights based on blur mode"""
    weights = np.ones(samples, dtype=np.float32)
    
    if mode == BlurMode.AVERAGE:
        # Equal weights
        pass
    
    elif mode == BlurMode.WEIGHTED:
        # Linear falloff from center
        center = (samples - 1) / 2
        for i in range(samples):
            dist = abs(i - center) / center
            weights[i] = 1.0 - dist * 0.5 * intensity
    
    elif mode == BlurMode.EXPONENTIAL:
        # Exponential decay (most natural motion blur)
        decay = 0.3 + 0.5 * (1 - intensity)  # Higher intensity = faster decay
        for i in range(samples):
            t = i / (samples - 1)
            weights[i] = np.exp(-t * 3 * intensity) + 0.1
    
    elif mode == BlurMode.FRONT_WEIGHTED:
        # More weight to destination (frame_b)
        for i in range(samples):
            t = i / (samples - 1)
            weights[i] = 0.3 + 0.7 * t * intensity
    
    elif mode == BlurMode.BACK_WEIGHTED:
        # More weight to source (frame_a) - trailing effect
        for i in range(samples):
            t = i / (samples - 1)
            weights[i] = 1.0 - 0.7 * t * intensity
    
    return weights


# =============================================================================
# Ghost/Trail Rendering
# =============================================================================

def ghost_trail(
    frames: List[np.ndarray],
    decay: float = 0.5,
    max_ghosts: int = 4,
    tint: Tuple[int, int, int] = None
) -> np.ndarray:
    """
    Render ghost trail from recent frames.
    
    Creates a fading trail effect by compositing recent frames
    with decreasing opacity.
    
    Args:
        frames: List of recent frames (most recent last)
        decay: Opacity decay per frame (0.5 = 50% each step)
        max_ghosts: Maximum number of ghost frames to use
        tint: Optional color tint for ghosts (RGB)
    
    Returns:
        Composited frame with ghost trail
    
    Example:
        # Keep last 5 frames in a buffer
        frame_buffer.append(current_frame)
        if len(frame_buffer) > 5:
            frame_buffer.pop(0)
        
        result = ghost_trail(frame_buffer, decay=0.6)
    """
    if not frames:
        return np.zeros((1, 1, 4), dtype=np.uint8)
    
    # Limit ghost count
    frames = frames[-max_ghosts:]
    
    # Start with most recent frame as base
    h, w = frames[-1].shape[:2]
    result = to_linear_premul(frames[-1])
    
    # Add ghosts from oldest to newest (excluding the last)
    num_ghosts = len(frames) - 1
    for i, frame in enumerate(frames[:-1]):
        ghost_age = num_ghosts - i  # Older = higher age
        opacity = decay ** ghost_age
        
        if opacity < 0.01:
            continue
        
        ghost_linear = to_linear_premul(frame)
        
        # Apply tint if specified
        if tint:
            tint_linear = np.array([(c / 255.0) ** GAMMA for c in tint], dtype=np.float32)
            ghost_linear[..., :3] *= tint_linear
        
        # Blend ghost under the result (ghosts are behind)
        ghost_alpha = ghost_linear[..., 3:4] * opacity
        result_alpha = result[..., 3:4]
        
        # Porter-Duff "over" compositing
        out_alpha = result_alpha + ghost_alpha * (1 - result_alpha)
        
        # Blend colors
        with np.errstate(divide='ignore', invalid='ignore'):
            out_rgb = np.where(
                out_alpha > 1e-10,
                (result[..., :3] + ghost_linear[..., :3] * opacity * (1 - result_alpha)) / np.maximum(out_alpha, 1e-10),
                result[..., :3]
            )
        
        result[..., :3] = out_rgb * out_alpha
        result[..., 3] = out_alpha[..., 0]
    
    return from_linear_premul(result)


# =============================================================================
# Velocity-Based Blur
# =============================================================================

def velocity_blur(
    frame: np.ndarray,
    velocity_x: Union[float, np.ndarray],
    velocity_y: Union[float, np.ndarray],
    samples: int = 4,
    intensity: float = 1.0
) -> np.ndarray:
    """
    Apply directional motion blur based on velocity.
    
    Blurs pixels along their motion vector. Velocity can be:
    - Scalar: Same blur direction for all pixels
    - Per-pixel array: Different blur for each pixel (flow field)
    
    Args:
        frame: Input frame (RGBA uint8)
        velocity_x: Horizontal velocity (pixels/frame)
        velocity_y: Vertical velocity (pixels/frame)
        samples: Blur samples along velocity vector
        intensity: Blur strength multiplier
    
    Returns:
        Velocity-blurred frame
    
    Example:
        # Uniform motion blur (sprite moving right)
        blurred = velocity_blur(frame, velocity_x=5, velocity_y=0)
        
        # Per-pixel velocity (from optical flow)
        blurred = velocity_blur(frame, vx_field, vy_field)
    """
    h, w = frame.shape[:2]
    linear = to_linear_premul(frame)
    
    # Convert scalar velocity to arrays
    if np.isscalar(velocity_x):
        vx = np.full((h, w), velocity_x * intensity, dtype=np.float32)
    else:
        vx = velocity_x.astype(np.float32) * intensity
    
    if np.isscalar(velocity_y):
        vy = np.full((h, w), velocity_y * intensity, dtype=np.float32)
    else:
        vy = velocity_y.astype(np.float32) * intensity
    
    # Accumulate samples along velocity vector
    accumulated = np.zeros_like(linear)
    
    for i in range(samples):
        # Sample position along velocity vector
        t = (i / (samples - 1) - 0.5) * 2  # -1 to 1
        
        offset_x = vx * t
        offset_y = vy * t
        
        # Sample with bilinear interpolation
        sampled = _bilinear_sample_vectorized(linear, offset_x, offset_y)
        
        # Weight (center samples more)
        weight = 1.0 - abs(t) * 0.3
        accumulated += sampled * weight
    
    accumulated /= samples
    
    return from_linear_premul(accumulated)


def _bilinear_sample_vectorized(
    image: np.ndarray,
    offset_x: np.ndarray,
    offset_y: np.ndarray
) -> np.ndarray:
    """Bilinear sample image with per-pixel offsets"""
    h, w = image.shape[:2]
    
    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Add offsets
    sample_x = x_coords + offset_x
    sample_y = y_coords + offset_y
    
    # Clamp to image bounds
    sample_x = np.clip(sample_x, 0, w - 1.001)
    sample_y = np.clip(sample_y, 0, h - 1.001)
    
    # Integer and fractional parts
    x0 = sample_x.astype(np.int32)
    y0 = sample_y.astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    
    fx = sample_x - x0
    fy = sample_y - y0
    
    # Expand for broadcasting with channels
    fx = fx[..., np.newaxis]
    fy = fy[..., np.newaxis]
    
    # Bilinear interpolation
    result = (
        image[y0, x0] * (1 - fx) * (1 - fy) +
        image[y0, x1] * fx * (1 - fy) +
        image[y1, x0] * (1 - fx) * fy +
        image[y1, x1] * fx * fy
    )
    
    return result


# =============================================================================
# Accumulation Buffer (Temporal Anti-Aliasing style)
# =============================================================================

class AccumulationBuffer:
    """
    Accumulation buffer for smooth temporal blending.
    
    Similar to TAA (Temporal Anti-Aliasing) in modern games.
    Blends current frame with history for smooth motion.
    
    Example:
        buffer = AccumulationBuffer(blend_factor=0.1)
        
        for frame in animation:
            smoothed = buffer.accumulate(frame)
            display(smoothed)
    """
    
    def __init__(
        self,
        blend_factor: float = 0.1,
        max_blend: float = 0.5
    ):
        """
        Args:
            blend_factor: How much of new frame to blend in (0.1 = 10%)
            max_blend: Maximum blend factor (prevents ghosting)
        """
        self.blend_factor = blend_factor
        self.max_blend = max_blend
        self.history: Optional[np.ndarray] = None
    
    def accumulate(self, frame: np.ndarray) -> np.ndarray:
        """
        Accumulate new frame with history.
        
        Args:
            frame: New frame (RGBA uint8)
        
        Returns:
            Blended frame
        """
        linear = to_linear_premul(frame)
        
        if self.history is None:
            self.history = linear.copy()
            return frame
        
        # Detect motion (simple pixel difference)
        diff = np.abs(linear - self.history).mean(axis=-1)
        
        # Adaptive blend factor (less blending where motion detected)
        motion_mask = np.clip(diff * 10, 0, 1)
        adaptive_blend = self.blend_factor + motion_mask * (self.max_blend - self.blend_factor)
        adaptive_blend = adaptive_blend[..., np.newaxis]
        
        # Blend
        self.history = self.history * (1 - adaptive_blend) + linear * adaptive_blend
        
        return from_linear_premul(self.history)
    
    def reset(self):
        """Clear history buffer"""
        self.history = None


# =============================================================================
# Frame Interpolation (In-betweening)
# =============================================================================

def interpolate_frames(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    num_frames: int = 3,
    easing: Callable[[float], float] = None
) -> List[np.ndarray]:
    """
    Generate intermediate frames between two keyframes.
    
    Creates smooth in-between frames for frame rate upscaling
    or smoother animations.
    
    Args:
        frame_a: Start frame (RGBA uint8)
        frame_b: End frame (RGBA uint8)
        num_frames: Number of intermediate frames to generate
        easing: Easing function for non-linear interpolation
    
    Returns:
        List of interpolated frames (excludes frame_a, includes frame_b)
    
    Example:
        # Double frame rate
        intermediates = interpolate_frames(frame1, frame2, num_frames=1)
        # Returns [frame_1.5]
        
        # With easing
        from src.core.easing import ease_in_out_cubic
        intermediates = interpolate_frames(frame1, frame2, 3, ease_in_out_cubic)
    """
    linear_a = to_linear_premul(frame_a)
    linear_b = to_linear_premul(frame_b)
    
    frames = []
    for i in range(1, num_frames + 1):
        t = i / (num_frames + 1)
        
        if easing:
            t = easing(t)
        
        # Interpolate in linear space
        interpolated = linear_a * (1 - t) + linear_b * t
        frames.append(from_linear_premul(interpolated))
    
    return frames


# =============================================================================
# Radial Motion Blur
# =============================================================================

def radial_blur(
    frame: np.ndarray,
    center: Tuple[float, float],
    strength: float = 0.1,
    samples: int = 8,
    inner_radius: float = 0.0
) -> np.ndarray:
    """
    Apply radial (zoom) motion blur.
    
    Creates blur emanating from a center point, useful for:
    - Speed effects (zoom blur)
    - Impact effects
    - Focus effects
    
    Args:
        frame: Input frame (RGBA uint8)
        center: Blur center point (x, y) in pixels
        strength: Blur strength (0.1 = 10% zoom)
        samples: Number of blur samples
        inner_radius: Radius where blur starts (0 = from center)
    
    Returns:
        Radially blurred frame
    """
    h, w = frame.shape[:2]
    linear = to_linear_premul(frame)
    
    # Create coordinate grids relative to center
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    dx = x_coords - center[0]
    dy = y_coords - center[1]
    
    # Distance from center
    dist = np.sqrt(dx * dx + dy * dy)
    
    # Normalize direction
    dist_safe = np.maximum(dist, 1e-10)
    dir_x = dx / dist_safe
    dir_y = dy / dist_safe
    
    # Blur strength increases with distance (outside inner radius)
    blur_amount = np.maximum(0, dist - inner_radius) * strength
    
    # Accumulate samples
    accumulated = np.zeros_like(linear)
    total_weight = 0.0
    
    for i in range(samples):
        t = (i / (samples - 1) - 0.5) * 2  # -1 to 1
        
        offset_x = dir_x * blur_amount * t
        offset_y = dir_y * blur_amount * t
        
        sampled = _bilinear_sample_vectorized(linear, offset_x, offset_y)
        
        weight = 1.0 - abs(t) * 0.5
        accumulated += sampled * weight
        total_weight += weight
    
    accumulated /= total_weight
    
    return from_linear_premul(accumulated)


# =============================================================================
# Rotational Motion Blur
# =============================================================================

def rotational_blur(
    frame: np.ndarray,
    center: Tuple[float, float],
    angle: float,
    samples: int = 8
) -> np.ndarray:
    """
    Apply rotational motion blur around a center point.
    
    Creates blur from rotation, useful for:
    - Spinning objects
    - Tornado/vortex effects
    - Wheel motion
    
    Args:
        frame: Input frame (RGBA uint8)
        center: Rotation center (x, y)
        angle: Total rotation angle in radians
        samples: Number of blur samples
    
    Returns:
        Rotationally blurred frame
    """
    h, w = frame.shape[:2]
    linear = to_linear_premul(frame)
    
    # Create coordinate grids relative to center
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    dx = x_coords - center[0]
    dy = y_coords - center[1]
    
    accumulated = np.zeros_like(linear)
    total_weight = 0.0
    
    for i in range(samples):
        t = (i / (samples - 1) - 0.5) * 2  # -1 to 1
        theta = angle * t
        
        # Rotate coordinates
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotated_x = dx * cos_t - dy * sin_t + center[0]
        rotated_y = dx * sin_t + dy * cos_t + center[1]
        
        # Calculate offset from original position
        offset_x = rotated_x - x_coords
        offset_y = rotated_y - y_coords
        
        sampled = _bilinear_sample_vectorized(linear, offset_x, offset_y)
        
        weight = 1.0 - abs(t) * 0.3
        accumulated += sampled * weight
        total_weight += weight
    
    accumulated /= total_weight
    
    return from_linear_premul(accumulated)


# =============================================================================
# Directional Motion Blur (Simple)
# =============================================================================

def directional_blur(
    frame: np.ndarray,
    angle: float,
    distance: float,
    samples: int = 8
) -> np.ndarray:
    """
    Apply uniform directional motion blur.
    
    Simple blur along a single direction.
    
    Args:
        frame: Input frame (RGBA uint8)
        angle: Blur direction in radians (0 = right, π/2 = down)
        distance: Blur distance in pixels
        samples: Number of blur samples
    
    Returns:
        Directionally blurred frame
    """
    vx = np.cos(angle) * distance
    vy = np.sin(angle) * distance
    
    return velocity_blur(frame, vx, vy, samples)
