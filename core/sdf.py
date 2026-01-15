"""
Signed Distance Fields (SDF)

Mathematical distance fields for high-quality effects:
- Perfect circular glows (not blocky pixel glow)
- Smooth outline scaling at any thickness
- Distance-based effects (dissolve from edges, reveal, etc.)
- Anti-aliased edges from pixel art

SDF Basics:
- Each pixel stores distance to nearest edge
- Negative = inside sprite, Positive = outside
- Zero = exactly on the edge
- Enables smooth gradients and mathematically correct falloffs
"""

import numpy as np
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass
from enum import Enum, auto

# Optional scipy for faster distance transform
try:
    from scipy import ndimage as scipy_ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =============================================================================
# Distance Transform (numpy-only fallback)
# =============================================================================

def _distance_transform_edt_numpy(mask: np.ndarray) -> np.ndarray:
    """
    Euclidean distance transform using numpy only.
    
    For each False pixel, computes distance to nearest True pixel.
    Uses efficient two-pass algorithm.
    """
    h, w = mask.shape
    
    # Initialize with large values for False, 0 for True
    dist = np.where(mask, 0.0, np.inf).astype(np.float32)
    
    # Forward pass (top-left to bottom-right)
    for y in range(h):
        for x in range(w):
            if dist[y, x] > 0:
                candidates = [dist[y, x]]
                if y > 0:
                    candidates.append(dist[y-1, x] + 1)
                if x > 0:
                    candidates.append(dist[y, x-1] + 1)
                if y > 0 and x > 0:
                    candidates.append(dist[y-1, x-1] + 1.414)
                if y > 0 and x < w - 1:
                    candidates.append(dist[y-1, x+1] + 1.414)
                dist[y, x] = min(candidates)
    
    # Backward pass (bottom-right to top-left)
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            if dist[y, x] > 0:
                candidates = [dist[y, x]]
                if y < h - 1:
                    candidates.append(dist[y+1, x] + 1)
                if x < w - 1:
                    candidates.append(dist[y, x+1] + 1)
                if y < h - 1 and x < w - 1:
                    candidates.append(dist[y+1, x+1] + 1.414)
                if y < h - 1 and x > 0:
                    candidates.append(dist[y+1, x-1] + 1.414)
                dist[y, x] = min(candidates)
    
    return dist


def _distance_transform_edt_fast(mask: np.ndarray) -> np.ndarray:
    """
    Faster EDT using vectorized operations.
    Uses Meijster's algorithm approximation.
    """
    h, w = mask.shape
    
    # Initialize
    inf = h + w  # Effectively infinite for our purposes
    dist = np.where(mask, 0, inf).astype(np.float32)
    
    # Row-wise pass (horizontal)
    for y in range(h):
        # Left to right
        for x in range(1, w):
            dist[y, x] = min(dist[y, x], dist[y, x-1] + 1)
        # Right to left
        for x in range(w - 2, -1, -1):
            dist[y, x] = min(dist[y, x], dist[y, x+1] + 1)
    
    # Column-wise pass (vertical) with proper Euclidean
    result = np.zeros_like(dist)
    
    for x in range(w):
        # Build parabola envelope for this column
        col = dist[:, x]
        for y in range(h):
            min_dist = float('inf')
            # Check nearby rows (limit search for performance)
            search_range = int(col.min()) + 10
            for y2 in range(max(0, y - search_range), min(h, y + search_range)):
                d = col[y2] ** 2 + (y - y2) ** 2
                if d < min_dist:
                    min_dist = d
            result[y, x] = np.sqrt(min_dist)
    
    return result


def distance_transform_edt(mask: np.ndarray) -> np.ndarray:
    """
    Euclidean distance transform.
    Uses scipy if available, otherwise numpy fallback.
    """
    if SCIPY_AVAILABLE:
        return scipy_ndimage.distance_transform_edt(mask).astype(np.float32)
    else:
        # Use faster numpy implementation
        return _distance_transform_edt_fast(mask)


def gaussian_filter(arr: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian blur using numpy only.
    """
    if SCIPY_AVAILABLE:
        return scipy_ndimage.gaussian_filter(arr, sigma=sigma)
    
    # Create Gaussian kernel
    size = int(6 * sigma + 1)
    if size % 2 == 0:
        size += 1
    
    x = np.arange(size) - size // 2
    kernel_1d = np.exp(-x**2 / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    
    # Separable convolution (faster)
    # Horizontal pass
    result = np.zeros_like(arr, dtype=np.float32)
    padded = np.pad(arr, ((0, 0), (size//2, size//2)), mode='reflect')
    for i in range(size):
        result += padded[:, i:i+arr.shape[1]] * kernel_1d[i]
    
    # Vertical pass
    padded = np.pad(result, ((size//2, size//2), (0, 0)), mode='reflect')
    result = np.zeros_like(arr, dtype=np.float32)
    for i in range(size):
        result += padded[i:i+arr.shape[0], :] * kernel_1d[i]
    
    return result


def sobel_filter(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Sobel edge detection filter.
    
    Args:
        arr: 2D input array
        axis: 0 for vertical edges (y gradient), 1 for horizontal (x gradient)
    """
    if SCIPY_AVAILABLE:
        return scipy_ndimage.sobel(arr, axis=axis)
    
    # Sobel kernels
    if axis == 0:
        kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    else:
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    
    return convolve2d(arr, kernel)


def convolve2d(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    2D convolution using numpy only.
    """
    if SCIPY_AVAILABLE:
        return scipy_ndimage.convolve(arr, kernel, mode='constant')
    
    h, w = arr.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # Pad array
    padded = np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    # Convolve
    result = np.zeros_like(arr, dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            result += padded[i:i+h, j:j+w] * kernel[i, j]
    
    return result


# =============================================================================
# SDF Generation
# =============================================================================

def generate_sdf(
    sprite: np.ndarray,
    threshold: float = 0.5,
    normalize: bool = True
) -> np.ndarray:
    """
    Generate a Signed Distance Field from sprite alpha.
    
    Args:
        sprite: RGBA image (H, W, 4) with values 0-255
        threshold: Alpha threshold for edge detection (0-1)
        normalize: If True, normalize to -1..1 range
        
    Returns:
        SDF array (H, W) where:
        - Negative values = inside sprite
        - Positive values = outside sprite  
        - Zero = on the edge
    """
    # Extract alpha and normalize
    if sprite.ndim == 3 and sprite.shape[2] >= 4:
        alpha = sprite[:, :, 3].astype(np.float32) / 255.0
    elif sprite.ndim == 3 and sprite.shape[2] == 3:
        # RGB - use luminance for mask
        alpha = (0.299 * sprite[:, :, 0] + 0.587 * sprite[:, :, 1] + 0.114 * sprite[:, :, 2]) / 255.0
    else:
        alpha = sprite.astype(np.float32) / 255.0
    
    # Create binary mask
    mask = alpha >= threshold
    
    # Calculate distance transform for inside and outside
    # dist_inside: 0 inside mask, >0 outside (distance to nearest inside pixel)
    dist_inside = distance_transform_edt(mask)
    # dist_outside: >0 inside mask (distance to nearest outside pixel), 0 outside
    dist_outside = distance_transform_edt(~mask)
    
    # Combine: negative inside, positive outside
    # Inside sprite: dist_inside=0, dist_outside>0 → want negative
    # Outside sprite: dist_inside>0, dist_outside=0 → want positive
    sdf = dist_inside - dist_outside
    
    if normalize:
        # Normalize to roughly -1..1 range based on max distance
        max_dist = max(dist_inside.max(), dist_outside.max())
        if max_dist > 0:
            sdf = sdf / max_dist
    
    return sdf.astype(np.float32)


def generate_sdf_from_edges(
    sprite: np.ndarray,
    edge_threshold: float = 0.1
) -> np.ndarray:
    """
    Generate SDF from detected edges (for sprites with internal details).
    
    Args:
        sprite: RGBA image
        edge_threshold: Sensitivity for edge detection
        
    Returns:
        SDF based on edges, not just alpha
    """
    # Convert to grayscale
    if sprite.ndim == 3:
        if sprite.shape[2] >= 3:
            gray = (0.299 * sprite[:, :, 0] + 0.587 * sprite[:, :, 1] + 0.114 * sprite[:, :, 2])
        else:
            gray = sprite[:, :, 0]
    else:
        gray = sprite
    
    gray = gray.astype(np.float32) / 255.0
    
    # Sobel edge detection (numpy-based)
    sobel_x = sobel_filter(gray, axis=1)
    sobel_y = sobel_filter(gray, axis=0)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Threshold to binary edges
    edge_mask = edges > edge_threshold
    
    # Distance from edges
    dist = distance_transform_edt(~edge_mask)
    
    # Sign based on alpha
    if sprite.ndim == 3 and sprite.shape[2] >= 4:
        alpha = sprite[:, :, 3].astype(np.float32) / 255.0
        inside = alpha > 0.5
        sdf = np.where(inside, -dist, dist)
    else:
        sdf = dist
    
    return sdf.astype(np.float32)


def generate_multi_channel_sdf(
    sprite: np.ndarray,
    spread: float = 4.0
) -> np.ndarray:
    """
    Generate multi-channel SDF for better anti-aliasing (MSDF-like).
    
    Stores distance information in RGB channels for sharper corners.
    
    Args:
        sprite: RGBA image
        spread: Distance spread in pixels
        
    Returns:
        (H, W, 3) array with distance in each channel
    """
    h, w = sprite.shape[:2]
    msdf = np.zeros((h, w, 3), dtype=np.float32)
    
    # Main SDF
    main_sdf = generate_sdf(sprite, normalize=False)
    
    # Clamp to spread range
    main_sdf = np.clip(main_sdf, -spread, spread) / spread
    
    # For pixel art, we can use directional distances
    # Red channel: horizontal distance bias
    # Green channel: vertical distance bias  
    # Blue channel: diagonal distance bias
    
    alpha = sprite[:, :, 3].astype(np.float32) / 255.0 if sprite.ndim == 3 else sprite / 255.0
    mask = alpha >= 0.5
    
    # Horizontal
    kernel_h = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])
    h_edges = convolve2d(mask.astype(float), kernel_h) > 0
    h_dist = distance_transform_edt(~(h_edges & mask))
    h_dist_out = distance_transform_edt(~(h_edges & ~mask))
    msdf[:, :, 0] = np.clip((h_dist_out - h_dist) / spread, -1, 1)
    
    # Vertical
    kernel_v = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
    v_edges = convolve2d(mask.astype(float), kernel_v) > 0
    v_dist = distance_transform_edt(~(v_edges & mask))
    v_dist_out = distance_transform_edt(~(v_edges & ~mask))
    msdf[:, :, 1] = np.clip((v_dist_out - v_dist) / spread, -1, 1)
    
    # Main SDF in blue
    msdf[:, :, 2] = main_sdf
    
    return msdf


# =============================================================================
# SDF-Based Effects
# =============================================================================

@dataclass
class GlowConfig:
    """Configuration for SDF glow effect"""
    radius: float = 4.0           # Glow radius in pixels
    color: Tuple[int, int, int] = (255, 200, 100)  # Glow color
    intensity: float = 1.0        # Glow brightness
    falloff: str = 'smooth'       # 'linear', 'smooth', 'exponential'
    inner: bool = False           # Inner glow instead of outer
    additive: bool = True         # Additive blending


def sdf_glow(
    sprite: np.ndarray,
    config: Optional[GlowConfig] = None,
    sdf: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply perfect circular glow using SDF.
    
    Args:
        sprite: RGBA image
        config: Glow configuration
        sdf: Pre-computed SDF (optional, will generate if not provided)
        
    Returns:
        RGBA image with glow applied
    """
    config = config or GlowConfig()
    
    if sdf is None:
        sdf = generate_sdf(sprite, normalize=False)
    
    h, w = sprite.shape[:2]
    result = sprite.copy().astype(np.float32)
    
    # Calculate glow mask based on distance
    if config.inner:
        # Inner glow: strongest at edge, fades toward center
        glow_dist = -sdf  # Flip sign for inner
        glow_mask = np.clip(glow_dist / config.radius, 0, 1)
    else:
        # Outer glow: strongest at edge, fades outward
        glow_dist = sdf
        glow_mask = np.clip(1 - glow_dist / config.radius, 0, 1)
    
    # Only apply to relevant areas
    if config.inner:
        glow_mask = np.where(sdf < 0, glow_mask, 0)  # Only inside
    else:
        glow_mask = np.where(sdf > 0, glow_mask, 0)  # Only outside
    
    # Apply falloff curve
    if config.falloff == 'smooth':
        # Smoothstep falloff
        glow_mask = glow_mask * glow_mask * (3 - 2 * glow_mask)
    elif config.falloff == 'exponential':
        # Exponential falloff (sharper)
        glow_mask = np.exp(-3 * (1 - glow_mask)) * glow_mask
    # 'linear' uses mask as-is
    
    glow_mask *= config.intensity
    
    # Create glow layer
    glow = np.zeros((h, w, 4), dtype=np.float32)
    glow[:, :, 0] = config.color[0]
    glow[:, :, 1] = config.color[1]
    glow[:, :, 2] = config.color[2]
    glow[:, :, 3] = glow_mask * 255
    
    # Composite
    if config.additive:
        # Additive blending for glow
        glow_rgb = glow[:, :, :3] * (glow_mask[:, :, np.newaxis])
        result[:, :, :3] = np.clip(result[:, :, :3] + glow_rgb, 0, 255)
        result[:, :, 3] = np.clip(result[:, :, 3] + glow[:, :, 3] * 0.5, 0, 255)
    else:
        # Normal alpha blending (behind sprite)
        sprite_alpha = result[:, :, 3:4] / 255.0
        glow_alpha = glow[:, :, 3:4] / 255.0
        
        # Composite glow behind sprite
        combined_alpha = sprite_alpha + glow_alpha * (1 - sprite_alpha)
        
        result[:, :, :3] = (
            result[:, :, :3] * sprite_alpha +
            glow[:, :, :3] * glow_alpha * (1 - sprite_alpha)
        ) / np.maximum(combined_alpha, 0.001)
        result[:, :, 3] = combined_alpha[:, :, 0] * 255
    
    return np.clip(result, 0, 255).astype(np.uint8)


@dataclass  
class OutlineConfig:
    """Configuration for SDF outline effect"""
    thickness: float = 1.0        # Outline thickness in pixels
    color: Tuple[int, int, int] = (0, 0, 0)  # Outline color
    softness: float = 0.5         # Edge softness (0=hard, 1=soft)
    inside: bool = False          # Draw outline inside instead of outside
    only_outline: bool = False    # Return only the outline, not the sprite


def sdf_outline(
    sprite: np.ndarray,
    config: Optional[OutlineConfig] = None,
    sdf: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generate smooth outline at any thickness using SDF.
    
    Args:
        sprite: RGBA image
        config: Outline configuration
        sdf: Pre-computed SDF
        
    Returns:
        RGBA image with outline
    """
    config = config or OutlineConfig()
    
    if sdf is None:
        sdf = generate_sdf(sprite, normalize=False)
    
    h, w = sprite.shape[:2]
    
    # Calculate outline band
    if config.inside:
        # Inside outline: from edge inward
        outline_start = -config.thickness
        outline_end = 0
    else:
        # Outside outline: from edge outward  
        outline_start = 0
        outline_end = config.thickness
    
    # Create outline mask
    in_band = (sdf >= outline_start) & (sdf <= outline_end)
    
    # Distance from center of band for softness
    band_center = (outline_start + outline_end) / 2
    band_half_width = (outline_end - outline_start) / 2
    
    if config.softness > 0 and band_half_width > 0:
        # Soft edges
        dist_from_center = np.abs(sdf - band_center) / band_half_width
        outline_alpha = np.clip(1 - dist_from_center * config.softness, 0, 1)
        outline_alpha = np.where(in_band, outline_alpha, 0)
    else:
        # Hard edges
        outline_alpha = in_band.astype(np.float32)
    
    # Create outline layer
    outline = np.zeros((h, w, 4), dtype=np.float32)
    outline[:, :, 0] = config.color[0]
    outline[:, :, 1] = config.color[1]
    outline[:, :, 2] = config.color[2]
    outline[:, :, 3] = outline_alpha * 255
    
    if config.only_outline:
        return np.clip(outline, 0, 255).astype(np.uint8)
    
    # Composite outline behind sprite
    result = np.zeros((h, w, 4), dtype=np.float32)
    sprite_f = sprite.astype(np.float32)
    
    sprite_alpha = sprite_f[:, :, 3:4] / 255.0
    outline_alpha_4 = outline[:, :, 3:4] / 255.0
    
    # Outline behind sprite
    combined_alpha = sprite_alpha + outline_alpha_4 * (1 - sprite_alpha)
    
    result[:, :, :3] = (
        sprite_f[:, :, :3] * sprite_alpha +
        outline[:, :, :3] * outline_alpha_4 * (1 - sprite_alpha)
    ) / np.maximum(combined_alpha, 0.001)
    result[:, :, 3] = combined_alpha[:, :, 0] * 255
    
    return np.clip(result, 0, 255).astype(np.uint8)


def sdf_dissolve(
    sprite: np.ndarray,
    progress: float,
    from_edge: bool = True,
    noise_amount: float = 0.3,
    edge_glow: bool = True,
    glow_color: Tuple[int, int, int] = (255, 200, 100),
    sdf: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Distance-based dissolve effect using SDF.
    
    Args:
        sprite: RGBA image
        progress: Dissolve progress 0-1 (0=full sprite, 1=dissolved)
        from_edge: True=dissolve from edges inward, False=center outward
        noise_amount: Random noise added to threshold (0-1)
        edge_glow: Add glow at dissolve edge
        glow_color: Color of edge glow
        sdf: Pre-computed SDF
        
    Returns:
        Partially dissolved sprite
    """
    if sdf is None:
        sdf = generate_sdf(sprite, normalize=False)
    
    h, w = sprite.shape[:2]
    result = sprite.copy().astype(np.float32)
    
    # Get distance range inside sprite (negative SDF values)
    inside_mask = sdf < 0
    if not inside_mask.any():
        return sprite.copy()
    
    min_dist = sdf[inside_mask].min()  # Most negative (deepest inside)
    
    # Calculate threshold based on progress
    # progress=0 means full sprite visible, progress=1 means fully dissolved
    # SDF is negative inside sprite (more negative = deeper inside)
    if from_edge:
        # Dissolve from edges inward
        # threshold goes from 0 (edge) toward min_dist (center)
        # pixels with sdf > threshold get dissolved
        threshold = min_dist * progress  # 0 at start, min_dist at end
    else:
        # Dissolve from center outward  
        # threshold goes from min_dist (center) toward 0 (edge)
        # pixels with sdf < threshold get dissolved
        threshold = min_dist * (1 - progress)  # min_dist at start, 0 at end
    
    # Add noise for organic look
    if noise_amount > 0:
        noise = np.random.random((h, w)) * noise_amount * abs(min_dist)
        if from_edge:
            threshold_map = threshold - noise  # subtract to push threshold deeper
        else:
            threshold_map = threshold + noise  # add to push threshold toward edge
    else:
        threshold_map = np.full((h, w), threshold)
    
    # Create dissolve mask
    if from_edge:
        dissolved = sdf > threshold_map
    else:
        dissolved = sdf < threshold_map
    
    # Apply dissolve
    result[dissolved, 3] = 0
    
    # Add edge glow at dissolve boundary
    if edge_glow and 0 < progress < 1:
        # Find pixels near the dissolve edge
        edge_band = 2.0  # Pixels
        edge_dist = np.abs(sdf - threshold_map)
        edge_mask = (edge_dist < edge_band) & ~dissolved & inside_mask
        
        edge_intensity = 1 - edge_dist / edge_band
        edge_intensity = np.where(edge_mask, edge_intensity, 0)
        edge_intensity = edge_intensity ** 0.5  # Boost
        
        # Add glow
        result[:, :, 0] = np.clip(result[:, :, 0] + glow_color[0] * edge_intensity * 0.5, 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 1] + glow_color[1] * edge_intensity * 0.5, 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2] + glow_color[2] * edge_intensity * 0.5, 0, 255)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def sdf_shadow(
    sprite: np.ndarray,
    offset: Tuple[float, float] = (2, 2),
    blur: float = 2.0,
    color: Tuple[int, int, int, int] = (0, 0, 0, 128),
    sdf: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    SDF-based drop shadow with proper blur falloff.
    
    Args:
        sprite: RGBA image
        offset: Shadow offset (x, y) in pixels
        blur: Shadow blur radius
        color: Shadow color with alpha
        sdf: Pre-computed SDF
        
    Returns:
        Sprite with shadow
    """
    if sdf is None:
        sdf = generate_sdf(sprite, normalize=False)
    
    h, w = sprite.shape[:2]
    
    # Offset the SDF for shadow position
    ox, oy = int(offset[0]), int(offset[1])
    shadow_sdf = np.full((h, w), 999.0, dtype=np.float32)
    
    # Shift SDF
    src_y1, src_y2 = max(0, -oy), min(h, h - oy)
    src_x1, src_x2 = max(0, -ox), min(w, w - ox)
    dst_y1, dst_y2 = max(0, oy), min(h, h + oy)
    dst_x1, dst_x2 = max(0, ox), min(w, w + ox)
    
    if dst_y2 > dst_y1 and dst_x2 > dst_x1:
        shadow_sdf[dst_y1:dst_y2, dst_x1:dst_x2] = sdf[src_y1:src_y2, src_x1:src_x2]
    
    # Create shadow alpha from SDF
    # Inside = shadow, with blur falloff
    shadow_alpha = np.clip(-shadow_sdf / max(blur, 0.1), 0, 1)
    
    # Apply gaussian blur for soft shadow
    if blur > 0:
        shadow_alpha = gaussian_filter(shadow_alpha, sigma=blur / 2)
    
    shadow_alpha *= color[3] / 255.0
    
    # Create shadow layer
    shadow = np.zeros((h, w, 4), dtype=np.float32)
    shadow[:, :, 0] = color[0]
    shadow[:, :, 1] = color[1]
    shadow[:, :, 2] = color[2]
    shadow[:, :, 3] = shadow_alpha * 255
    
    # Composite: shadow behind sprite
    sprite_f = sprite.astype(np.float32)
    sprite_alpha = sprite_f[:, :, 3:4] / 255.0
    shadow_alpha_4 = shadow[:, :, 3:4] / 255.0
    
    combined_alpha = sprite_alpha + shadow_alpha_4 * (1 - sprite_alpha)
    
    result = np.zeros((h, w, 4), dtype=np.float32)
    result[:, :, :3] = (
        sprite_f[:, :, :3] * sprite_alpha +
        shadow[:, :, :3] * shadow_alpha_4 * (1 - sprite_alpha)
    ) / np.maximum(combined_alpha, 0.001)
    result[:, :, 3] = combined_alpha[:, :, 0] * 255
    
    return np.clip(result, 0, 255).astype(np.uint8)


# =============================================================================
# SDF Animation Utilities
# =============================================================================

def sdf_reveal(
    sprite: np.ndarray,
    progress: float,
    direction: str = 'center',
    softness: float = 4.0,
    sdf: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Reveal sprite using SDF (opposite of dissolve).
    
    Args:
        sprite: RGBA image
        progress: Reveal progress 0-1
        direction: 'center' (inside out), 'edge' (outside in), 'left', 'right', 'top', 'bottom'
        softness: Edge softness in pixels
        sdf: Pre-computed SDF
        
    Returns:
        Partially revealed sprite
    """
    if sdf is None:
        sdf = generate_sdf(sprite, normalize=False)
    
    h, w = sprite.shape[:2]
    result = sprite.copy().astype(np.float32)
    
    # Calculate reveal threshold based on direction
    if direction == 'center':
        # Reveal from center (deepest point) outward
        min_dist = sdf.min()
        threshold = min_dist + progress * (-min_dist)
        reveal_mask = sdf <= threshold
    elif direction == 'edge':
        # Reveal from edge inward
        min_dist = sdf.min()
        threshold = progress * (-min_dist)
        reveal_mask = sdf >= -threshold
    elif direction in ('left', 'right', 'top', 'bottom'):
        # Directional reveal using position + SDF
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        if direction == 'left':
            pos_factor = x_coords / w
        elif direction == 'right':
            pos_factor = 1 - x_coords / w
        elif direction == 'top':
            pos_factor = y_coords / h
        else:  # bottom
            pos_factor = 1 - y_coords / h
        
        # Combine position with SDF for organic edge
        combined = pos_factor - (sdf / (sdf.min() if sdf.min() < 0 else 1)) * 0.2
        threshold = progress * 1.2  # Slight overshoot to ensure full reveal
        reveal_mask = combined <= threshold
    else:
        reveal_mask = np.ones((h, w), dtype=bool)
    
    # Apply softness
    if softness > 0:
        reveal_dist = distance_transform_edt(~reveal_mask)
        alpha_factor = np.clip(1 - reveal_dist / softness, 0, 1)
    else:
        alpha_factor = reveal_mask.astype(np.float32)
    
    result[:, :, 3] *= alpha_factor
    
    return np.clip(result, 0, 255).astype(np.uint8)


def sdf_pulse_glow(
    sprite: np.ndarray,
    time: float,
    base_radius: float = 2.0,
    pulse_amount: float = 2.0,
    color: Tuple[int, int, int] = (255, 200, 100),
    frequency: float = 1.0,
    sdf: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Animated pulsing glow using SDF.
    
    Args:
        sprite: RGBA image
        time: Animation time (0-1 for one cycle)
        base_radius: Minimum glow radius
        pulse_amount: Additional radius at peak
        color: Glow color
        frequency: Pulse speed multiplier
        sdf: Pre-computed SDF
        
    Returns:
        Sprite with pulsing glow
    """
    # Calculate current radius using sine wave
    pulse = (np.sin(time * 2 * np.pi * frequency) + 1) / 2
    current_radius = base_radius + pulse_amount * pulse
    current_intensity = 0.5 + 0.5 * pulse
    
    config = GlowConfig(
        radius=current_radius,
        color=color,
        intensity=current_intensity,
        falloff='smooth'
    )
    
    return sdf_glow(sprite, config, sdf)


def sdf_breathing(
    sprite: np.ndarray,
    time: float,
    amount: float = 0.1,
    sdf: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Subtle breathing/pulsing effect using SDF-based scaling.
    
    Args:
        sprite: RGBA image
        time: Animation time (0-1 for one cycle)
        amount: Scale amount (0.1 = 10% variation)
        sdf: Pre-computed SDF
        
    Returns:
        Scaled sprite with smooth edges
    """
    if sdf is None:
        sdf = generate_sdf(sprite, normalize=False)
    
    # Calculate scale factor
    scale = 1 + amount * np.sin(time * 2 * np.pi)
    
    # Scale the SDF (effectively scales the shape)
    scaled_sdf = sdf / scale
    
    # Reconstruct sprite from scaled SDF
    result = sprite.copy().astype(np.float32)
    
    # Inside the scaled shape = visible
    inside = scaled_sdf < 0
    
    # Smooth edge using SDF
    edge_softness = 1.0
    alpha_factor = np.clip(-scaled_sdf / edge_softness, 0, 1)
    alpha_factor = np.where(inside, 1.0, alpha_factor)
    
    result[:, :, 3] *= alpha_factor
    
    return np.clip(result, 0, 255).astype(np.uint8)


# =============================================================================
# SDF Analysis & Utilities
# =============================================================================

def get_sdf_bounds(sdf: np.ndarray) -> Tuple[float, float]:
    """Get min/max values of SDF (extent of sprite)"""
    return float(sdf.min()), float(sdf.max())


def get_sprite_thickness(sdf: np.ndarray) -> float:
    """Estimate sprite thickness from SDF (2x deepest point)"""
    return abs(sdf.min()) * 2


def get_edge_pixels(sdf: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Get mask of edge pixels (within threshold of zero)"""
    return np.abs(sdf) <= threshold


def dilate_sprite(
    sprite: np.ndarray,
    amount: float,
    sdf: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Dilate (expand) sprite by amount using SDF.
    
    Args:
        sprite: RGBA image
        amount: Dilation amount in pixels (negative = erode)
        sdf: Pre-computed SDF
        
    Returns:
        Dilated sprite with averaged colors at new edges
    """
    if sdf is None:
        sdf = generate_sdf(sprite, normalize=False)
    
    h, w = sprite.shape[:2]
    result = np.zeros((h, w, 4), dtype=np.float32)
    
    # New inside region (shifted by amount)
    new_inside = sdf < amount
    
    # For color, use nearest neighbor from original sprite
    # Find nearest inside pixel for each new pixel
    orig_inside = sdf < 0
    
    if amount > 0:
        # Expanding: new pixels need colors
        # Use average of nearby original colors
        for c in range(3):
            channel = sprite[:, :, c].astype(np.float32)
            channel = np.where(orig_inside, channel, 0)
            blurred = gaussian_filter(channel, sigma=amount)
            count = gaussian_filter(orig_inside.astype(np.float32), sigma=amount)
            result[:, :, c] = np.where(count > 0, blurred / np.maximum(count, 0.001), 0)
        
        # Original pixels keep original color
        result[orig_inside, :3] = sprite[orig_inside, :3]
    else:
        # Shrinking: just mask
        result[:, :, :3] = sprite[:, :, :3]
    
    # Alpha from SDF
    edge_softness = 0.5
    alpha = np.clip((amount - sdf) / edge_softness, 0, 1)
    result[:, :, 3] = alpha * 255
    
    return np.clip(result, 0, 255).astype(np.uint8)


def erode_sprite(
    sprite: np.ndarray,
    amount: float,
    sdf: Optional[np.ndarray] = None
) -> np.ndarray:
    """Erode (shrink) sprite by amount. Convenience wrapper for dilate_sprite."""
    return dilate_sprite(sprite, -amount, sdf)


# =============================================================================
# Convenience Functions
# =============================================================================

def add_glow(
    sprite: np.ndarray,
    radius: float = 4.0,
    color: Tuple[int, int, int] = (255, 200, 100),
    intensity: float = 1.0
) -> np.ndarray:
    """Quick glow effect"""
    config = GlowConfig(radius=radius, color=color, intensity=intensity)
    return sdf_glow(sprite, config)


def add_outline(
    sprite: np.ndarray,
    thickness: float = 1.0,
    color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """Fast outline effect using simple dilation (no SDF needed)"""
    h, w = sprite.shape[:2]
    
    # Get alpha mask
    alpha = sprite[:, :, 3] > 0
    
    # Dilate the mask to create outline area (fast vectorized)
    dilated = alpha.copy()
    thickness_int = max(1, int(thickness))
    for _ in range(thickness_int):
        new_dilated = dilated.copy()
        new_dilated[1:, :] |= dilated[:-1, :]
        new_dilated[:-1, :] |= dilated[1:, :]
        new_dilated[:, 1:] |= dilated[:, :-1]
        new_dilated[:, :-1] |= dilated[:, 1:]
        dilated = new_dilated
    
    # Outline = dilated minus original
    outline_mask = dilated & ~alpha
    
    # Create result with outline behind sprite
    result = np.zeros((h, w, 4), dtype=np.uint8)
    result[outline_mask, 0] = color[0]
    result[outline_mask, 1] = color[1]
    result[outline_mask, 2] = color[2]
    result[outline_mask, 3] = 255
    
    # Paste original sprite on top
    sprite_mask = sprite[:, :, 3] > 0
    result[sprite_mask] = sprite[sprite_mask]
    
    return result


def add_shadow(
    sprite: np.ndarray,
    offset: Tuple[float, float] = (2, 2),
    blur: float = 2.0,
    opacity: float = 0.5
) -> np.ndarray:
    """Quick drop shadow"""
    alpha = int(255 * opacity)
    return sdf_shadow(sprite, offset, blur, (0, 0, 0, alpha))
