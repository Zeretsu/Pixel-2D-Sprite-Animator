"""
Smear Frames - Fast Motion Distortion

Professional technique used in Dead Cells, Katana Zero, Hollow Knight.
For fast motion, sprites are stretched/distorted in motion direction.

Types of smears:
1. Stretch smear - Elongate sprite in motion direction
2. Multiple smear - Ghost copies along motion path
3. Motion lines - Speed lines behind sprite
4. Blur smear - Directional blur with shape preservation

This creates the illusion of speed that single frames can't capture.
"""

import numpy as np
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# Constants
# =============================================================================

GAMMA = 2.2
INV_GAMMA = 1.0 / GAMMA


# =============================================================================
# Smear Types
# =============================================================================

class SmearType(Enum):
    """Types of smear effects"""
    STRETCH = "stretch"           # Elongate in motion direction
    MULTIPLE = "multiple"         # Multiple ghost copies
    MOTION_LINES = "motion_lines" # Speed lines trailing
    BLUR = "blur"                 # Directional blur
    SQUASH = "squash"             # Compression (for impacts)
    ONION = "onion"               # Onion-skin style trail


# =============================================================================
# Color Space Helpers
# =============================================================================

def _to_linear_premul(pixels: np.ndarray) -> np.ndarray:
    """Convert RGBA to linear premultiplied"""
    linear = np.zeros(pixels.shape, dtype=np.float32)
    linear[..., :3] = (pixels[..., :3] / 255.0) ** GAMMA
    alpha = pixels[..., 3:4] / 255.0
    linear[..., 3] = alpha[..., 0]
    linear[..., :3] *= alpha
    return linear


def _from_linear_premul(linear_premul: np.ndarray) -> np.ndarray:
    """Convert linear premultiplied back to sRGB RGBA"""
    result = np.zeros(linear_premul.shape, dtype=np.uint8)
    alpha = linear_premul[..., 3:4]
    alpha_safe = np.where(alpha > 1e-10, alpha, 1.0)
    rgb_linear = np.clip(linear_premul[..., :3] / alpha_safe, 0, 1)
    result[..., :3] = (rgb_linear ** INV_GAMMA * 255).astype(np.uint8)
    result[..., 3] = (np.clip(linear_premul[..., 3], 0, 1) * 255).astype(np.uint8)
    return result


# =============================================================================
# Core Smear Functions
# =============================================================================

def add_smear_frame(
    sprite: np.ndarray,
    direction: Union[float, Tuple[float, float]],
    speed: float,
    smear_type: SmearType = SmearType.STRETCH,
    intensity: float = 1.0,
    preserve_original: bool = True
) -> np.ndarray:
    """
    Add smear effect to sprite for fast motion.
    
    Used in basically every professional pixel game (Dead Cells, Katana Zero).
    
    Args:
        sprite: Input sprite (RGBA uint8)
        direction: Motion direction - angle in radians OR (dx, dy) tuple
        speed: Motion speed (pixels per frame, affects smear length)
        smear_type: Type of smear effect
        intensity: Effect strength (0-1)
        preserve_original: Keep original sprite visible
    
    Returns:
        Smeared sprite (RGBA uint8)
    
    Example:
        # Sprite moving right at 10 pixels/frame
        smeared = add_smear_frame(sprite, direction=0, speed=10)
        
        # Sprite moving up-left
        smeared = add_smear_frame(sprite, direction=(−5, −8), speed=9.4)
    """
    # Parse direction
    if isinstance(direction, (int, float)):
        dx = np.cos(direction)
        dy = np.sin(direction)
    else:
        dx, dy = direction
        length = np.sqrt(dx * dx + dy * dy)
        if length > 0:
            dx, dy = dx / length, dy / length
        else:
            return sprite.copy()
    
    # Calculate smear amount
    smear_length = speed * intensity
    
    if smear_length < 0.5:
        return sprite.copy()
    
    # Apply smear based on type
    if smear_type == SmearType.STRETCH:
        return _stretch_smear(sprite, dx, dy, smear_length, preserve_original)
    elif smear_type == SmearType.MULTIPLE:
        return _multiple_smear(sprite, dx, dy, smear_length, preserve_original)
    elif smear_type == SmearType.MOTION_LINES:
        return _motion_lines_smear(sprite, dx, dy, smear_length, preserve_original)
    elif smear_type == SmearType.BLUR:
        return _blur_smear(sprite, dx, dy, smear_length)
    elif smear_type == SmearType.SQUASH:
        return _squash_smear(sprite, dx, dy, smear_length)
    elif smear_type == SmearType.ONION:
        return _onion_smear(sprite, dx, dy, smear_length, preserve_original)
    else:
        return sprite.copy()


def _stretch_smear(
    sprite: np.ndarray,
    dx: float,
    dy: float,
    smear_length: float,
    preserve_original: bool
) -> np.ndarray:
    """
    Stretch sprite in motion direction.
    
    The sprite is elongated along the motion vector.
    """
    h, w = sprite.shape[:2]
    
    # Calculate output size (may need to be larger)
    extra_w = int(abs(dx) * smear_length) + 1
    extra_h = int(abs(dy) * smear_length) + 1
    
    out_w = w + extra_w * 2
    out_h = h + extra_h * 2
    
    # Create output in linear space
    output = np.zeros((out_h, out_w, 4), dtype=np.float32)
    sprite_linear = _to_linear_premul(sprite)
    
    # Center offset
    cx, cy = extra_w, extra_h
    
    # Find sprite bounds (non-transparent pixels)
    mask = sprite[..., 3] > 10
    if not np.any(mask):
        return sprite.copy()
    
    ys, xs = np.where(mask)
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    sprite_cx = (min_x + max_x) / 2
    sprite_cy = (min_y + max_y) / 2
    
    # Create coordinate grids
    out_y, out_x = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing='ij')
    
    # For each output pixel, find source with stretch
    # The stretch maps: output_pos -> source_pos
    # Points in motion direction are compressed (come from same source region)
    
    # Calculate distance along motion direction from center
    rel_x = out_x - cx - sprite_cx
    rel_y = out_y - cy - sprite_cy
    
    # Project onto motion direction
    proj_along = rel_x * dx + rel_y * dy
    proj_perp_x = rel_x - proj_along * dx
    proj_perp_y = rel_y - proj_along * dy
    
    # Compress along motion direction (stretch effect)
    stretch_factor = 1.0 + smear_length / max(w, h)
    compressed_along = proj_along / stretch_factor
    
    # Reconstruct source coordinates
    src_x = sprite_cx + compressed_along * dx + proj_perp_x
    src_y = sprite_cy + compressed_along * dy + proj_perp_y
    
    # Bilinear sample from source
    output = _bilinear_sample_full(sprite_linear, src_x, src_y, out_h, out_w)
    
    # Optionally overlay original at center
    if preserve_original:
        for y in range(h):
            for x in range(w):
                out_y_pos = y + cy
                out_x_pos = x + cx
                if sprite[y, x, 3] > 10:
                    # Alpha blend original on top
                    src_alpha = sprite_linear[y, x, 3]
                    dst_alpha = output[out_y_pos, out_x_pos, 3]
                    out_alpha = src_alpha + dst_alpha * (1 - src_alpha)
                    if out_alpha > 0:
                        output[out_y_pos, out_x_pos, :3] = (
                            sprite_linear[y, x, :3] + 
                            output[out_y_pos, out_x_pos, :3] * (1 - src_alpha)
                        )
                        output[out_y_pos, out_x_pos, 3] = out_alpha
    
    # Trim to original size centered on sprite
    result = output[cy:cy+h, cx:cx+w]
    return _from_linear_premul(result)


def _multiple_smear(
    sprite: np.ndarray,
    dx: float,
    dy: float,
    smear_length: float,
    preserve_original: bool,
    num_copies: int = 4
) -> np.ndarray:
    """
    Multiple ghost copies along motion path.
    
    Creates several semi-transparent copies trailing behind.
    """
    h, w = sprite.shape[:2]
    sprite_linear = _to_linear_premul(sprite)
    output = np.zeros((h, w, 4), dtype=np.float32)
    
    # Add ghost copies from back to front
    for i in range(num_copies, -1, -1):
        if i == 0 and not preserve_original:
            continue
        
        t = i / num_copies  # 0 = current, 1 = furthest back
        
        # Offset for this copy (negative = behind)
        offset_x = -dx * smear_length * t
        offset_y = -dy * smear_length * t
        
        # Alpha falloff
        if i == 0:
            alpha_mult = 1.0
        else:
            alpha_mult = 0.6 * (1 - t * 0.7)  # Fade with distance
        
        # Sample sprite at offset
        ghost = _sample_offset(sprite_linear, offset_x, offset_y)
        ghost[..., 3] *= alpha_mult
        
        # Composite (over)
        output = _composite_over(ghost, output)
    
    return _from_linear_premul(output)


def _motion_lines_smear(
    sprite: np.ndarray,
    dx: float,
    dy: float,
    smear_length: float,
    preserve_original: bool,
    line_count: int = 5
) -> np.ndarray:
    """
    Speed lines trailing behind sprite.
    
    Draws lines from sprite edges in motion direction.
    """
    h, w = sprite.shape[:2]
    sprite_linear = _to_linear_premul(sprite)
    output = sprite_linear.copy() if preserve_original else np.zeros((h, w, 4), dtype=np.float32)
    
    # Find edge pixels (for line origins)
    mask = sprite[..., 3] > 10
    if not np.any(mask):
        return sprite.copy()
    
    # Simple edge detection
    padded = np.pad(mask, 1, mode='constant', constant_values=False)
    edges = mask & ~(
        padded[:-2, 1:-1] & padded[2:, 1:-1] & 
        padded[1:-1, :-2] & padded[1:-1, 2:]
    )
    
    edge_ys, edge_xs = np.where(edges)
    
    if len(edge_xs) == 0:
        return sprite.copy() if preserve_original else np.zeros_like(sprite)
    
    # Sample edge pixels for lines
    rng = np.random.default_rng(42)
    num_lines = min(line_count, len(edge_xs))
    indices = rng.choice(len(edge_xs), num_lines, replace=False)
    
    for idx in indices:
        start_x, start_y = edge_xs[idx], edge_ys[idx]
        
        # Get color from sprite
        color = sprite_linear[start_y, start_x, :3]
        
        # Draw line trailing behind
        line_length = int(smear_length * (0.5 + rng.random() * 0.5))
        
        for i in range(line_length):
            t = i / max(1, line_length - 1)
            
            lx = start_x - dx * i
            ly = start_y - dy * i
            
            if 0 <= int(lx) < w and 0 <= int(ly) < h:
                # Alpha fades along line
                alpha = 0.5 * (1 - t)
                
                # Add to output (additive for brightness)
                ix, iy = int(lx), int(ly)
                output[iy, ix, :3] += color * alpha * 0.5
                output[iy, ix, 3] = max(output[iy, ix, 3], alpha * 0.5)
    
    return _from_linear_premul(np.clip(output, 0, 1))


def _blur_smear(
    sprite: np.ndarray,
    dx: float,
    dy: float,
    smear_length: float,
    samples: int = 8
) -> np.ndarray:
    """
    Directional blur smear.
    
    Smooth blur along motion direction.
    """
    h, w = sprite.shape[:2]
    sprite_linear = _to_linear_premul(sprite)
    output = np.zeros((h, w, 4), dtype=np.float32)
    
    total_weight = 0.0
    
    for i in range(samples):
        t = (i / (samples - 1) - 0.5) * 2  # -1 to 1
        
        offset_x = dx * smear_length * t * 0.5
        offset_y = dy * smear_length * t * 0.5
        
        # Weight (more weight to center)
        weight = 1.0 - abs(t) * 0.5
        
        sampled = _sample_offset(sprite_linear, offset_x, offset_y)
        output += sampled * weight
        total_weight += weight
    
    output /= total_weight
    return _from_linear_premul(output)


def _squash_smear(
    sprite: np.ndarray,
    dx: float,
    dy: float,
    smear_amount: float
) -> np.ndarray:
    """
    Squash/compression smear for impacts.
    
    Compresses sprite perpendicular to motion (impact squash).
    """
    h, w = sprite.shape[:2]
    sprite_linear = _to_linear_premul(sprite)
    
    # Find sprite center
    mask = sprite[..., 3] > 10
    if not np.any(mask):
        return sprite.copy()
    
    ys, xs = np.where(mask)
    cx = (xs.min() + xs.max()) / 2
    cy = (ys.min() + ys.max()) / 2
    
    output = np.zeros((h, w, 4), dtype=np.float32)
    
    # Calculate squash factor
    squash_factor = 1.0 - min(smear_amount * 0.05, 0.5)
    stretch_factor = 1.0 / squash_factor  # Preserve volume
    
    # Create coordinate mapping
    for y in range(h):
        for x in range(w):
            # Relative to center
            rel_x = x - cx
            rel_y = y - cy
            
            # Project onto motion direction and perpendicular
            proj_along = rel_x * dx + rel_y * dy
            proj_perp = rel_x * (-dy) + rel_y * dx  # Perpendicular
            
            # Squash perpendicular, stretch along
            new_along = proj_along * stretch_factor
            new_perp = proj_perp * squash_factor
            
            # Back to cartesian
            src_x = cx + new_along * dx - new_perp * dy
            src_y = cy + new_along * dy + new_perp * dx
            
            # Bilinear sample
            if 0 <= src_x < w - 1 and 0 <= src_y < h - 1:
                x0, y0 = int(src_x), int(src_y)
                fx, fy = src_x - x0, src_y - y0
                
                output[y, x] = (
                    sprite_linear[y0, x0] * (1-fx) * (1-fy) +
                    sprite_linear[y0, min(x0+1, w-1)] * fx * (1-fy) +
                    sprite_linear[min(y0+1, h-1), x0] * (1-fx) * fy +
                    sprite_linear[min(y0+1, h-1), min(x0+1, w-1)] * fx * fy
                )
    
    return _from_linear_premul(output)


def _onion_smear(
    sprite: np.ndarray,
    dx: float,
    dy: float,
    smear_length: float,
    preserve_original: bool,
    layers: int = 3
) -> np.ndarray:
    """
    Onion-skin style trail.
    
    Multiple copies with color tinting (like animation onion skinning).
    """
    h, w = sprite.shape[:2]
    sprite_linear = _to_linear_premul(sprite)
    output = np.zeros((h, w, 4), dtype=np.float32)
    
    # Color tints for trail (cyan -> blue -> purple)
    tints = [
        np.array([0.8, 1.0, 1.0]),   # Cyan
        np.array([0.6, 0.7, 1.0]),   # Blue
        np.array([0.8, 0.5, 1.0]),   # Purple
    ]
    
    # Add layers from back to front
    for i in range(layers, -1, -1):
        t = i / layers
        
        offset_x = -dx * smear_length * t
        offset_y = -dy * smear_length * t
        
        layer = _sample_offset(sprite_linear, offset_x, offset_y)
        
        if i > 0:
            # Apply tint
            tint_idx = min(i - 1, len(tints) - 1)
            layer[..., :3] *= tints[tint_idx]
            layer[..., 3] *= 0.4 * (1 - t * 0.5)
        elif not preserve_original:
            continue
        
        output = _composite_over(layer, output)
    
    return _from_linear_premul(output)


# =============================================================================
# Helper Functions
# =============================================================================

def _sample_offset(
    linear: np.ndarray,
    offset_x: float,
    offset_y: float
) -> np.ndarray:
    """Sample image with pixel offset using bilinear interpolation."""
    h, w = linear.shape[:2]
    
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    src_x = np.clip(x_coords.astype(np.float32) - offset_x, 0, w - 1.001)
    src_y = np.clip(y_coords.astype(np.float32) - offset_y, 0, h - 1.001)
    
    x0 = src_x.astype(np.int32)
    y0 = src_y.astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    
    fx = (src_x - x0)[..., np.newaxis]
    fy = (src_y - y0)[..., np.newaxis]
    
    result = (
        linear[y0, x0] * (1 - fx) * (1 - fy) +
        linear[y0, x1] * fx * (1 - fy) +
        linear[y1, x0] * (1 - fx) * fy +
        linear[y1, x1] * fx * fy
    )
    
    return result


def _bilinear_sample_full(
    linear: np.ndarray,
    src_x: np.ndarray,
    src_y: np.ndarray,
    out_h: int,
    out_w: int
) -> np.ndarray:
    """Full bilinear sampling with out-of-bounds handling."""
    h, w = linear.shape[:2]
    output = np.zeros((out_h, out_w, 4), dtype=np.float32)
    
    # Mask for valid coordinates
    valid = (src_x >= 0) & (src_x < w - 1) & (src_y >= 0) & (src_y < h - 1)
    
    src_x_clipped = np.clip(src_x, 0, w - 1.001)
    src_y_clipped = np.clip(src_y, 0, h - 1.001)
    
    x0 = src_x_clipped.astype(np.int32)
    y0 = src_y_clipped.astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    
    fx = (src_x_clipped - x0)[..., np.newaxis]
    fy = (src_y_clipped - y0)[..., np.newaxis]
    
    sampled = (
        linear[y0, x0] * (1 - fx) * (1 - fy) +
        linear[y0, x1] * fx * (1 - fy) +
        linear[y1, x0] * (1 - fx) * fy +
        linear[y1, x1] * fx * fy
    )
    
    # Only use valid samples
    output[valid] = sampled[valid]
    
    return output


def _composite_over(
    src: np.ndarray,
    dst: np.ndarray
) -> np.ndarray:
    """Porter-Duff 'over' compositing in premultiplied alpha."""
    src_a = src[..., 3:4]
    result = np.zeros_like(dst)
    result[..., :3] = src[..., :3] + dst[..., :3] * (1 - src_a)
    result[..., 3] = src[..., 3] + dst[..., 3] * (1 - src_a[..., 0])
    return result


# =============================================================================
# High-Level Smear Generator
# =============================================================================

@dataclass
class SmearConfig:
    """Configuration for smear frame generation."""
    smear_type: SmearType = SmearType.STRETCH
    intensity: float = 1.0
    preserve_original: bool = True
    # Type-specific settings
    num_copies: int = 4      # For MULTIPLE
    line_count: int = 5      # For MOTION_LINES
    samples: int = 8         # For BLUR


def generate_smear_sequence(
    sprite: np.ndarray,
    velocities: List[Tuple[float, float]],
    config: SmearConfig = None
) -> List[np.ndarray]:
    """
    Generate smear frames from velocity sequence.
    
    Args:
        sprite: Base sprite (RGBA uint8)
        velocities: List of (vx, vy) per frame
        config: Smear configuration
    
    Returns:
        List of smeared frames
    
    Example:
        # Attack animation velocities
        velocities = [(0, 0), (5, 0), (15, 0), (8, 0), (2, 0), (0, 0)]
        
        frames = generate_smear_sequence(sprite, velocities)
    """
    config = config or SmearConfig()
    frames = []
    
    for vx, vy in velocities:
        speed = np.sqrt(vx * vx + vy * vy)
        
        if speed < 1.0:
            # No smear for slow motion
            frames.append(sprite.copy())
        else:
            smeared = add_smear_frame(
                sprite,
                direction=(vx, vy),
                speed=speed,
                smear_type=config.smear_type,
                intensity=config.intensity,
                preserve_original=config.preserve_original
            )
            frames.append(smeared)
    
    return frames


def create_attack_smear(
    sprite: np.ndarray,
    direction: float,
    peak_speed: float = 15.0,
    attack_frames: int = 3,
    recover_frames: int = 2
) -> List[np.ndarray]:
    """
    Create attack animation smear sequence.
    
    Generates anticipation -> attack -> recover with appropriate smears.
    
    Args:
        sprite: Base sprite
        direction: Attack direction (radians)
        peak_speed: Maximum speed during attack
        attack_frames: Frames for main attack
        recover_frames: Frames to recover
    
    Returns:
        List of smeared frames
    
    Example:
        # Horizontal slash
        frames = create_attack_smear(sprite, direction=0, peak_speed=20)
    """
    velocities = []
    
    # Build up (anticipation - slight backward)
    velocities.append((-np.cos(direction) * 2, -np.sin(direction) * 2))
    
    # Attack (fast forward)
    for i in range(attack_frames):
        t = (i + 1) / attack_frames
        speed = peak_speed * (1 - (1 - t) ** 2)  # Ease out
        velocities.append((np.cos(direction) * speed, np.sin(direction) * speed))
    
    # Recover (slow down)
    for i in range(recover_frames):
        t = (i + 1) / recover_frames
        speed = peak_speed * (1 - t) * 0.3
        velocities.append((np.cos(direction) * speed, np.sin(direction) * speed))
    
    return generate_smear_sequence(
        sprite,
        velocities,
        SmearConfig(smear_type=SmearType.STRETCH, intensity=1.2)
    )


def create_impact_smear(
    sprite: np.ndarray,
    impact_direction: float,
    impact_force: float = 10.0
) -> np.ndarray:
    """
    Create single impact smear frame.
    
    Args:
        sprite: Base sprite
        impact_direction: Direction of impact (radians)
        impact_force: Impact strength
    
    Returns:
        Squashed impact frame
    """
    dx = np.cos(impact_direction)
    dy = np.sin(impact_direction)
    
    return add_smear_frame(
        sprite,
        direction=(dx, dy),
        speed=impact_force,
        smear_type=SmearType.SQUASH,
        intensity=1.0,
        preserve_original=False
    )


def create_dash_smear(
    sprite: np.ndarray,
    direction: float,
    speed: float = 12.0,
    trail_type: SmearType = SmearType.ONION
) -> np.ndarray:
    """
    Create dash/dodge smear with trail.
    
    Args:
        sprite: Base sprite
        direction: Dash direction (radians)
        speed: Dash speed
        trail_type: Type of trail effect
    
    Returns:
        Smeared dash frame
    """
    return add_smear_frame(
        sprite,
        direction=direction,
        speed=speed,
        smear_type=trail_type,
        intensity=1.0,
        preserve_original=True
    )
