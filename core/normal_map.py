"""
Normal Map Support

Enable 3D-like lighting effects using normal maps:
- Load external normal maps (standard RGB encoding)
- Auto-generate normal maps from sprite (height estimation)
- Directional lighting with specular highlights
- Animated light sources for dynamic effects
- Rim lighting, ambient occlusion approximation

Normal Map Encoding (Standard):
- R channel = X normal (-1 to +1 mapped to 0-255)
- G channel = Y normal (-1 to +1 mapped to 0-255)  
- B channel = Z normal (0 to +1 mapped to 128-255, pointing out of screen)
- Neutral normal (flat surface facing camera) = RGB(128, 128, 255)
"""

import numpy as np
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image


# =============================================================================
# Normal Map Loading & Generation
# =============================================================================

def load_normal_map(path: Union[str, Path]) -> np.ndarray:
    """
    Load a normal map from file.
    
    Args:
        path: Path to normal map image (PNG recommended)
        
    Returns:
        Normal vectors array (H, W, 3) with values in range [-1, 1]
    """
    img = Image.open(path).convert('RGB')
    normals_raw = np.array(img, dtype=np.float32)
    
    # Decode from 0-255 to -1..1 range
    # Standard encoding: 128 = 0, 0 = -1, 255 = +1
    normals = (normals_raw / 255.0) * 2.0 - 1.0
    
    # Normalize vectors (they should already be normalized, but ensure it)
    magnitude = np.sqrt(np.sum(normals ** 2, axis=2, keepdims=True))
    normals = normals / np.maximum(magnitude, 0.001)
    
    return normals


def encode_normal_map(normals: np.ndarray) -> np.ndarray:
    """
    Encode normal vectors to standard RGB normal map.
    
    Args:
        normals: Normal vectors (H, W, 3) in range [-1, 1]
        
    Returns:
        RGB image (H, W, 3) with values 0-255
    """
    # Encode from -1..1 to 0-255
    encoded = ((normals + 1.0) / 2.0 * 255.0)
    return np.clip(encoded, 0, 255).astype(np.uint8)


def generate_normal_map_from_height(
    height_map: np.ndarray,
    strength: float = 1.0
) -> np.ndarray:
    """
    Generate normal map from a height/depth map.
    
    Args:
        height_map: 2D array of height values (higher = closer to camera)
        strength: Normal map intensity (higher = more pronounced bumps)
        
    Returns:
        Normal vectors (H, W, 3)
    """
    h, w = height_map.shape
    
    # Compute gradients using Sobel-like operators
    # X gradient (horizontal)
    gx = np.zeros_like(height_map)
    gx[:, :-1] = height_map[:, 1:] - height_map[:, :-1]
    
    # Y gradient (vertical) 
    gy = np.zeros_like(height_map)
    gy[:-1, :] = height_map[1:, :] - height_map[:-1, :]
    
    # Scale gradients by strength
    gx *= strength
    gy *= strength
    
    # Build normal vectors
    # Normal = normalize((-gx, -gy, 1))
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[:, :, 0] = -gx  # X
    normals[:, :, 1] = -gy  # Y (flip for standard convention)
    normals[:, :, 2] = 1.0  # Z (pointing out)
    
    # Normalize
    magnitude = np.sqrt(np.sum(normals ** 2, axis=2, keepdims=True))
    normals = normals / np.maximum(magnitude, 0.001)
    
    return normals


def generate_normal_map_from_sprite(
    sprite: np.ndarray,
    method: str = 'luminance',
    strength: float = 2.0,
    invert: bool = False
) -> np.ndarray:
    """
    Auto-generate a normal map from sprite appearance.
    
    Args:
        sprite: RGBA sprite image
        method: Height estimation method:
            - 'luminance': Brighter = higher (good for gems, metals)
            - 'alpha': Alpha edge detection (good for flat sprites)
            - 'sobel': Edge-based (emphasizes outlines)
            - 'spherical': Assumes spherical/rounded shape
        strength: Normal intensity
        invert: Invert height (darker = higher)
        
    Returns:
        Normal vectors (H, W, 3)
    """
    h, w = sprite.shape[:2]
    
    if method == 'luminance':
        # Use brightness as height
        if sprite.shape[2] >= 3:
            height = (
                0.299 * sprite[:, :, 0] +
                0.587 * sprite[:, :, 1] +
                0.114 * sprite[:, :, 2]
            ).astype(np.float32) / 255.0
        else:
            height = sprite[:, :, 0].astype(np.float32) / 255.0
        
        if invert:
            height = 1.0 - height
            
    elif method == 'alpha':
        # Use alpha channel - edges are lower
        if sprite.shape[2] >= 4:
            alpha = sprite[:, :, 3].astype(np.float32) / 255.0
        else:
            alpha = np.ones((h, w), dtype=np.float32)
        
        # Distance from edge as height using local SDF
        from .sdf import generate_sdf as sdf_generate
        sdf = sdf_generate(sprite, normalize=False)
        height = np.clip(-sdf / 10.0, 0, 1)  # Inside sprite = height
        
    elif method == 'sobel':
        # Edge detection for height
        if sprite.shape[2] >= 3:
            gray = (
                0.299 * sprite[:, :, 0] +
                0.587 * sprite[:, :, 1] +
                0.114 * sprite[:, :, 2]
            ).astype(np.float32) / 255.0
        else:
            gray = sprite[:, :, 0].astype(np.float32) / 255.0
        
        # Sobel gradients
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        
        height = np.sqrt(gx**2 + gy**2)
        height = 1.0 - np.clip(height * 5, 0, 1)  # Edges are lower
        
    elif method == 'spherical':
        # Assume center is highest, edges lowest (dome shape)
        if sprite.shape[2] >= 4:
            alpha = sprite[:, :, 3].astype(np.float32) / 255.0
        else:
            alpha = np.ones((h, w), dtype=np.float32)
        
        # Find centroid of visible pixels
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        total_alpha = alpha.sum()
        if total_alpha > 0:
            cx = (x_coords * alpha).sum() / total_alpha
            cy = (y_coords * alpha).sum() / total_alpha
        else:
            cx, cy = w / 2, h / 2
        
        # Distance from center
        dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        max_dist = max(dist[alpha > 0.5].max() if (alpha > 0.5).any() else 1, 1)
        
        # Spherical height profile
        normalized_dist = dist / max_dist
        height = np.sqrt(np.maximum(1 - normalized_dist**2, 0))
        height *= alpha  # Zero outside sprite
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return generate_normal_map_from_height(height, strength)


def create_flat_normal_map(shape: Tuple[int, int]) -> np.ndarray:
    """Create a flat normal map (all normals pointing at camera)."""
    h, w = shape
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[:, :, 2] = 1.0  # Z = 1 (pointing out)
    return normals


# =============================================================================
# Lighting Calculations
# =============================================================================

@dataclass
class LightSource:
    """Configuration for a light source"""
    direction: Tuple[float, float, float] = (0.5, -0.5, 1.0)  # Light direction (from light toward surface)
    color: Tuple[int, int, int] = (255, 255, 255)  # Light color
    intensity: float = 1.0  # Light brightness
    ambient: float = 0.3  # Ambient light level
    specular: float = 0.5  # Specular highlight intensity
    specular_power: int = 32  # Specular sharpness (higher = smaller highlight)


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector or array of vectors."""
    if v.ndim == 1:
        mag = np.sqrt(np.sum(v ** 2))
        return v / max(mag, 0.001)
    else:
        mag = np.sqrt(np.sum(v ** 2, axis=-1, keepdims=True))
        return v / np.maximum(mag, 0.001)


def apply_directional_light(
    sprite: np.ndarray,
    normals: np.ndarray,
    light: Optional[LightSource] = None
) -> np.ndarray:
    """
    Apply directional lighting to sprite using normal map.
    
    Args:
        sprite: RGBA sprite image
        normals: Normal map (H, W, 3) with vectors in [-1, 1]
        light: Light source configuration
        
    Returns:
        Lit RGBA sprite
    """
    light = light or LightSource()
    
    h, w = sprite.shape[:2]
    result = sprite.copy().astype(np.float32)
    
    # Normalize light direction (pointing FROM light TO surface)
    light_dir = normalize_vector(np.array(light.direction, dtype=np.float32))
    
    # Flip to get direction FROM surface TO light for dot product
    light_dir = -light_dir
    
    # Diffuse lighting: N · L
    # Dot product of normal and light direction
    diffuse = np.sum(normals * light_dir, axis=2)
    diffuse = np.clip(diffuse, 0, 1)  # Only positive (facing light)
    
    # Combine ambient and diffuse
    lighting = light.ambient + (1 - light.ambient) * diffuse * light.intensity
    
    # Apply to RGB channels
    light_color = np.array(light.color, dtype=np.float32) / 255.0
    for c in range(3):
        result[:, :, c] = result[:, :, c] * lighting * light_color[c]
    
    # Add specular highlights (Blinn-Phong)
    if light.specular > 0:
        # View direction (straight out of screen)
        view_dir = np.array([0, 0, 1], dtype=np.float32)
        
        # Half vector between light and view
        half_vec = normalize_vector(light_dir + view_dir)
        
        # Specular: (N · H)^power
        spec = np.sum(normals * half_vec, axis=2)
        spec = np.clip(spec, 0, 1)
        spec = np.power(spec, light.specular_power)
        spec *= light.specular * light.intensity
        
        # Add specular as white highlight
        for c in range(3):
            result[:, :, c] = np.clip(
                result[:, :, c] + spec * 255 * light_color[c],
                0, 255
            )
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_rim_light(
    sprite: np.ndarray,
    normals: np.ndarray,
    color: Tuple[int, int, int] = (200, 220, 255),
    intensity: float = 0.5,
    power: float = 2.0
) -> np.ndarray:
    """
    Apply rim/edge lighting effect.
    
    Rim light illuminates edges where the surface faces away from camera.
    
    Args:
        sprite: RGBA sprite image
        normals: Normal map
        color: Rim light color
        intensity: Rim brightness
        power: Edge sharpness (higher = thinner rim)
        
    Returns:
        Sprite with rim lighting
    """
    result = sprite.copy().astype(np.float32)
    
    # View direction (camera looks at sprite)
    view_dir = np.array([0, 0, 1], dtype=np.float32)
    
    # Fresnel-like effect: 1 - (N · V)
    n_dot_v = np.sum(normals * view_dir, axis=2)
    n_dot_v = np.clip(n_dot_v, 0, 1)
    
    # Rim is stronger where normal faces away from camera
    rim = np.power(1 - n_dot_v, power) * intensity
    
    # Apply rim light color
    rim_color = np.array(color, dtype=np.float32)
    for c in range(3):
        result[:, :, c] = np.clip(
            result[:, :, c] + rim * rim_color[c],
            0, 255
        )
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_ambient_occlusion(
    sprite: np.ndarray,
    normals: np.ndarray,
    strength: float = 0.3
) -> np.ndarray:
    """
    Apply simple ambient occlusion approximation.
    
    Darkens areas where normals point away from up direction,
    simulating light being occluded in crevices.
    
    Args:
        sprite: RGBA sprite image
        normals: Normal map
        strength: AO darkness strength
        
    Returns:
        Sprite with ambient occlusion
    """
    result = sprite.copy().astype(np.float32)
    
    # "Up" direction for AO
    up_dir = np.array([0, -1, 0.5], dtype=np.float32)
    up_dir = normalize_vector(up_dir)
    
    # Normals facing down/inward get darkened
    ao = np.sum(normals * up_dir, axis=2)
    ao = np.clip(ao, 0, 1)
    ao = 1 - (1 - ao) * strength
    
    # Apply darkening
    for c in range(3):
        result[:, :, c] *= ao
    
    return np.clip(result, 0, 255).astype(np.uint8)


# =============================================================================
# Animated Lighting
# =============================================================================

def animate_light_direction(
    sprite: np.ndarray,
    normals: np.ndarray,
    time: float,
    orbit_radius: float = 0.7,
    light_height: float = 0.5,
    light: Optional[LightSource] = None
) -> np.ndarray:
    """
    Animate light source orbiting around sprite.
    
    Args:
        sprite: RGBA sprite
        normals: Normal map
        time: Animation time (0-1 for one full orbit)
        orbit_radius: How far light orbits from center (0-1)
        light_height: Height of light above sprite plane
        light: Base light configuration
        
    Returns:
        Lit sprite at given time
    """
    light = light or LightSource()
    
    # Calculate orbiting light position
    angle = time * 2 * np.pi
    lx = np.cos(angle) * orbit_radius
    ly = np.sin(angle) * orbit_radius
    lz = light_height
    
    # Create new light with animated direction
    animated_light = LightSource(
        direction=(lx, ly, lz),
        color=light.color,
        intensity=light.intensity,
        ambient=light.ambient,
        specular=light.specular,
        specular_power=light.specular_power
    )
    
    return apply_directional_light(sprite, normals, animated_light)


def animate_light_color(
    sprite: np.ndarray,
    normals: np.ndarray,
    time: float,
    colors: List[Tuple[int, int, int]] = None,
    light: Optional[LightSource] = None
) -> np.ndarray:
    """
    Animate light color cycling.
    
    Args:
        sprite: RGBA sprite
        normals: Normal map
        time: Animation time (0-1 for one full cycle)
        colors: List of colors to cycle through
        light: Base light configuration
        
    Returns:
        Lit sprite with color at given time
    """
    if colors is None:
        colors = [
            (255, 200, 150),  # Warm
            (255, 255, 255),  # White
            (150, 200, 255),  # Cool
            (255, 255, 255),  # White
        ]
    
    light = light or LightSource()
    
    # Interpolate between colors
    num_colors = len(colors)
    segment = time * num_colors
    idx = int(segment) % num_colors
    next_idx = (idx + 1) % num_colors
    t = segment - int(segment)
    
    # Smooth interpolation
    t = t * t * (3 - 2 * t)  # Smoothstep
    
    color = tuple(
        int(colors[idx][c] * (1 - t) + colors[next_idx][c] * t)
        for c in range(3)
    )
    
    animated_light = LightSource(
        direction=light.direction,
        color=color,
        intensity=light.intensity,
        ambient=light.ambient,
        specular=light.specular,
        specular_power=light.specular_power
    )
    
    return apply_directional_light(sprite, normals, animated_light)


def animate_light_pulse(
    sprite: np.ndarray,
    normals: np.ndarray,
    time: float,
    min_intensity: float = 0.5,
    max_intensity: float = 1.5,
    light: Optional[LightSource] = None
) -> np.ndarray:
    """
    Animate pulsing light intensity.
    
    Args:
        sprite: RGBA sprite
        normals: Normal map
        time: Animation time (0-1 for one pulse cycle)
        min_intensity: Minimum light brightness
        max_intensity: Maximum light brightness
        light: Base light configuration
        
    Returns:
        Lit sprite with pulsing intensity
    """
    light = light or LightSource()
    
    # Sine wave intensity
    pulse = (np.sin(time * 2 * np.pi) + 1) / 2
    intensity = min_intensity + (max_intensity - min_intensity) * pulse
    
    animated_light = LightSource(
        direction=light.direction,
        color=light.color,
        intensity=intensity,
        ambient=light.ambient,
        specular=light.specular * intensity,
        specular_power=light.specular_power
    )
    
    return apply_directional_light(sprite, normals, animated_light)


# =============================================================================
# Effect Integration
# =============================================================================

def apply_normal_mapped_glow(
    sprite: np.ndarray,
    normals: np.ndarray,
    glow_color: Tuple[int, int, int] = (255, 200, 100),
    glow_intensity: float = 1.0,
    glow_direction: Tuple[float, float, float] = (0, 0, 1)
) -> np.ndarray:
    """
    Apply glow effect that follows surface normals.
    
    Glow is strongest where normals face the glow direction.
    
    Args:
        sprite: RGBA sprite
        normals: Normal map
        glow_color: Color of the glow
        glow_intensity: Glow brightness
        glow_direction: Direction glow emanates from
        
    Returns:
        Sprite with normal-mapped glow
    """
    h, w = sprite.shape[:2]
    result = sprite.copy().astype(np.float32)
    
    # Normalize glow direction
    glow_dir = normalize_vector(np.array(glow_direction, dtype=np.float32))
    
    # Glow intensity based on how much normal faces glow direction
    glow_factor = np.sum(normals * glow_dir, axis=2)
    glow_factor = np.clip(glow_factor, 0, 1)
    glow_factor = np.power(glow_factor, 0.5)  # Soften falloff
    glow_factor *= glow_intensity
    
    # Apply glow additively
    glow_rgb = np.array(glow_color, dtype=np.float32)
    for c in range(3):
        result[:, :, c] = np.clip(
            result[:, :, c] + glow_factor * glow_rgb[c] * 0.5,
            0, 255
        )
    
    return np.clip(result, 0, 255).astype(np.uint8)


def create_lit_animation(
    sprite: np.ndarray,
    normals: np.ndarray,
    frames: int = 8,
    animation: str = 'orbit',
    light: Optional[LightSource] = None,
    **kwargs
) -> List[np.ndarray]:
    """
    Create animation with moving light source.
    
    Args:
        sprite: RGBA sprite
        normals: Normal map
        frames: Number of animation frames
        animation: Animation type ('orbit', 'pulse', 'color')
        light: Base light configuration
        **kwargs: Additional arguments for specific animations
        
    Returns:
        List of lit frames
    """
    result_frames = []
    
    for i in range(frames):
        t = i / frames
        
        if animation == 'orbit':
            frame = animate_light_direction(
                sprite, normals, t,
                orbit_radius=kwargs.get('orbit_radius', 0.7),
                light_height=kwargs.get('light_height', 0.5),
                light=light
            )
        elif animation == 'pulse':
            frame = animate_light_pulse(
                sprite, normals, t,
                min_intensity=kwargs.get('min_intensity', 0.5),
                max_intensity=kwargs.get('max_intensity', 1.5),
                light=light
            )
        elif animation == 'color':
            frame = animate_light_color(
                sprite, normals, t,
                colors=kwargs.get('colors'),
                light=light
            )
        else:
            frame = apply_directional_light(sprite, normals, light)
        
        # Optional: add rim light
        if kwargs.get('rim_light', False):
            frame = apply_rim_light(
                frame, normals,
                intensity=kwargs.get('rim_intensity', 0.3)
            )
        
        result_frames.append(frame)
    
    return result_frames


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_light(
    sprite: np.ndarray,
    direction: str = 'top-left',
    intensity: float = 1.0,
    specular: float = 0.5
) -> np.ndarray:
    """
    Quick lighting with preset directions.
    
    Args:
        sprite: RGBA sprite
        direction: 'top-left', 'top', 'top-right', 'left', 'right', etc.
        intensity: Light brightness
        specular: Specular highlight amount
        
    Returns:
        Lit sprite (auto-generates normal map)
    """
    # Direction presets
    directions = {
        'top-left': (0.5, -0.5, 1.0),
        'top': (0, -0.7, 0.7),
        'top-right': (-0.5, -0.5, 1.0),
        'left': (0.7, 0, 0.7),
        'right': (-0.7, 0, 0.7),
        'bottom-left': (0.5, 0.5, 1.0),
        'bottom': (0, 0.7, 0.7),
        'bottom-right': (-0.5, 0.5, 1.0),
        'front': (0, 0, 1.0),
    }
    
    light_dir = directions.get(direction, directions['top-left'])
    
    # Auto-generate normal map
    normals = generate_normal_map_from_sprite(sprite, method='spherical')
    
    light = LightSource(
        direction=light_dir,
        intensity=intensity,
        specular=specular
    )
    
    return apply_directional_light(sprite, normals, light)


def quick_gem_lighting(
    sprite: np.ndarray,
    highlight_color: Tuple[int, int, int] = (255, 255, 255),
    ambient_color: Tuple[int, int, int] = (100, 100, 150)
) -> np.ndarray:
    """
    Quick gem/crystal lighting preset.
    
    Creates sparkly, faceted appearance with strong specular.
    """
    normals = generate_normal_map_from_sprite(sprite, method='luminance', strength=3.0)
    
    light = LightSource(
        direction=(0.4, -0.4, 1.0),
        color=highlight_color,
        intensity=1.2,
        ambient=0.4,
        specular=0.8,
        specular_power=64
    )
    
    result = apply_directional_light(sprite, normals, light)
    result = apply_rim_light(result, normals, color=ambient_color, intensity=0.4)
    
    return result


def quick_metal_lighting(
    sprite: np.ndarray,
    metal_color: Tuple[int, int, int] = (200, 200, 220)
) -> np.ndarray:
    """
    Quick metallic lighting preset.
    
    Creates smooth, reflective metal appearance.
    """
    normals = generate_normal_map_from_sprite(sprite, method='sobel', strength=1.5)
    
    light = LightSource(
        direction=(0.3, -0.5, 0.8),
        color=metal_color,
        intensity=1.0,
        ambient=0.35,
        specular=0.9,
        specular_power=48
    )
    
    result = apply_directional_light(sprite, normals, light)
    result = apply_rim_light(result, normals, color=(180, 180, 200), intensity=0.3)
    
    return result
