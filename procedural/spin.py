"""
Enhanced Spin Effect - Smooth rotation animation
Fixed center-of-mass rotation with proper alpha handling
"""

import numpy as np
from typing import List, Optional
from .base import BaseEffect, EffectConfig, PixelMath, Easing
from ..core.parser import Sprite


class SpinEffect(BaseEffect):
    """Creates smooth spinning/rotation effect with proper centering"""
    
    name = "spin"
    description = "Smooth rotation and spinning"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        
        self.rotations = self.config.extra.get('rotations', 1)
        self.direction = self.config.extra.get('direction', 1)
        self.ease_rotation = self.config.extra.get('ease', False)
        self.quality = self.config.extra.get('quality', 'high')
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        
        # Find the actual center of the sprite (center of mass of visible pixels)
        cx, cy = self._find_sprite_center(original)
        
        # Pre-compute coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Total rotation angle
        total_angle = 2 * np.pi * self.rotations * self.direction * self.config.intensity
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            
            # Calculate angle with optional easing
            if self.ease_rotation:
                eased_t = Easing.smooth_step(t)
                angle = eased_t * total_angle
            else:
                angle = t * total_angle
            
            frame_pixels = self._rotate_clean(original, angle, cx, cy, x_coords, y_coords)
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _find_sprite_center(self, pixels: np.ndarray) -> tuple:
        """Find the center of mass of visible pixels"""
        h, w = pixels.shape[:2]
        
        if pixels.shape[2] == 4:
            # Use alpha channel to find visible pixels
            alpha = pixels[:, :, 3].astype(np.float32)
        else:
            # Assume all non-black pixels are visible
            alpha = np.any(pixels[:, :, :3] > 0, axis=2).astype(np.float32)
        
        total_alpha = np.sum(alpha)
        
        if total_alpha < 1:
            # Fallback to geometric center
            return w / 2.0, h / 2.0
        
        # Calculate center of mass
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        cx = np.sum(x_coords * alpha) / total_alpha
        cy = np.sum(y_coords * alpha) / total_alpha
        
        return cx, cy
    
    def _rotate_clean(
        self, 
        pixels: np.ndarray, 
        angle: float, 
        cx: float, 
        cy: float,
        x_coords: np.ndarray,
        y_coords: np.ndarray
    ) -> np.ndarray:
        """Rotate with clean alpha handling - no trails"""
        h, w = pixels.shape[:2]
        
        # Inverse rotation to find source coordinates
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)
        
        # Translate to center, rotate, translate back
        dx = x_coords - cx
        dy = y_coords - cy
        
        src_x = dx * cos_a - dy * sin_a + cx
        src_y = dx * sin_a + dy * cos_a + cy
        
        # Create output with transparent background
        result = np.zeros_like(pixels)
        
        # Get valid source coordinates (within bounds)
        valid = (src_x >= 0) & (src_x < w - 1) & (src_y >= 0) & (src_y < h - 1)
        
        # Only sample where source is valid
        if not np.any(valid):
            return result
        
        # Bilinear interpolation manually for clean results
        x0 = np.floor(src_x).astype(np.int32)
        y0 = np.floor(src_y).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        
        # Fractional parts
        fx = src_x - x0
        fy = src_y - y0
        
        # Clamp to valid range for indexing
        x0_c = np.clip(x0, 0, w - 1)
        y0_c = np.clip(y0, 0, h - 1)
        x1_c = np.clip(x1, 0, w - 1)
        y1_c = np.clip(y1, 0, h - 1)
        
        # Get the four corner pixels
        p00 = pixels[y0_c, x0_c].astype(np.float32)
        p10 = pixels[y0_c, x1_c].astype(np.float32)
        p01 = pixels[y1_c, x0_c].astype(np.float32)
        p11 = pixels[y1_c, x1_c].astype(np.float32)
        
        # Bilinear weights
        w00 = ((1 - fx) * (1 - fy))[:, :, np.newaxis]
        w10 = (fx * (1 - fy))[:, :, np.newaxis]
        w01 = ((1 - fx) * fy)[:, :, np.newaxis]
        w11 = (fx * fy)[:, :, np.newaxis]
        
        # Interpolate all channels
        interpolated = p00 * w00 + p10 * w10 + p01 * w01 + p11 * w11
        
        # Apply only to valid pixels
        result[valid] = np.clip(interpolated[valid], 0, 255).astype(np.uint8)
        
        # For invalid source coords, keep transparent
        return result
