"""
Enhanced Wobble Effect - Smooth jelly/elastic deformation
Ultra-high quality with Lanczos resampling and gamma-correct blending
"""

import numpy as np
from typing import List, Optional
from .base import BaseEffect, EffectConfig, Easing, PixelMath
from ..core.parser import Sprite


class WobbleEffect(BaseEffect):
    """Creates smooth wobbly/jelly animation effect with artifact-free rendering"""
    
    name = "wobble"
    description = "Elastic jelly-like deformation with smooth interpolation"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        
        self.frequency = self.config.extra.get('frequency', 1.0)  # Use 1.0 for perfect loop
        self.amplitude = self.config.extra.get('amplitude', 2.5)
        self.squash = self.config.extra.get('squash', True)
        self.ripple = self.config.extra.get('ripple', True)
        self.quality = self.config.extra.get('quality', 'high')  # 'fast', 'high', or 'best'
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        mask = self._get_mask(original)
        
        # Find center of mass for natural deformation origin
        cy, cx = self._find_center(mask)
        
        # Pre-compute coordinate grids (optimization)
        y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
        dx_from_center = x_grid - cx
        dy_from_center = y_grid - cy
        dist_from_center = np.sqrt(dx_from_center**2 + dy_from_center**2 + 1e-6)
        angle_from_center = np.arctan2(dy_from_center, dx_from_center)
        
        # Compute normalized distance for falloff effects
        max_dist = np.sqrt(h**2 + w**2) / 2
        norm_dist = dist_from_center / max_dist
        
        for i in range(self.config.frame_count):
            # Use proper looping time (0 to 1, exclusive of 1 for seamless loop)
            t = i / self.config.frame_count
            
            frame_pixels = self._apply_smooth_wobble(
                original, mask, t, cx, cy, h, w,
                x_grid, y_grid, dx_from_center, dy_from_center,
                dist_from_center, angle_from_center, norm_dist
            )
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _find_center(self, mask: np.ndarray):
        """Find center of mass of masked region"""
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            h, w = mask.shape
            return h / 2.0, w / 2.0
        return float(np.mean(y_coords)), float(np.mean(x_coords))
    
    def _apply_smooth_wobble(
        self, pixels: np.ndarray, mask: np.ndarray,
        t: float, cx: float, cy: float, h: int, w: int,
        x_grid: np.ndarray, y_grid: np.ndarray,
        dx: np.ndarray, dy: np.ndarray,
        dist: np.ndarray, angle: np.ndarray, norm_dist: np.ndarray
    ) -> np.ndarray:
        """Vectorized smooth wobble with high-quality interpolation"""
        
        # Phase for seamless looping (full 2*pi cycle)
        phase = t * 2 * np.pi * self.frequency
        
        # Multi-frequency wobble for organic feel
        # Using integer frequency ratios for seamless looping
        # Add distance-based phase offset for wave propagation effect
        phase_offset = norm_dist * np.pi * 0.5
        
        wobble_radial = (
            np.sin(angle * 2 + phase + phase_offset) * 0.5 +
            np.sin(angle * 3 + phase * 2 + phase_offset * 1.5) * 0.3 +
            np.sin(angle * 5 + phase * 3 + phase_offset * 0.7) * 0.15 +
            np.sin(angle * 7 + phase * 4) * 0.05  # High frequency detail
        )
        
        # Amplitude falloff from center (more effect at edges)
        edge_factor = 0.3 + 0.7 * norm_dist
        
        # Compute displacement in X and Y
        intensity = self.amplitude * self.config.intensity * edge_factor
        wobble_x = wobble_radial * intensity * np.cos(angle + phase * 0.1)
        wobble_y = wobble_radial * intensity * np.sin(angle + phase * 0.1) * 0.8
        
        # Ripple effect (outward wave from center)
        if self.ripple:
            # Use distance-based ripple with proper phase for looping
            ripple_wave = np.sin(dist * 0.2 - phase * 2) * 0.35 * self.config.intensity
            # Gaussian falloff for ripple
            ripple_falloff = np.exp(-(norm_dist ** 2) * 2)
            ripple_x = ripple_wave * ripple_falloff * (dx / (dist + 1))
            ripple_y = ripple_wave * ripple_falloff * (dy / (dist + 1))
            wobble_x += ripple_x
            wobble_y += ripple_y
        
        # Squash and stretch (volume-preserving)
        if self.squash:
            # Use smooth breathing curve for squash
            squash_factor = Easing.breathing(t) * 0.05 * self.config.intensity
            scale_x = 1.0 + squash_factor
            scale_y = 1.0 - squash_factor  # Inverse for volume preservation
        else:
            scale_x = scale_y = 1.0
        
        # Compute source coordinates (inverse transform)
        src_x = cx + (dx / scale_x) - wobble_x
        src_y = cy + (dy / scale_y) - wobble_y
        
        # Use quality-selected sampling
        if self.quality == 'best':
            result = PixelMath.lanczos_sample(pixels, src_x, src_y, gamma_correct=True)
        elif self.quality == 'high':
            result = PixelMath.bilinear_sample(pixels, src_x, src_y, gamma_correct=True)
        else:
            result = PixelMath.bilinear_sample(pixels, src_x, src_y, gamma_correct=False)
        
        # Restore non-masked pixels from original
        result[~mask] = pixels[~mask]
        
        return result
