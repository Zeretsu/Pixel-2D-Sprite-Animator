"""
Enhanced Float Effect - Ultra-smooth hovering/bobbing animation
Professional-quality with physics-based motion and gamma-correct rendering
"""

import numpy as np
from typing import List, Optional
from .base import BaseEffect, EffectConfig, Easing, PixelMath
from ..core.parser import Sprite


class FloatEffect(BaseEffect):
    """Creates ultra-smooth floating/bobbing animation with physics-based motion"""
    
    name = "float"
    description = "Smooth floating and hovering motion"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        
        self.amplitude = self.config.extra.get('amplitude', 3.0)
        self.secondary_motion = self.config.extra.get('secondary', True)
        self.rotation = self.config.extra.get('rotation', True)
        self.horizontal_sway = self.config.extra.get('horizontal_sway', 0.5)
        self.quality = self.config.extra.get('quality', 'high')
        self.physics_mode = self.config.extra.get('physics', False)  # Use spring physics
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        original = sprite.pixels.copy()
        h, w = sprite.height, sprite.width
        
        # Pre-compute coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2.0, h / 2.0
        
        for i in range(self.config.frame_count):
            # Normalized time for seamless looping (0 to 1, exclusive of 1)
            t = i / self.config.frame_count
            
            if self.physics_mode:
                # Physics-based spring motion
                primary = Easing.spring(t, stiffness=10.0, damping=0.0)
            else:
                # Primary smooth bob using sine (seamless at t=0 and t=1)
                primary = np.sin(t * 2 * np.pi)
            
            # Secondary smaller motion for organic feel (use 2x frequency for seamless)
            secondary = 0.0
            if self.secondary_motion:
                # Add third harmonic for more natural motion
                secondary = (
                    np.sin(t * 4 * np.pi) * 0.12 +
                    np.sin(t * 6 * np.pi) * 0.03  # Subtle high frequency detail
                )
            
            # Combine motions with intensity
            dy = (primary + secondary) * self.amplitude * self.config.intensity
            
            # Subtle horizontal sway with phase offset
            dx = (
                np.sin(t * 2 * np.pi + np.pi / 4) * self.horizontal_sway +
                np.sin(t * 4 * np.pi + np.pi / 3) * self.horizontal_sway * 0.15
            ) * self.config.intensity
            
            # Subpixel-accurate shift using proper interpolation
            src_x = x_coords - dx
            src_y = y_coords - dy
            
            if self.quality == 'best':
                frame_pixels = PixelMath.lanczos_sample(original, src_x, src_y, gamma_correct=True)
            elif self.quality == 'high':
                frame_pixels = PixelMath.bilinear_sample(original, src_x, src_y, gamma_correct=True)
            else:
                frame_pixels = PixelMath.bilinear_sample(original, src_x, src_y, gamma_correct=False)
            
            # Optional subtle tilt synchronized with horizontal motion
            if self.rotation:
                tilt_angle = np.sin(t * 2 * np.pi) * 0.02 * self.config.intensity
                if abs(tilt_angle) > 0.001:
                    frame_pixels = self._apply_smooth_rotation(
                        frame_pixels, tilt_angle, cx, cy
                    )
            
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _apply_smooth_rotation(
        self, 
        pixels: np.ndarray, 
        angle: float, 
        cx: float, 
        cy: float
    ) -> np.ndarray:
        """Apply smooth rotation with proper interpolation"""
        h, w = pixels.shape[:2]
        
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        cos_a = np.cos(-angle)  # Negative for inverse transform
        sin_a = np.sin(-angle)
        
        # Translate to center, rotate, translate back
        dx = x_coords - cx
        dy = y_coords - cy
        
        src_x = dx * cos_a - dy * sin_a + cx
        src_y = dx * sin_a + dy * cos_a + cy
        
        if self.quality == 'best':
            return PixelMath.lanczos_sample(pixels, src_x, src_y, gamma_correct=True)
        else:
            return PixelMath.bilinear_sample(pixels, src_x, src_y, gamma_correct=True)
