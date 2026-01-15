"""
Enhanced Bounce Effect - Physics-accurate bouncing with squash/stretch
Professional-quality with proper elasticity and gamma-correct rendering
"""

import numpy as np
from typing import List, Optional
from .base import BaseEffect, EffectConfig, Easing, PixelMath
from ..core.parser import Sprite


class BounceEffect(BaseEffect):
    """Creates physics-accurate bouncing with squash/stretch deformation"""
    
    name = "bounce"
    description = "Physics-based bouncing with squash and stretch"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        
        self.height_amount = self.config.extra.get('height', 6.0)
        self.squash = self.config.extra.get('squash', True)
        self.bounce_count = self.config.extra.get('bounces', 1)
        self.elasticity = self.config.extra.get('elasticity', 0.85)
        self.quality = self.config.extra.get('quality', 'high')
        self.anticipation = self.config.extra.get('anticipation', True)
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        
        # Find bottom anchor (sprite's feet)
        mask = original[:, :, 3] > 0 if original.shape[2] == 4 else np.ones((h, w), bool)
        rows = np.any(mask, axis=1)
        anchor_y = float(np.where(rows)[0][-1]) if np.any(rows) else float(h - 1)
        
        # Pre-compute coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        cx = w / 2.0
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            
            # Use real physics-based bounce with decay
            bounce, velocity, impact = self._physics_bounce(t)
            
            # Height offset scaled by intensity
            dy = bounce * self.height_amount * self.config.intensity
            
            # Advanced squash/stretch with volume preservation
            if self.squash:
                scale_x, scale_y = self._calculate_deformation(impact, velocity)
            else:
                scale_x = scale_y = 1.0
            
            # Apply transformations using proper interpolation
            frame_pixels = self._smooth_transform(
                original, scale_x, scale_y, dy, anchor_y, cx,
                x_coords, y_coords
            )
            
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _physics_bounce(self, t: float) -> tuple:
        """
        Real physics bounce with parabolic trajectory.
        Returns: (height_normalized 0-1, velocity, impact_factor)
        """
        cycles = self.bounce_count
        cycle_t = (t * cycles) % 1.0
        bounce_num = int(t * cycles)
        
        # Decay factor (energy loss per bounce)
        decay = self.elasticity ** bounce_num
        
        # Parabolic trajectory: h = 1 - (2t-1)^2
        parabola = 1.0 - (2.0 * cycle_t - 1.0) ** 2
        
        bounce_height = parabola * decay
        
        # Velocity (derivative of parabola)
        velocity = -4.0 * (2.0 * cycle_t - 1.0) * decay
        
        # Impact factor: smooth transition near ground
        # Use cubic falloff for more cartoon feel
        impact = (1.0 - parabola) ** 0.5
        impact *= decay  # Also reduce impact with bounces
        
        return bounce_height, velocity, impact
    
    def _calculate_deformation(self, impact: float, velocity: float) -> tuple:
        """
        Calculate squash/stretch with volume preservation and anticipation.
        """
        base_squash = 0.25 * self.config.intensity
        
        # Impact squash (on landing)
        if impact > 0.5:
            squash_factor = (impact - 0.5) * 2.0  # 0 to 1
            squash_factor = Easing.ease_out_cubic(squash_factor)
            
            scale_y = 1.0 - squash_factor * base_squash
            # Volume preservation: area constant means x * y = 1
            scale_x = 1.0 / scale_y
        
        # Stretch when rising fast (anticipation)
        elif velocity < -0.5 and self.anticipation:
            stretch_factor = (abs(velocity) - 0.5) / 1.5
            stretch_factor = min(stretch_factor, 1.0)
            
            scale_y = 1.0 + stretch_factor * base_squash * 0.6
            scale_x = 1.0 / scale_y
        
        else:
            scale_x = scale_y = 1.0
        
        return scale_x, scale_y
    
    def _smooth_transform(
        self, pixels: np.ndarray, scale_x: float, scale_y: float,
        dy: float, anchor_y: float, cx: float,
        x_coords: np.ndarray, y_coords: np.ndarray
    ) -> np.ndarray:
        """Apply smooth squash/stretch anchored at bottom with high-quality interpolation"""
        # Inverse transform: where to sample from
        src_x = cx + (x_coords - cx) / scale_x
        src_y = anchor_y + (y_coords - anchor_y + dy) / scale_y
        
        # Sample with appropriate quality level
        if self.quality == 'best':
            return PixelMath.lanczos_sample(pixels, src_x, src_y, gamma_correct=True)
        elif self.quality == 'high':
            return PixelMath.bilinear_sample(pixels, src_x, src_y, gamma_correct=True)
        else:
            return PixelMath.bilinear_sample(pixels, src_x, src_y, gamma_correct=False)
