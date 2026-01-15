"""
Ripple Effect - Circular wave distortion
Magic/portal/impact wave effect
"""

import numpy as np
from typing import List, Optional
from .base import BaseEffect, EffectConfig, Easing, PixelMath
from ..core.parser import Sprite


class RippleEffect(BaseEffect):
    """Creates circular ripple/wave distortion from center"""
    
    name = "ripple"
    description = "Circular wave distortion for magic/portal effects"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        
        self.wave_count = self.config.extra.get('waves', 3)
        self.amplitude = self.config.extra.get('amplitude', 4.0)
        self.speed = self.config.extra.get('speed', 1.0)
        self.decay = self.config.extra.get('decay', True)  # Waves fade outward
        self.mode = self.config.extra.get('mode', 'expand')  # 'expand', 'contract', 'pulse'
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        
        # Find center
        cx, cy = self._find_center(original, w, h)
        
        # Pre-compute coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Distance and angle from center
        dx = x_coords - cx
        dy = y_coords - cy
        dist = np.sqrt(dx * dx + dy * dy)
        max_dist = np.sqrt(w * w + h * h) / 2
        
        # Direction vectors (normalized)
        with np.errstate(divide='ignore', invalid='ignore'):
            dir_x = np.where(dist > 0, dx / dist, 0)
            dir_y = np.where(dist > 0, dy / dist, 0)
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            
            frame_pixels = self._create_ripple_frame(
                original, t, dist, max_dist, dir_x, dir_y, x_coords, y_coords, h, w
            )
            
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _find_center(self, pixels: np.ndarray, w: int, h: int) -> tuple:
        """Find center of sprite"""
        if pixels.shape[2] == 4:
            alpha = pixels[:, :, 3].astype(np.float32)
        else:
            alpha = np.any(pixels > 0, axis=2).astype(np.float32)
        
        total = np.sum(alpha)
        if total < 1:
            return w / 2, h / 2
        
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        cx = np.sum(x_coords * alpha) / total
        cy = np.sum(y_coords * alpha) / total
        
        return cx, cy
    
    def _create_ripple_frame(
        self,
        original: np.ndarray,
        t: float,
        dist: np.ndarray,
        max_dist: float,
        dir_x: np.ndarray,
        dir_y: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        h: int,
        w: int
    ) -> np.ndarray:
        """Create a single ripple frame"""
        
        # Wave parameters
        wavelength = max_dist / self.wave_count
        wave_speed = self.speed * max_dist
        
        if self.mode == 'expand':
            # Waves expand outward
            phase = t * wave_speed
            wave_phase = (dist - phase) / wavelength * 2 * np.pi
        elif self.mode == 'contract':
            # Waves contract inward
            phase = (1 - t) * wave_speed
            wave_phase = (dist - phase) / wavelength * 2 * np.pi
        else:  # pulse
            # Standing wave that pulses
            wave_phase = dist / wavelength * 2 * np.pi - t * 2 * np.pi * self.speed
        
        # Calculate displacement
        displacement = np.sin(wave_phase) * self.amplitude * self.config.intensity
        
        # Apply decay (waves weaken with distance)
        if self.decay:
            if self.mode == 'expand':
                # Decay from wave front
                decay_factor = np.exp(-np.abs(dist - t * wave_speed) / (max_dist * 0.3))
            elif self.mode == 'contract':
                decay_factor = np.exp(-np.abs(dist - (1-t) * wave_speed) / (max_dist * 0.3))
            else:
                decay_factor = np.exp(-dist / max_dist)
            
            displacement *= decay_factor
        
        # Offset coordinates radially
        src_x = x_coords - dir_x * displacement
        src_y = y_coords - dir_y * displacement
        
        return PixelMath.bilinear_sample(original, src_x, src_y, gamma_correct=True)
