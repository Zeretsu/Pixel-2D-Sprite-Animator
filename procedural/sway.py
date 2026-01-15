"""
Enhanced Sway Effect - Physics-based organic swaying motion
Professional-quality with wind simulation and gamma-correct rendering
"""

import numpy as np
from typing import List, Optional
from .base import BaseEffect, EffectConfig, Easing, PixelMath
from ..core.parser import Sprite


class SwayEffect(BaseEffect):
    """Creates physics-based swaying animation (plants, candles, cloth)"""
    
    name = "sway"
    description = "Natural side-to-side swaying anchored at base"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        
        self.amplitude = self.config.extra.get('amplitude', 3.0)
        self.anchor = self.config.extra.get('anchor', 'bottom')
        self.wind_gusts = self.config.extra.get('wind_gusts', True)
        self.quality = self.config.extra.get('quality', 'high')
        self.spring_physics = self.config.extra.get('spring', True)  # Spring-mass simulation
        self.stiffness = self.config.extra.get('stiffness', 0.8)  # Higher = stiffer plant
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        mask = self._get_mask(original)
        
        anchor_y = self._get_anchor_y(mask, h)
        
        # Pre-compute coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Distance from anchor with non-linear falloff
        distance = np.abs(y_coords - anchor_y) / float(h)
        
        # Use cubic for more natural plant-like behavior
        # Stiffness affects how much the base moves vs tips
        sway_factor = np.power(distance, 2.0 + self.stiffness)
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            
            frame_pixels = self._apply_organic_sway(
                original, mask, t, h, w,
                x_coords, y_coords, sway_factor
            )
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _get_anchor_y(self, mask: np.ndarray, height: int) -> int:
        """Find anchor point based on anchor setting"""
        rows = np.any(mask, axis=1)
        if not np.any(rows):
            return height - 1
        
        y_indices = np.where(rows)[0]
        y_min, y_max = y_indices[0], y_indices[-1]
        
        if self.anchor == 'bottom':
            return y_max
        elif self.anchor == 'top':
            return y_min
        return (y_min + y_max) // 2
    
    def _apply_organic_sway(
        self, pixels: np.ndarray, mask: np.ndarray, 
        t: float, h: int, w: int,
        x_coords: np.ndarray, y_coords: np.ndarray,
        sway_factor: np.ndarray
    ) -> np.ndarray:
        """Vectorized organic sway with physics-based motion"""
        
        tau = t * 2 * np.pi
        
        if self.spring_physics:
            # Spring-mass pendulum physics
            # Natural frequency based on stiffness
            omega = 1.0 + self.stiffness * 0.5
            
            # Damped harmonic oscillator with multiple modes
            # Mode 1: Primary oscillation
            mode1 = np.sin(tau * omega) * 0.55
            
            # Mode 2: Second harmonic (faster, smaller)
            mode2 = np.sin(tau * omega * 2 + np.pi/4) * 0.28
            
            # Mode 3: Third harmonic (detail)
            mode3 = np.sin(tau * omega * 3 + np.pi/6) * 0.12
            
            # Phase variation along height for realistic wave propagation
            phase_offset = sway_factor * 0.5  # Higher parts lag slightly
            
            wave = (
                np.sin(tau * omega + phase_offset) * 0.55 +
                np.sin(tau * omega * 2 + phase_offset * 1.5 + np.pi/4) * 0.28 +
                np.sin(tau * omega * 3 + phase_offset * 2 + np.pi/6) * 0.12
            )
        else:
            # Simple multi-frequency wave
            wave = (
                np.sin(tau * 1) * 0.5 +
                np.sin(tau * 2) * 0.3 +
                np.sin(tau * 3) * 0.2
            )
        
        # Wind gust simulation
        if self.wind_gusts:
            # Gusts use smooth power function for natural feel
            gust_strength = np.power(np.clip(np.sin(tau * 4), 0, 1), 3)
            wave += gust_strength * 0.4 * np.sign(np.sin(tau * 4.5))
        
        # Calculate X offset
        offset = wave * self.amplitude * sway_factor * self.config.intensity
        
        # Source coordinates (inverse transform)
        src_x = x_coords - offset
        src_y = y_coords.copy()
        
        # Use appropriate quality sampling
        if self.quality == 'best':
            result = PixelMath.lanczos_sample(pixels, src_x, src_y, gamma_correct=True)
        elif self.quality == 'high':
            result = PixelMath.bilinear_sample(pixels, src_x, src_y, gamma_correct=True)
        else:
            result = PixelMath.bilinear_sample(pixels, src_x, src_y, gamma_correct=False)
        
        # Restore non-masked pixels from original
        result[~mask] = pixels[~mask]
        
        return result
