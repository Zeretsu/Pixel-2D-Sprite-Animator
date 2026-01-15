"""
Smoke Effect - Drifting smoke/cloud animation with pixel-perfect rendering
"""

import numpy as np
from typing import List, Optional
from .base import BaseEffect, EffectConfig, PixelMath
from .noise import NoiseGenerator
from ..core.parser import Sprite


class SmokeEffect(BaseEffect):
    """Creates smoke/cloud drifting animation with proper linear color sampling"""
    
    name = "smoke"
    description = "Soft drifting motion for smoke and clouds"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        self.noise = NoiseGenerator(self.config.seed)
        
        # Effect-specific settings
        self.drift_x = self.config.extra.get('drift_x', 1.0)
        self.drift_y = self.config.extra.get('drift_y', -0.5)  # Negative = upward
        self.turbulence = self.config.extra.get('turbulence', 0.3)
        self.fade = self.config.extra.get('fade', True)
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        """Apply smoke effect to sprite"""
        frames = []
        
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        
        # Pre-compute coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Pre-compute center distance for fading (vectorized)
        cy, cx = h / 2.0, w / 2.0
        dist_from_center = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        edge_factor = dist_from_center / max_dist
        
        for i in range(self.config.frame_count):
            t = i / max(1, self.config.frame_count - 1) if self.config.frame_count > 1 else 0.0
            
            # Calculate combined displacement with subpixel precision
            src_x, src_y = self._calculate_displacement(
                x_coords, y_coords, t, h, w
            )
            
            # Sample with proper bilinear interpolation and gamma correction
            frame_pixels = PixelMath.bilinear_sample(original, src_x, src_y, gamma_correct=True)
            
            # Apply fading at edges (vectorized)
            if self.fade:
                frame_pixels = self._apply_fade_vectorized(
                    frame_pixels, edge_factor, t
                )
            
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _calculate_displacement(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        t: float,
        h: int,
        w: int
    ) -> tuple:
        """Calculate combined drift and turbulence displacement"""
        # Smooth drift with subpixel precision (no integer truncation)
        drift_dx = np.sin(t * 2 * np.pi) * self.drift_x * self.config.intensity
        drift_dy = np.sin(t * 2 * np.pi) * self.drift_y * self.config.intensity
        
        # Generate smooth turbulence displacement
        noise_x = self.noise.perlin_2d(w, h, scale=8, time=t)
        noise_y = self.noise.perlin_2d(w, h, scale=8, time=t + 100)
        
        turb = self.turbulence * self.config.intensity * 3.0
        
        # Combine displacements (subpixel accurate)
        src_x = x_coords - drift_dx - noise_x * turb
        src_y = y_coords - drift_dy - noise_y * turb
        
        return src_x, src_y
    
    def _apply_fade_vectorized(
        self,
        pixels: np.ndarray,
        edge_factor: np.ndarray,
        t: float
    ) -> np.ndarray:
        """Apply alpha fading vectorized (no Python loops)"""
        if pixels.shape[2] != 4:
            return pixels
        
        result = pixels.copy()
        
        # Smooth pulsing fade factor
        fade_factor = 0.9 + np.sin(t * 2 * np.pi) * 0.1
        
        # Calculate alpha multiplier (more fade at edges)
        alpha_mult = fade_factor - edge_factor * 0.2
        alpha_mult = np.clip(alpha_mult, 0.0, 1.0)
        
        # Apply to alpha channel
        result[:, :, 3] = (result[:, :, 3].astype(np.float32) * alpha_mult).astype(np.uint8)
        
        return result
