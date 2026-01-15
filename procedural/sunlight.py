"""
Sunlight Effect - Realistic sunlight patches on surfaces
Creates soft, dappled light that gently shimmers like natural sunlight
"""

import numpy as np
from typing import List, Optional
from .base import BaseEffect, EffectConfig
from ..core.parser import Sprite


class SunlightEffect(BaseEffect):
    """Creates realistic sunlight patches that gently shimmer"""
    
    name = "sunlight"
    description = "Soft dappled sunlight patches that gently shimmer"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        self.rng = np.random.default_rng(self.config.seed or 42)
        
        # Sunlight settings
        self.patch_count = self.config.extra.get('patches', 3)
        self.warmth = self.config.extra.get('warmth', 0.1)  # Add warm tint
        self.shimmer_speed = self.config.extra.get('shimmer_speed', 1.0)
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        original = sprite.pixels.copy()
        h, w = sprite.height, sprite.width
        
        # Pre-generate sunlight patch positions and sizes
        patches = []
        for _ in range(self.patch_count):
            patches.append({
                'x': self.rng.random() * w,
                'y': self.rng.random() * h,
                'size_x': w * (0.2 + self.rng.random() * 0.4),
                'size_y': h * (0.2 + self.rng.random() * 0.4),
                'phase': self.rng.random() * 2 * np.pi,
                'intensity': 0.5 + self.rng.random() * 0.5
            })
        
        # Create coordinate grids
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            
            # Create sunlight mask
            sunlight = np.zeros((h, w), dtype=np.float32)
            
            for patch in patches:
                # Gentle shimmer - light intensity varies slowly
                shimmer = 0.7 + 0.3 * np.sin(t * 2 * np.pi * self.shimmer_speed + patch['phase'])
                
                # Soft elliptical patch with fuzzy edges
                dx = (xx - patch['x']) / patch['size_x']
                dy = (yy - patch['y']) / patch['size_y']
                dist = dx**2 + dy**2
                
                # Soft gaussian-like falloff
                patch_light = np.exp(-dist * 2) * patch['intensity'] * shimmer
                sunlight += patch_light
            
            # Normalize and apply intensity
            sunlight = np.clip(sunlight * self.config.intensity * 0.15, 0, 0.25)
            
            # Only apply to visible pixels
            alpha = original[:, :, 3] if original.shape[2] == 4 else np.ones((h, w)) * 255
            visible = alpha > 0
            sunlight = sunlight * visible
            
            # Apply sunlight
            frame_pixels = original.copy()
            
            # Brighten RGB channels
            for c in range(3):
                channel = frame_pixels[:, :, c].astype(np.float32)
                channel += sunlight * 255
                # Add warmth (more to red, less to blue)
                if c == 0:  # Red
                    channel += sunlight * 255 * self.warmth
                elif c == 2:  # Blue
                    channel -= sunlight * 255 * self.warmth * 0.5
                frame_pixels[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
            
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
