"""
Melt Effect - Melting/dripping animation
"""

import numpy as np
from typing import List, Optional
from .base import BaseEffect, EffectConfig
from .noise import NoiseGenerator
from ..core.parser import Sprite


class MeltEffect(BaseEffect):
    """Creates melting/dripping effect"""
    
    name = "melt"
    description = "Melting and dripping"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        self.noise = NoiseGenerator(self.config.seed)
        
        self.drip_speed = self.config.extra.get('drip_speed', 1.0)
        self.waviness = self.config.extra.get('waviness', 0.5)
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        
        # Pre-calculate drip pattern per column
        drip_pattern = self.noise.perlin_2d(w, 1, scale=8, time=0)[0]
        drip_pattern = (drip_pattern + 1) / 2  # Normalize to 0-1
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            
            frame_pixels = self._melt_frame(original, drip_pattern, t)
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _melt_frame(self, pixels: np.ndarray, drip_pattern: np.ndarray, t: float) -> np.ndarray:
        """Create single melt frame"""
        h, w = pixels.shape[:2]
        result = np.zeros_like(pixels)
        
        max_drip = int(h * 0.5 * t * self.config.intensity * self.drip_speed)
        
        for x in range(w):
            # Column-specific drip amount
            col_drip = int(max_drip * (0.5 + drip_pattern[x] * 0.5))
            
            # Add waviness
            wave = int(np.sin(x / 3 + t * 10) * self.waviness * 2 * self.config.intensity)
            col_drip += wave
            col_drip = max(0, col_drip)
            
            for y in range(h):
                src_y = y - col_drip
                
                if 0 <= src_y < h:
                    # Stretch effect - sample with distortion
                    stretch = 1.0 + t * 0.3 * self.config.intensity
                    actual_src_y = int((src_y - h/2) / stretch + h/2)
                    
                    if 0 <= actual_src_y < h:
                        result[y, x] = pixels[actual_src_y, x]
                        
                        # Fade bottom pixels
                        if y > h * 0.7:
                            fade = 1 - (y - h * 0.7) / (h * 0.3)
                            result[y, x, 3] = int(result[y, x, 3] * fade)
        
        return result
