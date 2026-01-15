"""
Flicker Effect - Light flicker / strobe animation
"""

import numpy as np
from typing import List, Optional
from .base import BaseEffect, EffectConfig
from ..core.parser import Sprite


class FlickerEffect(BaseEffect):
    """Creates flickering/strobe light effect"""
    
    name = "flicker"
    description = "Light flicker and strobe effect"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        self.rng = np.random.default_rng(self.config.seed)
        
        self.min_brightness = self.config.extra.get('min_brightness', 0.5)
        self.random_flicker = self.config.extra.get('random', True)
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        original = sprite.pixels.copy()
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            
            if self.random_flicker:
                # Random brightness
                brightness = self.min_brightness + self.rng.random() * (1 - self.min_brightness)
            else:
                # Sine wave flicker
                brightness = self.min_brightness + (1 - self.min_brightness) * \
                            (0.5 + 0.5 * np.sin(t * np.pi * 4 * self.config.intensity))
            
            frame_pixels = self._adjust_brightness(original, brightness)
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _adjust_brightness(self, pixels: np.ndarray, brightness: float) -> np.ndarray:
        """Adjust pixel brightness while preserving alpha"""
        result = pixels.copy()
        result[:, :, :3] = (result[:, :, :3] * brightness).clip(0, 255).astype(np.uint8)
        return result
