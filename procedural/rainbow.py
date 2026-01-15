"""
Rainbow Effect - Color cycling / hue shift animation
"""

import numpy as np
from typing import List, Optional
from .base import BaseEffect, EffectConfig
from ..core.parser import Sprite


class RainbowEffect(BaseEffect):
    """Creates rainbow/hue cycling effect"""
    
    name = "rainbow"
    description = "Rainbow color cycling"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        
        self.preserve_brightness = self.config.extra.get('preserve_brightness', True)
        self.saturation_boost = self.config.extra.get('saturation_boost', 1.2)
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        original = sprite.pixels.copy()
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            
            # Hue shift amount
            hue_shift = t * 360 * self.config.intensity
            
            frame_pixels = self._shift_hue(original, hue_shift)
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _shift_hue(self, pixels: np.ndarray, hue_shift: float) -> np.ndarray:
        """Shift hue of all pixels"""
        result = pixels.copy()
        h, w = pixels.shape[:2]
        
        for y in range(h):
            for x in range(w):
                if pixels[y, x, 3] == 0:
                    continue
                
                r, g, b = pixels[y, x, :3] / 255.0
                h_val, s, v = self._rgb_to_hsv(r, g, b)
                
                # Shift hue
                h_val = (h_val + hue_shift) % 360
                
                # Boost saturation
                s = min(1.0, s * self.saturation_boost)
                
                # Convert back
                r, g, b = self._hsv_to_rgb(h_val, s, v)
                result[y, x, :3] = (np.array([r, g, b]) * 255).astype(np.uint8)
        
        return result
    
    def _rgb_to_hsv(self, r: float, g: float, b: float) -> tuple:
        """Convert RGB to HSV"""
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        diff = max_c - min_c
        
        # Value
        v = max_c
        
        # Saturation
        s = 0 if max_c == 0 else diff / max_c
        
        # Hue
        if diff == 0:
            h = 0
        elif max_c == r:
            h = 60 * (((g - b) / diff) % 6)
        elif max_c == g:
            h = 60 * (((b - r) / diff) + 2)
        else:
            h = 60 * (((r - g) / diff) + 4)
        
        return h, s, v
    
    def _hsv_to_rgb(self, h: float, s: float, v: float) -> tuple:
        """Convert HSV to RGB"""
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        
        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return r + m, g + m, b + m
