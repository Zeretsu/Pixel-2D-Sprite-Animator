"""
Enhanced Pulse Effect - Organic breathing/pulsing animation
Professional-quality with gamma-correct brightness and multiple modes
"""

import numpy as np
from typing import List, Optional
from .base import BaseEffect, EffectConfig, Easing, PixelMath
from ..core.parser import Sprite


class PulseEffect(BaseEffect):
    """Creates organic pulsing/breathing animation with multiple physics modes"""
    
    name = "pulse"
    description = "Smooth breathing and pulsing effect"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        
        self.scale_amount = self.config.extra.get('scale_amount', 0.06)
        self.glow = self.config.extra.get('glow', True)
        self.brightness_pulse = self.config.extra.get('brightness', True)
        self.quality = self.config.extra.get('quality', 'high')
        self.mode = self.config.extra.get('mode', 'breathing')  # breathing, heartbeat, mechanical
        self.asymmetric = self.config.extra.get('asymmetric', True)  # Faster exhale
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        cx, cy = w / 2.0, h / 2.0
        
        # Pre-compute coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            
            # Get pulse value based on mode
            breath = self._get_pulse_value(t)
            
            # Scale factor
            scale = 1.0 + breath * self.scale_amount * self.config.intensity
            
            # Vectorized smooth scaling
            frame_pixels = self._smooth_scale(
                original, scale, cx, cy, x_coords, y_coords
            )
            
            # Brightness pulse with gamma-correct adjustment
            if self.brightness_pulse:
                brightness = 1.0 + breath * 0.12 * self.config.intensity
                frame_pixels = self._apply_brightness(frame_pixels, brightness)
            
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _get_pulse_value(self, t: float) -> float:
        """Get pulse value based on animation mode"""
        tau = t * 2 * np.pi
        
        if self.mode == 'heartbeat':
            # Double-peak heartbeat pattern
            # Primary beat
            beat1 = np.exp(-((t % 0.5) * 10) ** 2) * 0.8
            # Secondary beat (smaller, follows primary)
            beat2 = np.exp(-(((t + 0.15) % 0.5) * 10) ** 2) * 0.4
            return (beat1 + beat2) * 1.2
        
        elif self.mode == 'mechanical':
            # Sharp, mechanical pulse
            return Easing.ease_out_cubic(np.abs(np.sin(tau)))
        
        else:  # breathing (default)
            if self.asymmetric:
                # Realistic breathing: slower inhale, faster exhale
                # Inhale (first 60%), exhale (last 40%)
                if t < 0.6:
                    # Slow inhale with ease-in-out
                    local_t = t / 0.6
                    return Easing.ease_in_out_sine(local_t)
                else:
                    # Faster exhale with ease-out
                    local_t = (t - 0.6) / 0.4
                    return 1.0 - Easing.ease_out_quad(local_t)
            else:
                # Symmetric breathing
                return Easing.breathing(t)
    
    def _apply_brightness(self, pixels: np.ndarray, brightness: float) -> np.ndarray:
        """Apply brightness adjustment with gamma correction"""
        result = pixels.copy()
        
        if pixels.shape[2] == 4:
            mask = pixels[:, :, 3] > 0
        else:
            mask = np.ones(pixels.shape[:2], dtype=bool)
        
        # Convert to linear space, apply brightness, convert back
        rgb = result[mask, :3].astype(np.float32) / 255.0
        
        # Gamma decode (sRGB to linear)
        rgb_linear = np.power(rgb, 2.2)
        
        # Apply brightness in linear space
        rgb_linear = rgb_linear * brightness
        
        # Gamma encode (linear to sRGB)
        rgb = np.power(np.clip(rgb_linear, 0, 1), 1/2.2)
        
        result[mask, :3] = (rgb * 255).astype(np.uint8)
        return result
    
    def _smooth_scale(
        self, 
        pixels: np.ndarray, 
        scale: float, 
        cx: float, 
        cy: float,
        x_coords: np.ndarray,
        y_coords: np.ndarray
    ) -> np.ndarray:
        """Vectorized smooth scaling with quality selection"""
        # Inverse transform for proper scaling
        src_x = cx + (x_coords - cx) / scale
        src_y = cy + (y_coords - cy) / scale
        
        if self.quality == 'best':
            return PixelMath.lanczos_sample(pixels, src_x, src_y, gamma_correct=True)
        elif self.quality == 'high':
            return PixelMath.bilinear_sample(pixels, src_x, src_y, gamma_correct=True)
        else:
            return PixelMath.bilinear_sample(pixels, src_x, src_y, gamma_correct=False)
