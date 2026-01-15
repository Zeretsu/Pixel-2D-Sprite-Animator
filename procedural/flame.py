"""
Enhanced Flame Effect - Smooth, realistic fire animation
Fixed pixel interpolation and seamless looping
"""

import numpy as np
from typing import List, Optional
from .base import BaseEffect, EffectConfig, Easing, PixelMath
from .noise import NoiseGenerator
from ..core.parser import Sprite


class FlameEffect(BaseEffect):
    """Creates smooth, realistic fire/flame animation with artifact-free rendering"""
    
    name = "flame"
    description = "Smooth fire animation with flickering and upward drift"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        self.noise = NoiseGenerator(self.config.seed)
        
        # Tuned parameters for smooth animation
        self.drift_speed = self.config.extra.get('drift_speed', 1.0)
        self.flicker_amount = self.config.extra.get('flicker_amount', 0.2)
        self.wave_amount = self.config.extra.get('wave_amount', 2.0)
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        
        # Detect hot (warm-colored) pixels
        hot_mask = self._detect_hot_pixels(original)
        
        # Create height-based intensity (bottom = more effect)
        y_coords = np.arange(h).reshape(-1, 1)
        height_factor = 1 - (y_coords / h)  # 1 at top, 0 at bottom
        height_factor = np.tile(height_factor, (1, w))
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            frame_pixels = original.copy()
            
            # Multi-frequency wave for organic motion
            frame_pixels = self._apply_organic_wave(frame_pixels, hot_mask, height_factor, t)
            
            # Smooth brightness flicker
            frame_pixels = self._apply_flicker(frame_pixels, hot_mask, t)
            
            # Subtle color shift (orange -> yellow -> white at tips)
            frame_pixels = self._apply_heat_color(frame_pixels, hot_mask, t)
            
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _detect_hot_pixels(self, pixels: np.ndarray) -> np.ndarray:
        """Detect warm-colored pixels (reds, oranges, yellows) or all visible pixels as fallback"""
        r = pixels[:, :, 0].astype(float)
        g = pixels[:, :, 1].astype(float)
        b = pixels[:, :, 2].astype(float)
        a = pixels[:, :, 3] if pixels.shape[2] == 4 else np.ones_like(r) * 255
        
        # Visible pixels mask
        visible = a > 0
        
        # Warm = high red, medium green, low blue
        warmth = (r - b) / 255
        warmth = np.clip(warmth, 0, 1)
        
        # Also detect yellow (high red AND green)
        yellow = np.minimum(r, g) / 255 * (1 - b / 255)
        
        hot_score = warmth * 0.7 + yellow * 0.3
        hot_mask = (hot_score > 0.15) & visible
        
        # FALLBACK: If no hot pixels found, apply to ALL visible pixels
        # This ensures the effect works on neutral/gray sprites too
        if not hot_mask.any():
            return visible
        
        return hot_mask
    
    def _apply_organic_wave(
        self, pixels: np.ndarray, mask: np.ndarray, 
        height_factor: np.ndarray, t: float
    ) -> np.ndarray:
        """Apply organic wave distortion using smooth interpolation"""
        h, w = pixels.shape[:2]
        result = pixels.copy()
        
        intensity = self.config.intensity * self.wave_amount
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Phase for seamless looping (use integer frequency multipliers)
        phase = t * 2 * np.pi
        
        # Multiple overlapping waves for organic feel
        # Using frequencies that create seamless loops
        wave1 = np.sin(y_coords / 4.0 + phase * 1) * 1.5
        wave2 = np.sin(y_coords / 3.0 + phase * 2) * 0.8
        wave3 = np.sin(y_coords / 2.0 + phase * 3) * 0.3
        
        # Combine waves
        total_wave = (wave1 + wave2 + wave3) * intensity
        
        # More effect at top (where flames dance)
        total_wave *= (1.0 - height_factor * 0.5)
        
        # Source coordinates for smooth sampling
        src_x = x_coords - total_wave
        src_y = y_coords.copy()
        
        # Use bilinear sampling for smooth displacement
        sampled = PixelMath.bilinear_sample(pixels, src_x, src_y)
        
        # Only apply to hot pixels
        result[mask] = sampled[mask]
        
        return result
    
    def _apply_flicker(self, pixels: np.ndarray, mask: np.ndarray, t: float) -> np.ndarray:
        """Apply smooth brightness flickering with seamless looping"""
        result = pixels.copy()
        
        # Multiple sine waves for natural flicker (integer frequency ratios for looping)
        flicker = (
            0.85 + 
            0.08 * np.sin(t * 2 * np.pi * 4) +
            0.05 * np.sin(t * 2 * np.pi * 7) +
            0.02 * np.sin(t * 2 * np.pi * 13)
        )
        
        flicker = flicker ** (1.0 / max(0.1, self.config.intensity))
        flicker = np.clip(flicker, 0.7, 1.1)
        
        # Apply to RGB channels
        rgb = result[mask, :3].astype(np.float32)
        rgb = np.clip(rgb * flicker * (1.0 + self.flicker_amount), 0, 255)
        result[mask, :3] = rgb.astype(np.uint8)
        
        return result
    
    def _apply_heat_color(self, pixels: np.ndarray, mask: np.ndarray, t: float) -> np.ndarray:
        """Shift colors toward white/yellow at peaks with seamless looping"""
        result = pixels.copy()
        
        # Subtle color shift based on time (use integer frequency for seamless loop)
        shift = 0.5 + 0.5 * np.sin(t * 2 * np.pi * 2)
        shift *= self.config.intensity * 0.15
        
        # Boost yellow/white at flicker peaks
        r = result[mask, 0].astype(np.float32)
        g = result[mask, 1].astype(np.float32)
        
        # Push toward yellow (increase green toward red level)
        g = g + (r - g) * shift * 0.3
        
        result[mask, 1] = np.clip(g, 0, 255).astype(np.uint8)
        
        return result
