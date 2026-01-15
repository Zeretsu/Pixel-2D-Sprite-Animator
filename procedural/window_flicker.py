"""
Window Flicker Effect - Animated light dimming for lit windows
Targets warm/orange colored pixels and creates realistic indoor light flickering
"""

import numpy as np
from typing import List, Optional, Tuple
from .base import BaseEffect, EffectConfig, Easing
from ..core.parser import Sprite


class WindowFlickerEffect(BaseEffect):
    """Creates realistic window light dimming/flickering effect"""
    
    name = "window_flicker"
    description = "Animated light dimming for lit windows (targets warm/orange colors)"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        self.rng = np.random.default_rng(self.config.seed)
        
        # Configuration options
        self.min_brightness = self.config.extra.get('min_brightness', 0.6)
        self.max_brightness = self.config.extra.get('max_brightness', 1.0)
        self.flicker_style = self.config.extra.get('style', 'smooth')  # smooth, random, candle
        self.color_shift = self.config.extra.get('color_shift', True)  # Shift to warmer when dimmer
        
        # Color targeting - orange/warm window colors
        self.target_hue_min = self.config.extra.get('hue_min', 15)   # Orange-ish
        self.target_hue_max = self.config.extra.get('hue_max', 50)   # Yellow-orange
        self.saturation_min = self.config.extra.get('sat_min', 0.3)  # Minimum saturation
        self.brightness_min = self.config.extra.get('bright_min', 0.5)  # Minimum brightness to target
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        original = sprite.pixels.copy()
        
        # Create mask for window pixels (warm/orange lit areas)
        window_mask = self._create_window_mask(original)
        
        # Pre-compute random values for candle mode
        if self.flicker_style == 'candle':
            candle_values = self._generate_candle_pattern()
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            
            # Calculate brightness for this frame
            brightness = self._get_brightness(t, i if self.flicker_style == 'candle' else None,
                                              candle_values if self.flicker_style == 'candle' else None)
            
            # Apply effect to window pixels only
            frame_pixels = self._apply_window_dimming(original, window_mask, brightness)
            
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _create_window_mask(self, pixels: np.ndarray) -> np.ndarray:
        """Create a mask identifying warm/orange lit window pixels"""
        h, w = pixels.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Only process visible pixels
        alpha = pixels[:, :, 3] if pixels.shape[2] == 4 else np.ones((h, w)) * 255
        visible = alpha > 0
        
        # Convert RGB to HSV for color targeting
        rgb = pixels[:, :, :3].astype(np.float32) / 255.0
        
        # Calculate Value (brightness)
        v = np.max(rgb, axis=2)
        
        # Calculate Saturation
        c_min = np.min(rgb, axis=2)
        delta = v - c_min
        s = np.where(v > 0, delta / (v + 1e-8), 0)
        
        # Calculate Hue (0-360)
        h_arr = np.zeros((h, w), dtype=np.float32)
        
        # Red is max
        red_max = (rgb[:, :, 0] >= rgb[:, :, 1]) & (rgb[:, :, 0] >= rgb[:, :, 2]) & (delta > 0)
        h_arr[red_max] = 60 * (((rgb[:, :, 1] - rgb[:, :, 2]) / (delta + 1e-8))[red_max] % 6)
        
        # Green is max
        green_max = (rgb[:, :, 1] > rgb[:, :, 0]) & (rgb[:, :, 1] >= rgb[:, :, 2]) & (delta > 0)
        h_arr[green_max] = 60 * (((rgb[:, :, 2] - rgb[:, :, 0]) / (delta + 1e-8))[green_max] + 2)
        
        # Blue is max
        blue_max = (rgb[:, :, 2] > rgb[:, :, 0]) & (rgb[:, :, 2] > rgb[:, :, 1]) & (delta > 0)
        h_arr[blue_max] = 60 * (((rgb[:, :, 0] - rgb[:, :, 1]) / (delta + 1e-8))[blue_max] + 4)
        
        # Normalize hue to 0-360
        h_arr = h_arr % 360
        
        # Create mask based on color criteria
        # Target warm colors (orange/yellow hues) that are bright and saturated
        hue_match = ((h_arr >= self.target_hue_min) & (h_arr <= self.target_hue_max)) | (h_arr >= 350)  # Include red-ish
        sat_match = s >= self.saturation_min
        bright_match = v >= self.brightness_min
        
        # Combine all criteria
        target_pixels = visible & hue_match & sat_match & bright_match
        
        # Calculate intensity based on how "warm" and "bright" the pixel is
        # Brighter, more saturated orange pixels get stronger effect
        warmth = np.where(target_pixels, v * s * self.config.intensity, 0)
        
        # Apply Gaussian-like falloff at edges for smooth blending
        mask = np.clip(warmth, 0, 1)
        
        return mask
    
    def _generate_candle_pattern(self) -> np.ndarray:
        """Generate realistic candle-like flicker pattern"""
        # Use multiple sine waves with different frequencies for organic feel
        frames = self.config.frame_count
        values = np.zeros(frames)
        
        for i in range(frames):
            t = i / frames
            
            # Base slow oscillation
            base = 0.5 + 0.3 * np.sin(t * 2 * np.pi)
            
            # Add faster flickers
            flicker1 = 0.1 * np.sin(t * 7 * np.pi + self.rng.random() * 2)
            flicker2 = 0.05 * np.sin(t * 13 * np.pi + self.rng.random() * 3)
            
            # Random spikes (occasional bright flare)
            spike = self.rng.random() * 0.15 if self.rng.random() > 0.85 else 0
            
            values[i] = np.clip(base + flicker1 + flicker2 + spike, 0, 1)
        
        return values
    
    def _get_brightness(self, t: float, frame_idx: Optional[int] = None, 
                        candle_values: Optional[np.ndarray] = None) -> float:
        """Calculate brightness value for current frame"""
        
        if self.flicker_style == 'random':
            # Random brightness with some smoothness
            base = self.rng.random()
            return self.min_brightness + base * (self.max_brightness - self.min_brightness)
        
        elif self.flicker_style == 'candle' and candle_values is not None and frame_idx is not None:
            # Use pre-computed candle pattern
            value = candle_values[frame_idx]
            return self.min_brightness + value * (self.max_brightness - self.min_brightness)
        
        else:  # smooth (default)
            # Smooth breathing-like dimming
            # Use asymmetric timing: slower dim, faster brighten
            cycle_t = t * 2 * np.pi
            
            # Multiple harmonics for organic feel
            wave1 = np.sin(cycle_t)
            wave2 = 0.3 * np.sin(cycle_t * 2.17 + 0.5)  # Slight offset
            wave3 = 0.15 * np.sin(cycle_t * 3.31 + 1.2)
            
            combined = (wave1 + wave2 + wave3) / 1.45
            normalized = (combined + 1) / 2  # Map to 0-1
            
            return self.min_brightness + normalized * (self.max_brightness - self.min_brightness)
    
    def _apply_window_dimming(self, pixels: np.ndarray, mask: np.ndarray, 
                               brightness: float) -> np.ndarray:
        """Apply dimming effect to masked window pixels"""
        result = pixels.copy()
        
        # Calculate per-pixel brightness adjustment
        # Mask values determine how much of the effect applies to each pixel
        adjustment = 1.0 - mask * (1.0 - brightness)
        
        # Apply brightness to RGB channels
        for c in range(3):
            channel = result[:, :, c].astype(np.float32)
            channel *= adjustment
            result[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
        
        # Optional: shift color temperature when dimmer (warmer = more orange/red)
        if self.color_shift:
            # When brightness is lower, slightly increase red and reduce blue
            dim_factor = 1.0 - brightness
            shift_amount = dim_factor * 0.1 * mask  # Subtle shift
            
            # Boost red slightly
            r = result[:, :, 0].astype(np.float32)
            r += shift_amount * 20
            result[:, :, 0] = np.clip(r, 0, 255).astype(np.uint8)
            
            # Reduce blue slightly  
            b = result[:, :, 2].astype(np.float32)
            b -= shift_amount * 15
            result[:, :, 2] = np.clip(b, 0, 255).astype(np.uint8)
        
        return result
