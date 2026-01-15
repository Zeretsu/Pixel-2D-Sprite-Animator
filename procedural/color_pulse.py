"""
Color Pulse Effect - Pulse and hue shift specific colors in a sprite
Targets specific color ranges (e.g., blue, purple) and animates them
"""

import numpy as np
from typing import List, Optional, Tuple
from .base import BaseEffect, EffectConfig
from ..core.parser import Sprite


class ColorPulseEffect(BaseEffect):
    """Pulse and hue shift specific colors in a sprite"""
    
    name = "color_pulse"
    description = "Pulse and hue shift specific colors (e.g., blue magical glow)"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        
        # Target color configuration (default: blue/cyan for magic)
        self.target_hue_min = self.config.extra.get('hue_min', 180)  # Blue
        self.target_hue_max = self.config.extra.get('hue_max', 260)  # Through purple
        self.saturation_min = self.config.extra.get('sat_min', 0.2)
        self.brightness_min = self.config.extra.get('bright_min', 0.3)
        
        # Effect settings
        self.pulse_amount = self.config.extra.get('pulse_amount', 0.3)
        self.hue_shift_amount = self.config.extra.get('hue_shift', 20)  # Degrees
        self.add_shine = self.config.extra.get('shine', False)  # Disabled by default
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        original = sprite.pixels.copy()
        
        # Create mask for target color pixels
        color_mask = self._create_color_mask(original)
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            
            # Pulse brightness
            pulse = 0.5 + 0.5 * np.sin(t * 2 * np.pi)
            brightness_mod = 1.0 + pulse * self.pulse_amount * self.config.intensity
            
            # Hue shift
            hue_shift = np.sin(t * 2 * np.pi) * self.hue_shift_amount * self.config.intensity
            
            # Apply effects
            frame_pixels = self._apply_color_effect(original, color_mask, brightness_mod, hue_shift)
            
            # Add shine/sparkle
            if self.add_shine:
                frame_pixels = self._add_shine(frame_pixels, t)
            
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _create_color_mask(self, pixels: np.ndarray) -> np.ndarray:
        """Create mask for target color pixels"""
        h, w = pixels.shape[:2]
        
        alpha = pixels[:, :, 3] if pixels.shape[2] == 4 else np.ones((h, w)) * 255
        visible = alpha > 0
        
        # Convert to HSV
        rgb = pixels[:, :, :3].astype(np.float32) / 255.0
        
        v = np.max(rgb, axis=2)
        c_min = np.min(rgb, axis=2)
        delta = v - c_min
        s = np.where(v > 0, delta / (v + 1e-8), 0)
        
        # Calculate hue
        h_arr = np.zeros((h, w), dtype=np.float32)
        
        red_max = (rgb[:, :, 0] >= rgb[:, :, 1]) & (rgb[:, :, 0] >= rgb[:, :, 2]) & (delta > 0)
        h_arr[red_max] = 60 * (((rgb[:, :, 1] - rgb[:, :, 2]) / (delta + 1e-8))[red_max] % 6)
        
        green_max = (rgb[:, :, 1] > rgb[:, :, 0]) & (rgb[:, :, 1] >= rgb[:, :, 2]) & (delta > 0)
        h_arr[green_max] = 60 * (((rgb[:, :, 2] - rgb[:, :, 0]) / (delta + 1e-8))[green_max] + 2)
        
        blue_max = (rgb[:, :, 2] > rgb[:, :, 0]) & (rgb[:, :, 2] > rgb[:, :, 1]) & (delta > 0)
        h_arr[blue_max] = 60 * (((rgb[:, :, 0] - rgb[:, :, 1]) / (delta + 1e-8))[blue_max] + 4)
        
        h_arr = h_arr % 360
        
        # Match target hue range
        if self.target_hue_min <= self.target_hue_max:
            hue_match = (h_arr >= self.target_hue_min) & (h_arr <= self.target_hue_max)
        else:
            hue_match = (h_arr >= self.target_hue_min) | (h_arr <= self.target_hue_max)
        
        sat_match = s >= self.saturation_min
        bright_match = v >= self.brightness_min
        
        mask = visible & hue_match & sat_match & bright_match
        
        # Return float mask with intensity based on saturation and brightness
        return (mask.astype(np.float32) * s * v * self.config.intensity).clip(0, 1)
    
    def _apply_color_effect(self, pixels: np.ndarray, mask: np.ndarray, 
                            brightness_mod: float, hue_shift: float) -> np.ndarray:
        """Apply brightness and hue shift to masked pixels"""
        result = pixels.copy()
        h, w = pixels.shape[:2]
        
        # Apply brightness modification
        for c in range(3):
            channel = result[:, :, c].astype(np.float32)
            # Blend between original and brightened based on mask
            brightened = channel * brightness_mod
            channel = channel * (1 - mask) + brightened * mask
            result[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
        
        # Apply hue shift to masked pixels
        if abs(hue_shift) > 0.1:
            result = self._shift_hue(result, mask, hue_shift)
        
        return result
    
    def _shift_hue(self, pixels: np.ndarray, mask: np.ndarray, shift: float) -> np.ndarray:
        """Shift hue of masked pixels"""
        result = pixels.copy()
        
        # Only process pixels where mask > 0.1
        affected = mask > 0.1
        if not np.any(affected):
            return result
        
        # Convert affected pixels RGB -> HSV -> shift -> RGB
        rgb = pixels[affected, :3].astype(np.float32) / 255.0
        
        # RGB to HSV
        v = np.max(rgb, axis=1)
        c_min = np.min(rgb, axis=1)
        delta = v - c_min
        
        s = np.where(v > 0, delta / (v + 1e-8), 0)
        
        h_arr = np.zeros(len(rgb), dtype=np.float32)
        
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        
        red_max = (r >= g) & (r >= b) & (delta > 0)
        h_arr[red_max] = 60 * (((g - b) / (delta + 1e-8))[red_max] % 6)
        
        green_max = (g > r) & (g >= b) & (delta > 0)
        h_arr[green_max] = 60 * (((b - r) / (delta + 1e-8))[green_max] + 2)
        
        blue_max = (b > r) & (b > g) & (delta > 0)
        h_arr[blue_max] = 60 * (((r - g) / (delta + 1e-8))[blue_max] + 4)
        
        # Apply hue shift
        h_arr = (h_arr + shift) % 360
        
        # HSV to RGB
        c = v * s
        x = c * (1 - np.abs((h_arr / 60) % 2 - 1))
        m = v - c
        
        new_rgb = np.zeros_like(rgb)
        
        idx = (h_arr >= 0) & (h_arr < 60)
        new_rgb[idx] = np.column_stack([c[idx], x[idx], np.zeros(np.sum(idx))])
        
        idx = (h_arr >= 60) & (h_arr < 120)
        new_rgb[idx] = np.column_stack([x[idx], c[idx], np.zeros(np.sum(idx))])
        
        idx = (h_arr >= 120) & (h_arr < 180)
        new_rgb[idx] = np.column_stack([np.zeros(np.sum(idx)), c[idx], x[idx]])
        
        idx = (h_arr >= 180) & (h_arr < 240)
        new_rgb[idx] = np.column_stack([np.zeros(np.sum(idx)), x[idx], c[idx]])
        
        idx = (h_arr >= 240) & (h_arr < 300)
        new_rgb[idx] = np.column_stack([x[idx], np.zeros(np.sum(idx)), c[idx]])
        
        idx = (h_arr >= 300) & (h_arr < 360)
        new_rgb[idx] = np.column_stack([c[idx], np.zeros(np.sum(idx)), x[idx]])
        
        new_rgb = (new_rgb + m[:, np.newaxis]) * 255
        
        # Blend based on mask strength
        mask_vals = mask[affected][:, np.newaxis]
        original_rgb = pixels[affected, :3].astype(np.float32)
        blended = original_rgb * (1 - mask_vals) + new_rgb * mask_vals
        
        result[affected, :3] = np.clip(blended, 0, 255).astype(np.uint8)
        
        return result
    
    def _add_shine(self, pixels: np.ndarray, t: float) -> np.ndarray:
        """Add subtle shine/highlight that moves across the sprite"""
        result = pixels.copy()
        h, w = pixels.shape[:2]
        
        # Create a diagonal shine band that moves across
        x_coords = np.arange(w)
        y_coords = np.arange(h)
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # Diagonal position (moves over time)
        diag = (xx + yy) / (w + h)
        shine_pos = (t * 1.5) % 1.5 - 0.25  # Move across with pause
        
        # Gaussian-like shine band
        shine = np.exp(-((diag - shine_pos) ** 2) / 0.01) * 0.15
        
        # Only apply to visible pixels
        alpha = pixels[:, :, 3] if pixels.shape[2] == 4 else np.ones((h, w)) * 255
        shine = shine * (alpha > 0)
        
        # Add shine to RGB
        for c in range(3):
            channel = result[:, :, c].astype(np.float32)
            channel += shine * 255
            result[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
        
        return result
