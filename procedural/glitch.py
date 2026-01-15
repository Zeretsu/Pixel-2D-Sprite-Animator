"""
Enhanced Glitch Effect - Smooth digital corruption with controlled chaos
"""

import numpy as np
from typing import List, Optional
from .base import BaseEffect, EffectConfig, Easing
from ..core.parser import Sprite


class GlitchEffect(BaseEffect):
    """Creates controlled digital glitch effect with smooth transitions"""
    
    name = "glitch"
    description = "Digital glitch with RGB split, displacement, and scanlines"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        self.rng = np.random.default_rng(self.config.seed)
        
        self.rgb_split = self.config.extra.get('rgb_split', 3)
        self.scan_lines = self.config.extra.get('scan_lines', True)
        self.block_glitch = self.config.extra.get('block_glitch', True)
        self.chromatic = self.config.extra.get('chromatic', True)
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        
        # Pre-generate glitch patterns for consistency
        glitch_frames = self._generate_glitch_pattern()
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            frame_pixels = original.copy()
            
            # Smooth glitch intensity with occasional spikes
            base_glitch = Easing.sin_wave(t * 3) * 0.3
            spike = self.rng.random() < 0.15  # 15% chance of spike
            glitch_amount = base_glitch + (0.7 if spike else 0)
            glitch_amount *= self.config.intensity
            
            # Chromatic aberration (always subtle)
            if self.chromatic:
                frame_pixels = self._chromatic_aberration(frame_pixels, t, glitch_amount)
            
            # RGB split on glitch
            if glitch_amount > 0.3:
                frame_pixels = self._smooth_rgb_split(frame_pixels, t, glitch_amount)
            
            # Block displacement
            if self.block_glitch and glitch_amount > 0.5:
                frame_pixels = self._smooth_block_displacement(
                    frame_pixels, glitch_frames[i % len(glitch_frames)]
                )
            
            # Scan lines
            if self.scan_lines:
                frame_pixels = self._smooth_scan_lines(frame_pixels, t, glitch_amount)
            
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _generate_glitch_pattern(self) -> list:
        """Pre-generate glitch block patterns"""
        patterns = []
        for _ in range(8):  # 8 unique patterns
            num_blocks = self.rng.integers(1, 4)
            blocks = []
            for _ in range(num_blocks):
                blocks.append({
                    'y_ratio': self.rng.random(),
                    'height_ratio': self.rng.random() * 0.15,
                    'offset_ratio': (self.rng.random() - 0.5) * 0.3
                })
            patterns.append(blocks)
        return patterns
    
    def _chromatic_aberration(self, pixels: np.ndarray, t: float, amount: float) -> np.ndarray:
        """Subtle chromatic aberration at edges"""
        h, w = pixels.shape[:2]
        result = pixels.copy()
        
        # Small constant offset for subtle effect
        offset = max(1, int(amount * 2))
        
        # Shift red slightly left, blue slightly right
        if offset > 0 and w > offset * 2:
            result[:, offset:, 0] = pixels[:, :-offset, 0]
            result[:, :-offset, 2] = pixels[:, offset:, 2]
        
        return result
    
    def _smooth_rgb_split(self, pixels: np.ndarray, t: float, amount: float) -> np.ndarray:
        """Smooth RGB channel split with easing"""
        h, w = pixels.shape[:2]
        result = pixels.copy()
        
        # Smooth oscillating offset
        offset = int(Easing.sin_wave(t * 2) * self.rgb_split * amount)
        
        if offset != 0 and w > abs(offset) * 2:
            abs_off = abs(offset)
            if offset > 0:
                result[:, abs_off:, 0] = pixels[:, :-abs_off, 0]
                result[:, :-abs_off, 2] = pixels[:, abs_off:, 2]
            else:
                result[:, :-abs_off, 0] = pixels[:, abs_off:, 0]
                result[:, abs_off:, 2] = pixels[:, :-abs_off, 2]
        
        return result
    
    def _smooth_block_displacement(self, pixels: np.ndarray, blocks: list) -> np.ndarray:
        """Apply pre-generated block displacement"""
        h, w = pixels.shape[:2]
        result = pixels.copy()
        
        for block in blocks:
            y_start = int(block['y_ratio'] * h)
            block_height = max(2, int(block['height_ratio'] * h))
            offset = int(block['offset_ratio'] * w)
            
            y_end = min(y_start + block_height, h)
            
            if offset > 0 and w > offset:
                result[y_start:y_end, offset:] = pixels[y_start:y_end, :-offset]
            elif offset < 0 and w > -offset:
                result[y_start:y_end, :offset] = pixels[y_start:y_end, -offset:]
        
        return result
    
    def _smooth_scan_lines(self, pixels: np.ndarray, t: float, amount: float) -> np.ndarray:
        """Smooth moving scan line effect"""
        h, w = pixels.shape[:2]
        result = pixels.copy()
        
        # Moving scan position
        scan_pos = int(t * h * 3) % h
        
        # Create smooth scanline pattern
        y_coords = np.arange(h)
        
        # Darken every other line, with smooth falloff near scan position
        line_mask = (y_coords % 3 == 0)
        
        # Extra darkening near scan position
        dist_from_scan = np.abs(y_coords - scan_pos)
        scan_intensity = np.clip(1 - dist_from_scan / 10, 0, 1) * amount
        
        # Apply darkening
        for y in range(h):
            factor = 0.85 if line_mask[y] else 1.0
            factor -= scan_intensity[y] * 0.2
            result[y, :, :3] = np.clip(result[y, :, :3] * factor, 0, 255).astype(np.uint8)
        
        return result
        
        for c in range(3):
            result[:, :, c] = np.where(noise_mask, noise_color[:, :, c], result[:, :, c])
        
        return result
