"""
Enhanced Shake Effect - Film-quality camera shake with motion blur
Professional-grade temporal coherence and perlin-based randomness
"""

import numpy as np
from typing import List, Optional
from .base import BaseEffect, EffectConfig, Easing, PixelMath
from ..core.parser import Sprite


class ShakeEffect(BaseEffect):
    """Creates film-quality shake/vibration with optional motion blur"""
    
    name = "shake"
    description = "Screen shake and vibration"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        
        self.shake_x = self.config.extra.get('shake_x', 2.0)
        self.shake_y = self.config.extra.get('shake_y', 2.0)
        self.decay = self.config.extra.get('decay', False)
        self.smooth = self.config.extra.get('smooth', True)
        self.rotation_shake = self.config.extra.get('rotation', True)  # Add rotational component
        self.motion_blur = self.config.extra.get('motion_blur', False)
        self.blur_samples = self.config.extra.get('blur_samples', 5)
        self.quality = self.config.extra.get('quality', 'high')
        self.trauma_mode = self.config.extra.get('trauma', False)  # Film-style trauma shake
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        
        # Pre-compute coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2.0, h / 2.0
        
        # Pre-generate smooth shake offsets
        offsets_x, offsets_y, offsets_rot = self._generate_shake_pattern()
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            
            # Calculate shake amount with optional decay
            intensity = self.config.intensity
            if self.decay:
                # Exponential decay for more natural feel
                intensity *= np.exp(-3.0 * t)
            
            if self.trauma_mode:
                # Film-style trauma: intensity is squared for more impact
                intensity = intensity ** 2
            
            dx = offsets_x[i] * intensity
            dy = offsets_y[i] * intensity
            rot = offsets_rot[i] * intensity if self.rotation_shake else 0.0
            
            if self.motion_blur and (abs(dx) > 0.5 or abs(dy) > 0.5):
                # Apply motion blur for fast shakes
                frame_pixels = self._shake_with_blur(
                    original, dx, dy, rot, cx, cy, x_coords, y_coords
                )
            else:
                # Standard smooth shift
                frame_pixels = self._apply_shake(
                    original, dx, dy, rot, cx, cy, x_coords, y_coords
                )
            
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _generate_shake_pattern(self) -> tuple:
        """Generate temporally coherent shake pattern using layered frequencies"""
        offsets_x = []
        offsets_y = []
        offsets_rot = []
        
        # Frequency ratios for seamless looping (all integers)
        # Using prime-like spacing for natural feel
        freq_x = [3, 7, 13, 23]
        freq_y = [5, 11, 17, 29]
        freq_r = [4, 9, 15]
        
        # Amplitudes (higher frequencies = lower amplitude)
        amp = [0.45, 0.30, 0.18, 0.07]
        amp_r = [0.5, 0.35, 0.15]
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            tau = t * 2 * np.pi
            
            if self.smooth:
                # Multi-frequency layered noise
                dx = sum(a * np.sin(tau * f) for a, f in zip(amp, freq_x))
                dy = sum(a * np.sin(tau * f + np.pi/3) for a, f in zip(amp, freq_y))
                rot = sum(a * np.sin(tau * f + np.pi/6) for a, f in zip(amp_r, freq_r))
                
                dx *= self.shake_x
                dy *= self.shake_y
                rot *= 0.03  # Small rotation angle
            else:
                # Random with temporal smoothing
                dx = self.rng.uniform(-self.shake_x, self.shake_x)
                dy = self.rng.uniform(-self.shake_y, self.shake_y)
                rot = self.rng.uniform(-0.02, 0.02)
            
            offsets_x.append(dx)
            offsets_y.append(dy)
            offsets_rot.append(rot)
        
        # Apply temporal smoothing for random mode
        if not self.smooth and self.config.frame_count > 2:
            offsets_x = self._smooth_offsets(offsets_x)
            offsets_y = self._smooth_offsets(offsets_y)
            offsets_rot = self._smooth_offsets(offsets_rot)
        
        return offsets_x, offsets_y, offsets_rot
    
    def _smooth_offsets(self, offsets: list) -> list:
        """Apply Gaussian smoothing to offset list"""
        n = len(offsets)
        smoothed = []
        
        for i in range(n):
            # 3-tap Gaussian filter with wrap-around
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n
            
            val = offsets[prev_idx] * 0.25 + offsets[i] * 0.5 + offsets[next_idx] * 0.25
            smoothed.append(val)
        
        return smoothed
    
    def _apply_shake(
        self, pixels: np.ndarray, dx: float, dy: float, rot: float,
        cx: float, cy: float, x_coords: np.ndarray, y_coords: np.ndarray
    ) -> np.ndarray:
        """Apply shake transformation with rotation"""
        if abs(rot) < 0.0001:
            # Pure translation
            src_x = x_coords - dx
            src_y = y_coords - dy
        else:
            # Translation + rotation
            cos_r = np.cos(-rot)
            sin_r = np.sin(-rot)
            
            rel_x = x_coords - cx - dx
            rel_y = y_coords - cy - dy
            
            src_x = rel_x * cos_r - rel_y * sin_r + cx
            src_y = rel_x * sin_r + rel_y * cos_r + cy
        
        if self.quality == 'best':
            return PixelMath.lanczos_sample(pixels, src_x, src_y, gamma_correct=True)
        elif self.quality == 'high':
            return PixelMath.bilinear_sample(pixels, src_x, src_y, gamma_correct=True)
        else:
            return PixelMath.bilinear_sample(pixels, src_x, src_y, gamma_correct=False)
    
    def _shake_with_blur(
        self, pixels: np.ndarray, dx: float, dy: float, rot: float,
        cx: float, cy: float, x_coords: np.ndarray, y_coords: np.ndarray
    ) -> np.ndarray:
        """Apply shake with motion blur effect"""
        samples = []
        weights = []
        
        # Gaussian weights for motion blur samples
        sigma = self.blur_samples / 3.0
        
        for s in range(self.blur_samples):
            # Sample positions along motion path
            t = (s / (self.blur_samples - 1)) - 0.5 if self.blur_samples > 1 else 0
            
            sample_dx = dx * t * 2  # Full motion range
            sample_dy = dy * t * 2
            sample_rot = rot * t * 2
            
            # Gaussian weight (center samples weighted more)
            weight = np.exp(-0.5 * (t * 2) ** 2 / (sigma ** 2))
            
            sample = self._apply_shake(
                pixels, sample_dx, sample_dy, sample_rot,
                cx, cy, x_coords, y_coords
            )
            
            samples.append(sample.astype(np.float32))
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Weighted average in linear space
        result = np.zeros_like(samples[0])
        for sample, weight in zip(samples, weights):
            result += sample * weight
        
        return np.clip(result, 0, 255).astype(np.uint8)
