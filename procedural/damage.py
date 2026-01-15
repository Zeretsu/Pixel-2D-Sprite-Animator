"""
Damage Effect - Hit indicator with flash and shake
Combat feedback animation with pixel-perfect color blending
"""

import numpy as np
from typing import List, Optional
from .base import BaseEffect, EffectConfig, Easing, PixelMath
from ..core.parser import Sprite


class DamageEffect(BaseEffect):
    """Creates damage/hit effect with flash and shake"""
    
    name = "damage"
    description = "Hit indicator with flash and shake"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        
        self.flash_color = np.array(self.config.extra.get('flash_color', (255, 255, 255)), dtype=np.float64)
        self.hit_color = np.array(self.config.extra.get('hit_color', (255, 0, 0)), dtype=np.float64)
        self.shake_amount = self.config.extra.get('shake', 3.0)
        self.flash_frames = self.config.extra.get('flash_frames', 2)
        self.knockback = self.config.extra.get('knockback', True)
        self.blink = self.config.extra.get('blink', True)
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            
            frame_pixels = self._create_damage_frame(
                original, i, t, h, w
            )
            
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _create_damage_frame(
        self,
        original: np.ndarray,
        frame_idx: int,
        t: float,
        h: int,
        w: int
    ) -> np.ndarray:
        """Create a single damage frame with proper color math"""
        channels = original.shape[2]
        
        # Work in linear color space
        result = self._to_linear(original)
        
        # Get sprite mask
        if channels == 4:
            mask = original[:, :, 3] > 0
            alpha = original[:, :, 3].astype(np.float64) / 255.0
        else:
            mask = np.any(original > 0, axis=2)
            alpha = mask.astype(np.float64)
        
        # Convert effect colors to linear space
        flash_linear = np.power(self.flash_color / 255.0, PixelMath.GAMMA)
        hit_linear = np.power(self.hit_color / 255.0, PixelMath.GAMMA)
        
        # Phase 1: Initial white flash
        if frame_idx < self.flash_frames:
            # Quick flash that fades
            flash_progress = frame_idx / max(1, self.flash_frames)
            flash_intensity = (1.0 - flash_progress * 0.5) * self.config.intensity
            
            # Blend toward flash color in linear space
            for c in range(3):
                result[mask, c] = result[mask, c] * (1.0 - flash_intensity) + flash_linear[c] * flash_intensity
        
        # Phase 2: Red tint fading out with blink
        else:
            progress = (frame_idx - self.flash_frames) / max(1, self.config.frame_count - self.flash_frames)
            
            # Smooth ease-out for red tint
            red_intensity = (1.0 - Easing.ease_out_quad(progress)) * 0.5 * self.config.intensity
            
            # Blink effect (invincibility frames)
            visibility = 1.0
            if self.blink and progress < 0.7:
                # Smooth blink using sine wave (avoids harsh on/off)
                blink_phase = np.sin(t * 24.0 * np.pi)
                visibility = 0.5 + 0.5 * blink_phase
                red_intensity *= visibility
            
            # Apply red tint using proper color blending
            # Multiply blend: original * hit_color gives damage tint look
            for c in range(3):
                # Mix between original and red-tinted version
                tinted = result[mask, c] * (1.0 + hit_linear[c] * red_intensity)
                result[mask, c] = np.clip(tinted, 0, 1)
        
        # Apply shake with smooth decay
        shake_decay = 1.0 - Easing.ease_out_cubic(t)
        shake_strength = self.shake_amount * shake_decay * self.config.intensity
        
        if shake_strength > 0.1:
            # High frequency shake with different frequencies for x/y
            shake_x = np.sin(t * 60.0) * shake_strength
            shake_y = np.cos(t * 55.0) * shake_strength * 0.5
            
            # Knockback: initial push that returns
            if self.knockback:
                knockback_ease = Easing.ease_out_cubic(t)
                knockback_offset = (1.0 - knockback_ease) * 2.0 * self.config.intensity
                shake_x += knockback_offset
            
            # Create sampling coordinates
            y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float64)
            src_x = x_coords - shake_x
            src_y = y_coords - shake_y
            
            # Sample with bilinear interpolation in linear space
            result = self._sample_linear(result, alpha, src_x, src_y, h, w, channels)
        
        # Convert back to sRGB
        return self._from_linear(result, alpha, channels)
    
    def _to_linear(self, pixels: np.ndarray) -> np.ndarray:
        """Convert sRGB to linear color space"""
        return np.power(pixels[:, :, :3].astype(np.float64) / 255.0, PixelMath.GAMMA)
    
    def _from_linear(self, rgb_linear: np.ndarray, alpha: np.ndarray, channels: int) -> np.ndarray:
        """Convert linear RGB back to sRGB with alpha"""
        h, w = rgb_linear.shape[:2]
        result = np.zeros((h, w, channels), dtype=np.uint8)
        
        # Clamp and convert to sRGB
        rgb_srgb = np.power(np.clip(rgb_linear, 0, 1), PixelMath.INV_GAMMA) * 255.0
        result[:, :, :3] = np.clip(rgb_srgb, 0, 255).astype(np.uint8)
        
        if channels == 4:
            result[:, :, 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        
        return result
    
    def _sample_linear(
        self,
        rgb_linear: np.ndarray,
        alpha: np.ndarray,
        src_x: np.ndarray,
        src_y: np.ndarray,
        h: int,
        w: int,
        channels: int
    ) -> np.ndarray:
        """Bilinear sample RGB in linear space with premultiplied alpha"""
        # Premultiply for correct interpolation
        premul = rgb_linear * alpha[:, :, np.newaxis]
        
        # Integer and fractional parts
        x0 = np.floor(src_x).astype(np.int32)
        y0 = np.floor(src_y).astype(np.int32)
        fx = src_x - x0
        fy = src_y - y0
        
        # Clamp coordinates
        x0c = np.clip(x0, 0, w - 1)
        x1c = np.clip(x0 + 1, 0, w - 1)
        y0c = np.clip(y0, 0, h - 1)
        y1c = np.clip(y0 + 1, 0, h - 1)
        
        # Sample corners (premultiplied RGB)
        p00 = premul[y0c, x0c]
        p01 = premul[y0c, x1c]
        p10 = premul[y1c, x0c]
        p11 = premul[y1c, x1c]
        
        # Sample alpha
        a00 = alpha[y0c, x0c]
        a01 = alpha[y0c, x1c]
        a10 = alpha[y1c, x0c]
        a11 = alpha[y1c, x1c]
        
        # Bilinear weights
        w00 = (1.0 - fx) * (1.0 - fy)
        w01 = fx * (1.0 - fy)
        w10 = (1.0 - fx) * fy
        w11 = fx * fy
        
        # Interpolate
        rgb_result = (p00 * w00[:, :, np.newaxis] + p01 * w01[:, :, np.newaxis] +
                      p10 * w10[:, :, np.newaxis] + p11 * w11[:, :, np.newaxis])
        alpha_result = a00 * w00 + a01 * w01 + a10 * w10 + a11 * w11
        
        # Unpremultiply
        alpha_safe = np.maximum(alpha_result, 1e-10)
        rgb_result = rgb_result / alpha_safe[:, :, np.newaxis]
        
        # Handle out of bounds
        oob = (src_x < 0) | (src_x >= w) | (src_y < 0) | (src_y >= h)
        rgb_result[oob] = 0
        
        return rgb_result
