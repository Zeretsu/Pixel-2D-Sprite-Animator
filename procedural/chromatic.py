"""
Chromatic Effect - RGB channel separation
Retro/glitch aesthetic with color fringing and proper linear color handling
"""

import numpy as np
from typing import List, Optional
from .base import BaseEffect, EffectConfig, Easing, PixelMath
from ..core.parser import Sprite


class ChromaticEffect(BaseEffect):
    """Creates chromatic aberration with RGB channel split in linear color space"""
    
    name = "chromatic"
    description = "RGB channel separation for retro/glitch look"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        
        self.offset_amount = self.config.extra.get('offset', 3.0)
        self.mode = self.config.extra.get('mode', 'radial')  # 'radial', 'horizontal', 'animated'
        self.pulse = self.config.extra.get('pulse', True)
    
    def _to_linear(self, srgb: np.ndarray) -> np.ndarray:
        """Convert sRGB [0-255] to linear [0-1]"""
        return np.power(srgb.astype(np.float32) / 255.0, PixelMath.GAMMA)
    
    def _from_linear(self, linear: np.ndarray) -> np.ndarray:
        """Convert linear [0-1] to sRGB [0-255]"""
        return (np.power(np.clip(linear, 0.0, 1.0), PixelMath.INV_GAMMA) * 255.0 + 0.5).astype(np.uint8)
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        
        cx, cy = w / 2, h / 2
        
        # Convert original to linear space for proper sampling
        linear_rgb = self._to_linear(original[:, :, :3])
        if original.shape[2] == 4:
            alpha = original[:, :, 3].astype(np.float32) / 255.0
        else:
            alpha = np.where(np.any(original > 0, axis=2), 1.0, 0.0).astype(np.float32)
        
        for i in range(self.config.frame_count):
            t = i / max(1, self.config.frame_count - 1) if self.config.frame_count > 1 else 0.0
            
            frame_pixels = self._create_chromatic_frame(
                linear_rgb, alpha, t, cx, cy, h, w
            )
            
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _create_chromatic_frame(
        self,
        linear_rgb: np.ndarray,
        alpha: np.ndarray,
        t: float,
        cx: float,
        cy: float,
        h: int,
        w: int
    ) -> np.ndarray:
        """Create a single chromatic aberration frame in linear space"""
        result = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Calculate offset amount
        offset = self.offset_amount * self.config.intensity
        
        if self.pulse:
            # Smooth pulsing offset using sine wave
            pulse_factor = 0.5 + 0.5 * np.sin(t * 2 * np.pi)
            offset *= pulse_factor
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        if self.mode == 'radial':
            # Radial chromatic aberration (like lens distortion)
            dx = x_coords - cx
            dy = y_coords - cy
            dist = np.sqrt(dx * dx + dy * dy)
            max_dist = np.sqrt(cx * cx + cy * cy)
            
            # Normalize direction (avoid division by zero)
            safe_dist = np.where(dist > 1e-6, dist, 1.0)
            dir_x = dx / safe_dist
            dir_y = dy / safe_dist
            dir_x = np.where(dist > 1e-6, dir_x, 0.0)
            dir_y = np.where(dist > 1e-6, dir_y, 0.0)
            
            # Offset increases with distance from center (quadratic falloff)
            radial_factor = (dist / max_dist) ** 1.5
            
            # Red channel shifts outward
            r_offset_x = dir_x * offset * radial_factor
            r_offset_y = dir_y * offset * radial_factor
            
            # Blue channel shifts inward (opposite)
            b_offset_x = -dir_x * offset * radial_factor
            b_offset_y = -dir_y * offset * radial_factor
            
            # Green stays centered
            g_offset_x = np.zeros_like(x_coords)
            g_offset_y = np.zeros_like(y_coords)
            
        elif self.mode == 'horizontal':
            # Simple horizontal split
            r_offset_x = np.full_like(x_coords, offset)
            r_offset_y = np.zeros_like(y_coords)
            
            b_offset_x = np.full_like(x_coords, -offset)
            b_offset_y = np.zeros_like(y_coords)
            
            g_offset_x = np.zeros_like(x_coords)
            g_offset_y = np.zeros_like(y_coords)
            
        else:  # animated
            # Rotating chromatic aberration
            angle = t * 2 * np.pi
            
            r_offset_x = np.full_like(x_coords, np.cos(angle) * offset)
            r_offset_y = np.full_like(y_coords, np.sin(angle) * offset)
            
            b_offset_x = np.full_like(x_coords, np.cos(angle + 2*np.pi/3) * offset)
            b_offset_y = np.full_like(y_coords, np.sin(angle + 2*np.pi/3) * offset)
            
            g_offset_x = np.full_like(x_coords, np.cos(angle + 4*np.pi/3) * offset)
            g_offset_y = np.full_like(y_coords, np.sin(angle + 4*np.pi/3) * offset)
        
        # Sample each channel from offset positions (in linear space)
        r_src_x = x_coords - r_offset_x
        r_src_y = y_coords - r_offset_y
        
        g_src_x = x_coords - g_offset_x
        g_src_y = y_coords - g_offset_y
        
        b_src_x = x_coords - b_offset_x
        b_src_y = y_coords - b_offset_y
        
        # Sample channels with bilinear interpolation in linear space
        r_linear = self._sample_channel_linear(linear_rgb, alpha, r_src_x, r_src_y, 0, h, w)
        g_linear = self._sample_channel_linear(linear_rgb, alpha, g_src_x, g_src_y, 1, h, w)
        b_linear = self._sample_channel_linear(linear_rgb, alpha, b_src_x, b_src_y, 2, h, w)
        
        # Sample alpha for each channel
        r_alpha = self._sample_alpha(alpha, r_src_x, r_src_y, h, w)
        g_alpha = self._sample_alpha(alpha, g_src_x, g_src_y, h, w)
        b_alpha = self._sample_alpha(alpha, b_src_x, b_src_y, h, w)
        
        # Convert back to sRGB
        result[:, :, 0] = self._from_linear(r_linear)
        result[:, :, 1] = self._from_linear(g_linear)
        result[:, :, 2] = self._from_linear(b_linear)
        
        # Combined alpha: use max to preserve fringing visibility
        combined_alpha = np.maximum.reduce([r_alpha, g_alpha, b_alpha])
        result[:, :, 3] = (np.clip(combined_alpha, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        
        return result
    
    def _sample_channel_linear(
        self,
        linear_rgb: np.ndarray,
        alpha: np.ndarray,
        src_x: np.ndarray,
        src_y: np.ndarray,
        channel: int,
        h: int,
        w: int
    ) -> np.ndarray:
        """Sample a single channel with bilinear interpolation in linear space"""
        # Clamp coordinates
        src_x = np.clip(src_x, 0, w - 1.001)
        src_y = np.clip(src_y, 0, h - 1.001)
        
        # Bilinear interpolation coordinates
        x0 = np.floor(src_x).astype(np.int32)
        y0 = np.floor(src_y).astype(np.int32)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        
        fx = src_x - x0
        fy = src_y - y0
        
        # Get corner values (premultiplied by alpha for proper interpolation)
        v00 = linear_rgb[y0, x0, channel] * alpha[y0, x0]
        v10 = linear_rgb[y0, x1, channel] * alpha[y0, x1]
        v01 = linear_rgb[y1, x0, channel] * alpha[y1, x0]
        v11 = linear_rgb[y1, x1, channel] * alpha[y1, x1]
        
        # Get alpha for corners
        a00 = alpha[y0, x0]
        a10 = alpha[y0, x1]
        a01 = alpha[y1, x0]
        a11 = alpha[y1, x1]
        
        # Interpolate premultiplied values
        premul_interp = (
            v00 * (1 - fx) * (1 - fy) +
            v10 * fx * (1 - fy) +
            v01 * (1 - fx) * fy +
            v11 * fx * fy
        )
        
        # Interpolate alpha
        alpha_interp = (
            a00 * (1 - fx) * (1 - fy) +
            a10 * fx * (1 - fy) +
            a01 * (1 - fx) * fy +
            a11 * fx * fy
        )
        
        # Un-premultiply
        safe_alpha = np.where(alpha_interp > 1e-10, alpha_interp, 1.0)
        return premul_interp / safe_alpha
    
    def _sample_alpha(
        self,
        alpha: np.ndarray,
        src_x: np.ndarray,
        src_y: np.ndarray,
        h: int,
        w: int
    ) -> np.ndarray:
        """Sample alpha with bilinear interpolation"""
        # Clamp coordinates
        src_x = np.clip(src_x, 0, w - 1.001)
        src_y = np.clip(src_y, 0, h - 1.001)
        
        x0 = np.floor(src_x).astype(np.int32)
        y0 = np.floor(src_y).astype(np.int32)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        
        fx = src_x - x0
        fy = src_y - y0
        
        a00 = alpha[y0, x0]
        a10 = alpha[y0, x1]
        a01 = alpha[y1, x0]
        a11 = alpha[y1, x1]
        
        return (
            a00 * (1 - fx) * (1 - fy) +
            a10 * fx * (1 - fy) +
            a01 * (1 - fx) * fy +
            a11 * fx * fy
        )
