"""
Shadow Effect - Trailing afterimages for speed/dash effects
Creates motion trail with fading copies using pixel-perfect compositing
"""

import numpy as np
from typing import List, Optional, Tuple
from .base import BaseEffect, EffectConfig, Easing, PixelMath
from ..core.parser import Sprite


class ShadowEffect(BaseEffect):
    """Creates trailing afterimage/shadow effect for speed and motion"""
    
    name = "shadow"
    description = "Trailing afterimages for speed and dash effects"
    
    # Direction unit vectors (normalized)
    DIRECTIONS = {
        'right': (1.0, 0.0),
        'left': (-1.0, 0.0),
        'up': (0.0, -1.0),
        'down': (0.0, 1.0),
        'diagonal_dr': (0.7071, 0.7071),   # √2/2 for unit length
        'diagonal_dl': (-0.7071, 0.7071),
        'diagonal_ur': (0.7071, -0.7071),
        'diagonal_ul': (-0.7071, -0.7071),
    }
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        
        self.trail_count = self.config.extra.get('trails', 4)
        self.trail_spacing = self.config.extra.get('spacing', 3.0)
        self.direction = self.config.extra.get('direction', 'right')
        self.fade_style = self.config.extra.get('fade', 'both')  # opacity, color, both
        self.shadow_color = self.config.extra.get('shadow_color', None)
        self.oscillate = self.config.extra.get('oscillate', True)
        self.quality = self.config.extra.get('quality', 'high')  # low, medium, high
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        
        # Get normalized direction vector
        dx, dy = self.DIRECTIONS.get(self.direction, (1.0, 0.0))
        
        # Pre-compute shadow copies at different fade levels
        shadow_cache = self._precompute_shadows(original)
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            
            # Compute movement phase with smooth easing
            if self.oscillate:
                # Smooth sinusoidal back-and-forth
                phase = np.sin(t * 2.0 * np.pi)
            else:
                # Continuous motion with wrap-around
                raw_phase = t * 2.0
                phase = 2.0 * (raw_phase - np.floor(raw_phase)) - 1.0
            
            frame_pixels = self._composite_trail_frame(
                original, shadow_cache, phase, dx, dy, h, w
            )
            
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _precompute_shadows(self, original: np.ndarray) -> List[np.ndarray]:
        """Pre-compute shadow copies at different fade levels for efficiency"""
        shadows = []
        
        for trail_idx in range(self.trail_count):
            # trail_factor: 0 = closest to main sprite, 1 = furthest
            trail_factor = trail_idx / max(self.trail_count - 1, 1)
            shadow = self._create_shadow_copy(original, trail_factor)
            shadows.append(shadow)
        
        return shadows
    
    def _create_shadow_copy(self, original: np.ndarray, trail_factor: float) -> np.ndarray:
        """Create a shadow copy with proper color modification"""
        h, w = original.shape[:2]
        channels = original.shape[2]
        shadow = original.astype(np.float32)
        
        # Create mask for valid pixels
        if channels == 4:
            mask = original[:, :, 3] > 0
        else:
            mask = np.any(original > 0, axis=2)
        
        if self.fade_style in ['color', 'both']:
            if self.shadow_color is not None:
                # Use specified shadow color while preserving luminance variation
                sc = np.array(self.shadow_color, dtype=np.float32)
                
                # Compute original luminance to preserve some detail
                lum = (0.299 * shadow[:, :, 0] + 
                       0.587 * shadow[:, :, 1] + 
                       0.114 * shadow[:, :, 2]) / 255.0
                
                # Blend toward shadow color based on trail position
                blend = 0.7 + 0.3 * trail_factor  # More color as trail gets older
                
                for c in range(3):
                    # Preserve some luminance detail
                    target = sc[c] * (0.5 + 0.5 * lum)
                    shadow[mask, c] = shadow[mask, c] * (1 - blend) + target[mask] * blend
            else:
                # Darken and desaturate for natural shadow look
                # Use gamma-aware darkening
                darken = 0.4 - 0.3 * trail_factor  # Gets darker with distance
                
                # Convert to linear space for proper darkening
                linear = np.power(shadow[:, :, :3] / 255.0, PixelMath.GAMMA)
                linear *= darken
                shadow[:, :, :3] = np.power(linear, PixelMath.INV_GAMMA) * 255.0
                
                # Add slight blue tint for shadow realism
                shadow[mask, 2] = np.minimum(shadow[mask, 2] * 1.1, 255)
        
        return np.clip(shadow, 0, 255).astype(np.uint8)
    
    def _composite_trail_frame(
        self,
        original: np.ndarray,
        shadow_cache: List[np.ndarray],
        phase: float,
        dx: float,
        dy: float,
        h: int,
        w: int
    ) -> np.ndarray:
        """Composite all trails with pixel-perfect subpixel positioning"""
        
        channels = original.shape[2]
        
        # Work in linear color space with premultiplied alpha for correct compositing
        canvas = np.zeros((h, w, 4), dtype=np.float64)
        
        # Maximum movement distance for the main sprite
        max_offset = self.trail_spacing * self.config.intensity
        
        # Main sprite offset
        main_offset_x = phase * max_offset * dx
        main_offset_y = phase * max_offset * dy
        
        # Draw trails from back to front (oldest/furthest first)
        for trail_idx in range(self.trail_count - 1, -1, -1):
            # Trail position: behind the main sprite in movement direction
            trail_distance = (trail_idx + 1) * self.trail_spacing * 0.8
            
            offset_x = main_offset_x - dx * trail_distance
            offset_y = main_offset_y - dy * trail_distance
            
            # Opacity decreases with trail index (further = more transparent)
            trail_factor = trail_idx / max(self.trail_count - 1, 1)
            base_opacity = 0.6 - 0.5 * trail_factor  # 0.6 -> 0.1
            
            if self.fade_style in ['opacity', 'both']:
                opacity = base_opacity * self.config.intensity
            else:
                opacity = 0.5 * self.config.intensity
            
            shadow = shadow_cache[trail_idx]
            
            self._composite_subpixel(canvas, shadow, offset_x, offset_y, opacity, h, w)
        
        # Draw main sprite on top at full opacity
        self._composite_subpixel(canvas, original, main_offset_x, main_offset_y, 1.0, h, w)
        
        # Convert back from linear premultiplied to sRGB straight alpha
        return self._finalize_canvas(canvas, channels)
    
    def _composite_subpixel(
        self,
        canvas: np.ndarray,
        sprite: np.ndarray,
        offset_x: float,
        offset_y: float,
        opacity: float,
        h: int,
        w: int
    ) -> None:
        """
        Composite sprite onto canvas at subpixel position using bilinear sampling.
        Uses proper premultiplied alpha in linear color space.
        """
        if opacity < 0.001:
            return
        
        channels = sprite.shape[2]
        
        # Create coordinate grids for sampling
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float64)
        
        # Source coordinates (where to sample from sprite)
        # We're moving the sprite by offset, so we sample from (pos - offset)
        src_x = x_coords - offset_x
        src_y = y_coords - offset_y
        
        # Determine valid sampling region
        valid = (src_x >= 0) & (src_x < w - 1) & (src_y >= 0) & (src_y < h - 1)
        
        if not np.any(valid):
            return
        
        if self.quality == 'high':
            # Bilinear interpolation with gamma correction
            sampled = self._bilinear_sample_linear(sprite, src_x, src_y, valid)
        else:
            # Nearest neighbor (pixel-exact but no subpixel)
            sampled = self._nearest_sample_linear(sprite, src_x, src_y, valid)
        
        # Apply opacity to source alpha
        src_alpha = sampled[:, :, 3:4] * opacity
        
        # Porter-Duff "over" compositing in linear premultiplied space
        # out = src + dst * (1 - src_alpha)
        dst_alpha = canvas[:, :, 3:4]
        
        # Only composite where we have valid samples
        inv_src_alpha = 1.0 - src_alpha
        
        # RGB channels (already premultiplied)
        canvas[:, :, :3] = sampled[:, :, :3] * opacity + canvas[:, :, :3] * inv_src_alpha
        
        # Alpha channel
        canvas[:, :, 3:4] = src_alpha + dst_alpha * inv_src_alpha
    
    def _bilinear_sample_linear(
        self,
        sprite: np.ndarray,
        src_x: np.ndarray,
        src_y: np.ndarray,
        valid: np.ndarray
    ) -> np.ndarray:
        """
        Sample sprite with bilinear interpolation in linear color space.
        Returns premultiplied alpha in linear space.
        """
        h, w = sprite.shape[:2]
        channels = sprite.shape[2]
        
        # Initialize result
        result = np.zeros((h, w, 4), dtype=np.float64)
        
        # Integer and fractional parts
        x0 = np.floor(src_x).astype(np.int32)
        y0 = np.floor(src_y).astype(np.int32)
        fx = src_x - x0
        fy = src_y - y0
        
        # Clamp for safe indexing
        x0c = np.clip(x0, 0, w - 1)
        x1c = np.clip(x0 + 1, 0, w - 1)
        y0c = np.clip(y0, 0, h - 1)
        y1c = np.clip(y0 + 1, 0, h - 1)
        
        # Sample four corners and convert to linear premultiplied
        def get_linear_premul(py, px):
            samples = sprite[py, px].astype(np.float64)
            res = np.zeros((h, w, 4), dtype=np.float64)
            
            # Convert RGB to linear
            res[:, :, :3] = np.power(samples[:, :, :3] / 255.0, PixelMath.GAMMA)
            
            # Get alpha
            if channels == 4:
                res[:, :, 3] = samples[:, :, 3] / 255.0
            else:
                res[:, :, 3] = np.where(np.any(samples > 0, axis=2), 1.0, 0.0)
            
            # Premultiply
            res[:, :, :3] *= res[:, :, 3:4]
            
            return res
        
        p00 = get_linear_premul(y0c, x0c)
        p01 = get_linear_premul(y0c, x1c)
        p10 = get_linear_premul(y1c, x0c)
        p11 = get_linear_premul(y1c, x1c)
        
        # Bilinear weights
        w00 = ((1.0 - fx) * (1.0 - fy))[:, :, np.newaxis]
        w01 = (fx * (1.0 - fy))[:, :, np.newaxis]
        w10 = ((1.0 - fx) * fy)[:, :, np.newaxis]
        w11 = (fx * fy)[:, :, np.newaxis]
        
        # Interpolate (this is correct because we're in premultiplied space)
        result = p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11
        
        # Zero out invalid regions
        result[~valid] = 0
        
        return result
    
    def _nearest_sample_linear(
        self,
        sprite: np.ndarray,
        src_x: np.ndarray,
        src_y: np.ndarray,
        valid: np.ndarray
    ) -> np.ndarray:
        """
        Nearest-neighbor sampling in linear premultiplied space.
        Faster but no subpixel smoothing.
        """
        h, w = sprite.shape[:2]
        channels = sprite.shape[2]
        
        result = np.zeros((h, w, 4), dtype=np.float64)
        
        # Round to nearest pixel
        xi = np.clip(np.round(src_x).astype(np.int32), 0, w - 1)
        yi = np.clip(np.round(src_y).astype(np.int32), 0, h - 1)
        
        samples = sprite[yi, xi].astype(np.float64)
        
        # Convert to linear
        result[:, :, :3] = np.power(samples[:, :, :3] / 255.0, PixelMath.GAMMA)
        
        if channels == 4:
            result[:, :, 3] = samples[:, :, 3] / 255.0
        else:
            result[:, :, 3] = np.where(np.any(samples > 0, axis=2), 1.0, 0.0)
        
        # Premultiply
        result[:, :, :3] *= result[:, :, 3:4]
        
        # Zero out invalid
        result[~valid] = 0
        
        return result
    
    def _finalize_canvas(self, canvas: np.ndarray, original_channels: int) -> np.ndarray:
        """Convert from linear premultiplied back to sRGB straight alpha"""
        h, w = canvas.shape[:2]
        
        result = np.zeros((h, w, original_channels), dtype=np.uint8)
        
        alpha = canvas[:, :, 3]
        alpha_safe = np.maximum(alpha, 1e-10)
        
        # Unpremultiply RGB
        rgb_linear = np.zeros((h, w, 3), dtype=np.float64)
        for c in range(3):
            rgb_linear[:, :, c] = canvas[:, :, c] / alpha_safe
        
        # Convert to sRGB
        rgb_linear = np.clip(rgb_linear, 0, 1)
        rgb_srgb = np.power(rgb_linear, PixelMath.INV_GAMMA) * 255.0
        
        result[:, :, :3] = np.clip(rgb_srgb, 0, 255).astype(np.uint8)
        
        if original_channels == 4:
            result[:, :, 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        
        return result
