"""
Hologram Effect - Sci-fi holographic projection
Scan lines, flicker, and blue tint with pixel-perfect rendering
"""

import numpy as np
from typing import List, Optional
from .base import BaseEffect, EffectConfig, Easing, PixelMath
from ..core.parser import Sprite


class HologramEffect(BaseEffect):
    """Creates sci-fi hologram effect with proper linear color operations"""
    
    name = "hologram"
    description = "Holographic projection with scan lines"
    
    # Perceptually accurate luminance coefficients (Rec. 709)
    LUMA_R = 0.2126
    LUMA_G = 0.7152
    LUMA_B = 0.0722
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        
        self.holo_color = self.config.extra.get('color', (100, 200, 255))  # Cyan-blue
        self.scan_line_spacing = self.config.extra.get('scan_lines', 2)
        self.flicker = self.config.extra.get('flicker', True)
        self.glitch = self.config.extra.get('glitch', True)
        self.transparency = self.config.extra.get('transparency', 0.3)
        
        # Pre-convert hologram color to linear space
        self._holo_linear = np.array([
            (c / 255.0) ** PixelMath.GAMMA for c in self.holo_color
        ], dtype=np.float32)
        
        # Pre-generate noise patterns for consistent glitch
        self._init_glitch_patterns()
    
    def _init_glitch_patterns(self):
        """Pre-generate glitch patterns for consistency"""
        rng = np.random.RandomState(self.config.seed + 500)
        self._glitch_frames = []
        
        for _ in range(self.config.frame_count):
            frame_glitch = {
                'h_offset_y': rng.randint(0, 100),
                'h_offset_height': rng.randint(1, 4),
                'h_offset_amount': rng.randint(-5, 6),
                'h_offset_trigger': rng.random(),
                'channel_sep': rng.randint(0, 3),
                'channel_offset': rng.randint(-2, 3),
                'channel_trigger': rng.random(),
            }
            self._glitch_frames.append(frame_glitch)
    
    def _to_linear_premul(self, pixels: np.ndarray) -> np.ndarray:
        """Convert sRGB to linear premultiplied alpha"""
        result = np.zeros((pixels.shape[0], pixels.shape[1], 4), dtype=np.float32)
        
        # sRGB to linear
        rgb_norm = pixels[:, :, :3].astype(np.float32) / 255.0
        linear_rgb = np.power(rgb_norm, PixelMath.GAMMA)
        
        # Alpha
        if pixels.shape[2] == 4:
            alpha = pixels[:, :, 3].astype(np.float32) / 255.0
        else:
            alpha = np.where(np.any(pixels > 0, axis=2), 1.0, 0.0).astype(np.float32)
        
        # Premultiply
        result[:, :, 0] = linear_rgb[:, :, 0] * alpha
        result[:, :, 1] = linear_rgb[:, :, 1] * alpha
        result[:, :, 2] = linear_rgb[:, :, 2] * alpha
        result[:, :, 3] = alpha
        
        return result
    
    def _from_linear_premul(self, linear_premul: np.ndarray) -> np.ndarray:
        """Convert linear premultiplied to sRGB"""
        result = np.zeros((linear_premul.shape[0], linear_premul.shape[1], 4), dtype=np.uint8)
        
        alpha = linear_premul[:, :, 3]
        safe_alpha = np.where(alpha > 1e-10, alpha, 1.0)
        
        # Un-premultiply
        linear_rgb = np.zeros_like(linear_premul[:, :, :3])
        linear_rgb[:, :, 0] = linear_premul[:, :, 0] / safe_alpha
        linear_rgb[:, :, 1] = linear_premul[:, :, 1] / safe_alpha
        linear_rgb[:, :, 2] = linear_premul[:, :, 2] / safe_alpha
        
        # Linear to sRGB
        linear_rgb = np.clip(linear_rgb, 0.0, 1.0)
        srgb = np.power(linear_rgb, PixelMath.INV_GAMMA)
        
        result[:, :, :3] = (srgb * 255.0 + 0.5).astype(np.uint8)
        result[:, :, 3] = (np.clip(alpha, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        
        return result
    
    def _smoothstep(self, edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
        """Hermite smoothstep for anti-aliased transitions"""
        t = np.clip((x - edge0) / (edge1 - edge0 + 1e-10), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        h, w = sprite.height, sprite.width
        
        # Convert to linear premultiplied
        linear_premul = self._to_linear_premul(sprite.pixels)
        
        for i in range(self.config.frame_count):
            t = i / max(1, self.config.frame_count - 1) if self.config.frame_count > 1 else 0.0
            
            frame_linear = self._create_hologram_frame(
                linear_premul, t, i, h, w
            )
            
            frame_pixels = self._from_linear_premul(frame_linear)
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _create_hologram_frame(
        self,
        linear_premul: np.ndarray,
        t: float,
        frame_idx: int,
        h: int,
        w: int
    ) -> np.ndarray:
        """Create a single hologram frame with linear color operations"""
        result = linear_premul.copy()
        alpha = result[:, :, 3]
        mask = alpha > 0.01
        
        # Un-premultiply for color operations
        safe_alpha = np.where(alpha > 1e-10, alpha, 1.0)
        linear_rgb = np.zeros_like(result[:, :, :3])
        linear_rgb[:, :, 0] = result[:, :, 0] / safe_alpha
        linear_rgb[:, :, 1] = result[:, :, 1] / safe_alpha
        linear_rgb[:, :, 2] = result[:, :, 2] / safe_alpha
        
        # Calculate luminance in linear space (physically accurate)
        luminance = (
            linear_rgb[:, :, 0] * self.LUMA_R +
            linear_rgb[:, :, 1] * self.LUMA_G +
            linear_rgb[:, :, 2] * self.LUMA_B
        )
        
        # Apply hologram color tint in linear space
        color_strength = 0.7 * self.config.intensity
        tinted = np.zeros_like(linear_rgb)
        for c in range(3):
            # Preserve luminance variation while applying holo color
            tinted[:, :, c] = (
                linear_rgb[:, :, c] * (1 - color_strength) +
                self._holo_linear[c] * luminance * color_strength
            )
        
        # Add scan lines with anti-aliasing
        tinted = self._add_scan_lines(tinted, mask, t, h, w)
        
        # Flicker effect
        if self.flicker:
            flicker_intensity = self._calculate_flicker(t, frame_idx)
            tinted = tinted * flicker_intensity
        
        # Glitch effect (operates on premultiplied for proper shifting)
        # Re-premultiply for glitch
        premul_tinted = np.zeros_like(result)
        premul_tinted[:, :, 0] = np.clip(tinted[:, :, 0], 0, 1) * alpha
        premul_tinted[:, :, 1] = np.clip(tinted[:, :, 1], 0, 1) * alpha
        premul_tinted[:, :, 2] = np.clip(tinted[:, :, 2], 0, 1) * alpha
        premul_tinted[:, :, 3] = alpha
        
        if self.glitch:
            premul_tinted = self._add_glitch(premul_tinted, mask, t, frame_idx, h, w)
        
        # Apply transparency with smooth alpha variation
        alpha_mod = 1.0 - self.transparency * self.config.intensity
        
        # Scan line alpha variation (smoother than hard lines)
        y_coords = np.arange(h, dtype=np.float32).reshape(-1, 1)
        scan_phase = (y_coords / self.scan_line_spacing + t * 5) * 2 * np.pi
        scan_alpha = 0.75 + 0.25 * np.cos(scan_phase)  # Range 0.5 to 1.0
        
        premul_tinted[:, :, 3] = premul_tinted[:, :, 3] * alpha_mod * scan_alpha[:, :, 0] if scan_alpha.ndim == 3 else premul_tinted[:, :, 3] * alpha_mod * scan_alpha
        
        return premul_tinted
    
    def _add_scan_lines(
        self,
        linear_rgb: np.ndarray,
        mask: np.ndarray,
        t: float,
        h: int,
        w: int
    ) -> np.ndarray:
        """Add anti-aliased moving scan lines"""
        result = linear_rgb.copy()
        
        # Create smooth scan line pattern (sine wave instead of hard lines)
        y_coords = np.arange(h, dtype=np.float32).reshape(-1, 1)
        
        # Scan line frequency and phase
        scan_freq = 2 * np.pi / self.scan_line_spacing
        scan_offset = t * h * 2  # Moving scan lines
        
        # Smooth darkening pattern (0.5 to 1.0 range)
        scan_pattern = 0.75 + 0.25 * np.cos((y_coords + scan_offset) * scan_freq)
        
        # Apply to RGB
        result = result * scan_pattern
        
        # Add bright sweep line with smooth falloff
        sweep_y = (t * h * 1.5) % h
        sweep_width = 3.0
        
        # Distance from sweep line (handles wrap-around)
        dist_from_sweep = np.abs(y_coords - sweep_y)
        dist_from_sweep = np.minimum(dist_from_sweep, h - dist_from_sweep)  # Wrap
        
        # Gaussian falloff for sweep
        sweep_intensity = np.exp(-dist_from_sweep ** 2 / (2 * sweep_width ** 2))
        sweep_intensity = sweep_intensity * 0.5 * self.config.intensity
        
        # Add sweep glow (additive in linear space)
        for c in range(3):
            result[:, :, c] = result[:, :, c] + self._holo_linear[c] * sweep_intensity
        
        return result
    
    def _calculate_flicker(self, t: float, frame_idx: int) -> float:
        """Calculate smooth flicker intensity"""
        # Base smooth flicker (avoid jarring changes)
        flicker = 0.9 + 0.1 * np.sin(t * 30)
        
        # Use pre-generated random for deterministic occasional strong flicker
        if frame_idx < len(self._glitch_frames):
            glitch_data = self._glitch_frames[frame_idx]
            if glitch_data['h_offset_trigger'] < 0.1:
                # Smooth dip
                flicker *= 0.7 + 0.3 * glitch_data['channel_trigger']
        
        return flicker
    
    def _add_glitch(
        self,
        linear_premul: np.ndarray,
        mask: np.ndarray,
        t: float,
        frame_idx: int,
        h: int,
        w: int
    ) -> np.ndarray:
        """Add digital glitch artifacts with proper blending"""
        result = linear_premul.copy()
        
        if frame_idx >= len(self._glitch_frames):
            return result
        
        glitch_data = self._glitch_frames[frame_idx]
        
        # Horizontal offset glitch
        if glitch_data['h_offset_trigger'] < 0.15 * self.config.intensity:
            glitch_y = glitch_data['h_offset_y'] % h
            glitch_height = glitch_data['h_offset_height']
            glitch_offset = glitch_data['h_offset_amount']
            
            if glitch_offset != 0:
                for dy in range(glitch_height):
                    y = (glitch_y + dy) % h
                    
                    if glitch_offset > 0:
                        # Shift right - use proper interpolation for subpixel
                        result[y, glitch_offset:, :] = linear_premul[y, :-glitch_offset, :]
                        result[y, :glitch_offset, :] = 0
                    else:
                        # Shift left
                        result[y, :glitch_offset, :] = linear_premul[y, -glitch_offset:, :]
                        result[y, glitch_offset:, :] = 0
        
        # Color channel separation glitch (chromatic aberration style)
        if glitch_data['channel_trigger'] < 0.1 * self.config.intensity:
            channel = glitch_data['channel_sep']
            offset = glitch_data['channel_offset']
            
            if offset != 0:
                shifted = np.zeros_like(result[:, :, channel])
                if offset > 0:
                    shifted[:, offset:] = linear_premul[:, :-offset, channel]
                else:
                    shifted[:, :offset] = linear_premul[:, -offset:, channel]
                
                # Blend shifted channel (additive for chromatic aberration look)
                result[:, :, channel] = result[:, :, channel] * 0.5 + shifted * 0.5
        
        return result
