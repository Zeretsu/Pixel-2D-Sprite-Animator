"""
Teleport Effect - Scan-line materialization/dematerialization
Sci-fi style spawn/teleport with horizontal scan lines
Pixel-perfect with proper color blending
"""

import numpy as np
from typing import List, Optional
from .base import BaseEffect, EffectConfig, Easing, PixelMath
from ..core.parser import Sprite


class TeleportEffect(BaseEffect):
    """Creates sci-fi teleport/materialize effect with scan lines"""
    
    name = "teleport"
    description = "Scan-line materialization and dematerialization"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        
        self.mode = self.config.extra.get('mode', 'in')  # 'in', 'out', 'cycle'
        self.scan_lines = self.config.extra.get('scan_lines', 8)
        self.glow_color = np.array(self.config.extra.get('glow_color', (0, 255, 255)), dtype=np.float64)
        self.noise_amount = self.config.extra.get('noise', 0.3)
        self.vertical = self.config.extra.get('vertical', False)
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        
        # Pre-compute noise pattern (consistent across frames for less jitter)
        noise_pattern = self.rng.random((h, w)).astype(np.float64)
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            
            # Determine progress based on mode
            if self.mode == 'in':
                progress = Easing.ease_out_cubic(t)
            elif self.mode == 'out':
                progress = 1.0 - Easing.ease_in_cubic(t)
            else:  # cycle
                if t < 0.5:
                    progress = Easing.ease_out_cubic(t * 2)
                else:
                    progress = 1.0 - Easing.ease_in_cubic((t - 0.5) * 2)
            
            frame_pixels = self._create_teleport_frame(
                original, progress, t, noise_pattern, h, w
            )
            
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _create_teleport_frame(
        self,
        original: np.ndarray,
        progress: float,
        t: float,
        noise_pattern: np.ndarray,
        h: int,
        w: int
    ) -> np.ndarray:
        """Create a single teleport frame with scan lines"""
        channels = original.shape[2]
        
        # Work in linear color space with float64 precision
        result = np.zeros((h, w, 4), dtype=np.float64)
        
        # Convert original to linear premultiplied
        original_linear = self._to_linear_premul(original)
        
        # Get alpha mask
        if channels == 4:
            base_alpha = original[:, :, 3].astype(np.float64) / 255.0
        else:
            base_alpha = np.where(np.any(original > 0, axis=2), 1.0, 0.0)
        
        # Coordinate system for scan direction
        if self.vertical:
            scan_coord = np.arange(w, dtype=np.float64).reshape(1, -1)
            position = np.broadcast_to(scan_coord, (h, w))
            scan_size = float(w)
        else:
            scan_coord = np.arange(h, dtype=np.float64).reshape(-1, 1)
            position = np.broadcast_to(scan_coord, (h, w))
            scan_size = float(h)
        
        # Animated scan line pattern
        scan_phase = t * self.scan_lines * 2.0 * np.pi
        scan_pattern = 0.5 + 0.5 * np.sin(position / scan_size * self.scan_lines * 2.0 * np.pi + scan_phase)
        
        # Progress-based reveal with smooth edge
        reveal_line = progress * (scan_size + 10.0) - 5.0
        
        # Soft edge calculation using smoothstep
        edge_width = 4.0
        edge_start = reveal_line - edge_width
        edge_end = reveal_line + edge_width
        
        # Smoothstep for clean anti-aliased edge
        reveal_raw = (position - edge_start) / (edge_end - edge_start)
        reveal_raw = np.clip(reveal_raw, 0.0, 1.0)
        reveal_mask = reveal_raw * reveal_raw * (3.0 - 2.0 * reveal_raw)  # Smoothstep
        
        # Add controlled noise at the edge
        if self.noise_amount > 0:
            edge_zone = (reveal_mask > 0.05) & (reveal_mask < 0.95)
            noise_contribution = noise_pattern * self.noise_amount
            reveal_mask = np.where(
                edge_zone,
                np.clip(reveal_mask + noise_contribution - self.noise_amount * 0.5, 0.0, 1.0),
                reveal_mask
            )
        
        # Combine with scan pattern for edge flickering
        edge_zone = (reveal_mask > 0.1) & (reveal_mask < 0.9)
        reveal_mask = np.where(
            edge_zone,
            reveal_mask * (0.7 + 0.3 * scan_pattern),
            reveal_mask
        )
        
        # Final alpha combines base sprite alpha with reveal
        final_alpha = base_alpha * reveal_mask
        
        # Copy sprite with reveal mask applied
        result[:, :, :3] = original_linear[:, :, :3] * (reveal_mask[:, :, np.newaxis] / np.maximum(base_alpha[:, :, np.newaxis], 1e-10))
        result[:, :, 3] = final_alpha
        
        # Re-premultiply after alpha modification
        result[:, :, :3] *= final_alpha[:, :, np.newaxis]
        
        # Add glow at the scan edge
        result = self._add_edge_glow(result, reveal_mask, base_alpha, t, h, w)
        
        # Convert back to sRGB
        return self._from_linear_premul(result, channels)
    
    def _add_edge_glow(
        self,
        pixels: np.ndarray,
        reveal_mask: np.ndarray,
        base_alpha: np.ndarray,
        t: float,
        h: int,
        w: int
    ) -> np.ndarray:
        """Add glowing edge effect at the scan line"""
        result = pixels.copy()
        
        # Edge intensity is strongest where reveal_mask is around 0.5
        edge_intensity = np.exp(-((reveal_mask - 0.5) ** 2) / 0.03)
        edge_intensity *= base_alpha * self.config.intensity
        
        # Convert glow color to linear
        glow_linear = np.power(self.glow_color / 255.0, PixelMath.GAMMA)
        
        # Add glow (additive blending in premultiplied space)
        for c in range(3):
            result[:, :, c] += glow_linear[c] * edge_intensity
        
        # Glow contributes to alpha
        result[:, :, 3] = np.maximum(result[:, :, 3], edge_intensity * 0.8)
        
        return result
    
    def _to_linear_premul(self, pixels: np.ndarray) -> np.ndarray:
        """Convert sRGB straight alpha to linear premultiplied"""
        h, w = pixels.shape[:2]
        channels = pixels.shape[2]
        
        result = np.zeros((h, w, 4), dtype=np.float64)
        
        # Convert RGB to linear
        result[:, :, :3] = np.power(pixels[:, :, :3].astype(np.float64) / 255.0, PixelMath.GAMMA)
        
        # Get alpha
        if channels == 4:
            result[:, :, 3] = pixels[:, :, 3].astype(np.float64) / 255.0
        else:
            result[:, :, 3] = np.where(np.any(pixels > 0, axis=2), 1.0, 0.0)
        
        # Premultiply
        result[:, :, :3] *= result[:, :, 3:4]
        
        return result
    
    def _from_linear_premul(self, pixels: np.ndarray, output_channels: int) -> np.ndarray:
        """Convert linear premultiplied back to sRGB straight alpha"""
        h, w = pixels.shape[:2]
        
        result = np.zeros((h, w, output_channels), dtype=np.uint8)
        
        alpha = pixels[:, :, 3]
        alpha_safe = np.maximum(alpha, 1e-10)
        
        # Unpremultiply
        rgb_linear = pixels[:, :, :3] / alpha_safe[:, :, np.newaxis]
        
        # Convert to sRGB
        rgb_linear = np.clip(rgb_linear, 0.0, 1.0)
        rgb_srgb = np.power(rgb_linear, PixelMath.INV_GAMMA) * 255.0
        
        result[:, :, :3] = np.clip(rgb_srgb, 0, 255).astype(np.uint8)
        
        if output_channels == 4:
            result[:, :, 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        
        return result
