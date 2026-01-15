"""
Dissolve Effect - Particle dissolve/materialize animation with pixel-perfect rendering
"""

import numpy as np
from typing import List, Optional
from .base import BaseEffect, EffectConfig, PixelMath
from ..core.parser import Sprite


class DissolveEffect(BaseEffect):
    """Creates dissolve/materialize particle effect with vectorized operations"""
    
    name = "dissolve"
    description = "Particle dissolve and materialize"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        self.rng = np.random.default_rng(self.config.seed)
        
        self.direction = self.config.extra.get('direction', 'out')  # 'in' or 'out'
        self.scatter = self.config.extra.get('scatter', 3)
    
    def _smoothstep(self, edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
        """Hermite smoothstep for anti-aliased transitions"""
        t = np.clip((x - edge0) / (edge1 - edge0 + 1e-10), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    
    def _to_linear_premul(self, pixels: np.ndarray) -> np.ndarray:
        """Convert sRGB to linear premultiplied alpha"""
        result = np.zeros((pixels.shape[0], pixels.shape[1], 4), dtype=np.float32)
        rgb_norm = pixels[:, :, :3].astype(np.float32) / 255.0
        linear_rgb = np.power(rgb_norm, PixelMath.GAMMA)
        
        if pixels.shape[2] == 4:
            alpha = pixels[:, :, 3].astype(np.float32) / 255.0
        else:
            alpha = np.where(np.any(pixels > 0, axis=2), 1.0, 0.0).astype(np.float32)
        
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
        
        linear_rgb = np.zeros_like(linear_premul[:, :, :3])
        linear_rgb[:, :, 0] = linear_premul[:, :, 0] / safe_alpha
        linear_rgb[:, :, 1] = linear_premul[:, :, 1] / safe_alpha
        linear_rgb[:, :, 2] = linear_premul[:, :, 2] / safe_alpha
        
        linear_rgb = np.clip(linear_rgb, 0.0, 1.0)
        srgb = np.power(linear_rgb, PixelMath.INV_GAMMA)
        
        result[:, :, :3] = (srgb * 255.0 + 0.5).astype(np.uint8)
        result[:, :, 3] = (np.clip(alpha, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        return result
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        
        # Convert to linear premultiplied
        linear_premul = self._to_linear_premul(original)
        original_alpha = linear_premul[:, :, 3]
        
        # Create smooth threshold map using multi-octave noise
        threshold_map = self._generate_dissolve_pattern(h, w)
        
        # Pre-generate scatter offsets for all pixels (vectorized)
        scatter_dx = self.rng.integers(-self.scatter, self.scatter + 1, size=(h, w))
        scatter_dy = self.rng.integers(-self.scatter, self.scatter + 1, size=(h, w))
        
        for i in range(self.config.frame_count):
            t = i / max(1, self.config.frame_count - 1) if self.config.frame_count > 1 else 0.0
            
            if self.direction == 'in':
                t = 1 - t
            
            frame_linear = self._dissolve_frame_vectorized(
                linear_premul, original_alpha, threshold_map,
                scatter_dx, scatter_dy, t, h, w
            )
            
            frame_pixels = self._from_linear_premul(frame_linear)
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _generate_dissolve_pattern(self, h: int, w: int) -> np.ndarray:
        """Generate organic dissolve pattern using layered noise"""
        # Base random pattern
        pattern = self.rng.random((h, w)).astype(np.float32)
        
        # Add smooth noise for more organic dissolve
        # Multi-octave noise simulation
        for scale in [4, 8, 16]:
            noise_h = max(2, h // scale + 2)
            noise_w = max(2, w // scale + 2)
            
            small_noise = self.rng.random((noise_h, noise_w)).astype(np.float32)
            
            # Bilinear upsample
            y_coords = np.linspace(0, noise_h - 1.001, h)
            x_coords = np.linspace(0, noise_w - 1.001, w)
            
            y0 = y_coords.astype(int)
            x0 = x_coords.astype(int)
            y1 = np.minimum(y0 + 1, noise_h - 1)
            x1 = np.minimum(x0 + 1, noise_w - 1)
            
            fy = (y_coords - y0).reshape(-1, 1)
            fx = (x_coords - x0).reshape(1, -1)
            
            # Smoothstep interpolation
            fy = fy * fy * (3 - 2 * fy)
            fx = fx * fx * (3 - 2 * fx)
            
            upsampled = (
                small_noise[y0][:, x0] * (1 - fy) * (1 - fx) +
                small_noise[y1][:, x0] * fy * (1 - fx) +
                small_noise[y0][:, x1] * (1 - fy) * fx +
                small_noise[y1][:, x1] * fy * fx
            )
            
            pattern = pattern * 0.7 + upsampled * 0.3
        
        return pattern
    
    def _dissolve_frame_vectorized(
        self,
        linear_premul: np.ndarray,
        original_alpha: np.ndarray,
        threshold_map: np.ndarray,
        scatter_dx: np.ndarray,
        scatter_dy: np.ndarray,
        t: float,
        h: int,
        w: int
    ) -> np.ndarray:
        """Create single dissolve frame - fully vectorized"""
        result = np.zeros_like(linear_premul)
        
        # Calculate dissolve threshold with smooth edges
        visible_threshold = t * self.config.intensity
        
        # Smooth visibility using smoothstep (anti-aliased edge)
        edge_width = 0.1
        visibility = 1.0 - self._smoothstep(
            visible_threshold - edge_width,
            visible_threshold + edge_width,
            threshold_map
        )
        
        # Mask for pixels that have content
        has_content = original_alpha > 0.01
        
        # VISIBLE PIXELS: Copy with visibility factor
        visible_mask = (visibility > 0.01) & has_content
        
        for c in range(4):
            result[:, :, c] = np.where(
                visible_mask,
                linear_premul[:, :, c] * visibility,
                result[:, :, c]
            )
        
        # SCATTERED PIXELS: Particles at dissolve edge
        scatter_zone = (visibility > 0.01) & (visibility < 0.5) & has_content
        
        if np.any(scatter_zone):
            # Get source coordinates for scattered pixels
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            
            # Calculate scatter destinations
            dest_x = np.clip(x_coords + scatter_dx, 0, w - 1)
            dest_y = np.clip(y_coords + scatter_dy, 0, h - 1)
            
            # Scatter intensity based on how close to threshold
            scatter_intensity = visibility * (1.0 - visibility) * 4  # Peak at 0.5
            
            # Apply scatter using advanced indexing
            scatter_y = dest_y[scatter_zone]
            scatter_x = dest_x[scatter_zone]
            scatter_int = scatter_intensity[scatter_zone]
            
            # Add scattered particles (using np.add.at for accumulation)
            for c in range(3):
                values = linear_premul[:, :, c][scatter_zone] * scatter_int * 0.5
                np.add.at(result[:, :, c], (scatter_y, scatter_x), values)
            
            # Update alpha for scattered
            alpha_values = original_alpha[scatter_zone] * scatter_int * 0.5
            np.maximum.at(result[:, :, 3], (scatter_y, scatter_x), alpha_values)
        
        return result
