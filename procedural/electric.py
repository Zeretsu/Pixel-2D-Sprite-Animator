"""
Electric Effect - Lightning/electricity crackling animation with pixel-perfect rendering
"""

import numpy as np
from typing import List, Optional, Tuple
from .base import BaseEffect, EffectConfig, PixelMath
from ..core.parser import Sprite


class ElectricEffect(BaseEffect):
    """Creates electric/lightning effect with anti-aliased arcs and linear color blending"""
    
    name = "electric"
    description = "Electricity and lightning"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        self.rng = np.random.default_rng(self.config.seed)
        
        self.arc_count = self.config.extra.get('arc_count', 3)
        self.flash = self.config.extra.get('flash', True)
        
        # Pre-convert electric colors to linear space
        self._colors_linear = [
            np.array([(c / 255.0) ** PixelMath.GAMMA for c in color], dtype=np.float32)
            for color in [
                (200, 230, 255),  # Light blue
                (150, 200, 255),  # Blue
                (255, 255, 255),  # White
            ]
        ]
    
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
        
        # Get edge pixels vectorized
        edge_pixels = self._get_edge_pixels_vectorized(original)
        
        # Pre-generate arc data for consistency
        arc_data = self._pregenerate_arcs(edge_pixels, h, w)
        
        for i in range(self.config.frame_count):
            t = i / max(1, self.config.frame_count - 1) if self.config.frame_count > 1 else 0.0
            
            # Convert to linear premultiplied
            frame_linear = self._to_linear_premul(original)
            
            # Flash effect in linear space
            if self.flash:
                flash_intensity = self._calculate_flash(i)
                if flash_intensity > 1.0:
                    frame_linear[:, :, :3] *= flash_intensity
            
            # Draw anti-aliased electric arcs
            frame_linear = self._draw_arcs_aa(frame_linear, arc_data, i, h, w)
            
            frame_pixels = self._from_linear_premul(frame_linear)
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _get_edge_pixels_vectorized(self, pixels: np.ndarray) -> List[Tuple[int, int]]:
        """Get edge pixels using vectorized operations"""
        alpha = pixels[:, :, 3]
        visible = alpha > 0
        
        # Vectorized edge detection using shifts
        padded = np.pad(visible, 1, mode='constant', constant_values=False)
        
        edge_mask = visible & (
            ~padded[:-2, 1:-1] |  # Top neighbor empty
            ~padded[2:, 1:-1] |   # Bottom neighbor empty
            ~padded[1:-1, :-2] |  # Left neighbor empty
            ~padded[1:-1, 2:]     # Right neighbor empty
        )
        
        # Get coordinates
        y_coords, x_coords = np.where(edge_mask)
        return list(zip(x_coords, y_coords))
    
    def _pregenerate_arcs(self, edge_pixels: List[Tuple[int, int]], h: int, w: int) -> List[dict]:
        """Pre-generate arc data for all frames"""
        arc_data = []
        
        for frame_idx in range(self.config.frame_count):
            frame_arcs = []
            
            for _ in range(self.arc_count):
                if edge_pixels and self.rng.random() < 0.5:
                    start_idx = self.rng.integers(len(edge_pixels))
                    start = edge_pixels[start_idx]
                    
                    # Generate arc path
                    arc_path = self._generate_arc_path(start, h, w)
                    color_idx = self.rng.integers(len(self._colors_linear))
                    
                    frame_arcs.append({
                        'path': arc_path,
                        'color': self._colors_linear[color_idx]
                    })
            
            arc_data.append(frame_arcs)
        
        return arc_data
    
    def _generate_arc_path(self, start: Tuple[int, int], h: int, w: int) -> List[Tuple[float, float]]:
        """Generate smooth arc path with subpixel positions"""
        path = []
        x, y = float(start[0]), float(start[1])
        arc_length = self.rng.integers(3, 8)
        
        for _ in range(arc_length):
            path.append((x, y))
            
            # Smooth random direction
            dx = self.rng.uniform(-2, 2)
            dy = self.rng.uniform(-2, 2)
            
            x = np.clip(x + dx, 0, w - 1)
            y = np.clip(y + dy, 0, h - 1)
        
        return path
    
    def _calculate_flash(self, frame_idx: int) -> float:
        """Calculate flash intensity for frame"""
        # Deterministic flash based on frame
        seed = (self.config.seed or 0) + frame_idx
        rng = np.random.default_rng(seed)
        if rng.random() < 0.3 * self.config.intensity:
            return 1.3
        return 1.0
    
    def _draw_arcs_aa(self, linear_premul: np.ndarray, arc_data: List[dict], 
                      frame_idx: int, h: int, w: int) -> np.ndarray:
        """Draw anti-aliased electric arcs"""
        result = linear_premul.copy()
        
        if frame_idx >= len(arc_data):
            return result
        
        for arc in arc_data[frame_idx]:
            path = arc['path']
            color = arc['color']
            
            # Draw each segment with anti-aliased line
            for i in range(len(path) - 1):
                x0, y0 = path[i]
                x1, y1 = path[i + 1]
                self._draw_aa_glow_line(result, x0, y0, x1, y1, color, h, w)
            
            # Draw glow at each point
            for px, py in path:
                self._draw_glow_point(result, px, py, color, h, w)
        
        return result
    
    def _draw_aa_glow_line(self, canvas: np.ndarray, x0: float, y0: float,
                           x1: float, y1: float, color: np.ndarray, h: int, w: int):
        """Draw anti-aliased glowing line segment"""
        # Bresenham-like with anti-aliasing
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy), 1)
        
        for i in range(int(steps) + 1):
            t = i / steps if steps > 0 else 0
            px = x0 + dx * t
            py = y0 + dy * t
            
            # Draw with Gaussian splat for anti-aliasing
            self._splat_pixel(canvas, px, py, color, 1.0, h, w)
    
    def _draw_glow_point(self, canvas: np.ndarray, px: float, py: float,
                         color: np.ndarray, h: int, w: int):
        """Draw glowing point with smooth falloff"""
        radius = 1.5
        
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ix = int(px + 0.5) + dx
                iy = int(py + 0.5) + dy
                
                if 0 <= ix < w and 0 <= iy < h:
                    dist_sq = (ix - px) ** 2 + (iy - py) ** 2
                    falloff = np.exp(-dist_sq / (2 * (radius * 0.6) ** 2))
                    
                    if falloff > 0.01:
                        # Additive blend in linear space
                        intensity = falloff * self.config.intensity
                        for c in range(3):
                            canvas[iy, ix, c] += color[c] * intensity
                        canvas[iy, ix, 3] = max(canvas[iy, ix, 3], intensity)
    
    def _splat_pixel(self, canvas: np.ndarray, px: float, py: float,
                     color: np.ndarray, intensity: float, h: int, w: int):
        """Splat a single pixel with bilinear distribution"""
        x0, y0 = int(px), int(py)
        fx, fy = px - x0, py - y0
        
        weights = [
            ((1 - fx) * (1 - fy), x0, y0),
            (fx * (1 - fy), x0 + 1, y0),
            ((1 - fx) * fy, x0, y0 + 1),
            (fx * fy, x0 + 1, y0 + 1),
        ]
        
        for weight, ix, iy in weights:
            if 0 <= ix < w and 0 <= iy < h and weight > 0.01:
                contrib = weight * intensity * self.config.intensity
                for c in range(3):
                    canvas[iy, ix, c] += color[c] * contrib
                canvas[iy, ix, 3] = max(canvas[iy, ix, 3], contrib)
