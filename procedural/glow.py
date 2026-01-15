"""
Enhanced Glow Effect - Smooth pulsing aura with vectorized rendering
Fixed seamless looping and improved glow quality
"""

import numpy as np
from typing import List, Optional, Tuple
from .base import BaseEffect, EffectConfig, Easing
from ..core.parser import Sprite


class GlowEffect(BaseEffect):
    """Creates smooth pulsing glow/aura effect with artifact-free rendering"""
    
    name = "glow"
    description = "Smooth pulsing glow and aura"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        
        self.glow_size = self.config.extra.get('glow_size', 3)
        self.glow_color = self.config.extra.get('glow_color', None)
        self.inner_glow = self.config.extra.get('inner_glow', True)
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        
        # Detect color and create masks
        glow_color = self._detect_glow_color(original) if self.glow_color is None else self.glow_color
        edge_mask = self._create_edge_mask(original)
        distance_map = self._create_distance_map(edge_mask, original[:, :, 3] > 0)
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            
            # Smooth breathing pulse
            pulse = Easing.breathing(t)
            glow_intensity = 0.4 + pulse * 0.6 * self.config.intensity
            
            frame_pixels = self._apply_smooth_glow(
                original, distance_map, glow_color, glow_intensity
            )
            
            # Inner brightness pulse
            if self.inner_glow:
                frame_pixels = self._apply_inner_glow(frame_pixels, pulse)
            
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _detect_glow_color(self, pixels: np.ndarray) -> Tuple[int, int, int]:
        """Detect brightest color for glow"""
        mask = pixels[:, :, 3] > 0
        if not mask.any():
            return (255, 255, 255)
        
        brightness = pixels[:, :, :3].sum(axis=2).astype(float)
        brightness[~mask] = 0
        
        threshold = np.percentile(brightness[mask], 85)
        bright_mask = brightness >= threshold
        
        if bright_mask.any():
            r = int(pixels[bright_mask, 0].mean())
            g = int(pixels[bright_mask, 1].mean())
            b = int(pixels[bright_mask, 2].mean())
            factor = min(2.0, 255 / max(r, g, b, 1))
            return (
                min(255, int(r * factor)),
                min(255, int(g * factor)),
                min(255, int(b * factor))
            )
        
        return (255, 255, 255)
    
    def _create_edge_mask(self, pixels: np.ndarray) -> np.ndarray:
        """Create mask of sprite edges"""
        alpha = pixels[:, :, 3]
        visible = alpha > 0
        
        # Vectorized edge detection
        edge_mask = np.zeros_like(visible)
        
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = np.roll(np.roll(visible, dy, axis=0), dx, axis=1)
            edge_mask |= (visible & ~shifted)
        
        return edge_mask
    
    def _create_distance_map(self, edge_mask: np.ndarray, visible: np.ndarray) -> np.ndarray:
        """Create distance map from edges (vectorized)"""
        h, w = edge_mask.shape
        
        try:
            from scipy.ndimage import distance_transform_edt
            # Distance from edge for outside glow
            outside = ~visible
            dist_outside = distance_transform_edt(~edge_mask & outside)
            dist_outside[visible] = 0
            return dist_outside
        except ImportError:
            # Fallback: simple expansion
            dist = np.zeros((h, w), dtype=float)
            dist[~visible] = float('inf')
            
            current = edge_mask.copy()
            for d in range(1, self.glow_size + 2):
                expanded = np.zeros_like(current)
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    expanded |= np.roll(np.roll(current, dy, axis=0), dx, axis=1)
                new_area = expanded & ~visible & (dist == float('inf'))
                dist[new_area] = d
                current = expanded
            
            dist[dist == float('inf')] = self.glow_size + 2
            return dist
    
    def _apply_smooth_glow(
        self, pixels: np.ndarray, distance_map: np.ndarray,
        color: Tuple[int, int, int], intensity: float
    ) -> np.ndarray:
        """Apply smooth glow using vectorized operations"""
        h, w = pixels.shape[:2]
        result = pixels.copy()
        
        max_dist = self.glow_size * intensity + 1
        
        # Glow area: outside sprite, within range
        glow_area = (pixels[:, :, 3] == 0) & (distance_map > 0) & (distance_map <= max_dist)
        
        if not glow_area.any():
            return result
        
        # Smooth falloff using smoother_step
        normalized_dist = distance_map[glow_area] / max_dist
        falloff = 1 - Easing.smoother_step(normalized_dist)
        
        # Apply glow color with falloff alpha
        result[glow_area, 0] = color[0]
        result[glow_area, 1] = color[1]
        result[glow_area, 2] = color[2]
        result[glow_area, 3] = (falloff * 180 * intensity).astype(np.uint8)
        
        return result
    
    def _apply_inner_glow(self, pixels: np.ndarray, pulse: float) -> np.ndarray:
        """Subtle inner brightness pulse"""
        result = pixels.copy()
        visible = pixels[:, :, 3] > 0
        
        brightness = 1.0 + pulse * 0.15 * self.config.intensity
        result[visible, :3] = np.clip(
            result[visible, :3] * brightness, 0, 255
        ).astype(np.uint8)
        
        return result
