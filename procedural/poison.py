"""
Poison Effect - Toxic dripping with bubbles
Poisoned status effect animation with anti-aliased particles
"""

import numpy as np
from typing import List, Optional
from .base import BaseEffect, EffectConfig, Easing, PixelMath
from ..core.parser import Sprite


class PoisonEffect(BaseEffect):
    """Creates poison/toxic effect with drips and bubbles"""
    
    name = "poison"
    description = "Toxic dripping with bubbles"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        
        self.poison_color = np.array(self.config.extra.get('color', (120, 200, 50)), dtype=np.float64)
        self.drip_color = np.array(self.config.extra.get('drip_color', (80, 160, 30)), dtype=np.float64)
        self.bubble_count = self.config.extra.get('bubbles', 8)
        self.drip_count = self.config.extra.get('drips', 4)
        self.pulse = self.config.extra.get('pulse', True)
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        
        # Pre-generate bubble and drip data
        bubbles = self._generate_bubbles(original, h, w)
        drips = self._generate_drips(original, h, w)
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            
            frame_pixels = self._create_poison_frame(
                original, bubbles, drips, t, h, w
            )
            
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _generate_bubbles(self, pixels: np.ndarray, h: int, w: int) -> list:
        """Generate bubble particle data"""
        bubbles = []
        
        if pixels.shape[2] == 4:
            mask = pixels[:, :, 3] > 0
        else:
            mask = np.any(pixels > 0, axis=2)
        
        cols = np.any(mask, axis=0)
        rows = np.any(mask, axis=1)
        
        if not np.any(cols) or not np.any(rows):
            return bubbles
        
        x_min, x_max = np.where(cols)[0][[0, -1]]
        y_min, y_max = np.where(rows)[0][[0, -1]]
        
        for i in range(self.bubble_count):
            bubbles.append({
                'x': self.rng.uniform(x_min, x_max),
                'y_start': self.rng.uniform(y_min + (y_max - y_min) * 0.3, y_max),
                'speed': self.rng.uniform(0.5, 1.5),
                'size': self.rng.uniform(1.0, 2.5),
                'phase': self.rng.uniform(0, 1),
                'wobble': self.rng.uniform(0.5, 1.5),
            })
        
        return bubbles
    
    def _generate_drips(self, pixels: np.ndarray, h: int, w: int) -> list:
        """Generate drip positions from bottom edge"""
        drips = []
        
        if pixels.shape[2] == 4:
            mask = pixels[:, :, 3] > 0
        else:
            mask = np.any(pixels > 0, axis=2)
        
        # Find bottom edge points
        bottom_points = []
        for x in range(w):
            col = mask[:, x]
            if np.any(col):
                y = np.where(col)[0][-1]
                bottom_points.append((x, y))
        
        if not bottom_points:
            return drips
        
        # Select drip points
        if len(bottom_points) > self.drip_count:
            indices = self.rng.choice(len(bottom_points), self.drip_count, replace=False)
            selected = [bottom_points[i] for i in indices]
        else:
            selected = bottom_points
        
        for x, y in selected:
            drips.append({
                'x': float(x),
                'y_start': float(y),
                'speed': self.rng.uniform(0.8, 1.2),
                'phase': self.rng.uniform(0, 1),
                'length': self.rng.uniform(4, 10),
            })
        
        return drips
    
    def _create_poison_frame(
        self,
        original: np.ndarray,
        bubbles: list,
        drips: list,
        t: float,
        h: int,
        w: int
    ) -> np.ndarray:
        """Create a single poison frame with proper color blending"""
        channels = original.shape[2]
        
        if channels == 4:
            mask = original[:, :, 3] > 0
            alpha = original[:, :, 3].astype(np.float64) / 255.0
        else:
            mask = np.any(original > 0, axis=2)
            alpha = mask.astype(np.float64)
        
        # Work in linear color space
        rgb_linear = np.power(original[:, :, :3].astype(np.float64) / 255.0, PixelMath.GAMMA)
        poison_linear = np.power(self.poison_color / 255.0, PixelMath.GAMMA)
        
        # Pulsing tint strength
        pulse_factor = 1.0
        if self.pulse:
            pulse_factor = 0.7 + 0.3 * np.sin(t * 4.0 * np.pi)
        
        tint_strength = 0.35 * self.config.intensity * pulse_factor
        
        # Apply green tint in linear space
        result = rgb_linear.copy()
        for c in range(3):
            tint_weight = tint_strength if c != 1 else tint_strength * 0.5  # Less tint on green channel
            result[mask, c] = result[mask, c] * (1 - tint_weight) + poison_linear[c] * tint_weight
        
        # Create canvas for particles (premultiplied linear)
        canvas = np.zeros((h, w, 4), dtype=np.float64)
        canvas[:, :, :3] = result * alpha[:, :, np.newaxis]
        canvas[:, :, 3] = alpha
        
        # Draw anti-aliased bubbles
        canvas = self._draw_bubbles(canvas, bubbles, t, h, w)
        
        # Draw anti-aliased drips
        canvas = self._draw_drips(canvas, drips, t, h, w)
        
        # Convert back to sRGB
        return self._from_linear_premul(canvas, channels)
    
    def _draw_bubbles(
        self,
        canvas: np.ndarray,
        bubbles: list,
        t: float,
        h: int,
        w: int
    ) -> np.ndarray:
        """Draw anti-aliased rising bubbles"""
        result = canvas.copy()
        
        poison_linear = np.power(self.poison_color / 255.0, PixelMath.GAMMA)
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float64)
        
        for b in bubbles:
            # Bubble position
            progress = (t * b['speed'] + b['phase']) % 1.0
            rise_distance = h * 0.4
            
            y = b['y_start'] - progress * rise_distance
            x = b['x'] + np.sin(progress * 6.0 * np.pi) * b['wobble'] * 2.0
            
            # Fade as it rises
            bubble_alpha = (1.0 - progress) * self.config.intensity * 0.8
            
            if y < -5 or y >= h + 5:
                continue
            
            # Anti-aliased bubble using Gaussian
            dist_sq = (x_coords - x) ** 2 + (y_coords - y) ** 2
            size = b['size']
            
            # Hollow bubble: ring shape
            dist = np.sqrt(dist_sq)
            ring_center = size * 0.7
            ring_width = size * 0.4
            ring_intensity = np.exp(-((dist - ring_center) ** 2) / (ring_width ** 2))
            
            # Inner highlight
            inner_intensity = np.exp(-dist_sq / (size * 0.3) ** 2) * 0.3
            
            bubble_shape = (ring_intensity + inner_intensity) * bubble_alpha
            affect_mask = bubble_shape > 0.005
            
            # Additive blend
            for c in range(3):
                result[affect_mask, c] += poison_linear[c] * bubble_shape[affect_mask]
            result[affect_mask, 3] = np.maximum(result[affect_mask, 3], bubble_shape[affect_mask])
        
        return result
    
    def _draw_drips(
        self,
        canvas: np.ndarray,
        drips: list,
        t: float,
        h: int,
        w: int
    ) -> np.ndarray:
        """Draw anti-aliased dripping poison"""
        result = canvas.copy()
        
        drip_linear = np.power(self.drip_color / 255.0, PixelMath.GAMMA)
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float64)
        
        for d in drips:
            progress = (t * d['speed'] + d['phase']) % 1.0
            
            # Drip head position
            fall_distance = d['length'] + 8
            drip_y = d['y_start'] + progress * fall_distance
            drip_x = d['x']
            
            # Drip trail length decreases as it falls
            trail_length = d['length'] * (1.0 - progress * 0.5)
            
            # Draw drip as elongated gaussian blob
            drip_alpha = (1.0 - progress * 0.5) * self.config.intensity * 0.9
            
            # Distance from drip center line
            dx = x_coords - drip_x
            
            # Y distance from drip (tapered trail)
            dy = y_coords - drip_y
            
            # Drip is at dy=0, trail extends upward (negative dy)
            # Taper width based on position in trail
            trail_pos = np.clip(-dy / max(trail_length, 1), 0, 1)
            width = 1.0 + trail_pos * 0.5  # Wider at top
            
            # Gaussian in x, linear falloff in y for trail
            x_falloff = np.exp(-(dx ** 2) / (width ** 2))
            y_falloff = np.where(
                (dy <= 0) & (dy >= -trail_length),
                1.0 - trail_pos * 0.5,  # Trail fades toward top
                np.where(dy > 0, np.exp(-(dy ** 2) / 2), 0)  # Droplet below
            )
            
            drip_shape = x_falloff * y_falloff * drip_alpha
            affect_mask = drip_shape > 0.005
            
            # Over blend (drips are more opaque)
            for c in range(3):
                result[affect_mask, c] = (
                    drip_linear[c] * drip_shape[affect_mask] +
                    result[affect_mask, c] * (1.0 - drip_shape[affect_mask])
                )
            result[affect_mask, 3] = np.maximum(result[affect_mask, 3], drip_shape[affect_mask])
        
        return result
    
    def _from_linear_premul(self, pixels: np.ndarray, output_channels: int) -> np.ndarray:
        """Convert linear premultiplied to sRGB straight alpha"""
        h, w = pixels.shape[:2]
        result = np.zeros((h, w, output_channels), dtype=np.uint8)
        
        alpha = np.clip(pixels[:, :, 3], 0, 1)
        alpha_safe = np.maximum(alpha, 1e-10)
        
        rgb_linear = np.clip(pixels[:, :, :3] / alpha_safe[:, :, np.newaxis], 0, 1)
        rgb_srgb = np.power(rgb_linear, PixelMath.INV_GAMMA) * 255.0
        
        result[:, :, :3] = np.clip(rgb_srgb, 0, 255).astype(np.uint8)
        if output_channels == 4:
            result[:, :, 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        
        return result
