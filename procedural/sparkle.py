"""
Enhanced Sparkle Effect - Magic particles with pixel-perfect anti-aliased rendering
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from .base import BaseEffect, EffectConfig, PixelMath
from ..core.parser import Sprite
from ..core.utils import MathUtils


class SparkleEffect(BaseEffect):
    """Creates sparkle/magic particle animation with anti-aliased rendering in linear color space"""
    
    name = "sparkle"
    description = "Glittering particles and magical sparkles with smooth animation"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        
        # Effect-specific settings
        self.density = self.config.extra.get('density', 0.12)
        self.fade_speed = self.config.extra.get('fade_speed', 0.3)
        self.twinkle_speed = self.config.extra.get('twinkle_speed', 2.0)
        self.drift = self.config.extra.get('drift', True)
        self.trails = self.config.extra.get('trails', False)
        
        # Color palettes
        self.palette_name = self.config.extra.get('palette', 'magic')
        self.colors = self._get_palette(self.palette_name)
        
        # Pre-convert colors to linear space
        self._colors_linear = [
            np.array([(c / 255.0) ** PixelMath.GAMMA for c in color], dtype=np.float32)
            for color in self.colors
        ]
        
        # Pre-generated sparkle data
        self.sparkle_data = []
    
    def _get_palette(self, name: str) -> List[Tuple[int, int, int]]:
        """Get color palette by name"""
        palettes = {
            'magic': [
                (255, 255, 255),
                (255, 255, 220),
                (220, 220, 255),
                (255, 220, 255),
                (220, 255, 255),
            ],
            'gold': [
                (255, 215, 0),
                (255, 230, 100),
                (255, 200, 50),
                (255, 245, 180),
            ],
            'ice': [
                (200, 240, 255),
                (180, 220, 255),
                (255, 255, 255),
                (220, 250, 255),
            ],
            'fire': [
                (255, 200, 100),
                (255, 180, 50),
                (255, 220, 150),
                (255, 255, 200),
            ],
            'rainbow': [
                (255, 100, 100),
                (255, 200, 100),
                (255, 255, 100),
                (100, 255, 100),
                (100, 200, 255),
                (200, 100, 255),
            ],
        }
        return palettes.get(name, palettes['magic'])
    
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
        alpha = np.clip(linear_premul[:, :, 3], 0.0, 1.0)
        safe_alpha = np.where(alpha > 1e-10, alpha, 1.0)
        
        linear_rgb = np.zeros_like(linear_premul[:, :, :3])
        linear_rgb[:, :, 0] = linear_premul[:, :, 0] / safe_alpha
        linear_rgb[:, :, 1] = linear_premul[:, :, 1] / safe_alpha
        linear_rgb[:, :, 2] = linear_premul[:, :, 2] / safe_alpha
        
        linear_rgb = np.clip(linear_rgb, 0.0, 1.0)
        srgb = np.power(linear_rgb, PixelMath.INV_GAMMA)
        
        result[:, :, :3] = (srgb * 255.0 + 0.5).astype(np.uint8)
        result[:, :, 3] = (alpha * 255.0 + 0.5).astype(np.uint8)
        return result
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        """Apply sparkle effect to sprite"""
        frames = []
        
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        mask = self._get_mask(original)
        
        # Detect dominant colors for auto-palette
        self._auto_adjust_palette(original, mask)
        
        # Update linear colors after palette adjustment
        self._colors_linear = [
            np.array([(c / 255.0) ** PixelMath.GAMMA for c in color], dtype=np.float32)
            for color in self.colors
        ]
        
        # Generate sparkle positions
        self._generate_sparkles(sprite, mask)
        
        # Convert original to linear premultiplied
        linear_premul = self._to_linear_premul(original)
        
        for i in range(self.config.frame_count):
            t = i / max(1, self.config.frame_count - 1) if self.config.frame_count > 1 else 0.0
            
            frame_linear = linear_premul.copy()
            
            # Draw sparkle trails if enabled
            if self.trails:
                frame_linear = self._draw_trails_aa(frame_linear, t, h, w)
            
            # Draw active sparkles with anti-aliasing
            frame_linear = self._draw_sparkles_aa(frame_linear, t, h, w)
            
            frame_pixels = self._from_linear_premul(frame_linear)
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _auto_adjust_palette(self, pixels: np.ndarray, mask: np.ndarray):
        """Adjust palette based on sprite colors"""
        if not np.any(mask):
            return
        
        # Sample visible pixel colors
        visible = pixels[mask]
        
        # Calculate average brightness and hue
        avg_r = np.mean(visible[:, 0])
        avg_g = np.mean(visible[:, 1])
        avg_b = np.mean(visible[:, 2])
        
        # Detect if sprite is warm or cool toned
        warmth = (avg_r - avg_b) / 255
        
        # Auto-select palette if not specified
        if self.palette_name == 'magic':
            if warmth > 0.2:
                self.colors = self._get_palette('gold')
            elif warmth < -0.2:
                self.colors = self._get_palette('ice')
    
    def _generate_sparkles(self, sprite: Sprite, mask: np.ndarray):
        """Generate sparkle positions with varied behaviors"""
        h, w = sprite.height, sprite.width
        
        # Calculate sparkle count
        visible_pixels = np.sum(mask)
        if visible_pixels == 0:
            self.sparkle_data = []
            return
        
        num_sparkles = max(4, int(visible_pixels * self.density * self.config.intensity))
        
        # Get valid positions
        valid_y, valid_x = np.where(mask)
        
        self.sparkle_data = []
        
        for i in range(num_sparkles):
            idx = self.rng.integers(0, len(valid_x))
            x, y = valid_x[idx], valid_y[idx]
            
            # Random timing and properties
            start_time = self.rng.random()
            duration = 0.15 + self.rng.random() * 0.35
            
            # Sparkle type affects behavior
            sparkle_type = self.rng.choice(['point', 'star', 'glow', 'burst'])
            
            # Select color and pre-convert to linear
            color = self.colors[self.rng.integers(0, len(self.colors))]
            color_linear = np.array([(c / 255.0) ** PixelMath.GAMMA for c in color], dtype=np.float32)
            
            self.sparkle_data.append({
                'x': float(x),
                'y': float(y),
                'start': start_time,
                'duration': duration,
                'color': color,
                'color_linear': color_linear,
                'size': self.rng.integers(1, 4),
                'type': sparkle_type,
                'phase_offset': self.rng.random() * np.pi * 2,
                'drift_x': (self.rng.random() - 0.5) * 2 if self.drift else 0,
                'drift_y': (self.rng.random() - 0.5) * 2 if self.drift else 0,
                'twinkle_freq': 1 + self.rng.random() * 3,
            })
    
    def _draw_sparkles_aa(
        self,
        linear_premul: np.ndarray,
        t: float,
        h: int,
        w: int
    ) -> np.ndarray:
        """Draw active sparkles with anti-aliased Gaussian rendering"""
        result = linear_premul.copy()
        
        for sparkle in self.sparkle_data:
            # Calculate local time (looping)
            local_t = (t - sparkle['start']) % 1.0
            
            if local_t > sparkle['duration']:
                continue
            
            # Calculate progress and intensity
            progress = local_t / sparkle['duration']
            
            # Smooth fade in/out using smoothstep
            if progress < 0.3:
                intensity = self._smoothstep(0, 0.3, progress)
            elif progress > 0.7:
                intensity = 1.0 - self._smoothstep(0.7, 1.0, progress)
            else:
                intensity = 1.0
            
            # Add twinkle effect
            twinkle = (np.sin(t * 2 * np.pi * sparkle['twinkle_freq'] * self.twinkle_speed + sparkle['phase_offset']) + 1) / 2
            intensity *= 0.6 + 0.4 * twinkle
            
            # Calculate position with subpixel drift
            x = sparkle['x'] + sparkle['drift_x'] * local_t * 3
            y = sparkle['y'] + sparkle['drift_y'] * local_t * 3
            
            color_linear = sparkle['color_linear']
            size = sparkle['size']
            sparkle_type = sparkle['type']
            
            # Draw based on type with anti-aliasing
            self._draw_sparkle_shape_aa(result, x, y, color_linear, size, intensity, sparkle_type, progress, h, w)
        
        return result
    
    def _smoothstep(self, edge0: float, edge1: float, x: float) -> float:
        """Hermite smoothstep"""
        t = np.clip((x - edge0) / (edge1 - edge0 + 1e-10), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    
    def _draw_sparkle_shape_aa(
        self,
        linear_premul: np.ndarray,
        x: float,
        y: float,
        color_linear: np.ndarray,
        size: int,
        intensity: float,
        sparkle_type: str,
        progress: float,
        h: int,
        w: int
    ):
        """Draw anti-aliased sparkle shape"""
        if sparkle_type == 'point':
            self._draw_point_aa(linear_premul, x, y, color_linear, intensity, h, w)
            
        elif sparkle_type == 'star':
            self._draw_star_aa(linear_premul, x, y, color_linear, size, intensity, progress, h, w)
            
        elif sparkle_type == 'glow':
            self._draw_glow_aa(linear_premul, x, y, color_linear, size, intensity, h, w)
            
        elif sparkle_type == 'burst':
            self._draw_burst_aa(linear_premul, x, y, color_linear, size, intensity, progress, h, w)
    
    def _draw_point_aa(
        self,
        linear_premul: np.ndarray,
        x: float,
        y: float,
        color_linear: np.ndarray,
        intensity: float,
        h: int,
        w: int
    ):
        """Draw anti-aliased single point sparkle using bilinear splat"""
        x0, y0 = int(x), int(y)
        fx, fy = x - x0, y - y0
        
        weights = [
            ((1 - fx) * (1 - fy), x0, y0),
            (fx * (1 - fy), x0 + 1, y0),
            ((1 - fx) * fy, x0, y0 + 1),
            (fx * fy, x0 + 1, y0 + 1),
        ]
        
        for weight, ix, iy in weights:
            if 0 <= ix < w and 0 <= iy < h and weight > 0.01:
                contrib = weight * intensity * self.config.intensity
                # Additive blend in linear space
                for c in range(3):
                    linear_premul[iy, ix, c] += color_linear[c] * contrib
                linear_premul[iy, ix, 3] = max(linear_premul[iy, ix, 3], contrib)
    
    def _draw_star_aa(
        self,
        linear_premul: np.ndarray,
        x: float,
        y: float,
        color_linear: np.ndarray,
        size: int,
        intensity: float,
        progress: float,
        h: int,
        w: int
    ):
        """Draw anti-aliased 4-point star sparkle"""
        # Center with Gaussian
        self._draw_gaussian_splat(linear_premul, x, y, color_linear, intensity, 0.8, h, w)
        
        # Star arms with length based on progress
        arm_length = size + size * np.sin(progress * np.pi)
        
        for d in np.linspace(0.5, arm_length, int(arm_length * 2)):
            arm_intensity = intensity * (1 - d / (arm_length + 1)) * 0.7
            
            if arm_intensity < 0.01:
                continue
            
            # Horizontal arms
            self._draw_gaussian_splat(linear_premul, x + d, y, color_linear, arm_intensity, 0.5, h, w)
            self._draw_gaussian_splat(linear_premul, x - d, y, color_linear, arm_intensity, 0.5, h, w)
            
            # Vertical arms
            self._draw_gaussian_splat(linear_premul, x, y + d, color_linear, arm_intensity, 0.5, h, w)
            self._draw_gaussian_splat(linear_premul, x, y - d, color_linear, arm_intensity, 0.5, h, w)
    
    def _draw_glow_aa(
        self,
        linear_premul: np.ndarray,
        x: float,
        y: float,
        color_linear: np.ndarray,
        size: int,
        intensity: float,
        h: int,
        w: int
    ):
        """Draw anti-aliased soft glow sparkle"""
        self._draw_gaussian_splat(linear_premul, x, y, color_linear, intensity, size * 0.7, h, w)
    
    def _draw_burst_aa(
        self,
        linear_premul: np.ndarray,
        x: float,
        y: float,
        color_linear: np.ndarray,
        size: int,
        intensity: float,
        progress: float,
        h: int,
        w: int
    ):
        """Draw anti-aliased expanding burst sparkle"""
        # Ring expands outward
        ring_radius = size * progress * 2
        
        # Draw ring using multiple points
        num_points = max(8, int(ring_radius * 4))
        for i in range(num_points):
            angle = i * 2 * np.pi / num_points
            px = x + np.cos(angle) * ring_radius
            py = y + np.sin(angle) * ring_radius
            
            self._draw_gaussian_splat(linear_premul, px, py, color_linear, intensity * 0.6, 0.6, h, w)
        
        # Center point fades
        center_intensity = intensity * (1 - progress * 0.5)
        self._draw_gaussian_splat(linear_premul, x, y, color_linear, center_intensity, 0.8, h, w)
    
    def _draw_gaussian_splat(
        self,
        linear_premul: np.ndarray,
        x: float,
        y: float,
        color_linear: np.ndarray,
        intensity: float,
        sigma: float,
        h: int,
        w: int
    ):
        """Draw anti-aliased Gaussian splat"""
        radius = int(np.ceil(sigma * 3)) + 1
        x_int, y_int = int(x + 0.5), int(y + 0.5)
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ix, iy = x_int + dx, y_int + dy
                
                if 0 <= ix < w and 0 <= iy < h:
                    dist_sq = (ix - x) ** 2 + (iy - y) ** 2
                    falloff = np.exp(-dist_sq / (2 * sigma ** 2))
                    
                    if falloff > 0.01:
                        contrib = falloff * intensity * self.config.intensity
                        # Additive blend
                        for c in range(3):
                            linear_premul[iy, ix, c] += color_linear[c] * contrib
                        linear_premul[iy, ix, 3] = max(linear_premul[iy, ix, 3], contrib)
    
    def _draw_trails_aa(
        self,
        linear_premul: np.ndarray,
        t: float,
        h: int,
        w: int
    ) -> np.ndarray:
        """Draw trailing sparkle effects with anti-aliasing"""
        result = linear_premul.copy()
        
        for sparkle in self.sparkle_data:
            local_t = (t - sparkle['start']) % 1.0
            
            if local_t > sparkle['duration'] * 1.5:
                continue
            
            # Trail fades quickly
            trail_intensity = max(0, 1 - local_t / (sparkle['duration'] * 1.5)) * 0.3
            
            if trail_intensity < 0.05:
                continue
            
            x, y = float(sparkle['x']), float(sparkle['y'])
            color_linear = sparkle['color_linear']
            
            # Draw faded trail point with anti-aliasing
            self._draw_gaussian_splat(result, x, y, color_linear, trail_intensity, 0.6, h, w)
        
        return result
