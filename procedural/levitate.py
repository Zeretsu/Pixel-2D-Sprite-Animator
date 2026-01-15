"""
Levitate Effect - Floating with rotating particles
Magic/psychic hover animation with pixel-perfect rendering
"""

import numpy as np
from typing import List, Optional, Tuple
from .base import BaseEffect, EffectConfig, Easing, PixelMath
from ..core.parser import Sprite


class LevitateEffect(BaseEffect):
    """Creates magic levitation with floating motion and anti-aliased orbiting particles"""
    
    name = "levitate"
    description = "Magic floating with orbiting particles"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        
        self.float_height = self.config.extra.get('height', 3.0)
        self.particle_count = self.config.extra.get('particles', 6)
        self.particle_color = self.config.extra.get('color', (255, 220, 100))  # Golden
        self.glow = self.config.extra.get('glow', True)
        self.tilt = self.config.extra.get('tilt', True)  # Subtle rotation
        
        # Pre-convert particle color to linear space
        self._particle_linear = np.array([
            (c / 255.0) ** PixelMath.GAMMA for c in self.particle_color
        ], dtype=np.float32)
    
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
        
        # Find sprite center and bounds
        cx, cy = self._find_center(original, w, h)
        bounds = self._find_bounds(original, w, h)
        
        # Pre-generate particle orbits
        particles = self._generate_particles(cx, cy, bounds)
        
        # Convert to linear premultiplied for effects
        linear_premul = self._to_linear_premul(original)
        
        for i in range(self.config.frame_count):
            t = i / max(1, self.config.frame_count - 1) if self.config.frame_count > 1 else 0.0
            
            frame_linear = self._create_levitate_frame(
                original, linear_premul, t, particles, cx, cy, h, w
            )
            
            frame_pixels = self._from_linear_premul(frame_linear)
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _find_center(self, pixels: np.ndarray, w: int, h: int) -> Tuple[float, float]:
        """Find center of sprite"""
        if pixels.shape[2] == 4:
            alpha = pixels[:, :, 3].astype(np.float32)
        else:
            alpha = np.any(pixels > 0, axis=2).astype(np.float32)
        
        total = np.sum(alpha)
        if total < 1:
            return w / 2, h / 2
        
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        cx = np.sum(x_coords * alpha) / total
        cy = np.sum(y_coords * alpha) / total
        
        return cx, cy
    
    def _find_bounds(self, pixels: np.ndarray, w: int, h: int) -> dict:
        """Find sprite bounding box"""
        if pixels.shape[2] == 4:
            mask = pixels[:, :, 3] > 0
        else:
            mask = np.any(pixels > 0, axis=2)
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return {'x_min': 0, 'x_max': w, 'y_min': 0, 'y_max': h}
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        return {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'width': x_max - x_min,
            'height': y_max - y_min
        }
    
    def _generate_particles(self, cx: float, cy: float, bounds: dict) -> list:
        """Generate orbiting particle data"""
        particles = []
        
        # Orbit radius based on sprite size
        orbit_radius = max(bounds.get('width', 10), bounds.get('height', 10)) * 0.7
        
        for i in range(self.particle_count):
            angle_offset = (i / self.particle_count) * 2 * np.pi
            
            particles.append({
                'angle_offset': angle_offset,
                'radius': orbit_radius * self.rng.uniform(0.8, 1.2),
                'speed': self.rng.uniform(0.8, 1.2),
                'size': self.rng.uniform(1.5, 3),
                'y_offset': self.rng.uniform(-0.3, 0.3),
                'brightness': self.rng.uniform(0.7, 1.0),
            })
        
        return particles
    
    def _create_levitate_frame(
        self,
        original: np.ndarray,
        linear_premul: np.ndarray,
        t: float,
        particles: list,
        cx: float,
        cy: float,
        h: int,
        w: int
    ) -> np.ndarray:
        """Create a single levitate frame"""
        
        # Calculate float offset (smooth up/down bob)
        float_offset = np.sin(t * 2 * np.pi) * self.float_height * self.config.intensity
        
        # Secondary smaller bob
        float_offset += np.sin(t * 4 * np.pi) * self.float_height * 0.2 * self.config.intensity
        
        # Subtle horizontal sway
        sway_offset = np.sin(t * 2 * np.pi + np.pi/4) * self.float_height * 0.3 * self.config.intensity
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Apply floating offset
        src_x = x_coords - sway_offset
        src_y = y_coords + float_offset
        
        # Optional subtle tilt
        if self.tilt:
            tilt_angle = np.sin(t * 2 * np.pi) * 0.03 * self.config.intensity
            
            cos_a = np.cos(-tilt_angle)
            sin_a = np.sin(-tilt_angle)
            
            dx = src_x - cx
            dy = src_y - cy
            
            src_x = dx * cos_a - dy * sin_a + cx
            src_y = dx * sin_a + dy * cos_a + cy
        
        # Sample sprite at new position (uses gamma_correct=True)
        result_srgb = PixelMath.bilinear_sample(original, src_x, src_y, gamma_correct=True)
        
        # Convert to linear premultiplied for effects
        result = self._to_linear_premul(result_srgb)
        
        # Add glow underneath (levitation aura)
        if self.glow:
            result = self._add_levitation_glow(result, original, cx, cy, t, h, w)
        
        # Draw orbiting particles with anti-aliasing
        result = self._draw_particles_aa(result, particles, t, cx, cy - float_offset, h, w)
        
        return result
    
    def _add_levitation_glow(
        self,
        result: np.ndarray,
        original: np.ndarray,
        cx: float,
        cy: float,
        t: float,
        h: int,
        w: int
    ) -> np.ndarray:
        """Add glowing aura effect in linear space"""
        # Find sprite bottom
        if original.shape[2] == 4:
            mask = original[:, :, 3] > 0
        else:
            mask = np.any(original > 0, axis=2)
        
        rows = np.any(mask, axis=1)
        if not np.any(rows):
            return result
        
        y_max = np.where(rows)[0][-1]
        
        # Create glow beneath sprite
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Elliptical glow at sprite's base
        glow_cy = y_max + 3
        
        # Smooth elliptical falloff
        dx_norm = (x_coords - cx) / 15.0
        dy_norm = (y_coords - glow_cy) / 5.0
        glow_dist_sq = dx_norm ** 2 + dy_norm ** 2
        
        # Gaussian falloff for smooth glow
        pulse = 0.6 + 0.4 * np.sin(t * 4 * np.pi)
        glow_intensity = np.exp(-glow_dist_sq * 2.0) * pulse * 0.5 * self.config.intensity
        
        # Only apply to areas below sprite and not on sprite
        glow_mask = (y_coords > y_max) & (glow_intensity > 0.01) & ~mask
        
        # Additive blend in linear space (premultiplied)
        glow_alpha = glow_intensity * glow_mask
        for c in range(3):
            result[:, :, c] = result[:, :, c] + self._particle_linear[c] * glow_alpha
        result[:, :, 3] = np.maximum(result[:, :, 3], glow_alpha * 0.6)
        
        return result
    
    def _draw_particles_aa(
        self,
        linear_premul: np.ndarray,
        particles: list,
        t: float,
        cx: float,
        cy: float,
        h: int,
        w: int
    ) -> np.ndarray:
        """Draw anti-aliased orbiting magic particles"""
        result = linear_premul.copy()
        
        for p in particles:
            # Particle orbits around center
            angle = t * 2 * np.pi * p['speed'] + p['angle_offset']
            
            # 3D-ish orbit (y varies with angle to simulate depth)
            px = cx + np.cos(angle) * p['radius']
            py = cy + np.sin(angle) * p['radius'] * 0.4 + p['y_offset'] * p['radius']
            
            # Depth-based brightness (brighter when in front)
            depth_brightness = 0.5 + 0.5 * np.sin(angle)
            
            # Skip if far out of bounds
            if px < -5 or px >= w + 5 or py < -5 or py >= h + 5:
                continue
            
            # Draw anti-aliased glowing particle using Gaussian splat
            size = p['size']
            intensity = p['brightness'] * depth_brightness * self.config.intensity
            
            # Calculate region to affect
            radius_int = int(np.ceil(size * 2)) + 1
            px_int = int(px + 0.5)
            py_int = int(py + 0.5)
            
            for dy in range(-radius_int, radius_int + 1):
                for dx in range(-radius_int, radius_int + 1):
                    ppx = px_int + dx
                    ppy = py_int + dy
                    
                    if 0 <= ppx < w and 0 <= ppy < h:
                        # Distance from particle center (subpixel accurate)
                        dist_sq = (ppx - px) ** 2 + (ppy - py) ** 2
                        sigma = size * 0.5
                        
                        # Gaussian falloff
                        falloff = np.exp(-dist_sq / (2 * sigma ** 2))
                        pixel_alpha = intensity * falloff
                        
                        # Additive blend in linear premultiplied space
                        for c in range(3):
                            result[ppy, ppx, c] += self._particle_linear[c] * pixel_alpha
                        result[ppy, ppx, 3] = max(result[ppy, ppx, 3], pixel_alpha * 0.8)
        
        return result
