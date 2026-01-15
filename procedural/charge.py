"""
Charge Effect - Energy buildup with particles and glow
Power-up animation for attacks, abilities, charging
Pixel-perfect with anti-aliased particles and proper color blending
"""

import numpy as np
from typing import List, Optional
from .base import BaseEffect, EffectConfig, Easing, PixelMath
from ..core.parser import Sprite


class ChargeEffect(BaseEffect):
    """Creates energy charge/power-up effect with particles and glow"""
    
    name = "charge"
    description = "Energy buildup with particles and glow"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        
        self.charge_color = np.array(self.config.extra.get('color', (255, 200, 50)), dtype=np.float64)
        self.particle_count = self.config.extra.get('particles', 12)
        self.buildup = self.config.extra.get('buildup', True)
        self.shake = self.config.extra.get('shake', True)
        self.aura = self.config.extra.get('aura', True)
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        
        # Find sprite center using center of mass
        cx, cy = self._find_center(original)
        
        # Pre-generate particle data
        particles = self._generate_particles(cx, cy, w, h)
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            
            # Charge intensity
            if self.buildup:
                charge = Easing.ease_in_quad(t)
            else:
                charge = 0.5 + 0.5 * np.sin(t * 2.0 * np.pi)
            
            frame_pixels = self._create_charge_frame(
                original, t, charge, particles, cx, cy, h, w
            )
            
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _find_center(self, pixels: np.ndarray) -> tuple:
        """Find center of mass of visible pixels"""
        h, w = pixels.shape[:2]
        
        if pixels.shape[2] == 4:
            alpha = pixels[:, :, 3].astype(np.float64)
        else:
            alpha = np.any(pixels > 0, axis=2).astype(np.float64) * 255.0
        
        total = np.sum(alpha)
        if total < 1:
            return w / 2.0, h / 2.0
        
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float64)
        cx = np.sum(x_coords * alpha) / total
        cy = np.sum(y_coords * alpha) / total
        
        return cx, cy
    
    def _generate_particles(self, cx: float, cy: float, w: int, h: int) -> list:
        """Generate particle properties for spiral animation"""
        particles = []
        radius = max(w, h) * 0.8
        
        for i in range(self.particle_count):
            base_angle = (i / self.particle_count) * 2.0 * np.pi
            angle_offset = self.rng.uniform(-0.3, 0.3)
            
            particles.append({
                'angle': base_angle + angle_offset,
                'radius': radius * self.rng.uniform(0.8, 1.2),
                'speed': self.rng.uniform(0.8, 1.2),
                'size': self.rng.uniform(1.5, 3.0),
                'phase': self.rng.uniform(0, 2.0 * np.pi),
            })
        
        return particles
    
    def _create_charge_frame(
        self,
        original: np.ndarray,
        t: float,
        charge: float,
        particles: list,
        cx: float,
        cy: float,
        h: int,
        w: int
    ) -> np.ndarray:
        """Create a single charge frame"""
        channels = original.shape[2]
        
        # Work in linear premultiplied space
        canvas = self._to_linear_premul(original)
        
        # Get sprite mask
        if channels == 4:
            mask = original[:, :, 3] > 0
        else:
            mask = np.any(original > 0, axis=2)
        
        # Apply shake at high charge
        if self.shake and charge > 0.5:
            shake_amount = (charge - 0.5) * 4.0 * self.config.intensity
            shake_x = np.sin(t * 50.0) * shake_amount
            shake_y = np.cos(t * 47.0) * shake_amount
            
            y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float64)
            src_x = x_coords - shake_x
            src_y = y_coords - shake_y
            
            # Sample with bilinear interpolation
            canvas = self._sample_linear_premul(canvas, src_x, src_y)
        
        # Add aura glow
        if self.aura:
            canvas = self._add_aura(canvas, mask, charge, cx, cy, h, w)
        
        # Draw anti-aliased particles
        canvas = self._draw_particles(canvas, particles, t, charge, cx, cy, h, w)
        
        # Add color tint to sprite
        tint_strength = charge * 0.4 * self.config.intensity
        tint_linear = np.power(self.charge_color / 255.0, PixelMath.GAMMA)
        
        # Apply tint only to sprite pixels (unpremultiply, tint, re-premultiply)
        alpha = canvas[:, :, 3]
        alpha_safe = np.maximum(alpha, 1e-10)
        
        for c in range(3):
            rgb = canvas[mask, c] / alpha_safe[mask]
            tinted = rgb * (1.0 - tint_strength) + tint_linear[c] * tint_strength
            canvas[mask, c] = tinted * alpha[mask]
        
        return self._from_linear_premul(canvas, channels)
    
    def _add_aura(
        self,
        canvas: np.ndarray,
        mask: np.ndarray,
        charge: float,
        cx: float,
        cy: float,
        h: int,
        w: int
    ) -> np.ndarray:
        """Add glowing aura around sprite"""
        result = canvas.copy()
        
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float64)
        
        # Radial distance from center
        dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
        max_dist = np.sqrt(w ** 2 + h ** 2) / 2.0
        
        # Pulsing aura with smooth animation
        pulse = 0.7 + 0.3 * np.sin(charge * 10.0)
        aura_radius = max_dist * 0.4 * (1.0 + charge * 0.5)
        
        # Gaussian falloff for smooth aura
        aura_intensity = np.exp(-(dist / aura_radius) ** 2) * charge * pulse * self.config.intensity
        
        # Only apply outside sprite
        aura_mask = ~mask & (aura_intensity > 0.005)
        
        # Convert color to linear
        aura_linear = np.power(self.charge_color / 255.0, PixelMath.GAMMA)
        
        # Additive blending in premultiplied space
        for c in range(3):
            result[aura_mask, c] += aura_linear[c] * aura_intensity[aura_mask]
        
        # Aura alpha
        result[aura_mask, 3] = np.maximum(result[aura_mask, 3], aura_intensity[aura_mask] * 0.7)
        
        return result
    
    def _draw_particles(
        self,
        canvas: np.ndarray,
        particles: list,
        t: float,
        charge: float,
        cx: float,
        cy: float,
        h: int,
        w: int
    ) -> np.ndarray:
        """Draw anti-aliased energy particles spiraling toward center"""
        result = canvas.copy()
        
        # Convert particle color to linear
        color_linear = np.power(self.charge_color / 255.0, PixelMath.GAMMA)
        
        # Create coordinate grids for distance calculations
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float64)
        
        for p in particles:
            # Particle position along spiral
            progress = (t * p['speed'] + p['phase'] / (2.0 * np.pi)) % 1.0
            
            # Radius decreases toward center
            current_radius = p['radius'] * (1.0 - progress * 0.9)
            
            # Spiral angle increases
            spiral_angle = p['angle'] + progress * 4.0 * np.pi
            
            # Particle position
            px = cx + np.cos(spiral_angle) * current_radius
            py = cy + np.sin(spiral_angle) * current_radius
            
            # Skip if far outside bounds
            if px < -5 or px >= w + 5 or py < -5 or py >= h + 5:
                continue
            
            # Particle size and intensity
            size = p['size'] * (1.0 + charge * 0.5)
            intensity = (1.0 - progress) * charge * self.config.intensity
            
            if intensity < 0.01:
                continue
            
            # Distance from particle center for each pixel
            dist_sq = (x_coords - px) ** 2 + (y_coords - py) ** 2
            
            # Gaussian particle with anti-aliased edges
            sigma = size / 2.0
            particle_alpha = np.exp(-dist_sq / (2.0 * sigma ** 2)) * intensity
            
            # Only affect pixels within reasonable range
            affect_mask = particle_alpha > 0.005
            
            # Additive blending for glow effect
            for c in range(3):
                result[affect_mask, c] += color_linear[c] * particle_alpha[affect_mask]
            
            # Update alpha
            result[affect_mask, 3] = np.maximum(result[affect_mask, 3], particle_alpha[affect_mask])
        
        return result
    
    def _to_linear_premul(self, pixels: np.ndarray) -> np.ndarray:
        """Convert sRGB to linear premultiplied"""
        h, w = pixels.shape[:2]
        channels = pixels.shape[2]
        
        result = np.zeros((h, w, 4), dtype=np.float64)
        result[:, :, :3] = np.power(pixels[:, :, :3].astype(np.float64) / 255.0, PixelMath.GAMMA)
        
        if channels == 4:
            result[:, :, 3] = pixels[:, :, 3].astype(np.float64) / 255.0
        else:
            result[:, :, 3] = np.where(np.any(pixels > 0, axis=2), 1.0, 0.0)
        
        result[:, :, :3] *= result[:, :, 3:4]
        return result
    
    def _from_linear_premul(self, pixels: np.ndarray, output_channels: int) -> np.ndarray:
        """Convert linear premultiplied to sRGB"""
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
    
    def _sample_linear_premul(
        self,
        pixels: np.ndarray,
        src_x: np.ndarray,
        src_y: np.ndarray
    ) -> np.ndarray:
        """Bilinear sample in linear premultiplied space"""
        h, w = pixels.shape[:2]
        
        x0 = np.floor(src_x).astype(np.int32)
        y0 = np.floor(src_y).astype(np.int32)
        fx = src_x - x0
        fy = src_y - y0
        
        x0c = np.clip(x0, 0, w - 1)
        x1c = np.clip(x0 + 1, 0, w - 1)
        y0c = np.clip(y0, 0, h - 1)
        y1c = np.clip(y0 + 1, 0, h - 1)
        
        p00 = pixels[y0c, x0c]
        p01 = pixels[y0c, x1c]
        p10 = pixels[y1c, x0c]
        p11 = pixels[y1c, x1c]
        
        w00 = ((1.0 - fx) * (1.0 - fy))[:, :, np.newaxis]
        w01 = (fx * (1.0 - fy))[:, :, np.newaxis]
        w10 = ((1.0 - fx) * fy)[:, :, np.newaxis]
        w11 = (fx * fy)[:, :, np.newaxis]
        
        result = p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11
        
        # Handle out of bounds
        oob = (src_x < 0) | (src_x >= w) | (src_y < 0) | (src_y >= h)
        result[oob] = 0
        
        return result
