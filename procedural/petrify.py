"""
Petrify Effect - Stone/statue conversion
Petrified status effect animation with pixel-perfect rendering
"""

import numpy as np
from typing import List, Optional, Tuple
from .base import BaseEffect, EffectConfig, Easing, PixelMath
from ..core.parser import Sprite


class PetrifyEffect(BaseEffect):
    """Creates petrification/stone effect with proper linear color space operations"""
    
    name = "petrify"
    description = "Turn to stone with grayscale conversion"
    
    # Perceptually accurate luminance coefficients (Rec. 709)
    LUMA_R = 0.2126
    LUMA_G = 0.7152
    LUMA_B = 0.0722
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        
        self.stone_color = self.config.extra.get('stone_color', (140, 135, 130))
        self.mode = self.config.extra.get('mode', 'spread')  # 'spread', 'instant', 'crumble'
        self.cracks = self.config.extra.get('cracks', True)
        self.dust = self.config.extra.get('dust', True)
        
        # Pre-convert stone color to linear space
        self._stone_linear = np.array([
            (c / 255.0) ** PixelMath.GAMMA for c in self.stone_color
        ], dtype=np.float32)
    
    def _smoothstep(self, edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
        """Hermite smoothstep for anti-aliased transitions"""
        t = np.clip((x - edge0) / (edge1 - edge0 + 1e-10), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    
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
    
    def _perlin_noise_2d(self, shape: Tuple[int, int], scale: float, octaves: int = 4) -> np.ndarray:
        """Multi-octave smooth noise for organic patterns"""
        h, w = shape
        result = np.zeros((h, w), dtype=np.float32)
        amplitude = 1.0
        total_amplitude = 0.0
        
        for octave in range(octaves):
            freq = scale * (2 ** octave)
            # Generate smooth noise at this frequency
            noise_h = max(2, int(h / freq) + 2)
            noise_w = max(2, int(w / freq) + 2)
            
            noise_grid = self.rng.random((noise_h, noise_w)).astype(np.float32)
            
            # Bilinear interpolation for smoothness
            y_coords = np.linspace(0, noise_h - 1.001, h)
            x_coords = np.linspace(0, noise_w - 1.001, w)
            
            y0 = y_coords.astype(int)
            x0 = x_coords.astype(int)
            y1 = np.minimum(y0 + 1, noise_h - 1)
            x1 = np.minimum(x0 + 1, noise_w - 1)
            
            fy = y_coords - y0
            fx = x_coords - x0
            
            # Smoothstep interpolation weights
            fy = fy * fy * (3 - 2 * fy)
            fx = fx * fx * (3 - 2 * fx)
            
            # Bilinear sample
            fy = fy.reshape(-1, 1)
            fx = fx.reshape(1, -1)
            
            c00 = noise_grid[y0][:, x0]
            c10 = noise_grid[y1][:, x0]
            c01 = noise_grid[y0][:, x1]
            c11 = noise_grid[y1][:, x1]
            
            interp = (c00 * (1 - fy) * (1 - fx) +
                     c10 * fy * (1 - fx) +
                     c01 * (1 - fy) * fx +
                     c11 * fy * fx)
            
            result += interp * amplitude
            total_amplitude += amplitude
            amplitude *= 0.5
        
        return result / total_amplitude
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        h, w = sprite.height, sprite.width
        
        # Convert to linear premultiplied
        linear_premul = self._to_linear_premul(sprite.pixels)
        alpha_mask = linear_premul[:, :, 3] > 0.01
        
        # Generate organic stone spread pattern
        stone_pattern = self._generate_stone_pattern(linear_premul, h, w)
        
        # Pre-generate crack pattern (consistent across frames)
        crack_pattern = self._generate_crack_pattern(h, w, alpha_mask)
        
        for i in range(self.config.frame_count):
            t = i / max(1, self.config.frame_count - 1) if self.config.frame_count > 1 else 1.0
            
            if self.mode == 'spread':
                petrify_progress = Easing.ease_out_quad(t)
            elif self.mode == 'crumble':
                petrify_progress = Easing.ease_out_quad(min(t * 2, 1.0))
            else:  # instant
                petrify_progress = 1.0
            
            frame_linear = self._create_petrify_frame(
                linear_premul, stone_pattern, crack_pattern, petrify_progress, h, w
            )
            
            # Crumble effect
            if self.mode == 'crumble' and t > 0.5:
                crumble_t = (t - 0.5) / 0.5
                frame_linear = self._apply_crumble(frame_linear, crumble_t, h, w)
            
            frame_pixels = self._from_linear_premul(frame_linear)
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _generate_stone_pattern(self, linear_premul: np.ndarray, h: int, w: int) -> np.ndarray:
        """Generate organic petrification spread pattern"""
        alpha = linear_premul[:, :, 3]
        mask = alpha > 0.01
        
        if not np.any(mask):
            return np.ones((h, w), dtype=np.float32)
        
        # Find sprite bounds
        rows = np.any(mask, axis=1)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        sprite_height = y_max - y_min + 1
        
        # Base pattern: spread from bottom up
        y_coords = np.arange(h, dtype=np.float32).reshape(-1, 1)
        base_pattern = (y_max - y_coords) / max(1, sprite_height)
        base_pattern = np.clip(base_pattern, 0, 1)
        
        # Add organic noise for natural stone spread
        noise = self._perlin_noise_2d((h, w), scale=8.0, octaves=3)
        noise = noise * 0.3  # Noise amplitude
        
        pattern = base_pattern + noise
        pattern = np.clip(pattern, 0, 1)
        
        # Apply smoothstep for cleaner transitions
        pattern = self._smoothstep(0.0, 1.0, pattern)
        
        # Only affect visible pixels
        pattern = np.where(mask, pattern, 1.0)
        
        return pattern.astype(np.float32)
    
    def _generate_crack_pattern(self, h: int, w: int, mask: np.ndarray) -> np.ndarray:
        """Pre-generate anti-aliased crack pattern"""
        cracks = np.zeros((h, w), dtype=np.float32)
        
        if not self.cracks:
            return cracks
        
        crack_seed = self.config.seed + 100
        rng = np.random.RandomState(crack_seed)
        
        # Find starting points for cracks
        crack_count = int(np.sum(mask) * 0.002 * self.config.intensity)
        crack_count = max(3, min(crack_count, 20))
        
        mask_y, mask_x = np.where(mask)
        if len(mask_y) < crack_count:
            return cracks
        
        # Generate crack lines with anti-aliasing
        for _ in range(crack_count):
            idx = rng.randint(len(mask_y))
            start_x = float(mask_x[idx])
            start_y = float(mask_y[idx])
            
            # Random crack direction with vertical bias
            angle = rng.uniform(-0.4, 0.4) + np.pi / 2  # Mostly downward
            length = rng.uniform(3, 8)
            
            # Draw anti-aliased line using Wu's algorithm
            end_x = start_x + np.cos(angle) * length
            end_y = start_y + np.sin(angle) * length
            
            self._draw_aa_line(cracks, start_x, start_y, end_x, end_y, mask, h, w)
            
            # Branch with probability
            if rng.random() < 0.4:
                branch_angle = angle + rng.uniform(-0.8, 0.8)
                branch_len = length * 0.5
                branch_end_x = end_x + np.cos(branch_angle) * branch_len
                branch_end_y = end_y + np.sin(branch_angle) * branch_len
                self._draw_aa_line(cracks, end_x, end_y, branch_end_x, branch_end_y, mask, h, w)
        
        return cracks
    
    def _draw_aa_line(self, canvas: np.ndarray, x0: float, y0: float, 
                      x1: float, y1: float, mask: np.ndarray, h: int, w: int):
        """Draw anti-aliased line using Wu's algorithm"""
        steep = abs(y1 - y0) > abs(x1 - x0)
        
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
        
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        
        dx = x1 - x0
        dy = y1 - y0
        gradient = dy / dx if dx > 0.001 else 1.0
        
        # First endpoint
        xend = round(x0)
        yend = y0 + gradient * (xend - x0)
        xgap = 1.0 - ((x0 + 0.5) % 1.0)
        xpxl1 = int(xend)
        ypxl1 = int(yend)
        
        def plot(px: int, py: int, intensity: float):
            if steep:
                px, py = py, px
            if 0 <= px < w and 0 <= py < h and mask[py, px]:
                canvas[py, px] = max(canvas[py, px], intensity)
        
        plot(xpxl1, ypxl1, (1.0 - (yend % 1.0)) * xgap)
        plot(xpxl1, ypxl1 + 1, (yend % 1.0) * xgap)
        
        intery = yend + gradient
        
        # Second endpoint
        xend = round(x1)
        yend = y1 + gradient * (xend - x1)
        xgap = (x1 + 0.5) % 1.0
        xpxl2 = int(xend)
        ypxl2 = int(yend)
        
        plot(xpxl2, ypxl2, (1.0 - (yend % 1.0)) * xgap)
        plot(xpxl2, ypxl2 + 1, (yend % 1.0) * xgap)
        
        # Main loop
        for x in range(xpxl1 + 1, xpxl2):
            plot(x, int(intery), 1.0 - (intery % 1.0))
            plot(x, int(intery) + 1, intery % 1.0)
            intery += gradient
    
    def _create_petrify_frame(
        self,
        linear_premul: np.ndarray,
        stone_pattern: np.ndarray,
        crack_pattern: np.ndarray,
        petrify_progress: float,
        h: int,
        w: int
    ) -> np.ndarray:
        """Create petrify frame with proper linear color blending"""
        result = linear_premul.copy()
        alpha = result[:, :, 3]
        mask = alpha > 0.01
        
        # Calculate petrification amount per pixel (smooth transition)
        # Use smoothstep for anti-aliased edge
        edge_width = 0.15
        petrify_amount = self._smoothstep(
            petrify_progress - edge_width,
            petrify_progress,
            stone_pattern
        )
        petrify_amount = 1.0 - petrify_amount  # Invert so lower pattern values = more petrified
        petrify_amount = np.where(mask, petrify_amount, 0.0)
        
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
        
        # Desaturate toward grayscale
        stone_blend = 0.7 * self.config.intensity
        desaturated = np.zeros_like(linear_rgb)
        desaturated[:, :, 0] = luminance
        desaturated[:, :, 1] = luminance
        desaturated[:, :, 2] = luminance
        
        # Blend grayscale with stone color
        stone_colored = np.zeros_like(linear_rgb)
        for c in range(3):
            stone_colored[:, :, c] = (
                desaturated[:, :, c] * (1 - stone_blend) +
                self._stone_linear[c] * stone_blend
            )
        
        # Darken for stone appearance
        stone_colored *= 0.85
        
        # Blend original with stone based on petrify amount
        petrify_3d = petrify_amount[:, :, np.newaxis]
        blended_rgb = linear_rgb * (1 - petrify_3d) + stone_colored * petrify_3d
        
        # Add edge highlight (bright line at petrification front)
        edge_highlight = self._smoothstep(
            petrify_progress - 0.05, petrify_progress, stone_pattern
        ) * self._smoothstep(
            petrify_progress + 0.05, petrify_progress, stone_pattern
        )
        edge_highlight = edge_highlight * mask * 0.3
        blended_rgb += edge_highlight[:, :, np.newaxis]
        
        # Apply cracks (darken crack areas)
        crack_intensity = crack_pattern * petrify_amount * 0.7
        crack_3d = crack_intensity[:, :, np.newaxis]
        blended_rgb = blended_rgb * (1.0 - crack_3d)
        
        # Re-premultiply
        result[:, :, 0] = np.clip(blended_rgb[:, :, 0], 0, 1) * alpha
        result[:, :, 1] = np.clip(blended_rgb[:, :, 1], 0, 1) * alpha
        result[:, :, 2] = np.clip(blended_rgb[:, :, 2], 0, 1) * alpha
        
        return result
    
    def _apply_crumble(
        self,
        linear_premul: np.ndarray,
        crumble_t: float,
        h: int,
        w: int
    ) -> np.ndarray:
        """Apply physics-based crumble effect"""
        result = linear_premul.copy()
        alpha = result[:, :, 3]
        mask = alpha > 0.01
        
        if not np.any(mask):
            return result
        
        # Find center of mass
        mask_y, mask_x = np.where(mask)
        cx = np.mean(mask_x)
        cy = np.mean(mask_y)
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Distance from center (normalized)
        dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
        max_dist = np.max(dist[mask]) if np.any(mask) else 1
        norm_dist = dist / max_dist
        
        # Smooth crumble threshold (outer areas crumble first)
        crumble_threshold = 1.0 - crumble_t
        crumble_amount = self._smoothstep(
            crumble_threshold - 0.1,
            crumble_threshold + 0.1,
            norm_dist
        )
        crumble_amount = np.where(mask, crumble_amount, 0.0)
        
        # Fade alpha smoothly
        result[:, :, 3] = alpha * (1.0 - crumble_amount)
        
        # Update RGB for premultiplied alpha
        for c in range(3):
            safe_old_alpha = np.where(alpha > 1e-10, alpha, 1.0)
            unpremul = result[:, :, c] / safe_old_alpha
            result[:, :, c] = unpremul * result[:, :, 3]
        
        # Add dust particles
        if self.dust:
            result = self._add_dust_particles(result, mask, crumble_amount, crumble_t, h, w)
        
        return result
    
    def _add_dust_particles(
        self,
        linear_premul: np.ndarray,
        original_mask: np.ndarray,
        crumble_amount: np.ndarray,
        crumble_t: float,
        h: int,
        w: int
    ) -> np.ndarray:
        """Add anti-aliased dust particles"""
        result = linear_premul.copy()
        
        # Get crumbling pixels as dust sources
        crumbling = crumble_amount > 0.3
        if not np.any(crumbling):
            return result
        
        dust_count = int(np.sum(crumbling) * 0.15 * self.config.intensity)
        dust_count = max(5, min(dust_count, 50))
        
        crumble_y, crumble_x = np.where(crumbling)
        if len(crumble_y) < dust_count:
            dust_count = len(crumble_y)
        
        indices = self.rng.choice(len(crumble_y), dust_count, replace=False)
        
        # Physics: gravity + spread
        gravity = 15.0
        
        for idx in indices:
            src_y = float(crumble_y[idx])
            src_x = float(crumble_x[idx])
            
            # Each dust particle has random initial velocity
            vx = self.rng.uniform(-2, 2)
            vy = self.rng.uniform(-1, 2)  # Slight upward possible
            
            # Position with physics
            t_local = crumble_t * self.rng.uniform(0.5, 1.0)  # Stagger
            px = src_x + vx * t_local
            py = src_y + vy * t_local + 0.5 * gravity * t_local * t_local
            
            # Fade out over time
            life = 1.0 - crumble_t
            particle_alpha = life * 0.6
            
            # Particle size (Gaussian splat)
            radius = 1.5
            
            # Draw anti-aliased particle
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    px_i = int(px + 0.5) + dx
                    py_i = int(py + 0.5) + dy
                    
                    if 0 <= px_i < w and 0 <= py_i < h:
                        # Distance from particle center
                        dist_sq = (px_i - px) ** 2 + (py_i - py) ** 2
                        falloff = np.exp(-dist_sq / (2 * (radius * 0.5) ** 2))
                        
                        # Blend dust (premultiplied)
                        dust_a = particle_alpha * falloff
                        for c in range(3):
                            result[py_i, px_i, c] += self._stone_linear[c] * dust_a
                        result[py_i, px_i, 3] = max(result[py_i, px_i, 3], dust_a)
        
        return result
