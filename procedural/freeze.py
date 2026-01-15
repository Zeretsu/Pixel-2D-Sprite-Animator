"""
Freeze Effect - Ice crystallization spreading over sprite
Frozen status effect animation with pixel-perfect rendering
"""

import numpy as np
from typing import List, Optional
from .base import BaseEffect, EffectConfig, Easing, PixelMath
from ..core.parser import Sprite


class FreezeEffect(BaseEffect):
    """Creates ice freeze effect with crystallization spread"""
    
    name = "freeze"
    description = "Ice crystallization spreading effect"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        
        self.ice_color = np.array(self.config.extra.get('ice_color', (180, 220, 255)), dtype=np.float64)
        self.frost_color = np.array(self.config.extra.get('frost_color', (220, 240, 255)), dtype=np.float64)
        self.mode = self.config.extra.get('mode', 'spread')
        self.sparkle = self.config.extra.get('sparkle', True)
        self.crack_lines = self.config.extra.get('cracks', True)
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        
        # Pre-calculate ice spread pattern with smooth noise
        ice_pattern = self._generate_ice_pattern(original, h, w)
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            
            if self.mode == 'spread':
                freeze_progress = Easing.ease_out_quad(t)
            elif self.mode == 'shatter':
                freeze_progress = Easing.ease_out_quad(min(t / 0.6, 1.0))
            else:
                freeze_progress = 1.0
            
            frame_pixels = self._create_freeze_frame(
                original, ice_pattern, freeze_progress, t, h, w
            )
            
            if self.mode == 'shatter' and t > 0.6:
                shatter_t = (t - 0.6) / 0.4
                frame_pixels = self._apply_shatter(frame_pixels, original, shatter_t, h, w)
            
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _generate_ice_pattern(self, pixels: np.ndarray, h: int, w: int) -> np.ndarray:
        """Generate smooth crystallization spread pattern"""
        channels = pixels.shape[2]
        
        if channels == 4:
            mask = pixels[:, :, 3] > 0
        else:
            mask = np.any(pixels > 0, axis=2)
        
        # Distance from top (ice spreads from top down)
        y_coords = np.arange(h, dtype=np.float64).reshape(-1, 1)
        pattern = y_coords / max(h - 1, 1)
        
        # Add smooth Perlin-like noise for organic spread
        # Using multiple octaves of sine waves
        x_coords = np.arange(w, dtype=np.float64).reshape(1, -1)
        noise = np.zeros((h, w), dtype=np.float64)
        
        for octave in range(3):
            freq = 2 ** octave
            amp = 0.15 / (octave + 1)
            noise += amp * np.sin(y_coords * freq * 0.5 + self.rng.uniform(0, 6.28))
            noise += amp * np.sin(x_coords * freq * 0.3 + self.rng.uniform(0, 6.28))
        
        pattern = np.broadcast_to(pattern, (h, w)) + noise
        
        # Only apply to visible pixels
        pattern = np.where(mask, pattern, 1.0)
        
        return np.clip(pattern, 0, 1)
    
    def _create_freeze_frame(
        self,
        original: np.ndarray,
        ice_pattern: np.ndarray,
        freeze_progress: float,
        t: float,
        h: int,
        w: int
    ) -> np.ndarray:
        """Create a single freeze frame with proper color blending"""
        channels = original.shape[2]
        
        if channels == 4:
            mask = original[:, :, 3] > 0
            alpha = original[:, :, 3].astype(np.float64) / 255.0
        else:
            mask = np.any(original > 0, axis=2)
            alpha = mask.astype(np.float64)
        
        # Work in linear color space
        rgb_linear = np.power(original[:, :, :3].astype(np.float64) / 255.0, PixelMath.GAMMA)
        ice_linear = np.power(self.ice_color / 255.0, PixelMath.GAMMA)
        frost_linear = np.power(self.frost_color / 255.0, PixelMath.GAMMA)
        
        # Determine frozen regions with smooth transitions
        freeze_threshold = freeze_progress
        
        # Smooth transition zone
        transition_width = 0.15
        frozen_amount = np.clip((freeze_threshold - ice_pattern) / transition_width + 0.5, 0, 1)
        frozen_amount = frozen_amount * frozen_amount * (3 - 2 * frozen_amount)  # Smoothstep
        
        # Edge zone (frost effect at boundary)
        edge_amount = np.exp(-((ice_pattern - freeze_threshold) ** 2) / 0.01)
        edge_amount *= mask
        
        # Apply ice effect to frozen pixels
        # 1. Desaturate (convert to luminance)
        luminance = 0.299 * rgb_linear[:, :, 0] + 0.587 * rgb_linear[:, :, 1] + 0.114 * rgb_linear[:, :, 2]
        
        # 2. Blend toward ice color
        ice_blend = 0.6 * self.config.intensity
        result = np.zeros_like(rgb_linear)
        
        for c in range(3):
            # Frozen: desaturated + ice tinted + brightened
            frozen_color = luminance * (1 - ice_blend) + ice_linear[c] * ice_blend
            frozen_color *= 1.2  # Brighten for icy look
            
            # Edge: frost highlight
            frost_color = rgb_linear[:, :, c] * 0.5 + frost_linear[c] * 0.5
            
            # Blend based on frozen amount
            result[:, :, c] = (
                rgb_linear[:, :, c] * (1 - frozen_amount) +
                frozen_color * frozen_amount +
                frost_color * edge_amount * 0.5
            )
        
        # Add sparkles
        if self.sparkle and freeze_progress > 0.3:
            result = self._add_sparkles(result, mask, frozen_amount, t, h, w)
        
        # Convert back to sRGB
        result = np.clip(result, 0, 1)
        rgb_srgb = np.power(result, PixelMath.INV_GAMMA) * 255.0
        
        output = np.zeros((h, w, channels), dtype=np.uint8)
        output[:, :, :3] = np.clip(rgb_srgb, 0, 255).astype(np.uint8)
        if channels == 4:
            output[:, :, 3] = original[:, :, 3]
        
        return output
    
    def _add_sparkles(
        self,
        rgb_linear: np.ndarray,
        mask: np.ndarray,
        frozen_amount: np.ndarray,
        t: float,
        h: int,
        w: int
    ) -> np.ndarray:
        """Add twinkling ice sparkles using Gaussian spots"""
        result = rgb_linear.copy()
        
        # Generate consistent sparkle positions based on time
        sparkle_seed = int(t * 8) + self.config.seed
        rng = np.random.RandomState(sparkle_seed)
        
        frozen_mask = (frozen_amount > 0.5) & mask
        sparkle_count = int(np.sum(frozen_mask) * 0.015 * self.config.intensity)
        
        if sparkle_count < 1:
            return result
        
        frozen_y, frozen_x = np.where(frozen_mask)
        if len(frozen_y) < sparkle_count:
            sparkle_count = len(frozen_y)
        
        indices = rng.choice(len(frozen_y), sparkle_count, replace=False)
        
        # Create coordinate grids for distance calculation
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float64)
        
        for idx in indices:
            sy, sx = frozen_y[idx], frozen_x[idx]
            
            # Sparkle intensity varies with time phase
            phase = rng.uniform(0, 2 * np.pi)
            intensity = 0.5 + 0.5 * np.sin(t * 20 + phase)
            intensity *= rng.uniform(0.5, 1.0) * self.config.intensity
            
            # Gaussian sparkle (anti-aliased)
            dist_sq = (x_coords - sx) ** 2 + (y_coords - sy) ** 2
            sparkle = np.exp(-dist_sq / 1.5) * intensity
            
            # Add white sparkle
            for c in range(3):
                result[:, :, c] = np.minimum(1.0, result[:, :, c] + sparkle)
        
        return result
    
    def _apply_shatter(
        self,
        pixels: np.ndarray,
        original: np.ndarray,
        shatter_t: float,
        h: int,
        w: int
    ) -> np.ndarray:
        """Apply shattering effect with proper fragment physics"""
        channels = pixels.shape[2]
        result = np.zeros_like(pixels)
        
        if channels == 4:
            mask = original[:, :, 3] > 0
        else:
            mask = np.any(original > 0, axis=2)
        
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float64)
        
        # Find center
        if np.any(mask):
            mask_y, mask_x = np.where(mask)
            cx, cy = np.mean(mask_x), np.mean(mask_y)
        else:
            cx, cy = w / 2.0, h / 2.0
        
        # Create fragment grid
        frag_size = 6
        frag_x_id = (x_coords / frag_size).astype(np.int32)
        frag_y_id = (y_coords / frag_size).astype(np.int32)
        fragment_id = frag_x_id + frag_y_id * ((w // frag_size) + 1)
        
        # Each fragment has outward velocity + gravity
        unique_frags = np.unique(fragment_id[mask])
        
        for frag_id in unique_frags:
            frag_mask = (fragment_id == frag_id) & mask
            if not np.any(frag_mask):
                continue
            
            # Fragment center
            frag_y, frag_x = np.where(frag_mask)
            fcx, fcy = np.mean(frag_x), np.mean(frag_y)
            
            # Velocity away from center + random component
            rng = np.random.RandomState(self.config.seed + int(frag_id))
            vx = (fcx - cx) * 0.1 + rng.uniform(-2, 2)
            vy = (fcy - cy) * 0.1 + rng.uniform(-3, 0)
            
            # Physics: position = initial + velocity*t + 0.5*gravity*t^2
            gravity = 15.0
            offset_x = vx * shatter_t * self.config.intensity
            offset_y = vy * shatter_t + 0.5 * gravity * shatter_t ** 2
            
            # Rotation
            rotation = rng.uniform(-1, 1) * shatter_t * 2
            
            # Source coordinates for this fragment
            for fy, fx in zip(frag_y, frag_x):
                # Apply rotation around fragment center
                dx, dy = fx - fcx, fy - fcy
                cos_r, sin_r = np.cos(rotation), np.sin(rotation)
                rx = dx * cos_r - dy * sin_r + fcx
                ry = dx * sin_r + dy * cos_r + fcy
                
                # Final position
                dest_x = int(rx + offset_x)
                dest_y = int(ry + offset_y)
                
                if 0 <= dest_x < w and 0 <= dest_y < h:
                    # Fade out
                    fade = 1.0 - shatter_t
                    result[dest_y, dest_x, :3] = (pixels[fy, fx, :3] * fade).astype(np.uint8)
                    if channels == 4:
                        result[dest_y, dest_x, 3] = int(pixels[fy, fx, 3] * fade)
        
        return result
