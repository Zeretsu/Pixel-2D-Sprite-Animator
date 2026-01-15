"""
Enhanced Water Effect - Physics-based wave simulation
Professional-quality with Gerstner waves and gamma-correct rendering
"""

import numpy as np
from typing import List, Optional, Tuple
from .base import BaseEffect, EffectConfig, Easing, PixelMath
from .noise import NoiseGenerator
from ..core.parser import Sprite
from ..core.utils import MathUtils


class WaterEffect(BaseEffect):
    """Creates realistic water/wave animation with physics-based waves"""
    
    name = "water"
    description = "Animated water with physics-based waves, ripples, and reflections"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        self.noise = NoiseGenerator(self.config.seed)
        
        # Effect-specific settings
        self.wave_amplitude = self.config.extra.get('wave_amplitude', 2.0)
        self.wave_frequency = self.config.extra.get('wave_frequency', 1.0)
        self.wave_layers = self.config.extra.get('wave_layers', 2)
        self.ripple_intensity = self.config.extra.get('ripple_intensity', 0.3)
        self.caustics = self.config.extra.get('caustics', True)
        self.foam = self.config.extra.get('foam', False)
        self.quality = self.config.extra.get('quality', 'high')
        self.gerstner = self.config.extra.get('gerstner', True)  # Use Gerstner waves
        
        # Pre-calculated noise cache
        self._noise_cache = {}
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        """Apply water effect to sprite"""
        frames = []
        
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        
        # Get mask of water pixels
        water_mask = self._get_water_mask(original)
        water_depth = self._calculate_depth_map(water_mask)
        
        # Pre-calculate noise for all frames
        self._precalculate_noise(w, h)
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            
            frame_pixels = original.copy()
            
            # Apply layered wave distortion for depth
            frame_pixels = self._apply_layered_waves(frame_pixels, water_mask, water_depth, t)
            
            # Apply smooth color ripples
            frame_pixels = self._apply_smooth_ripples(frame_pixels, water_mask, water_depth, t)
            
            # Add caustic light patterns
            if self.caustics:
                frame_pixels = self._add_caustics(frame_pixels, water_mask, t)
            
            # Add moving highlight reflections
            frame_pixels = self._add_smooth_highlights(frame_pixels, water_mask, t)
            
            # Add foam at edges if enabled
            if self.foam:
                frame_pixels = self._add_foam(frame_pixels, water_mask, t)
            
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _precalculate_noise(self, w: int, h: int):
        """Pre-calculate noise patterns for smooth animation"""
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            self._noise_cache[i] = {
                'wave1': self.noise.perlin_2d(w, h, scale=4, octaves=2, time=t * self.wave_frequency),
                'wave2': self.noise.perlin_2d(w, h, scale=8, octaves=1, time=t * self.wave_frequency * 0.7),
                'ripple': self.noise.perlin_2d(w, h, scale=6, octaves=2, time=t * 2),
                'caustic': self.noise.turbulence(w, h, scale=5, octaves=3, time=t * 1.5),
            }
    
    def _get_water_mask(self, pixels: np.ndarray) -> np.ndarray:
        """Identify water-colored pixels with better detection and fallback"""
        r = pixels[:, :, 0].astype(float)
        g = pixels[:, :, 1].astype(float)
        b = pixels[:, :, 2].astype(float)
        
        # Visible pixels mask
        if pixels.shape[2] == 4:
            visible = pixels[:, :, 3] > 0
        else:
            visible = np.ones((pixels.shape[0], pixels.shape[1]), dtype=bool)
        
        # Water detection criteria:
        # 1. Blue-dominant pixels
        blue_dominant = b > r
        
        # 2. Coolness (more blue than red)
        coolness = (b - r) / 255
        
        # 3. Cyan-ish (blue and green both high, red lower)
        cyan_factor = (b + g - r * 2) / (2 * 255)
        
        # Combine criteria
        water_score = coolness * 0.6 + cyan_factor * 0.4
        
        water_mask = (water_score > 0.08) & visible
        
        # FALLBACK: If no water pixels found, apply to ALL visible pixels
        # This ensures the effect works on neutral/gray sprites too
        if not water_mask.any():
            return visible
        
        return water_mask
    
    def _calculate_depth_map(self, water_mask: np.ndarray) -> np.ndarray:
        """Calculate depth map based on distance from edges"""
        try:
            from scipy.ndimage import distance_transform_edt
            # Use distance transform for depth
            depth = distance_transform_edt(water_mask)
            
            # Normalize to 0-1
            if np.max(depth) > 0:
                depth = depth / np.max(depth)
            
            return depth
        except (ImportError, ModuleNotFoundError):
            # Fallback: simple vertical gradient
            h, w = water_mask.shape
            y_coords = np.arange(h).reshape(-1, 1)
            depth = np.tile(y_coords / h, (1, w))
            depth[~water_mask] = 0
            return depth
    
    def _apply_layered_waves(
        self,
        pixels: np.ndarray,
        water_mask: np.ndarray,
        depth_map: np.ndarray,
        t: float
    ) -> np.ndarray:
        """Apply multi-layer wave distortion using Gerstner waves for realism"""
        h, w = pixels.shape[:2]
        
        frame_idx = int(t * self.config.frame_count) % self.config.frame_count
        cache = self._noise_cache.get(frame_idx, {})
        
        wave1 = cache.get('wave1', self.noise.perlin_2d(w, h, scale=4, time=t))
        wave2 = cache.get('wave2', self.noise.perlin_2d(w, h, scale=8, time=t * 0.7))
        
        # Combine multiple wave layers
        combined_wave = wave1 * 0.7 + wave2 * 0.3
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Wave phase with integer frequencies for seamless looping
        tau = t * 2 * np.pi
        
        if self.gerstner:
            # Gerstner wave simulation - more realistic wave shape
            # Waves peak sharply and have flat troughs
            steepness = 0.4 * self.config.intensity
            
            # Layer 1: Primary wave
            k1 = 2 * np.pi / 8.0  # Wave number
            phase1 = k1 * y_coords - tau * self.wave_frequency
            gerstner_x1 = steepness * np.cos(phase1) * self.wave_amplitude
            
            # Layer 2: Secondary wave at different angle
            k2 = 2 * np.pi / 12.0
            phase2 = k2 * (y_coords * 0.7 + x_coords * 0.3) - tau * self.wave_frequency * 2
            gerstner_x2 = steepness * 0.5 * np.cos(phase2) * self.wave_amplitude
            
            # Layer 3: Detail wave
            k3 = 2 * np.pi / 5.0
            phase3 = k3 * y_coords - tau * self.wave_frequency * 3
            gerstner_x3 = steepness * 0.25 * np.cos(phase3) * self.wave_amplitude
            
            total_offset = gerstner_x1 + gerstner_x2 + gerstner_x3
        else:
            # Simple sine waves
            phase1 = tau * self.wave_frequency
            phase2 = tau * self.wave_frequency * 2
            
            primary_wave = np.sin(y_coords / 4.0 + phase1)
            secondary_wave = np.sin(y_coords / 6.0 + phase2) * 0.4
            
            total_offset = (primary_wave + secondary_wave) * self.wave_amplitude
        
        # Add noise modulation for natural variation
        total_offset += combined_wave * 0.5 * self.wave_amplitude
        
        # Scale by depth (deeper = more wave effect)
        wave_strength = depth_map * self.config.intensity
        offset = total_offset * wave_strength
        
        # Soft falloff at edges
        offset *= (wave_strength ** 0.5)
        
        # Source coordinates
        src_x = x_coords - offset
        src_y = y_coords.copy()
        
        # Use appropriate quality sampling
        if self.quality == 'best':
            result = PixelMath.lanczos_sample(pixels, src_x, src_y, gamma_correct=True)
        elif self.quality == 'high':
            result = PixelMath.bilinear_sample(pixels, src_x, src_y, gamma_correct=True)
        else:
            result = PixelMath.bilinear_sample(pixels, src_x, src_y, gamma_correct=False)
        
        # Restore non-water pixels
        result[~water_mask] = pixels[~water_mask]
        
        return result
    
    def _apply_smooth_ripples(
        self,
        pixels: np.ndarray,
        water_mask: np.ndarray,
        depth_map: np.ndarray,
        t: float
    ) -> np.ndarray:
        """Apply smooth color ripple effect"""
        result = pixels.copy()
        
        h, w = pixels.shape[:2]
        frame_idx = int(t * self.config.frame_count) % self.config.frame_count
        
        ripple = self._noise_cache.get(frame_idx, {}).get('ripple')
        if ripple is None:
            ripple = self.noise.perlin_2d(w, h, scale=6, octaves=2, time=t * 2)
        
        # Smooth the ripple intensity over time
        time_mod = MathUtils.smooth_wave(t, frequency=2.0)
        
        ripple_amount = ripple * self.ripple_intensity * (0.7 + 0.3 * time_mod)
        
        # Apply to blue and green channels
        for y in range(h):
            for x in range(w):
                if not water_mask[y, x]:
                    continue
                
                depth = depth_map[y, x]
                mod = ripple_amount[y, x] * (0.5 + depth * 0.5)
                
                # Shift blue channel
                b_val = result[y, x, 2].astype(float) + mod * 40
                result[y, x, 2] = int(np.clip(b_val, 0, 255))
                
                # Slight green shift for aqua tones
                g_val = result[y, x, 1].astype(float) + mod * 15
                result[y, x, 1] = int(np.clip(g_val, 0, 255))
        
        return result
    
    def _add_caustics(
        self,
        pixels: np.ndarray,
        water_mask: np.ndarray,
        t: float
    ) -> np.ndarray:
        """Add caustic light patterns (underwater light refractions)"""
        result = pixels.copy()
        
        h, w = pixels.shape[:2]
        frame_idx = int(t * self.config.frame_count) % self.config.frame_count
        
        caustic = self._noise_cache.get(frame_idx, {}).get('caustic')
        if caustic is None:
            caustic = self.noise.turbulence(w, h, scale=5, octaves=3, time=t * 1.5)
        
        # Threshold for caustic bright spots
        caustic_threshold = 0.6
        
        for y in range(h):
            for x in range(w):
                if not water_mask[y, x]:
                    continue
                
                c_val = caustic[y, x]
                
                if c_val > caustic_threshold:
                    # Intensity of caustic
                    intensity = (c_val - caustic_threshold) / (1 - caustic_threshold)
                    intensity = MathUtils.ease_out(intensity) * 0.3 * self.config.intensity
                    
                    # Brighten all channels slightly
                    for c in range(3):
                        val = result[y, x, c].astype(float)
                        val += intensity * 50
                        result[y, x, c] = int(np.clip(val, 0, 255))
        
        return result
    
    def _add_smooth_highlights(
        self,
        pixels: np.ndarray,
        water_mask: np.ndarray,
        t: float
    ) -> np.ndarray:
        """Add moving highlight reflections with smooth animation"""
        result = pixels.copy()
        
        h, w = pixels.shape[:2]
        
        # Multiple highlight waves for natural look
        for wave_idx in range(2):
            freq = 1.5 + wave_idx * 0.5
            offset = wave_idx * np.pi / 3
            
            for y in range(h):
                for x in range(w):
                    if not water_mask[y, x]:
                        continue
                    
                    # Moving highlight based on position and time
                    highlight = np.sin((x + y) / (3 + wave_idx) + t * 2 * np.pi * freq + offset)
                    
                    # Only show highlights at peaks
                    if highlight > 0.75:
                        # Smooth the highlight intensity
                        intensity = MathUtils.ease_in_out((highlight - 0.75) / 0.25) * 0.35
                        
                        # Add white-ish highlight
                        for c in range(3):
                            val = result[y, x, c].astype(float)
                            val += 40 * intensity
                            result[y, x, c] = int(np.clip(val, 0, 255))
        
        return result
    
    def _add_foam(
        self,
        pixels: np.ndarray,
        water_mask: np.ndarray,
        t: float
    ) -> np.ndarray:
        """Add foam effect at water edges"""
        result = pixels.copy()
        
        h, w = pixels.shape[:2]
        
        # Find edge pixels
        edge_mask = self._find_edges(water_mask)
        
        # Animated foam
        phase = t * 2 * np.pi
        
        for y in range(h):
            for x in range(w):
                if not edge_mask[y, x]:
                    continue
                
                # Foam visibility oscillates
                foam_vis = (np.sin(x / 2 + phase) + 1) / 2
                foam_vis *= (np.sin(y / 3 + phase * 0.7) + 1) / 2
                
                if foam_vis > 0.5:
                    intensity = (foam_vis - 0.5) * 2 * 0.6
                    
                    # Add white foam color
                    for c in range(3):
                        val = result[y, x, c].astype(float)
                        val = val * (1 - intensity) + 255 * intensity
                        result[y, x, c] = int(np.clip(val, 0, 255))
        
        return result
    
    def _find_edges(self, mask: np.ndarray) -> np.ndarray:
        """Find edge pixels of a mask"""
        h, w = mask.shape
        edges = np.zeros_like(mask, dtype=bool)
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if mask[y, x]:
                    # Check 4-connected neighbors
                    if not (mask[y-1, x] and mask[y+1, x] and 
                            mask[y, x-1] and mask[y, x+1]):
                        edges[y, x] = True
        
        return edges
