"""
Enhanced Noise generators for procedural effects
Improved fallback implementation for smoother results
"""

import numpy as np
from typing import Optional

# Try to import noise library, fall back to enhanced Perlin
try:
    from noise import pnoise2, snoise2
    HAS_NOISE_LIB = True
except ImportError:
    HAS_NOISE_LIB = False


class NoiseGenerator:
    """Generates various types of noise for procedural effects - Enhanced version"""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.base = self.rng.integers(0, 10000)
        
        # Pre-generate permutation table for Perlin noise
        self._perm = self._generate_permutation()
        self._grad = self._generate_gradients()
    
    def _generate_permutation(self) -> np.ndarray:
        """Generate permutation table for noise"""
        perm = np.arange(256, dtype=np.int32)
        self.rng.shuffle(perm)
        return np.concatenate([perm, perm])
    
    def _generate_gradients(self) -> np.ndarray:
        """Generate gradient vectors"""
        angles = self.rng.random(256) * 2 * np.pi
        return np.column_stack([np.cos(angles), np.sin(angles)])
    
    def _fade(self, t: np.ndarray) -> np.ndarray:
        """Improved fade function for smoother noise"""
        # 6t^5 - 15t^4 + 10t^3 (Ken Perlin's improved version)
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def _gradient_dot(self, hash_val: int, x: float, y: float) -> float:
        """Calculate dot product with gradient"""
        g = self._grad[hash_val % 256]
        return g[0] * x + g[1] * y
    
    def perlin_2d(
        self,
        width: int,
        height: int,
        scale: float = 10.0,
        octaves: int = 1,
        time: float = 0.0,
        persistence: float = 0.5,
        lacunarity: float = 2.0
    ) -> np.ndarray:
        """Generate 2D Perlin noise with multiple octaves"""
        if HAS_NOISE_LIB:
            return self._perlin_noise_lib(width, height, scale, octaves, time)
        return self._perlin_enhanced(width, height, scale, octaves, time, persistence, lacunarity)
    
    def _perlin_noise_lib(
        self,
        width: int,
        height: int,
        scale: float,
        octaves: int,
        time: float
    ) -> np.ndarray:
        """Use the noise library for high quality Perlin noise - vectorized"""
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        x_scaled = x_coords / scale + time
        y_scaled = y_coords / scale
        
        # Vectorize pnoise2 call
        result = np.vectorize(lambda x, y: pnoise2(x, y, octaves=octaves, base=self.base))(x_scaled, y_scaled)
        return result
    
    def _perlin_enhanced(
        self,
        width: int,
        height: int,
        scale: float,
        octaves: int,
        time: float,
        persistence: float,
        lacunarity: float
    ) -> np.ndarray:
        """Enhanced Perlin noise implementation"""
        result = np.zeros((height, width))
        
        amplitude = 1.0
        max_amplitude = 0.0
        freq = 1.0
        
        for _ in range(octaves):
            noise = self._perlin_single_octave(width, height, scale / freq, time * freq)
            result += noise * amplitude
            max_amplitude += amplitude
            amplitude *= persistence
            freq *= lacunarity
        
        if max_amplitude > 0:
            result /= max_amplitude
        
        return result
    
    def _perlin_single_octave(
        self,
        width: int,
        height: int,
        scale: float,
        time: float
    ) -> np.ndarray:
        """Single octave Perlin noise - fully vectorized"""
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Scale and offset coordinates
        px = (x_coords / scale + time) % 256
        py = (y_coords / scale) % 256
        
        # Grid cell coordinates
        x0 = np.floor(px).astype(np.int32) & 255
        y0 = np.floor(py).astype(np.int32) & 255
        x1 = (x0 + 1) & 255
        y1 = (y0 + 1) & 255
        
        # Relative position in cell
        rx = px - np.floor(px)
        ry = py - np.floor(py)
        
        # Fade curves (vectorized)
        u = rx * rx * rx * (rx * (rx * 6 - 15) + 10)
        v = ry * ry * ry * (ry * (ry * 6 - 15) + 10)
        
        # Hash coordinates (vectorized lookup)
        aa = self._perm[self._perm[x0] + y0]
        ab = self._perm[self._perm[x0] + y1]
        ba = self._perm[self._perm[x1] + y0]
        bb = self._perm[self._perm[x1] + y1]
        
        # Gradient dots (vectorized)
        def grad_dot(hash_arr, dx, dy):
            g = self._grad[hash_arr % 256]
            return g[:, :, 0] * dx + g[:, :, 1] * dy
        
        g_aa = grad_dot(aa, rx, ry)
        g_ab = grad_dot(ab, rx, ry - 1)
        g_ba = grad_dot(ba, rx - 1, ry)
        g_bb = grad_dot(bb, rx - 1, ry - 1)
        
        # Interpolate (vectorized)
        x1_interp = g_aa + u * (g_ba - g_aa)
        x2_interp = g_ab + u * (g_bb - g_ab)
        result = x1_interp + v * (x2_interp - x1_interp)
        
        return result
    
    def simplex_2d(
        self,
        width: int,
        height: int,
        scale: float = 10.0,
        time: float = 0.0
    ) -> np.ndarray:
        """Generate 2D Simplex noise - vectorized"""
        if HAS_NOISE_LIB:
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            x_scaled = x_coords / scale + time
            y_scaled = y_coords / scale
            result = np.vectorize(lambda x, y: snoise2(x, y, base=self.base))(x_scaled, y_scaled)
            return result
        return self._perlin_single_octave(width, height, scale, time)
    
    def white_noise(self, width: int, height: int) -> np.ndarray:
        """Generate white noise"""
        return self.rng.random((height, width)) * 2 - 1
    
    def value_noise(
        self,
        width: int,
        height: int,
        scale: float = 10.0,
        time: float = 0.0
    ) -> np.ndarray:
        """Generate smooth value noise (simpler than Perlin) - vectorized"""
        # Create low-res random grid
        grid_w = int(np.ceil(width / scale)) + 2
        grid_h = int(np.ceil(height / scale)) + 2
        
        # Use consistent grid based on base seed and time
        rng = np.random.default_rng(self.base + int(time * 1000) % 10000)
        grid = rng.random((grid_h, grid_w))
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Grid position
        gx = (x_coords / scale + time) % (grid_w - 1)
        gy = y_coords / scale % (grid_h - 1)
        
        # Grid cell
        x0 = np.floor(gx).astype(np.int32) % (grid_w - 1)
        y0 = np.floor(gy).astype(np.int32) % (grid_h - 1)
        x1 = (x0 + 1) % grid_w
        y1 = (y0 + 1) % grid_h
        
        # Fractional parts with smooth interpolation
        fx = gx - np.floor(gx)
        fy = gy - np.floor(gy)
        
        # Apply smoothstep
        fx = fx * fx * (3 - 2 * fx)
        fy = fy * fy * (3 - 2 * fy)
        
        # Bilinear interpolation (vectorized)
        result = (
            grid[y0, x0] * (1 - fx) * (1 - fy) +
            grid[y0, x1] * fx * (1 - fy) +
            grid[y1, x0] * (1 - fx) * fy +
            grid[y1, x1] * fx * fy
        ) * 2 - 1
        
        return result
    
    def turbulence(
        self,
        width: int,
        height: int,
        scale: float = 10.0,
        octaves: int = 4,
        time: float = 0.0
    ) -> np.ndarray:
        """Generate turbulence (absolute value of multiple octaves)"""
        result = np.zeros((height, width))
        
        amplitude = 1.0
        max_amplitude = 0.0
        freq_scale = scale
        
        for _ in range(octaves):
            noise = self.perlin_2d(width, height, freq_scale, 1, time)
            result += np.abs(noise) * amplitude
            max_amplitude += amplitude
            amplitude *= 0.5
            freq_scale *= 0.5
        
        if max_amplitude > 0:
            result /= max_amplitude
        
        return result
    
    def ridged(
        self,
        width: int,
        height: int,
        scale: float = 10.0,
        octaves: int = 4,
        time: float = 0.0
    ) -> np.ndarray:
        """Generate ridged noise (inverted turbulence)"""
        turb = self.turbulence(width, height, scale, octaves, time)
        return 1.0 - turb
