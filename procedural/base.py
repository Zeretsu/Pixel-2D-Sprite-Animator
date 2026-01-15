"""
Base Effect - Abstract base class for all procedural effects
Enhanced with advanced pixel math, gamma-correct interpolation, and smooth animation utilities
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from ..core.parser import Sprite


@dataclass
class EffectConfig:
    """Configuration for an effect"""
    frame_count: int = 8
    speed: float = 1.0
    intensity: float = 1.0
    seed: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class PixelMath:
    """
    Advanced pixel manipulation utilities for artifact-free animations.
    Includes gamma-correct interpolation, Lanczos resampling, and proper alpha handling.
    """
    
    # sRGB gamma correction constants
    GAMMA = 2.2
    INV_GAMMA = 1.0 / 2.2
    
    @staticmethod
    def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
        """Convert sRGB values (0-255) to linear color space for correct interpolation"""
        normalized = srgb.astype(np.float32) / 255.0
        # Simplified sRGB to linear (accurate approximation)
        return np.power(normalized, PixelMath.GAMMA)
    
    @staticmethod
    def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
        """Convert linear values back to sRGB (0-255)"""
        # Clamp to avoid negative values from interpolation overshoot
        linear = np.clip(linear, 0, 1)
        srgb = np.power(linear, PixelMath.INV_GAMMA)
        return (srgb * 255).astype(np.uint8)
    
    @staticmethod
    def premultiply_alpha(pixels: np.ndarray) -> np.ndarray:
        """Convert to premultiplied alpha for correct blending"""
        if pixels.shape[2] != 4:
            return pixels.astype(np.float32)
        
        result = pixels.astype(np.float32).copy()
        alpha = result[:, :, 3:4] / 255.0
        result[:, :, :3] *= alpha
        return result
    
    @staticmethod
    def unpremultiply_alpha(pixels: np.ndarray) -> np.ndarray:
        """Convert from premultiplied alpha back to straight alpha"""
        if pixels.shape[2] != 4:
            return np.clip(pixels, 0, 255).astype(np.uint8)
        
        result = pixels.copy()
        alpha = result[:, :, 3:4]
        # Avoid division by zero
        mask = alpha > 0.001
        result[:, :, :3] = np.where(mask, result[:, :, :3] / (alpha / 255.0 + 0.001), 0)
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def lanczos_kernel(x: np.ndarray, a: int = 3) -> np.ndarray:
        """
        Lanczos kernel for high-quality resampling.
        a = 3 is a good balance between quality and performance.
        """
        x = np.abs(x)
        result = np.zeros_like(x)
        
        # For x == 0, sinc(0) = 1
        zero_mask = x < 1e-8
        result[zero_mask] = 1.0
        
        # For 0 < x < a
        valid = (~zero_mask) & (x < a)
        px = np.pi * x[valid]
        result[valid] = (a * np.sin(px) * np.sin(px / a)) / (px * px)
        
        return result
    
    @staticmethod
    def lanczos_sample(
        pixels: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        a: int = 3,
        gamma_correct: bool = True
    ) -> np.ndarray:
        """
        Highest quality resampling using Lanczos filter.
        Slower but produces sharper results with minimal ringing.
        """
        h, w = pixels.shape[:2]
        channels = pixels.shape[2]
        
        # Convert to linear color space for correct interpolation
        if gamma_correct and channels >= 3:
            linear = PixelMath.srgb_to_linear(pixels[:, :, :3])
            work_pixels = np.dstack([linear, pixels[:, :, 3:4].astype(np.float32) / 255.0]) if channels == 4 else linear
        else:
            work_pixels = pixels.astype(np.float32) / 255.0
        
        # Premultiply alpha
        if channels == 4:
            alpha = work_pixels[:, :, 3:4]
            work_pixels[:, :, :3] *= alpha
        
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        
        result = np.zeros((*x.shape, channels), dtype=np.float32)
        weight_sum = np.zeros(x.shape, dtype=np.float32)
        
        # Sample (2a)x(2a) neighborhood
        for j in range(-a + 1, a + 1):
            for i in range(-a + 1, a + 1):
                # Lanczos weights
                wx = PixelMath.lanczos_kernel(x - (x0 + i), a)
                wy = PixelMath.lanczos_kernel(y - (y0 + j), a)
                weight = wx * wy
                
                # Sample coordinates (clamped)
                sx = np.clip(x0 + i, 0, w - 1)
                sy = np.clip(y0 + j, 0, h - 1)
                
                # Accumulate
                result += work_pixels[sy, sx] * weight[:, :, np.newaxis]
                weight_sum += weight
        
        # Normalize
        weight_sum = np.maximum(weight_sum, 1e-8)[:, :, np.newaxis]
        result /= weight_sum
        
        # Unpremultiply alpha
        if channels == 4:
            alpha_out = result[:, :, 3:4]
            alpha_safe = np.maximum(alpha_out, 1e-8)
            result[:, :, :3] /= alpha_safe
            result[:, :, 3] = np.clip(result[:, :, 3], 0, 1)
        
        # Convert back to sRGB
        if gamma_correct and channels >= 3:
            result[:, :, :3] = PixelMath.linear_to_srgb(np.clip(result[:, :, :3], 0, 1)).astype(np.float32) / 255.0
        
        # Handle out-of-bounds
        out_of_bounds = (x < 0) | (x >= w) | (y < 0) | (y >= h)
        if channels == 4:
            result[out_of_bounds, 3] = 0
        else:
            result[out_of_bounds] = 0
        
        return np.clip(result * 255, 0, 255).astype(np.uint8)
    
    @staticmethod
    def bilinear_sample(
        pixels: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        default: Optional[np.ndarray] = None,
        gamma_correct: bool = False
    ) -> np.ndarray:
        """
        High-quality bilinear interpolation with optional gamma correction.
        Properly handles alpha channel and out-of-bounds coordinates.
        """
        h, w = pixels.shape[:2]
        channels = pixels.shape[2]
        
        # Compute integer and fractional parts
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        
        # Fractional parts for interpolation weights (use smoothstep for better quality)
        fx = (x - x0).astype(np.float32)
        fy = (y - y0).astype(np.float32)
        
        # Optional: Use smoothstep for smoother interpolation
        # fx = fx * fx * (3 - 2 * fx)
        # fy = fy * fy * (3 - 2 * fy)
        
        # Clamp coordinates for sampling
        x0c = np.clip(x0, 0, w - 1)
        x1c = np.clip(x1, 0, w - 1)
        y0c = np.clip(y0, 0, h - 1)
        y1c = np.clip(y1, 0, h - 1)
        
        # Sample four corners
        p00 = pixels[y0c, x0c].astype(np.float32)
        p01 = pixels[y0c, x1c].astype(np.float32)
        p10 = pixels[y1c, x0c].astype(np.float32)
        p11 = pixels[y1c, x1c].astype(np.float32)
        
        # Gamma correct if requested (important for smooth color gradients)
        if gamma_correct and channels >= 3:
            p00[:, :, :3] = np.power(p00[:, :, :3] / 255.0, PixelMath.GAMMA) * 255.0
            p01[:, :, :3] = np.power(p01[:, :, :3] / 255.0, PixelMath.GAMMA) * 255.0
            p10[:, :, :3] = np.power(p10[:, :, :3] / 255.0, PixelMath.GAMMA) * 255.0
            p11[:, :, :3] = np.power(p11[:, :, :3] / 255.0, PixelMath.GAMMA) * 255.0
        
        # Premultiply alpha for correct blending
        if channels == 4:
            a00 = p00[:, :, 3:4] / 255.0
            a01 = p01[:, :, 3:4] / 255.0
            a10 = p10[:, :, 3:4] / 255.0
            a11 = p11[:, :, 3:4] / 255.0
            p00[:, :, :3] *= a00
            p01[:, :, :3] *= a01
            p10[:, :, :3] *= a10
            p11[:, :, :3] *= a11
        
        # Compute bilinear weights
        w00 = ((1.0 - fx) * (1.0 - fy))[:, :, np.newaxis]
        w01 = (fx * (1.0 - fy))[:, :, np.newaxis]
        w10 = ((1.0 - fx) * fy)[:, :, np.newaxis]
        w11 = (fx * fy)[:, :, np.newaxis]
        
        # Bilinear interpolation
        result = (p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11)
        
        # Unpremultiply alpha
        if channels == 4:
            alpha_result = result[:, :, 3:4]
            alpha_safe = np.maximum(alpha_result, 0.001)
            result[:, :, :3] = np.where(alpha_result > 0.001, 
                                         result[:, :, :3] / (alpha_safe / 255.0), 
                                         0)
        
        # Gamma uncorrect
        if gamma_correct and channels >= 3:
            result[:, :, :3] = np.power(np.clip(result[:, :, :3] / 255.0, 0, 1), 
                                        PixelMath.INV_GAMMA) * 255.0
        
        # Handle out-of-bounds by setting alpha to 0
        out_of_bounds = (x < 0) | (x >= w) | (y < 0) | (y >= h)
        if channels == 4:
            result[out_of_bounds, 3] = 0
        elif default is not None:
            result[out_of_bounds] = default
        else:
            result[out_of_bounds] = 0
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def edge_aware_sample(
        pixels: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        edge_threshold: float = 0.3
    ) -> np.ndarray:
        """
        Edge-aware sampling that preserves sharp edges while smoothing gradients.
        Uses bilateral-style weighting based on color similarity.
        """
        h, w = pixels.shape[:2]
        channels = pixels.shape[2]
        
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        fx = (x - x0).astype(np.float32)
        fy = (y - y0).astype(np.float32)
        
        # Clamp coordinates
        x0c = np.clip(x0, 0, w - 1)
        x1c = np.clip(x0 + 1, 0, w - 1)
        y0c = np.clip(y0, 0, h - 1)
        y1c = np.clip(y0 + 1, 0, h - 1)
        
        # Sample four corners
        p00 = pixels[y0c, x0c].astype(np.float32)
        p01 = pixels[y0c, x1c].astype(np.float32)
        p10 = pixels[y1c, x0c].astype(np.float32)
        p11 = pixels[y1c, x1c].astype(np.float32)
        
        # Compute color differences to detect edges
        def color_diff(a, b):
            if channels == 4:
                # Weight by alpha
                diff = np.abs(a[:, :, :3] - b[:, :, :3]).mean(axis=2)
                alpha_diff = np.abs(a[:, :, 3] - b[:, :, 3])
                return diff / 255.0 + alpha_diff / 255.0 * 0.5
            return np.abs(a - b).mean(axis=2) / 255.0
        
        # Detect if we're crossing an edge
        h_edge = np.maximum(color_diff(p00, p01), color_diff(p10, p11))
        v_edge = np.maximum(color_diff(p00, p10), color_diff(p01, p11))
        
        # Use nearest neighbor near edges, bilinear in smooth areas
        is_edge = (h_edge > edge_threshold) | (v_edge > edge_threshold)
        
        # Bilinear result
        w00 = ((1.0 - fx) * (1.0 - fy))[:, :, np.newaxis]
        w01 = (fx * (1.0 - fy))[:, :, np.newaxis]
        w10 = ((1.0 - fx) * fy)[:, :, np.newaxis]
        w11 = (fx * fy)[:, :, np.newaxis]
        bilinear = p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11
        
        # Nearest neighbor result
        nearest_x = np.where(fx < 0.5, x0c, x1c)
        nearest_y = np.where(fy < 0.5, y0c, y1c)
        nearest = pixels[nearest_y, nearest_x].astype(np.float32)
        
        # Blend based on edge detection
        result = np.where(is_edge[:, :, np.newaxis], nearest, bilinear)
        
        # Handle out-of-bounds
        out_of_bounds = (x < 0) | (x >= w) | (y < 0) | (y >= h)
        if channels == 4:
            result[out_of_bounds, 3] = 0
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def smooth_displacement(
        pixels: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
        mask: Optional[np.ndarray] = None,
        quality: str = 'high'
    ) -> np.ndarray:
        """
        Apply smooth displacement with selectable quality.
        quality: 'fast' (bilinear), 'high' (gamma-correct bilinear), 'best' (Lanczos)
        """
        h, w = pixels.shape[:2]
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Source coordinates (inverse mapping)
        src_x = x_coords - dx
        src_y = y_coords - dy
        
        # Sample with selected quality
        if quality == 'best':
            result = PixelMath.lanczos_sample(pixels, src_x, src_y)
        elif quality == 'high':
            result = PixelMath.bilinear_sample(pixels, src_x, src_y, gamma_correct=True)
        else:
            result = PixelMath.bilinear_sample(pixels, src_x, src_y, gamma_correct=False)
        
        # Apply mask if provided
        if mask is not None:
            result = np.where(mask[:, :, np.newaxis], result, pixels)
        
        return result
    
    @staticmethod
    def temporal_blend(
        frames: List[np.ndarray],
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Blend multiple frames for temporal anti-aliasing / motion blur effect.
        """
        if not frames:
            raise ValueError("No frames to blend")
        
        if weights is None:
            weights = [1.0 / len(frames)] * len(frames)
        
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
        
        result = np.zeros_like(frames[0], dtype=np.float32)
        for frame, weight in zip(frames, weights):
            result += frame.astype(np.float32) * weight
        
        return np.clip(result, 0, 255).astype(np.uint8)


class Easing:
    """Collection of easing functions for smooth animations"""
    
    @staticmethod
    def linear(t: float) -> float:
        return t
    
    @staticmethod
    def ease_in_quad(t: float) -> float:
        return t * t
    
    @staticmethod
    def ease_out_quad(t: float) -> float:
        return 1 - (1 - t) * (1 - t)
    
    @staticmethod
    def ease_in_out_quad(t: float) -> float:
        return 2 * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 2) / 2
    
    @staticmethod
    def ease_in_cubic(t: float) -> float:
        return t * t * t
    
    @staticmethod
    def ease_out_cubic(t: float) -> float:
        return 1 - pow(1 - t, 3)
    
    @staticmethod
    def ease_in_out_cubic(t: float) -> float:
        return 4 * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2
    
    @staticmethod
    def ease_in_sine(t: float) -> float:
        return 1 - np.cos((t * np.pi) / 2)
    
    @staticmethod
    def ease_out_sine(t: float) -> float:
        return np.sin((t * np.pi) / 2)
    
    @staticmethod
    def ease_in_out_sine(t: float) -> float:
        return -(np.cos(np.pi * t) - 1) / 2
    
    @staticmethod
    def ease_in_elastic(t: float) -> float:
        if t == 0 or t == 1:
            return t
        return -pow(2, 10 * t - 10) * np.sin((t * 10 - 10.75) * (2 * np.pi / 3))
    
    @staticmethod
    def ease_out_elastic(t: float) -> float:
        if t == 0 or t == 1:
            return t
        return pow(2, -10 * t) * np.sin((t * 10 - 0.75) * (2 * np.pi / 3)) + 1
    
    @staticmethod
    def ease_out_bounce(t: float) -> float:
        n1, d1 = 7.5625, 2.75
        if t < 1 / d1:
            return n1 * t * t
        elif t < 2 / d1:
            t -= 1.5 / d1
            return n1 * t * t + 0.75
        elif t < 2.5 / d1:
            t -= 2.25 / d1
            return n1 * t * t + 0.9375
        else:
            t -= 2.625 / d1
            return n1 * t * t + 0.984375
    
    @staticmethod
    def smooth_step(t: float) -> float:
        """Hermite interpolation (smoothstep)"""
        return t * t * (3 - 2 * t)
    
    @staticmethod
    def smoother_step(t: float) -> float:
        """Ken Perlin's improved smoothstep"""
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    @staticmethod
    def sin_wave(t: float, frequency: float = 1.0) -> float:
        """Smooth sine wave oscillation (-1 to 1)"""
        return np.sin(t * frequency * 2 * np.pi)
    
    @staticmethod
    def cos_wave(t: float, frequency: float = 1.0) -> float:
        """Smooth cosine wave oscillation (-1 to 1)"""
        return np.cos(t * frequency * 2 * np.pi)
    
    @staticmethod
    def loop_sin(t: float, frequency: float = 1.0) -> float:
        """
        Seamless looping sine wave.
        Guaranteed to start and end at the same value.
        """
        return np.sin(t * frequency * 2 * np.pi)
    
    @staticmethod
    def loop_cos(t: float, frequency: float = 1.0) -> float:
        """
        Seamless looping cosine wave.
        Guaranteed to start and end at the same value.
        """
        return np.cos(t * frequency * 2 * np.pi)
    
    @staticmethod
    def breathing(t: float) -> float:
        """Natural breathing rhythm (0 to 1, smooth loop)"""
        # Use sine that starts at 0, peaks at 0.5, returns to 0 at 1.0
        return (1 - np.cos(t * 2 * np.pi)) / 2
    
    @staticmethod
    def heartbeat(t: float) -> float:
        """
        Heartbeat-style pulse with two peaks.
        Mimics natural cardiac rhythm.
        """
        # Two quick pulses followed by rest
        t_mod = t % 1.0
        if t_mod < 0.15:
            return Easing.smooth_step(t_mod / 0.15)
        elif t_mod < 0.3:
            return 1 - Easing.smooth_step((t_mod - 0.15) / 0.15) * 0.5
        elif t_mod < 0.35:
            return 0.5 + Easing.smooth_step((t_mod - 0.3) / 0.05) * 0.3
        elif t_mod < 0.5:
            return 0.8 - Easing.smooth_step((t_mod - 0.35) / 0.15) * 0.8
        return 0.0
    
    @staticmethod
    def flicker(t: float, rng: np.random.Generator, smoothness: float = 0.7) -> float:
        """Natural flickering (like fire/candles) with temporal coherence"""
        # Multiple sine waves for organic flickering
        base = (np.sin(t * 7 * 2 * np.pi) * 0.3 + 
                np.sin(t * 13 * 2 * np.pi) * 0.2 + 
                np.sin(t * 23 * 2 * np.pi) * 0.1)
        noise = rng.random() * (1 - smoothness)
        return 0.6 + base * 0.4 + noise
    
    @staticmethod
    def wobble(t: float, frequency: float = 1.0, damping: float = 0.0) -> float:
        """
        Elastic wobble animation.
        Optional damping for settling animations.
        """
        decay = np.exp(-damping * t) if damping > 0 else 1.0
        return np.sin(t * frequency * 2 * np.pi) * decay
    
    @staticmethod
    def multi_frequency(t: float, frequencies: list = None, weights: list = None) -> float:
        """
        Combine multiple sine waves for organic motion.
        Ensures seamless looping by using integer frequency ratios.
        """
        if frequencies is None:
            frequencies = [1.0, 2.0, 3.0]
        if weights is None:
            weights = [0.6, 0.3, 0.1]
        
        total = 0.0
        weight_sum = sum(weights)
        for freq, weight in zip(frequencies, weights):
            total += np.sin(t * freq * 2 * np.pi) * weight
        
        return total / weight_sum if weight_sum > 0 else 0.0
    
    @staticmethod
    def pendulum(t: float, damping: float = 0.0) -> float:
        """
        Pendulum-style swing motion with optional damping.
        More physically accurate than simple sine wave.
        """
        decay = np.exp(-damping * t * 3) if damping > 0 else 1.0
        # Slight asymmetry for more natural feel
        return np.sin(t * 2 * np.pi) * decay * (1 + 0.1 * np.sin(t * 4 * np.pi))
    
    @staticmethod
    def spring(t: float, stiffness: float = 15.0, damping: float = 0.5) -> float:
        """
        Spring physics simulation for elastic motion.
        Higher stiffness = faster oscillation, higher damping = faster settling.
        """
        omega = np.sqrt(stiffness)
        decay = np.exp(-damping * t * 5)
        return np.sin(t * omega * 2 * np.pi) * decay
    
    @staticmethod
    def bounce_decay(t: float, bounces: int = 3) -> float:
        """
        Multiple bounces with decreasing height.
        Physically accurate bounce timing.
        """
        # Each bounce takes less time (sqrt relationship)
        remaining = 1.0
        accumulated_time = 0.0
        height = 1.0
        
        for i in range(bounces + 1):
            bounce_duration = remaining * 0.5
            if t < accumulated_time + bounce_duration:
                # In this bounce
                local_t = (t - accumulated_time) / bounce_duration
                # Parabolic trajectory
                return height * 4 * local_t * (1 - local_t)
            accumulated_time += bounce_duration
            remaining -= bounce_duration
            height *= 0.5
        
        return 0.0
    
    @staticmethod
    def perlin_ease(t: float, seed: int = 0) -> float:
        """
        Perlin-noise-like smooth random variation.
        Useful for organic, non-repetitive motion.
        """
        # Use multiple sine waves with irrational frequency ratios
        # for pseudo-random but smooth motion
        golden = (1 + np.sqrt(5)) / 2
        return (
            np.sin(t * 2 * np.pi) * 0.5 +
            np.sin(t * 2 * np.pi * golden + seed) * 0.3 +
            np.sin(t * 2 * np.pi * golden * golden + seed * 2) * 0.2
        )
    
    @staticmethod
    def anticipation(t: float, wind_up: float = 0.2) -> float:
        """
        Animation anticipation curve - slight movement backward before forward.
        Classic animation principle for more natural motion.
        """
        if t < wind_up:
            # Wind-up phase (negative movement)
            local_t = t / wind_up
            return -0.2 * Easing.ease_in_quad(local_t)
        else:
            # Main movement with overshoot compensation
            local_t = (t - wind_up) / (1 - wind_up)
            return -0.2 + 1.2 * Easing.ease_out_cubic(local_t)
    
    @staticmethod
    def overshoot(t: float, amount: float = 0.1) -> float:
        """
        Overshoot and settle curve.
        Goes past target then settles back.
        """
        if t < 0.7:
            # Main movement with overshoot
            local_t = t / 0.7
            return (1 + amount) * Easing.ease_out_quad(local_t)
        else:
            # Settle back
            local_t = (t - 0.7) / 0.3
            return 1.0 + amount * (1 - Easing.ease_in_out_quad(local_t))
    
    @staticmethod
    def wave_packet(t: float, frequency: float = 5.0, width: float = 0.3) -> float:
        """
        Wave packet - localized wave that appears and disappears.
        Useful for pulse effects that travel.
        """
        # Gaussian envelope centered at t=0.5
        envelope = np.exp(-((t - 0.5) ** 2) / (2 * width ** 2))
        return envelope * np.sin(t * frequency * 2 * np.pi)
    
    @staticmethod
    def chirp(t: float, start_freq: float = 1.0, end_freq: float = 5.0) -> float:
        """
        Chirp - wave with increasing/decreasing frequency.
        Creates accelerating or decelerating oscillation.
        """
        # Linear frequency sweep
        freq = start_freq + (end_freq - start_freq) * t
        phase = 2 * np.pi * (start_freq * t + 0.5 * (end_freq - start_freq) * t * t)
        return np.sin(phase)


class BaseEffect(ABC):
    """Abstract base class for procedural animation effects"""
    
    # Effect metadata
    name: str = "base"
    description: str = "Base effect"
    
    # Default parameters
    DEFAULT_FRAME_COUNT = 8
    DEFAULT_SPEED = 1.0
    DEFAULT_INTENSITY = 1.0
    
    def __init__(self, config: Optional[EffectConfig] = None):
        self.config = config or EffectConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.easing = Easing()
    
    @abstractmethod
    def apply(self, sprite: Sprite) -> List[Sprite]:
        """Apply the effect to a sprite and return animation frames."""
        pass
    
    def _create_frame(self, sprite: Sprite, pixels: np.ndarray) -> Sprite:
        """Helper to create a new frame from pixels"""
        return Sprite(
            width=sprite.width,
            height=sprite.height,
            pixels=pixels.astype(np.uint8),
            name=f"{sprite.name}_frame",
            source_path=sprite.source_path
        )
    
    def _get_progress(self, frame_idx: int) -> float:
        """Get normalized progress for a frame (0-1)"""
        return frame_idx / self.config.frame_count
    
    def _get_loop_progress(self, frame_idx: int, smooth: bool = True) -> float:
        """Get looping progress (0 -> 1 -> 0) with optional smoothing"""
        t = self._get_progress(frame_idx)
        if t < 0.5:
            loop_t = t * 2
        else:
            loop_t = (1 - t) * 2
        return Easing.smooth_step(loop_t) if smooth else loop_t
    
    def _get_wave(self, frame_idx: int, frequency: float = 1.0) -> float:
        """Get sine wave value for frame (-1 to 1)"""
        t = self._get_progress(frame_idx)
        return np.sin(t * frequency * 2 * np.pi)
    
    def _lerp(self, a: float, b: float, t: float) -> float:
        """Linear interpolation between two values"""
        return a + (b - a) * t
    
    def _lerp_pixels(
        self, 
        pixels1: np.ndarray, 
        pixels2: np.ndarray, 
        t: float
    ) -> np.ndarray:
        """Linear interpolation between two pixel arrays"""
        return (pixels1 * (1 - t) + pixels2 * t).astype(np.uint8)
    
    def _shift_pixels(
        self,
        pixels: np.ndarray,
        dx: int,
        dy: int,
        wrap: bool = False
    ) -> np.ndarray:
        """Shift pixels by dx, dy"""
        result = np.zeros_like(pixels)
        
        h, w = pixels.shape[:2]
        
        # Calculate source and destination slices
        src_y = slice(max(0, -dy), min(h, h - dy))
        src_x = slice(max(0, -dx), min(w, w - dx))
        dst_y = slice(max(0, dy), min(h, h + dy))
        dst_x = slice(max(0, dx), min(w, w + dx))
        
        if wrap:
            result = np.roll(np.roll(pixels, dy, axis=0), dx, axis=1)
        else:
            result[dst_y, dst_x] = pixels[src_y, src_x]
        
        return result
    
    def _get_mask(self, pixels: np.ndarray) -> np.ndarray:
        """Get binary mask of non-transparent pixels"""
        if pixels.shape[2] == 4:
            return pixels[:, :, 3] > 0
        return np.ones(pixels.shape[:2], dtype=bool)
    
    def _apply_mask(
        self,
        modified: np.ndarray,
        original: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Apply mask to blend modified and original pixels"""
        result = original.copy()
        result[mask] = modified[mask]
        return result


class SimpleEffect(BaseEffect):
    """Base for effects that just transform each frame independently"""
    
    @abstractmethod
    def transform_frame(self, sprite: Sprite, frame_idx: int) -> np.ndarray:
        """Transform pixels for a single frame"""
        pass
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        """Apply transformation to all frames"""
        frames = []
        for i in range(self.config.frame_count):
            pixels = self.transform_frame(sprite, i)
            frames.append(self._create_frame(sprite, pixels))
        return frames
