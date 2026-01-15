"""
Utility functions for color manipulation and math operations
Enhanced with better easing and interpolation
"""

import numpy as np
from typing import Tuple, List, Optional
import colorsys


class ColorUtils:
    """Color analysis and manipulation utilities"""
    
    # Color category definitions (HSV ranges) - expanded for better detection
    COLOR_CATEGORIES = {
        'red': {'h': [(0, 15), (345, 360)], 's': (0.3, 1), 'v': (0.3, 1)},
        'orange': {'h': [(15, 45)], 's': (0.4, 1), 'v': (0.4, 1)},
        'yellow': {'h': [(45, 70)], 's': (0.3, 1), 'v': (0.5, 1)},
        'lime': {'h': [(70, 90)], 's': (0.3, 1), 'v': (0.4, 1)},
        'green': {'h': [(90, 160)], 's': (0.3, 1), 'v': (0.3, 1)},
        'cyan': {'h': [(160, 200)], 's': (0.3, 1), 'v': (0.3, 1)},
        'blue': {'h': [(200, 260)], 's': (0.3, 1), 'v': (0.3, 1)},
        'purple': {'h': [(260, 290)], 's': (0.3, 1), 'v': (0.2, 1)},
        'magenta': {'h': [(290, 330)], 's': (0.3, 1), 'v': (0.3, 1)},
        'pink': {'h': [(330, 345)], 's': (0.2, 0.7), 'v': (0.6, 1)},
        'white': {'h': [(0, 360)], 's': (0, 0.15), 'v': (0.85, 1)},
        'light_gray': {'h': [(0, 360)], 's': (0, 0.1), 'v': (0.6, 0.85)},
        'gray': {'h': [(0, 360)], 's': (0, 0.15), 'v': (0.3, 0.6)},
        'dark_gray': {'h': [(0, 360)], 's': (0, 0.15), 'v': (0.15, 0.3)},
        'black': {'h': [(0, 360)], 's': (0, 1), 'v': (0, 0.15)},
        'brown': {'h': [(15, 45)], 's': (0.3, 0.8), 'v': (0.2, 0.5)},
    }
    
    @staticmethod
    def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
        """Convert RGB (0-255) to HSV (0-360, 0-1, 0-1)"""
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        return (h * 360, s, v)
    
    @staticmethod
    def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
        """Convert HSV (0-360, 0-1, 0-1) to RGB (0-255)"""
        r, g, b = colorsys.hsv_to_rgb(h/360, s, v)
        return (int(r * 255), int(g * 255), int(b * 255))
    
    @classmethod
    def categorize_color(cls, r: int, g: int, b: int) -> str:
        """Categorize an RGB color into a named category"""
        h, s, v = cls.rgb_to_hsv(r, g, b)
        
        for category, ranges in cls.COLOR_CATEGORIES.items():
            s_min, s_max = ranges['s']
            v_min, v_max = ranges['v']
            
            if not (s_min <= s <= s_max and v_min <= v <= v_max):
                continue
            
            # Check hue ranges (can have multiple for wrap-around)
            for h_min, h_max in ranges['h']:
                if h_min <= h <= h_max:
                    return category
        
        return 'unknown'
    
    @classmethod
    def get_color_distribution(cls, pixels: np.ndarray) -> dict:
        """Analyze color distribution in an image with weighted importance"""
        flat = pixels.reshape(-1, pixels.shape[2])
        
        # Filter out transparent pixels
        if pixels.shape[2] == 4:
            visible = flat[flat[:, 3] > 10]  # More lenient threshold
        else:
            visible = flat
        
        if len(visible) == 0:
            return {}
        
        # Categorize each pixel with brightness weighting
        categories = {}
        weights = {}
        
        for pixel in visible:
            cat = cls.categorize_color(pixel[0], pixel[1], pixel[2])
            brightness = cls.get_brightness(pixel[0], pixel[1], pixel[2])
            
            # Weight brighter pixels more (they're more visually important)
            weight = 0.5 + brightness * 0.5
            
            categories[cat] = categories.get(cat, 0) + 1
            weights[cat] = weights.get(cat, 0) + weight
        
        # Convert to percentages using weights
        total_weight = sum(weights.values())
        return {k: v / total_weight for k, v in weights.items()}
    
    @staticmethod
    def get_brightness(r: int, g: int, b: int) -> float:
        """Calculate perceived brightness (0-1)"""
        return (0.299 * r + 0.587 * g + 0.114 * b) / 255
    
    @staticmethod
    def get_saturation(r: int, g: int, b: int) -> float:
        """Calculate color saturation (0-1)"""
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        if max_c == 0:
            return 0
        return (max_c - min_c) / max_c
    
    @staticmethod
    def get_warmth(r: int, g: int, b: int) -> float:
        """Calculate color warmth (-1 cold to 1 warm)"""
        return (r - b) / 255
    
    @classmethod
    def analyze_gradient(cls, pixels: np.ndarray) -> dict:
        """Analyze if sprite has gradient patterns (important for flame detection)"""
        if pixels.shape[2] == 4:
            mask = pixels[:, :, 3] > 0
        else:
            mask = np.ones(pixels.shape[:2], dtype=bool)
        
        if not np.any(mask):
            return {'has_vertical_gradient': False, 'has_horizontal_gradient': False}
        
        h, w = pixels.shape[:2]
        
        # Check vertical brightness gradient
        top_third = pixels[:h//3, :, :3][mask[:h//3, :]]
        bottom_third = pixels[2*h//3:, :, :3][mask[2*h//3:, :]]
        
        if len(top_third) > 0 and len(bottom_third) > 0:
            top_brightness = np.mean([cls.get_brightness(*p) for p in top_third[:100]])
            bottom_brightness = np.mean([cls.get_brightness(*p) for p in bottom_third[:100]])
            vertical_gradient = top_brightness - bottom_brightness
        else:
            vertical_gradient = 0
        
        return {
            'has_vertical_gradient': abs(vertical_gradient) > 0.1,
            'gradient_direction': 'up' if vertical_gradient > 0 else 'down',
            'gradient_strength': abs(vertical_gradient)
        }
    
    @classmethod
    def is_fire_palette(cls, pixels: np.ndarray) -> float:
        """Score how likely the palette represents fire (0-1)"""
        dist = cls.get_color_distribution(pixels)
        gradient = cls.analyze_gradient(pixels)
        
        fire_colors = dist.get('red', 0) + dist.get('orange', 0) + dist.get('yellow', 0)
        cold_colors = dist.get('blue', 0) + dist.get('cyan', 0) + dist.get('green', 0)
        
        if fire_colors + cold_colors == 0:
            return 0
        
        base_score = fire_colors / (fire_colors + cold_colors + 0.001)
        
        # Bonus for upward brightness gradient (fire is brighter at base)
        if gradient['has_vertical_gradient'] and gradient['gradient_direction'] == 'down':
            base_score += 0.15
        
        return min(1.0, base_score)
    
    @classmethod
    def is_water_palette(cls, pixels: np.ndarray) -> float:
        """Score how likely the palette represents water (0-1)"""
        dist = cls.get_color_distribution(pixels)
        
        water_colors = dist.get('blue', 0) + dist.get('cyan', 0)
        warm_colors = dist.get('red', 0) + dist.get('orange', 0) + dist.get('yellow', 0)
        
        if water_colors + warm_colors == 0:
            return 0
        
        base_score = water_colors / (water_colors + warm_colors + 0.001)
        
        # Bonus for white highlights (reflections)
        if dist.get('white', 0) > 0.02:
            base_score += 0.1
        
        return min(1.0, base_score)
    
    @classmethod
    def is_magic_palette(cls, pixels: np.ndarray) -> float:
        """Score how likely the palette represents magic/sparkle (0-1)"""
        dist = cls.get_color_distribution(pixels)
        
        magic_colors = (
            dist.get('purple', 0) * 1.5 +
            dist.get('magenta', 0) * 1.3 +
            dist.get('cyan', 0) * 0.8 +
            dist.get('white', 0) * 0.7 +
            dist.get('pink', 0) * 0.6
        )
        
        return min(1.0, magic_colors * 1.5)


class MathUtils:
    """Math utilities for animation calculations - enhanced easing"""
    
    @staticmethod
    def lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation between a and b"""
        return a + (b - a) * t
    
    @staticmethod
    def smooth_lerp(a: float, b: float, t: float) -> float:
        """Smooth interpolation with ease in-out"""
        t = t * t * (3 - 2 * t)
        return a + (b - a) * t
    
    @staticmethod
    def smoother_lerp(a: float, b: float, t: float) -> float:
        """Even smoother interpolation (quintic)"""
        t = t * t * t * (t * (t * 6 - 15) + 10)
        return a + (b - a) * t
    
    @staticmethod
    def ease_in_out(t: float) -> float:
        """Smooth ease in-out curve (cubic)"""
        return t * t * (3 - 2 * t)
    
    @staticmethod
    def ease_in_out_quad(t: float) -> float:
        """Quadratic ease in-out"""
        if t < 0.5:
            return 2 * t * t
        return 1 - pow(-2 * t + 2, 2) / 2
    
    @staticmethod
    def ease_in_out_cubic(t: float) -> float:
        """Cubic ease in-out"""
        if t < 0.5:
            return 4 * t * t * t
        return 1 - pow(-2 * t + 2, 3) / 2
    
    @staticmethod
    def ease_in_out_elastic(t: float, amplitude: float = 1.0) -> float:
        """Elastic ease in-out for bouncy effects"""
        if t == 0 or t == 1:
            return t
        
        if t < 0.5:
            return -(amplitude * pow(2, 20 * t - 10) * np.sin((20 * t - 11.125) * (2 * np.pi) / 4.5)) / 2
        return (amplitude * pow(2, -20 * t + 10) * np.sin((20 * t - 11.125) * (2 * np.pi) / 4.5)) / 2 + 1
    
    @staticmethod
    def ease_in(t: float) -> float:
        """Ease in curve (slow start)"""
        return t * t
    
    @staticmethod
    def ease_out(t: float) -> float:
        """Ease out curve (slow end)"""
        return 1 - (1 - t) * (1 - t)
    
    @staticmethod
    def ease_out_back(t: float, overshoot: float = 1.70158) -> float:
        """Ease out with slight overshoot"""
        return 1 + (overshoot + 1) * pow(t - 1, 3) + overshoot * pow(t - 1, 2)
    
    @staticmethod
    def bounce(t: float) -> float:
        """Bounce effect"""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2
    
    @staticmethod
    def bounce_out(t: float) -> float:
        """Bounce out effect"""
        n1 = 7.5625
        d1 = 2.75
        
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
    def wave(t: float, frequency: float = 1.0) -> float:
        """Sine wave oscillation"""
        return np.sin(t * frequency * 2 * np.pi)
    
    @staticmethod
    def smooth_wave(t: float, frequency: float = 1.0) -> float:
        """Smoother wave using cosine for seamless looping"""
        return (1 - np.cos(t * frequency * 2 * np.pi)) / 2
    
    @staticmethod
    def triangle_wave(t: float, frequency: float = 1.0) -> float:
        """Triangle wave oscillation"""
        t = (t * frequency) % 1.0
        if t < 0.5:
            return t * 2
        return 2 - t * 2
    
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max"""
        return max(min_val, min(max_val, value))
    
    @staticmethod
    def remap(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
        """Remap value from one range to another"""
        if in_max == in_min:
            return out_min
        return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)
    
    @staticmethod
    def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Euclidean distance between two points"""
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    @staticmethod
    def normalize(value: float, min_val: float, max_val: float) -> float:
        """Normalize value to 0-1 range"""
        if max_val == min_val:
            return 0
        return (value - min_val) / (max_val - min_val)
    
    @staticmethod
    def bilinear_interpolate(
        pixels: np.ndarray,
        x: float,
        y: float
    ) -> np.ndarray:
        """Bilinear interpolation for smooth subpixel sampling"""
        h, w = pixels.shape[:2]
        
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)
        
        x0 = max(0, min(x0, w - 1))
        y0 = max(0, min(y0, h - 1))
        
        fx = x - np.floor(x)
        fy = y - np.floor(y)
        
        # Bilinear interpolation
        result = (
            pixels[y0, x0].astype(float) * (1 - fx) * (1 - fy) +
            pixels[y0, x1].astype(float) * fx * (1 - fy) +
            pixels[y1, x0].astype(float) * (1 - fx) * fy +
            pixels[y1, x1].astype(float) * fx * fy
        )
        
        return result.astype(np.uint8)
