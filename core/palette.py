"""
Palette Cycling & Color Ramping

Classic pixel art technique: animate colors, not just pixels.
Used in retro games for water, fire, lava, magic, and more.

Techniques:
1. Palette Cycling - Rotate colors in sequence (water shimmer)
2. Color Ramping - Shift along gradient by intensity (fire glow)
3. Hue Shifting - Rotate hue over time (rainbow, magic)
4. Intensity Mapping - Remap brightness to new colors (heat maps)

Examples:
- Fire: orange → yellow → bright yellow cycle
- Water: blue → cyan → white at wave peaks
- Lava: dark red → orange → yellow pulsing
- Magic: purple → pink → white sparkle
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import colorsys


# =============================================================================
# Color Types & Constants
# =============================================================================

Color = Tuple[int, int, int]          # RGB 0-255
ColorF = Tuple[float, float, float]   # RGB 0-1

GAMMA = 2.2
INV_GAMMA = 1.0 / GAMMA


# =============================================================================
# Color Space Conversions
# =============================================================================

def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert RGB (0-255) to HSV (0-1, 0-1, 0-1)"""
    return colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)


def hsv_to_rgb(h: float, s: float, v: float) -> Color:
    """Convert HSV (0-1) to RGB (0-255)"""
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, np.clip(s, 0, 1), np.clip(v, 0, 1))
    return (int(r * 255), int(g * 255), int(b * 255))


def rgb_to_hsl(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert RGB (0-255) to HSL (0-1, 0-1, 0-1)"""
    return colorsys.rgb_to_hls(r / 255, g / 255, b / 255)


def hsl_to_rgb(h: float, s: float, l: float) -> Color:
    """Convert HSL (0-1) to RGB (0-255)"""
    r, g, b = colorsys.hls_to_rgb(h % 1.0, np.clip(l, 0, 1), np.clip(s, 0, 1))
    return (int(r * 255), int(g * 255), int(b * 255))


def to_linear(srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB to linear"""
    return (srgb / 255.0) ** GAMMA


def to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear to sRGB"""
    return (np.clip(linear, 0, 1) ** INV_GAMMA * 255).astype(np.uint8)


def get_luminance(r: int, g: int, b: int) -> float:
    """Get perceptual luminance (0-1)"""
    # Rec. 709 coefficients
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255


def color_distance(c1: Color, c2: Color) -> float:
    """Euclidean distance between colors"""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


# =============================================================================
# Palette Definitions
# =============================================================================

@dataclass
class ColorRamp:
    """
    A gradient of colors for ramping effects.
    
    Colors are ordered from dark/low to bright/high intensity.
    """
    colors: List[Color]
    name: str = "custom"
    
    def sample(self, t: float) -> Color:
        """
        Sample color at position t (0-1) along ramp.
        
        Uses linear interpolation between colors.
        """
        t = np.clip(t, 0.0, 1.0)
        
        if len(self.colors) == 1:
            return self.colors[0]
        
        # Find segment
        segment_count = len(self.colors) - 1
        scaled_t = t * segment_count
        idx = int(scaled_t)
        local_t = scaled_t - idx
        
        if idx >= segment_count:
            return self.colors[-1]
        
        # Interpolate
        c1 = self.colors[idx]
        c2 = self.colors[idx + 1]
        
        return (
            int(c1[0] + (c2[0] - c1[0]) * local_t),
            int(c1[1] + (c2[1] - c1[1]) * local_t),
            int(c1[2] + (c2[2] - c1[2]) * local_t)
        )
    
    def sample_linear(self, t: float) -> Color:
        """Sample with gamma-correct interpolation"""
        t = np.clip(t, 0.0, 1.0)
        
        if len(self.colors) == 1:
            return self.colors[0]
        
        segment_count = len(self.colors) - 1
        scaled_t = t * segment_count
        idx = int(scaled_t)
        local_t = scaled_t - idx
        
        if idx >= segment_count:
            return self.colors[-1]
        
        # Convert to linear, interpolate, convert back
        c1 = np.array(self.colors[idx]) / 255.0
        c2 = np.array(self.colors[idx + 1]) / 255.0
        
        c1_lin = c1 ** GAMMA
        c2_lin = c2 ** GAMMA
        
        result_lin = c1_lin + (c2_lin - c1_lin) * local_t
        result = result_lin ** INV_GAMMA * 255
        
        return (int(result[0]), int(result[1]), int(result[2]))
    
    def __len__(self):
        return len(self.colors)


# =============================================================================
# Preset Palettes & Ramps
# =============================================================================

# Fire ramps (dark → bright)
FIRE_RAMP = ColorRamp([
    (60, 20, 10),      # Dark ember
    (180, 50, 20),     # Deep red
    (255, 100, 0),     # Orange
    (255, 150, 50),    # Yellow-orange
    (255, 200, 100),   # Bright yellow
    (255, 240, 200),   # White-hot
], name="fire")

EMBER_RAMP = ColorRamp([
    (40, 10, 5),
    (120, 30, 10),
    (200, 60, 20),
    (255, 100, 30),
    (255, 160, 80),
], name="ember")

# Water ramps
WATER_RAMP = ColorRamp([
    (20, 40, 80),      # Deep blue
    (40, 80, 140),     # Mid blue
    (60, 120, 180),    # Light blue
    (100, 180, 220),   # Cyan
    (180, 230, 255),   # Light cyan
    (240, 250, 255),   # White foam
], name="water")

OCEAN_RAMP = ColorRamp([
    (10, 30, 60),
    (20, 60, 100),
    (40, 100, 150),
    (80, 160, 200),
    (150, 210, 240),
], name="ocean")

# Lava ramps
LAVA_RAMP = ColorRamp([
    (30, 10, 5),       # Black crust
    (80, 20, 10),      # Dark red
    (160, 40, 10),     # Red
    (220, 80, 10),     # Orange-red
    (255, 140, 30),    # Orange
    (255, 200, 80),    # Yellow
], name="lava")

# Magic/energy ramps
MAGIC_RAMP = ColorRamp([
    (40, 20, 80),      # Dark purple
    (80, 40, 140),     # Purple
    (140, 60, 180),    # Magenta
    (200, 100, 220),   # Pink
    (240, 180, 255),   # Light pink
    (255, 240, 255),   # White
], name="magic")

ELECTRIC_RAMP = ColorRamp([
    (20, 40, 80),
    (40, 80, 180),
    (80, 140, 255),
    (160, 200, 255),
    (220, 240, 255),
    (255, 255, 255),
], name="electric")

ICE_RAMP = ColorRamp([
    (60, 80, 120),
    (100, 140, 180),
    (150, 190, 220),
    (200, 230, 250),
    (240, 250, 255),
], name="ice")

POISON_RAMP = ColorRamp([
    (20, 40, 10),
    (40, 80, 20),
    (80, 140, 40),
    (120, 200, 60),
    (180, 240, 100),
], name="poison")

GOLD_RAMP = ColorRamp([
    (80, 50, 20),
    (140, 90, 30),
    (200, 140, 40),
    (240, 190, 80),
    (255, 230, 150),
], name="gold")

# Grayscale for intensity mapping
GRAYSCALE_RAMP = ColorRamp([
    (0, 0, 0),
    (64, 64, 64),
    (128, 128, 128),
    (192, 192, 192),
    (255, 255, 255),
], name="grayscale")

# Heat map
HEAT_RAMP = ColorRamp([
    (0, 0, 0),         # Cold (black)
    (0, 0, 128),       # Blue
    (128, 0, 128),     # Purple
    (255, 0, 0),       # Red
    (255, 128, 0),     # Orange
    (255, 255, 0),     # Yellow
    (255, 255, 255),   # White hot
], name="heat")

# Preset dictionary
RAMP_PRESETS = {
    'fire': FIRE_RAMP,
    'ember': EMBER_RAMP,
    'water': WATER_RAMP,
    'ocean': OCEAN_RAMP,
    'lava': LAVA_RAMP,
    'magic': MAGIC_RAMP,
    'electric': ELECTRIC_RAMP,
    'ice': ICE_RAMP,
    'poison': POISON_RAMP,
    'gold': GOLD_RAMP,
    'grayscale': GRAYSCALE_RAMP,
    'heat': HEAT_RAMP,
}


def get_ramp(name: str) -> ColorRamp:
    """Get a preset color ramp by name"""
    if name not in RAMP_PRESETS:
        available = ', '.join(RAMP_PRESETS.keys())
        raise ValueError(f"Unknown ramp '{name}'. Available: {available}")
    return RAMP_PRESETS[name]


# =============================================================================
# Palette Cycling
# =============================================================================

class PaletteCycler:
    """
    Cycle through colors in a palette over time.
    
    Classic effect for water shimmer, fire flicker, etc.
    
    Example:
        cycler = PaletteCycler([
            (255, 100, 0),   # Orange
            (255, 150, 50),  # Yellow-orange
            (255, 200, 100), # Bright yellow
        ])
        
        for frame in range(total_frames):
            t = frame / total_frames
            cycler.advance(1 / fps)
            
            # Map original colors to cycled colors
            output = cycler.apply(sprite, original_palette)
    """
    
    def __init__(
        self,
        colors: List[Color],
        cycle_speed: float = 1.0,
        interpolate: bool = True
    ):
        """
        Args:
            colors: List of colors to cycle through
            cycle_speed: Cycles per second
            interpolate: Smooth interpolation between colors
        """
        self.colors = colors
        self.cycle_speed = cycle_speed
        self.interpolate = interpolate
        self.phase = 0.0
    
    def advance(self, dt: float):
        """Advance cycle by dt seconds"""
        self.phase = (self.phase + dt * self.cycle_speed) % 1.0
    
    def set_phase(self, phase: float):
        """Set cycle phase directly (0-1)"""
        self.phase = phase % 1.0
    
    def get_current_color(self, index: int = 0) -> Color:
        """
        Get current color for palette index.
        
        Index 0 gets the current "front" color, index 1 gets the next, etc.
        """
        n = len(self.colors)
        
        if self.interpolate:
            # Smooth interpolation
            scaled = self.phase * n
            idx1 = int(scaled) % n
            idx2 = (idx1 + 1) % n
            t = scaled - int(scaled)
            
            # Offset by requested index
            idx1 = (idx1 + index) % n
            idx2 = (idx2 + index) % n
            
            c1 = self.colors[idx1]
            c2 = self.colors[idx2]
            
            return (
                int(c1[0] + (c2[0] - c1[0]) * t),
                int(c1[1] + (c2[1] - c1[1]) * t),
                int(c1[2] + (c2[2] - c1[2]) * t)
            )
        else:
            # Step through colors
            idx = (int(self.phase * n) + index) % n
            return self.colors[idx]
    
    def get_palette(self) -> List[Color]:
        """Get current full palette (all colors in cycle order)"""
        return [self.get_current_color(i) for i in range(len(self.colors))]
    
    def apply_to_image(
        self,
        pixels: np.ndarray,
        source_palette: List[Color],
        tolerance: int = 10
    ) -> np.ndarray:
        """
        Apply palette cycling to an image.
        
        Replaces colors matching source_palette with cycled versions.
        
        Args:
            pixels: Input image (RGB or RGBA)
            source_palette: Colors to replace (in order)
            tolerance: Color matching tolerance
        
        Returns:
            Image with cycled colors
        """
        result = pixels.copy()
        current_palette = self.get_palette()
        
        # Pad palettes to match
        max_len = max(len(source_palette), len(current_palette))
        while len(current_palette) < max_len:
            current_palette.append(current_palette[-1])
        
        for i, src_color in enumerate(source_palette):
            if i >= len(current_palette):
                break
            
            dst_color = current_palette[i]
            
            # Find matching pixels
            if pixels.shape[-1] == 4:
                # RGBA
                mask = (
                    (np.abs(pixels[..., 0].astype(int) - src_color[0]) <= tolerance) &
                    (np.abs(pixels[..., 1].astype(int) - src_color[1]) <= tolerance) &
                    (np.abs(pixels[..., 2].astype(int) - src_color[2]) <= tolerance) &
                    (pixels[..., 3] > 10)  # Only visible pixels
                )
            else:
                # RGB
                mask = (
                    (np.abs(pixels[..., 0].astype(int) - src_color[0]) <= tolerance) &
                    (np.abs(pixels[..., 1].astype(int) - src_color[1]) <= tolerance) &
                    (np.abs(pixels[..., 2].astype(int) - src_color[2]) <= tolerance)
                )
            
            # Replace colors
            result[mask, 0] = dst_color[0]
            result[mask, 1] = dst_color[1]
            result[mask, 2] = dst_color[2]
        
        return result


# =============================================================================
# Color Ramping
# =============================================================================

class ColorRamper:
    """
    Remap colors based on intensity/brightness to a color ramp.
    
    Great for: fire glow, water highlights, energy effects.
    
    Example:
        ramper = ColorRamper(FIRE_RAMP)
        
        # Brighter pixels become more yellow/white
        output = ramper.apply_by_brightness(sprite)
    """
    
    def __init__(
        self,
        ramp: Union[ColorRamp, str],
        gamma_correct: bool = True
    ):
        if isinstance(ramp, str):
            ramp = get_ramp(ramp)
        self.ramp = ramp
        self.gamma_correct = gamma_correct
    
    def apply_by_brightness(
        self,
        pixels: np.ndarray,
        min_brightness: float = 0.0,
        max_brightness: float = 1.0,
        preserve_alpha: bool = True
    ) -> np.ndarray:
        """
        Remap colors based on pixel brightness.
        
        Brighter pixels map to higher positions in the ramp.
        """
        result = pixels.copy()
        
        # Calculate brightness for each pixel
        if pixels.shape[-1] == 4:
            r, g, b, a = pixels[..., 0], pixels[..., 1], pixels[..., 2], pixels[..., 3]
        else:
            r, g, b = pixels[..., 0], pixels[..., 1], pixels[..., 2]
            a = np.full_like(r, 255)
        
        # Perceptual luminance
        brightness = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
        
        # Normalize to ramp range
        brightness = (brightness - min_brightness) / (max_brightness - min_brightness + 1e-10)
        brightness = np.clip(brightness, 0, 1)
        
        # Apply ramp
        for y in range(pixels.shape[0]):
            for x in range(pixels.shape[1]):
                if a[y, x] > 10:  # Only visible pixels
                    if self.gamma_correct:
                        color = self.ramp.sample_linear(brightness[y, x])
                    else:
                        color = self.ramp.sample(brightness[y, x])
                    result[y, x, 0] = color[0]
                    result[y, x, 1] = color[1]
                    result[y, x, 2] = color[2]
        
        return result
    
    def apply_by_brightness_vectorized(
        self,
        pixels: np.ndarray,
        min_brightness: float = 0.0,
        max_brightness: float = 1.0
    ) -> np.ndarray:
        """Vectorized brightness remapping (faster for large images)"""
        result = pixels.copy()
        
        # Get brightness
        brightness = (
            0.2126 * pixels[..., 0] + 
            0.7152 * pixels[..., 1] + 
            0.0722 * pixels[..., 2]
        ) / 255
        
        # Normalize
        brightness = (brightness - min_brightness) / (max_brightness - min_brightness + 1e-10)
        brightness = np.clip(brightness, 0, 1)
        
        # Pre-sample ramp at many points
        num_samples = 256
        lut = np.array([self.ramp.sample(i / (num_samples - 1)) for i in range(num_samples)])
        
        # Map brightness to LUT indices
        indices = (brightness * (num_samples - 1)).astype(np.int32)
        
        # Apply LUT
        mask = pixels[..., 3] > 10 if pixels.shape[-1] == 4 else np.ones(pixels.shape[:2], bool)
        
        for c in range(3):
            result[..., c] = np.where(mask, lut[indices, c], result[..., c])
        
        return result
    
    def apply_by_channel(
        self,
        pixels: np.ndarray,
        channel: int = 0,
        preserve_alpha: bool = True
    ) -> np.ndarray:
        """
        Remap based on a specific channel (R=0, G=1, B=2).
        
        Useful when a channel encodes intensity (like alpha-based fire).
        """
        result = pixels.copy()
        
        intensity = pixels[..., channel] / 255.0
        
        num_samples = 256
        lut = np.array([self.ramp.sample(i / (num_samples - 1)) for i in range(num_samples)])
        indices = (intensity * (num_samples - 1)).astype(np.int32)
        
        if pixels.shape[-1] == 4:
            mask = pixels[..., 3] > 10
        else:
            mask = np.ones(pixels.shape[:2], bool)
        
        for c in range(3):
            result[..., c] = np.where(mask, lut[indices, c], result[..., c])
        
        return result


# =============================================================================
# Hue Shifting
# =============================================================================

class HueShifter:
    """
    Shift hue over time for rainbow/magic effects.
    
    Example:
        shifter = HueShifter(speed=0.5)  # Half rotation per second
        
        for frame in range(total_frames):
            shifter.advance(1/fps)
            output = shifter.apply(sprite)
    """
    
    def __init__(
        self,
        speed: float = 1.0,
        preserve_saturation: bool = True,
        preserve_value: bool = True
    ):
        self.speed = speed
        self.preserve_saturation = preserve_saturation
        self.preserve_value = preserve_value
        self.hue_offset = 0.0
    
    def advance(self, dt: float):
        """Advance hue shift by dt seconds"""
        self.hue_offset = (self.hue_offset + dt * self.speed) % 1.0
    
    def set_offset(self, offset: float):
        """Set hue offset directly (0-1)"""
        self.hue_offset = offset % 1.0
    
    def apply(
        self,
        pixels: np.ndarray,
        mask: np.ndarray = None
    ) -> np.ndarray:
        """Apply hue shift to image"""
        result = pixels.copy()
        
        if mask is None:
            if pixels.shape[-1] == 4:
                mask = pixels[..., 3] > 10
            else:
                mask = np.ones(pixels.shape[:2], bool)
        
        ys, xs = np.where(mask)
        
        for y, x in zip(ys, xs):
            r, g, b = pixels[y, x, 0], pixels[y, x, 1], pixels[y, x, 2]
            h, s, v = rgb_to_hsv(r, g, b)
            
            # Shift hue
            h = (h + self.hue_offset) % 1.0
            
            new_r, new_g, new_b = hsv_to_rgb(h, s, v)
            result[y, x, 0] = new_r
            result[y, x, 1] = new_g
            result[y, x, 2] = new_b
        
        return result
    
    def apply_vectorized(
        self,
        pixels: np.ndarray,
        mask: np.ndarray = None
    ) -> np.ndarray:
        """Vectorized hue shift (faster)"""
        result = pixels.copy()
        
        if mask is None:
            if pixels.shape[-1] == 4:
                mask = pixels[..., 3] > 10
            else:
                mask = np.ones(pixels.shape[:2], bool)
        
        # Convert to float
        r = pixels[..., 0].astype(np.float32) / 255
        g = pixels[..., 1].astype(np.float32) / 255
        b = pixels[..., 2].astype(np.float32) / 255
        
        # RGB to HSV (vectorized)
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        diff = max_c - min_c
        
        # Hue calculation
        h = np.zeros_like(r)
        
        # Where max == r
        mask_r = (max_c == r) & (diff > 0)
        h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360
        
        # Where max == g
        mask_g = (max_c == g) & (diff > 0)
        h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360
        
        # Where max == b
        mask_b = (max_c == b) & (diff > 0)
        h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360
        
        h = h / 360  # Normalize to 0-1
        
        # Saturation and value
        s = np.where(max_c > 0, diff / max_c, 0)
        v = max_c
        
        # Shift hue
        h = (h + self.hue_offset) % 1.0
        
        # HSV to RGB (vectorized)
        h6 = h * 6
        i = np.floor(h6).astype(int)
        f = h6 - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        
        i = i % 6
        
        new_r = np.zeros_like(r)
        new_g = np.zeros_like(g)
        new_b = np.zeros_like(b)
        
        m0 = i == 0
        new_r[m0], new_g[m0], new_b[m0] = v[m0], t[m0], p[m0]
        m1 = i == 1
        new_r[m1], new_g[m1], new_b[m1] = q[m1], v[m1], p[m1]
        m2 = i == 2
        new_r[m2], new_g[m2], new_b[m2] = p[m2], v[m2], t[m2]
        m3 = i == 3
        new_r[m3], new_g[m3], new_b[m3] = p[m3], q[m3], v[m3]
        m4 = i == 4
        new_r[m4], new_g[m4], new_b[m4] = t[m4], p[m4], v[m4]
        m5 = i == 5
        new_r[m5], new_g[m5], new_b[m5] = v[m5], p[m5], q[m5]
        
        # Apply only to masked pixels
        result[..., 0] = np.where(mask, (new_r * 255).astype(np.uint8), result[..., 0])
        result[..., 1] = np.where(mask, (new_g * 255).astype(np.uint8), result[..., 1])
        result[..., 2] = np.where(mask, (new_b * 255).astype(np.uint8), result[..., 2])
        
        return result


# =============================================================================
# Intensity Pulsing
# =============================================================================

class IntensityPulser:
    """
    Pulse color intensity over time.
    
    Shifts colors along a ramp based on a time-varying intensity.
    Great for: glowing effects, heartbeats, magic auras.
    
    Example:
        pulser = IntensityPulser(MAGIC_RAMP, speed=2.0)
        
        for frame in range(total_frames):
            pulser.advance(1/fps)
            output = pulser.apply(sprite)
    """
    
    def __init__(
        self,
        ramp: Union[ColorRamp, str],
        speed: float = 1.0,
        min_intensity: float = 0.3,
        max_intensity: float = 1.0,
        wave_shape: str = "sine"  # sine, triangle, square, sawtooth
    ):
        if isinstance(ramp, str):
            ramp = get_ramp(ramp)
        self.ramp = ramp
        self.speed = speed
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.wave_shape = wave_shape
        self.phase = 0.0
        
        self._ramper = ColorRamper(ramp)
    
    def advance(self, dt: float):
        """Advance pulse by dt seconds"""
        self.phase = (self.phase + dt * self.speed) % 1.0
    
    def get_intensity(self) -> float:
        """Get current intensity value"""
        if self.wave_shape == "sine":
            wave = (np.sin(self.phase * 2 * np.pi) + 1) / 2
        elif self.wave_shape == "triangle":
            wave = 1 - abs(2 * self.phase - 1)
        elif self.wave_shape == "square":
            wave = 1.0 if self.phase < 0.5 else 0.0
        elif self.wave_shape == "sawtooth":
            wave = self.phase
        else:
            wave = (np.sin(self.phase * 2 * np.pi) + 1) / 2
        
        return self.min_intensity + wave * (self.max_intensity - self.min_intensity)
    
    def apply(self, pixels: np.ndarray) -> np.ndarray:
        """Apply pulsing intensity to image"""
        intensity = self.get_intensity()
        
        # Remap brightness range based on intensity
        return self._ramper.apply_by_brightness_vectorized(
            pixels,
            min_brightness=0.0,
            max_brightness=1.0 / intensity
        )


# =============================================================================
# Palette Extraction & Matching
# =============================================================================

def extract_palette(
    pixels: np.ndarray,
    max_colors: int = 8,
    min_alpha: int = 10
) -> List[Color]:
    """
    Extract dominant colors from an image.
    
    Returns:
        List of (R, G, B) tuples sorted by frequency
    """
    # Get visible pixels
    if pixels.shape[-1] == 4:
        mask = pixels[..., 3] > min_alpha
        colors = pixels[mask, :3]
    else:
        colors = pixels.reshape(-1, 3)
    
    if len(colors) == 0:
        return [(128, 128, 128)]
    
    # Simple color quantization via unique colors
    unique_colors, counts = np.unique(colors, axis=0, return_counts=True)
    
    # Sort by frequency
    sorted_indices = np.argsort(-counts)
    
    # Return top colors
    result = []
    for idx in sorted_indices[:max_colors]:
        c = unique_colors[idx]
        result.append((int(c[0]), int(c[1]), int(c[2])))
    
    return result


def match_to_palette(
    pixels: np.ndarray,
    palette: List[Color],
    dithering: bool = False
) -> np.ndarray:
    """
    Match image colors to a fixed palette.
    
    Args:
        pixels: Input image
        palette: Target palette
        dithering: Apply Floyd-Steinberg dithering
    
    Returns:
        Palettized image
    """
    result = pixels.copy().astype(np.float32)
    palette_arr = np.array(palette)
    
    h, w = pixels.shape[:2]
    
    for y in range(h):
        for x in range(w):
            if pixels.shape[-1] == 4 and pixels[y, x, 3] < 10:
                continue
            
            old_color = result[y, x, :3]
            
            # Find nearest palette color
            distances = np.sqrt(np.sum((palette_arr - old_color) ** 2, axis=1))
            nearest_idx = np.argmin(distances)
            new_color = palette_arr[nearest_idx]
            
            result[y, x, :3] = new_color
            
            if dithering:
                # Floyd-Steinberg error diffusion
                error = old_color - new_color
                
                if x + 1 < w:
                    result[y, x + 1, :3] += error * 7 / 16
                if y + 1 < h:
                    if x > 0:
                        result[y + 1, x - 1, :3] += error * 3 / 16
                    result[y + 1, x, :3] += error * 5 / 16
                    if x + 1 < w:
                        result[y + 1, x + 1, :3] += error * 1 / 16
    
    return np.clip(result, 0, 255).astype(np.uint8)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_fire_cycler(speed: float = 2.0) -> PaletteCycler:
    """Create fire palette cycler"""
    return PaletteCycler(FIRE_RAMP.colors, cycle_speed=speed)


def create_water_cycler(speed: float = 1.0) -> PaletteCycler:
    """Create water palette cycler"""
    return PaletteCycler(WATER_RAMP.colors, cycle_speed=speed)


def create_magic_cycler(speed: float = 1.5) -> PaletteCycler:
    """Create magic palette cycler"""
    return PaletteCycler(MAGIC_RAMP.colors, cycle_speed=speed)


def apply_fire_ramp(pixels: np.ndarray) -> np.ndarray:
    """Quick fire color ramp application"""
    return ColorRamper(FIRE_RAMP).apply_by_brightness_vectorized(pixels)


def apply_water_ramp(pixels: np.ndarray) -> np.ndarray:
    """Quick water color ramp application"""
    return ColorRamper(WATER_RAMP).apply_by_brightness_vectorized(pixels)


def shift_hue(pixels: np.ndarray, offset: float) -> np.ndarray:
    """Quick hue shift"""
    shifter = HueShifter()
    shifter.set_offset(offset)
    return shifter.apply_vectorized(pixels)
