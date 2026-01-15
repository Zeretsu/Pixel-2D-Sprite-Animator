"""
Elemental Effects - Add fire, water, ice elements to sprites

Enhanced version with:
- NO sprite distortion by default (sword stays perfectly still)
- Smooth HSV color gradients for better hue transitions
- Improved particle physics and math
- Better visual quality and smoothness
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from .base import BaseEffect, EffectConfig, PixelMath
from .noise import NoiseGenerator
from ..core.parser import Sprite


# =============================================================================
# COLOR UTILITIES - HSV for smooth gradients
# =============================================================================

def rgb_to_hsv(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """Convert RGB (0-255) to HSV (h=0-1, s=0-1, v=0-1)."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    diff = max_c - min_c
    
    if diff == 0:
        h = 0
    elif max_c == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif max_c == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360
    
    s = 0 if max_c == 0 else diff / max_c
    v = max_c
    
    return h / 360.0, s, v


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """Convert HSV (0-1) to RGB (0-255)."""
    h = (h % 1.0) * 360
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))


def color_lerp_hsv(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    """Interpolate between colors in HSV space for smooth, natural gradients."""
    h1, s1, v1 = rgb_to_hsv(*c1)
    h2, s2, v2 = rgb_to_hsv(*c2)
    
    # Handle hue wrapping (take shortest path around color wheel)
    if abs(h2 - h1) > 0.5:
        if h1 > h2:
            h2 += 1.0
        else:
            h1 += 1.0
    
    t = max(0.0, min(1.0, t))
    # Smoothstep for more natural transitions
    t_smooth = t * t * (3 - 2 * t)
    
    h = (h1 + (h2 - h1) * t_smooth) % 1.0
    s = s1 + (s2 - s1) * t_smooth
    v = v1 + (v2 - v1) * t_smooth
    
    return hsv_to_rgb(h, s, v)


def ease_out_quad(t: float) -> float:
    """Quadratic ease out for smooth deceleration."""
    return 1 - (1 - t) * (1 - t)


def ease_in_out_sine(t: float) -> float:
    """Sinusoidal ease in-out for smooth animations."""
    return -(np.cos(np.pi * t) - 1) / 2


# =============================================================================
# FIRE ELEMENT
# =============================================================================

@dataclass
class FireElementConfig(EffectConfig):
    """Configuration for fire element effect."""
    flame_height: float = 0.6
    flame_density: float = 0.8
    ember_count: int = 12
    glow_intensity: float = 0.7
    # Fire color palette (optimized for HSV blending)
    color_core: Tuple[int, int, int] = (255, 255, 220)   # White-yellow core
    color_mid: Tuple[int, int, int] = (255, 160, 40)     # Bright orange
    color_outer: Tuple[int, int, int] = (220, 60, 15)    # Deep orange-red
    color_tip: Tuple[int, int, int] = (120, 25, 8)       # Dark red tip
    animate_sprite: bool = False  # DISABLED by default - sword stays still!


class FireElement(BaseEffect):
    """Add fire/flame elemental effect to sprite."""
    
    name = "fire_element"
    description = "Add flames and fire effects emanating from sprite edges"
    config_class = FireElementConfig
    
    def __init__(self, config: Optional[FireElementConfig] = None):
        if config is None:
            config = FireElementConfig()
        super().__init__(config)
        self.noise = NoiseGenerator(self.config.seed)
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        
        visible_mask = original[:, :, 3] > 10 if original.shape[2] == 4 else np.any(original > 0, axis=2)
        edge_mask = self._find_edges(visible_mask)
        
        flame_origins = self._init_flame_origins(edge_mask)
        embers = self._init_embers(visible_mask, h, w)
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            
            expand = int(h * self.config.flame_height * 0.7) + 6
            canvas_h = h + expand * 2
            canvas_w = w + expand * 2
            
            canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
            
            # Layer 1: Ambient glow
            canvas = self._add_ambient_glow(canvas, w // 2 + expand, h // 2 + expand, t, expand)
            
            # Layer 2: Flames (behind sprite)
            visible_canvas = np.zeros((canvas_h, canvas_w), dtype=bool)
            visible_canvas[expand:expand+h, expand:expand+w] = visible_mask
            canvas = self._add_flames(canvas, flame_origins, t, expand, h, w)
            
            # Layer 3: Sprite ON TOP
            self._composite_sprite(canvas, original, expand, h, w)
            
            # Layer 4: Edge glow on sprite
            canvas = self._add_edge_glow(canvas, visible_canvas, t)
            
            # Layer 5: Embers (front)
            canvas = self._add_embers(canvas, embers, t, expand, h)
            
            # Optional sprite shimmer (OFF by default)
            if self.config.animate_sprite:
                canvas = self._apply_shimmer(canvas, visible_canvas, t)
            
            margin = max(4, expand // 2)
            final = canvas[expand-margin:expand+h+margin, expand-margin:expand+w+margin]
            
            frame = Sprite(
                width=final.shape[1], height=final.shape[0],
                pixels=final, name=f"{sprite.name}_fire_{i}",
                source_path=sprite.source_path
            )
            frames.append(frame)
        
        return frames
    
    def _find_edges(self, mask: np.ndarray) -> np.ndarray:
        padded = np.pad(mask.astype(np.uint8), 1, mode='constant', constant_values=0)
        eroded = (padded[:-2, 1:-1] & padded[2:, 1:-1] &
                  padded[1:-1, :-2] & padded[1:-1, 2:] & padded[1:-1, 1:-1])
        return mask & ~eroded
    
    def _init_flame_origins(self, edge_mask: np.ndarray) -> List[dict]:
        edge_ys, edge_xs = np.where(edge_mask)
        if len(edge_xs) == 0:
            return []
        
        n_flames = int(len(edge_xs) * self.config.flame_density * 0.35)
        n_flames = max(8, min(n_flames, 50))
        
        np.random.seed(self.config.seed if self.config.seed else 42)
        indices = np.linspace(0, len(edge_xs) - 1, n_flames, dtype=int)
        
        flames = []
        for idx in indices:
            flames.append({
                'x': edge_xs[idx], 'y': edge_ys[idx],
                'phase': np.random.uniform(0, 2 * np.pi),
                'speed': np.random.uniform(0.8, 1.3),
                'height_mult': np.random.uniform(0.7, 1.3),
                'width': np.random.uniform(0.8, 1.4),
            })
        return flames
    
    def _init_embers(self, mask: np.ndarray, h: int, w: int) -> List[dict]:
        if mask.any():
            ys, xs = np.where(mask)
            min_x, max_x = xs.min(), xs.max()
            min_y, max_y = ys.min(), ys.max()
        else:
            min_x, max_x, min_y, max_y = 0, w, 0, h
        
        np.random.seed((self.config.seed if self.config.seed else 42) + 100)
        
        embers = []
        for _ in range(self.config.ember_count):
            embers.append({
                'x': np.random.uniform(min_x, max_x),
                'y': np.random.uniform(min_y, max_y),
                'speed': np.random.uniform(0.6, 1.6),
                'drift': np.random.uniform(-0.8, 0.8),
                'size': np.random.uniform(1.2, 2.5),
                'phase': np.random.uniform(0, 2 * np.pi),
                'life_offset': np.random.uniform(0, 1),
            })
        return embers
    
    def _add_ambient_glow(self, canvas: np.ndarray, cx: int, cy: int, t: float, expand: int) -> np.ndarray:
        result = canvas.copy()
        h, w = canvas.shape[:2]
        
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        
        pulse = 0.8 + 0.2 * np.sin(t * 2 * np.pi * 2)
        max_dist = expand * 1.5
        glow = np.clip(1 - dist / max_dist, 0, 1) * pulse * 0.25 * self.config.glow_intensity
        
        result[:, :, 0] = np.clip(result[:, :, 0] + glow * 70, 0, 255).astype(np.uint8)
        result[:, :, 1] = np.clip(result[:, :, 1] + glow * 25, 0, 255).astype(np.uint8)
        result[:, :, 2] = np.clip(result[:, :, 2] + glow * 5, 0, 255).astype(np.uint8)
        result[:, :, 3] = np.clip(result[:, :, 3] + glow * 90, 0, 255).astype(np.uint8)
        
        return result
    
    def _add_flames(self, canvas: np.ndarray, flames: List[dict], t: float, 
                    expand: int, orig_h: int, orig_w: int) -> np.ndarray:
        result = canvas.copy()
        h, w = canvas.shape[:2]
        phase = t * 2 * np.pi
        
        for flame in flames:
            fx = flame['x'] + expand
            fy = flame['y'] + expand
            
            # Smooth animated height
            noise1 = np.sin(phase * flame['speed'] * 2 + flame['phase'])
            noise2 = np.sin(phase * flame['speed'] * 3.7 + flame['phase'] * 1.3) * 0.4
            height_anim = 0.5 + 0.5 * ease_in_out_sine((noise1 * 0.65 + noise2 * 0.35 + 1) / 2)
            
            flame_len = int(expand * self.config.flame_height * flame['height_mult'] * height_anim)
            flame_len = max(4, flame_len)
            
            for dist in range(flame_len):
                progress = dist / flame_len
                
                # Multi-frequency wobble for organic motion
                wobble = (np.sin(phase * 3 + flame['phase'] + dist * 0.18) * 2.0 +
                          np.sin(phase * 5.3 + flame['phase'] * 0.7 + dist * 0.3) * 1.2 +
                          np.sin(phase * 8 + dist * 0.12) * progress * 1.5)
                wobble *= flame['width'] * (1 + progress * 0.4)
                
                px = int(fx + wobble)
                py = int(fy - dist)
                
                if 0 <= px < w and 0 <= py < h:
                    # HSV gradient: core -> mid -> outer -> tip
                    if progress < 0.15:
                        color = color_lerp_hsv(self.config.color_core, self.config.color_mid, progress / 0.15)
                    elif progress < 0.45:
                        color = color_lerp_hsv(self.config.color_mid, self.config.color_outer, (progress - 0.15) / 0.3)
                    else:
                        color = color_lerp_hsv(self.config.color_outer, self.config.color_tip, (progress - 0.45) / 0.55)
                    
                    # Smooth alpha falloff
                    alpha_curve = 1 - ease_out_quad(progress)
                    flicker = 0.85 + 0.15 * np.sin(phase * 9 + flame['phase'] + dist * 0.25)
                    alpha = int(255 * alpha_curve * flicker * self.config.intensity)
                    
                    base_width = 2 + int((1 - progress) * 2 * flame['width'])
                    
                    for dx in range(-base_width, base_width + 1):
                        for dy in range(-1, 2):
                            ppx, ppy = px + dx, py + dy
                            if 0 <= ppx < w and 0 <= ppy < h:
                                edge_falloff = 1 - (abs(dx) / (base_width + 0.5)) ** 1.5
                                final_alpha = int(alpha * edge_falloff)
                                
                                if final_alpha > 8 and final_alpha > result[ppy, ppx, 3] * 0.5:
                                    blend = final_alpha / 255.0
                                    for c in range(3):
                                        result[ppy, ppx, c] = int(result[ppy, ppx, c] * (1-blend) + color[c] * blend)
                                    result[ppy, ppx, 3] = max(result[ppy, ppx, 3], final_alpha)
        
        return result
    
    def _composite_sprite(self, canvas: np.ndarray, original: np.ndarray, expand: int, h: int, w: int):
        """Composite sprite onto canvas with proper alpha blending."""
        region = canvas[expand:expand+h, expand:expand+w]
        alpha = original[:, :, 3:4] / 255.0
        for c in range(3):
            region[:, :, c] = (original[:, :, c] * alpha[:, :, 0] + 
                              region[:, :, c] * (1 - alpha[:, :, 0])).astype(np.uint8)
        region[:, :, 3] = np.maximum(region[:, :, 3], original[:, :, 3])
    
    def _add_edge_glow(self, canvas: np.ndarray, visible_mask: np.ndarray, t: float) -> np.ndarray:
        result = canvas.copy()
        if not visible_mask.any():
            return result
        
        edges = self._find_edges(visible_mask)
        edge_ys, edge_xs = np.where(edges)
        phase = t * 2 * np.pi
        
        for ey, ex in zip(edge_ys, edge_xs):
            intensity = self.config.glow_intensity * (0.6 + 0.4 * np.sin(phase * 3.5 + ex * 0.07 + ey * 0.05))
            
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    py, px = ey + dy, ex + dx
                    if 0 <= py < canvas.shape[0] and 0 <= px < canvas.shape[1]:
                        dist = np.sqrt(dx*dx + dy*dy)
                        falloff = max(0, 1 - dist / 2.5)
                        
                        if result[py, px, 3] > 0:
                            strength = intensity * falloff * 0.35
                            for c in range(3):
                                blended = result[py, px, c] * (1 - strength) + self.config.color_mid[c] * strength
                                result[py, px, c] = int(np.clip(blended, 0, 255))
        
        return result
    
    def _add_embers(self, canvas: np.ndarray, embers: List[dict], t: float, 
                    expand: int, orig_h: int) -> np.ndarray:
        result = canvas.copy()
        h, w = canvas.shape[:2]
        phase = t * 2 * np.pi
        
        for ember in embers:
            cycle = (t * ember['speed'] + ember['life_offset']) % 1.0
            
            # Smooth acceleration upward
            rise = ease_out_quad(cycle) * expand * 2.2
            
            drift_x = (np.sin(phase * 2 + ember['phase']) * 3.5 +
                       np.sin(phase * 3.5 + ember['phase'] * 0.7) * 2) * ember['drift']
            
            ex = int(ember['x'] + expand + drift_x)
            ey = int(ember['y'] + expand - rise)
            
            # Smooth fade
            fade = 1 - ease_out_quad(cycle)
            brightness = fade * (0.75 + 0.25 * np.sin(phase * 12 + ember['phase']))
            alpha = int(255 * brightness * self.config.intensity)
            
            if 0 <= ex < w and 0 <= ey < h and alpha > 12:
                color = color_lerp_hsv(self.config.color_mid, self.config.color_outer, cycle * 0.8)
                size = ember['size'] * (1 - cycle * 0.25)
                size_int = max(1, int(size))
                
                for dy in range(-size_int, size_int + 1):
                    for dx in range(-size_int, size_int + 1):
                        d = np.sqrt(dx*dx + dy*dy)
                        if d <= size:
                            px, py = ex + dx, ey + dy
                            if 0 <= px < w and 0 <= py < h:
                                falloff = 1 - d / (size + 0.5)
                                final_alpha = int(alpha * falloff)
                                if final_alpha > result[py, px, 3] * 0.35:
                                    result[py, px, :3] = color
                                    result[py, px, 3] = max(result[py, px, 3], final_alpha)
        
        return result
    
    def _apply_shimmer(self, canvas: np.ndarray, visible_mask: np.ndarray, t: float) -> np.ndarray:
        """Very subtle heat shimmer (only if enabled)."""
        result = canvas.copy()
        h, w = canvas.shape[:2]
        if not visible_mask.any():
            return result
        
        phase = t * 2 * np.pi
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        wobble_x = np.sin(y_coords / 5 + phase * 2) * 0.25 * self.config.intensity
        wobble_y = np.cos(x_coords / 6 + phase * 3) * 0.15 * self.config.intensity
        
        src_x = np.clip(x_coords + wobble_x, 0, w - 1)
        src_y = np.clip(y_coords + wobble_y, 0, h - 1)
        
        sampled = PixelMath.bilinear_sample(canvas, src_x, src_y)
        result[visible_mask] = sampled[visible_mask]
        return result


# =============================================================================
# WATER ELEMENT
# =============================================================================

@dataclass
class WaterElementConfig(EffectConfig):
    """Configuration for water element effect."""
    splash_intensity: float = 0.7
    ripple_count: int = 4
    droplet_count: int = 14
    stream_count: int = 8
    flow_speed: float = 1.0
    # Water color palette
    color_highlight: Tuple[int, int, int] = (230, 248, 255)
    color_light: Tuple[int, int, int] = (160, 215, 255)
    color_mid: Tuple[int, int, int] = (90, 170, 245)
    color_dark: Tuple[int, int, int] = (50, 110, 200)
    animate_sprite: bool = False  # OFF by default!


class WaterElement(BaseEffect):
    """Add water elemental effect to sprite."""
    
    name = "water_element"
    description = "Add flowing water, ripples and splash effects around sprite"
    config_class = WaterElementConfig
    
    def __init__(self, config: Optional[WaterElementConfig] = None):
        if config is None:
            config = WaterElementConfig()
        super().__init__(config)
        self.noise = NoiseGenerator(self.config.seed)
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        
        visible_mask = original[:, :, 3] > 10 if original.shape[2] == 4 else np.any(original > 0, axis=2)
        edge_mask = self._find_edges(visible_mask)
        
        droplets = self._init_droplets(visible_mask, h, w)
        streams = self._init_streams(edge_mask)
        
        if visible_mask.any():
            ys, xs = np.where(visible_mask)
            center_x, center_y = np.mean(xs), np.mean(ys)
        else:
            center_x, center_y = w // 2, h // 2
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            
            expand = max(10, int(max(h, w) * 0.35))
            canvas_h = h + expand * 2
            canvas_w = w + expand * 2
            
            canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
            
            # Ripples (behind)
            canvas = self._add_ripples(canvas, center_x + expand, center_y + expand, t, expand)
            
            # Streams
            visible_canvas = np.zeros((canvas_h, canvas_w), dtype=bool)
            visible_canvas[expand:expand+h, expand:expand+w] = visible_mask
            canvas = self._add_streams(canvas, streams, visible_canvas, t, expand)
            
            # Sprite ON TOP
            self._composite_sprite(canvas, original, expand, h, w)
            
            # Water glow
            canvas = self._add_water_glow(canvas, visible_canvas, t)
            
            # Droplets (front)
            canvas = self._add_droplets(canvas, droplets, t, expand, h)
            
            if self.config.animate_sprite:
                canvas = self._apply_wave(canvas, visible_canvas, t)
            
            margin = max(4, expand // 2)
            final = canvas[expand-margin:expand+h+margin, expand-margin:expand+w+margin]
            
            frame = Sprite(
                width=final.shape[1], height=final.shape[0],
                pixels=final, name=f"{sprite.name}_water_{i}",
                source_path=sprite.source_path
            )
            frames.append(frame)
        
        return frames
    
    def _find_edges(self, mask: np.ndarray) -> np.ndarray:
        padded = np.pad(mask.astype(np.uint8), 1, mode='constant', constant_values=0)
        eroded = (padded[:-2, 1:-1] & padded[2:, 1:-1] &
                  padded[1:-1, :-2] & padded[1:-1, 2:] & padded[1:-1, 1:-1])
        return mask & ~eroded
    
    def _init_droplets(self, mask: np.ndarray, h: int, w: int) -> List[dict]:
        if mask.any():
            ys, xs = np.where(mask)
            min_x, max_x = xs.min(), xs.max()
            min_y, max_y = ys.min(), ys.max()
        else:
            min_x, max_x, min_y, max_y = 0, w, 0, h
        
        np.random.seed(self.config.seed if self.config.seed else 42)
        
        droplets = []
        for _ in range(self.config.droplet_count):
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(1.5, 3.5)
            droplets.append({
                'start_x': np.random.uniform(min_x, max_x),
                'start_y': np.random.uniform(min_y, max_y),
                'vx': np.cos(angle) * speed,
                'vy': np.sin(angle) * speed - 1.5,
                'size': np.random.uniform(1.5, 3),
                'life_offset': np.random.uniform(0, 1),
                'phase': np.random.uniform(0, 2 * np.pi),
            })
        return droplets
    
    def _init_streams(self, edge_mask: np.ndarray) -> List[dict]:
        edge_ys, edge_xs = np.where(edge_mask)
        if len(edge_xs) == 0:
            return []
        
        np.random.seed((self.config.seed if self.config.seed else 42) + 200)
        n_streams = min(self.config.stream_count, len(edge_xs))
        indices = np.random.choice(len(edge_xs), n_streams, replace=False)
        
        streams = []
        for idx in indices:
            streams.append({
                'x': edge_xs[idx], 'y': edge_ys[idx],
                'phase': np.random.uniform(0, 2 * np.pi),
                'speed': np.random.uniform(0.7, 1.3),
                'length': np.random.uniform(0.5, 1.1),
            })
        return streams
    
    def _add_ripples(self, canvas: np.ndarray, cx: float, cy: float, t: float, expand: int) -> np.ndarray:
        result = canvas.copy()
        h, w = canvas.shape[:2]
        
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        phase = t * 2 * np.pi
        
        for ring in range(self.config.ripple_count):
            ring_t = (t + ring / self.config.ripple_count) % 1.0
            ring_radius = ring_t * expand * 1.8
            ring_width = 3 + ring * 0.5
            
            ring_dist = np.abs(dist - ring_radius)
            in_ring = ring_dist < ring_width
            
            ring_alpha = np.clip(1 - ring_dist / ring_width, 0, 1)
            ring_alpha *= (1 - ease_out_quad(ring_t)) * self.config.splash_intensity * 0.55
            
            where_ring = np.where(in_ring & (ring_alpha > 0.02))
            for py, px in zip(*where_ring):
                alpha = int(ring_alpha[py, px] * 190)
                angle = np.arctan2(py - cy, px - cx)
                highlight = 0.5 + 0.5 * np.sin(angle * 4 + phase * 2.5)
                color = color_lerp_hsv(self.config.color_mid, self.config.color_light, highlight)
                
                if alpha > result[py, px, 3] * 0.45:
                    result[py, px, :3] = color
                    result[py, px, 3] = max(result[py, px, 3], alpha)
        
        return result
    
    def _add_streams(self, canvas: np.ndarray, streams: List[dict], 
                     visible_mask: np.ndarray, t: float, expand: int) -> np.ndarray:
        result = canvas.copy()
        h, w = canvas.shape[:2]
        phase = t * 2 * np.pi
        
        for stream in streams:
            sx = stream['x'] + expand
            sy = stream['y'] + expand
            
            anim = 0.5 + 0.5 * np.sin(phase * stream['speed'] + stream['phase'])
            stream_len = int(expand * 0.75 * stream['length'] * anim)
            
            for dist in range(max(1, stream_len)):
                progress = dist / max(1, stream_len)
                
                wave = (np.sin(phase * 2 * self.config.flow_speed + stream['phase'] + dist * 0.25) * 2 +
                        np.sin(phase * 3.7 * self.config.flow_speed + dist * 0.18) * 1.2)
                
                px = int(sx + wave * (1 + progress))
                py = int(sy + dist)
                
                if 0 <= px < w and 0 <= py < h and not visible_mask[py, px]:
                    if progress < 0.25:
                        color = color_lerp_hsv(self.config.color_highlight, self.config.color_light, progress / 0.25)
                    else:
                        color = color_lerp_hsv(self.config.color_light, self.config.color_mid, (progress - 0.25) / 0.75)
                    
                    alpha = int(210 * (1 - ease_out_quad(progress) * 0.65) * self.config.splash_intensity)
                    width = max(1, int(2.5 * (1 - progress * 0.4)))
                    
                    for dx in range(-width, width + 1):
                        ppx = px + dx
                        if 0 <= ppx < w:
                            falloff = 1 - abs(dx) / (width + 0.5)
                            final_alpha = int(alpha * falloff)
                            if final_alpha > result[py, ppx, 3] * 0.55:
                                result[py, ppx, :3] = color
                                result[py, ppx, 3] = max(result[py, ppx, 3], final_alpha)
        
        return result
    
    def _composite_sprite(self, canvas: np.ndarray, original: np.ndarray, expand: int, h: int, w: int):
        region = canvas[expand:expand+h, expand:expand+w]
        alpha = original[:, :, 3:4] / 255.0
        for c in range(3):
            region[:, :, c] = (original[:, :, c] * alpha[:, :, 0] + 
                              region[:, :, c] * (1 - alpha[:, :, 0])).astype(np.uint8)
        region[:, :, 3] = np.maximum(region[:, :, 3], original[:, :, 3])
    
    def _add_water_glow(self, canvas: np.ndarray, visible_mask: np.ndarray, t: float) -> np.ndarray:
        result = canvas.copy()
        if not visible_mask.any():
            return result
        
        edges = self._find_edges(visible_mask)
        edge_ys, edge_xs = np.where(edges)
        phase = t * 2 * np.pi
        
        for ey, ex in zip(edge_ys, edge_xs):
            intensity = (0.4 + 0.2 * np.sin(phase * 2.5 + ex * 0.09 + ey * 0.07)) * self.config.intensity
            
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    py, px = ey + dy, ex + dx
                    if 0 <= py < canvas.shape[0] and 0 <= px < canvas.shape[1]:
                        dist = np.sqrt(dx*dx + dy*dy)
                        falloff = max(0, 1 - dist / 2.5)
                        
                        if result[py, px, 3] > 0:
                            strength = intensity * falloff * 0.3
                            for c in range(3):
                                blended = result[py, px, c] * (1 - strength) + self.config.color_light[c] * strength
                                result[py, px, c] = int(np.clip(blended, 0, 255))
        
        return result
    
    def _add_droplets(self, canvas: np.ndarray, droplets: List[dict], t: float, 
                      expand: int, orig_h: int) -> np.ndarray:
        result = canvas.copy()
        h, w = canvas.shape[:2]
        gravity = 2.5
        
        for drop in droplets:
            cycle = (t * self.config.flow_speed + drop['life_offset']) % 1.0
            
            dx = drop['start_x'] + expand + drop['vx'] * cycle * expand * 0.35
            dy = drop['start_y'] + expand + drop['vy'] * cycle * expand * 0.35
            dy += gravity * cycle * cycle * expand * 0.22
            
            dx, dy = int(dx), int(dy)
            
            brightness = (1 - ease_out_quad(cycle) * 0.55) * self.config.splash_intensity
            sparkle = 0.7 + 0.3 * np.sin(t * 18 * np.pi + drop['phase'])
            alpha = int(220 * brightness * sparkle)
            
            if 0 <= dx < w and 0 <= dy < h and alpha > 12:
                size = drop['size'] * (1 - cycle * 0.15)
                color = color_lerp_hsv(self.config.color_highlight, self.config.color_light, 0.25)
                
                size_int = max(1, int(size))
                for ddy in range(-size_int, size_int + 1):
                    for ddx in range(-size_int, size_int + 1):
                        d = np.sqrt(ddx*ddx + ddy*ddy)
                        if d <= size:
                            px, py = dx + ddx, dy + ddy
                            if 0 <= px < w and 0 <= py < h:
                                falloff = 1 - d / (size + 0.5)
                                final_alpha = int(alpha * falloff)
                                if final_alpha > result[py, px, 3] * 0.35:
                                    result[py, px, :3] = color
                                    result[py, px, 3] = max(result[py, px, 3], final_alpha)
        
        return result
    
    def _apply_wave(self, canvas: np.ndarray, visible_mask: np.ndarray, t: float) -> np.ndarray:
        result = canvas.copy()
        h, w = canvas.shape[:2]
        if not visible_mask.any():
            return result
        
        phase = t * 2 * np.pi * self.config.flow_speed
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        wave_x = np.sin(y_coords / 5.5 + phase * 1.5) * 0.35 * self.config.intensity
        wave_y = np.cos(x_coords / 6.5 + phase) * 0.25 * self.config.intensity
        
        src_x = np.clip(x_coords + wave_x, 0, w - 1)
        src_y = np.clip(y_coords + wave_y, 0, h - 1)
        
        sampled = PixelMath.bilinear_sample(canvas, src_x, src_y)
        result[visible_mask] = sampled[visible_mask]
        return result


# =============================================================================
# ICE ELEMENT
# =============================================================================

@dataclass
class IceElementConfig(EffectConfig):
    """Configuration for ice element effect."""
    frost_intensity: float = 0.6
    crystal_count: int = 8
    sparkle_count: int = 18
    shard_count: int = 5
    # Ice color palette
    color_white: Tuple[int, int, int] = (245, 252, 255)
    color_ice: Tuple[int, int, int] = (200, 235, 255)
    color_frost: Tuple[int, int, int] = (140, 195, 245)
    color_deep: Tuple[int, int, int] = (70, 140, 215)
    animate_sprite: bool = False


class IceElement(BaseEffect):
    """Add ice/frost elemental effect to sprite."""
    
    name = "ice_element"
    description = "Add ice crystals, frost and cold effects around sprite"
    config_class = IceElementConfig
    
    def __init__(self, config: Optional[IceElementConfig] = None):
        if config is None:
            config = IceElementConfig()
        super().__init__(config)
        self.noise = NoiseGenerator(self.config.seed)
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        
        visible_mask = original[:, :, 3] > 10 if original.shape[2] == 4 else np.any(original > 0, axis=2)
        edge_mask = self._find_edges(visible_mask)
        
        np.random.seed(self.config.seed if self.config.seed else 42)
        
        if visible_mask.any():
            ys, xs = np.where(visible_mask)
            bounds = (xs.min(), xs.max(), ys.min(), ys.max())
            center = (np.mean(xs), np.mean(ys))
        else:
            bounds = (0, w, 0, h)
            center = (w // 2, h // 2)
        
        crystals = self._init_crystals(edge_mask, bounds)
        sparkles = self._init_sparkles(bounds)
        shards = self._init_shards(center)
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            
            expand = max(8, int(max(h, w) * 0.25))
            canvas_h = h + expand * 2
            canvas_w = w + expand * 2
            
            canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
            
            # Frost aura
            canvas = self._add_frost_aura(canvas, center[0] + expand, center[1] + expand, t, expand)
            
            # Crystals
            canvas = self._add_crystals(canvas, crystals, t, expand)
            
            # Sprite
            self._composite_sprite(canvas, original, expand, h, w)
            
            visible_canvas = np.zeros((canvas_h, canvas_w), dtype=bool)
            visible_canvas[expand:expand+h, expand:expand+w] = visible_mask
            
            # Frost glow
            canvas = self._add_frost_glow(canvas, visible_canvas, t)
            
            # Ice tint
            canvas = self._apply_ice_tint(canvas, visible_canvas)
            
            # Shards
            canvas = self._add_shards(canvas, shards, t, expand)
            
            # Sparkles
            canvas = self._add_sparkles(canvas, sparkles, t, expand)
            
            margin = max(4, expand // 2)
            final = canvas[expand-margin:expand+h+margin, expand-margin:expand+w+margin]
            
            frame = Sprite(
                width=final.shape[1], height=final.shape[0],
                pixels=final, name=f"{sprite.name}_ice_{i}",
                source_path=sprite.source_path
            )
            frames.append(frame)
        
        return frames
    
    def _find_edges(self, mask: np.ndarray) -> np.ndarray:
        padded = np.pad(mask.astype(np.uint8), 1, mode='constant', constant_values=0)
        eroded = (padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:] & padded[1:-1, 1:-1])
        return mask & ~eroded
    
    def _init_crystals(self, edge_mask: np.ndarray, bounds: tuple) -> List[dict]:
        edge_ys, edge_xs = np.where(edge_mask)
        crystals = []
        
        if len(edge_xs) > 0:
            n = min(self.config.crystal_count, len(edge_xs))
            indices = np.random.choice(len(edge_xs), n, replace=False)
            
            for idx in indices:
                crystals.append({
                    'x': edge_xs[idx], 'y': edge_ys[idx],
                    'angle': np.random.uniform(-40, 40),
                    'size': np.random.uniform(4, 9),
                    'phase': np.random.uniform(0, 2 * np.pi),
                })
        return crystals
    
    def _init_sparkles(self, bounds: tuple) -> List[dict]:
        min_x, max_x, min_y, max_y = bounds
        sparkles = []
        
        for _ in range(self.config.sparkle_count):
            sparkles.append({
                'x': np.random.uniform(min_x - 4, max_x + 4),
                'y': np.random.uniform(min_y - 4, max_y + 4),
                'phase': np.random.uniform(0, 2 * np.pi),
                'speed': np.random.uniform(1.2, 2.8),
                'size': np.random.uniform(1, 2),
            })
        return sparkles
    
    def _init_shards(self, center: tuple) -> List[dict]:
        cx, cy = center
        shards = []
        
        for _ in range(self.config.shard_count):
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(6, 14)
            shards.append({
                'base_x': cx + np.cos(angle) * dist,
                'base_y': cy + np.sin(angle) * dist,
                'orbit_radius': np.random.uniform(3, 7),
                'orbit_speed': np.random.uniform(0.4, 1.2),
                'phase': angle,
                'size': np.random.uniform(2.5, 5),
            })
        return shards
    
    def _add_frost_aura(self, canvas: np.ndarray, cx: float, cy: float, t: float, expand: int) -> np.ndarray:
        result = canvas.copy()
        h, w = canvas.shape[:2]
        
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        
        pulse = 0.85 + 0.15 * np.sin(t * 2 * np.pi * 1.5)
        max_dist = expand * 1.7
        glow = np.clip(1 - dist / max_dist, 0, 1) * pulse * 0.22 * self.config.frost_intensity
        
        result[:, :, 0] = np.clip(result[:, :, 0] + glow * 35, 0, 255).astype(np.uint8)
        result[:, :, 1] = np.clip(result[:, :, 1] + glow * 55, 0, 255).astype(np.uint8)
        result[:, :, 2] = np.clip(result[:, :, 2] + glow * 85, 0, 255).astype(np.uint8)
        result[:, :, 3] = np.clip(result[:, :, 3] + glow * 75, 0, 255).astype(np.uint8)
        
        return result
    
    def _add_crystals(self, canvas: np.ndarray, crystals: List[dict], t: float, expand: int) -> np.ndarray:
        result = canvas.copy()
        h, w = canvas.shape[:2]
        phase = t * 2 * np.pi
        
        for crystal in crystals:
            cx = int(crystal['x'] + expand)
            cy = int(crystal['y'] + expand)
            
            size_anim = 0.85 + 0.15 * np.sin(phase * 2 + crystal['phase'])
            size = int(crystal['size'] * size_anim)
            
            # Hexagonal crystal
            for i in range(-size, size + 1):
                progress = abs(i) / (size + 0.1)
                width = int(size * (1 - progress * 0.65))
                
                for j in range(-width, width + 1):
                    px, py = cx + j, cy + i
                    if 0 <= px < w and 0 <= py < h:
                        d = np.sqrt(i*i + j*j) / (size + 0.1)
                        
                        if d < 0.25:
                            color = color_lerp_hsv(self.config.color_white, self.config.color_ice, d / 0.25)
                        elif d < 0.6:
                            color = color_lerp_hsv(self.config.color_ice, self.config.color_frost, (d - 0.25) / 0.35)
                        else:
                            color = color_lerp_hsv(self.config.color_frost, self.config.color_deep, (d - 0.6) / 0.4)
                        
                        alpha = int(210 * (1 - d * 0.35) * self.config.frost_intensity)
                        
                        if alpha > result[py, px, 3] * 0.45:
                            result[py, px, :3] = color
                            result[py, px, 3] = max(result[py, px, 3], alpha)
        
        return result
    
    def _composite_sprite(self, canvas: np.ndarray, original: np.ndarray, expand: int, h: int, w: int):
        region = canvas[expand:expand+h, expand:expand+w]
        alpha = original[:, :, 3:4] / 255.0
        for c in range(3):
            region[:, :, c] = (original[:, :, c] * alpha[:, :, 0] + 
                              region[:, :, c] * (1 - alpha[:, :, 0])).astype(np.uint8)
        region[:, :, 3] = np.maximum(region[:, :, 3], original[:, :, 3])
    
    def _add_frost_glow(self, canvas: np.ndarray, visible_mask: np.ndarray, t: float) -> np.ndarray:
        result = canvas.copy()
        edges = self._find_edges(visible_mask)
        edge_ys, edge_xs = np.where(edges)
        phase = t * 2 * np.pi
        
        for ey, ex in zip(edge_ys, edge_xs):
            intensity = (0.4 + 0.2 * np.sin(phase * 2 + ex * 0.07 + ey * 0.05)) * self.config.frost_intensity
            
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    py, px = ey + dy, ex + dx
                    if 0 <= py < canvas.shape[0] and 0 <= px < canvas.shape[1]:
                        dist = np.sqrt(dx*dx + dy*dy)
                        falloff = max(0, 1 - dist / 2.5)
                        
                        if result[py, px, 3] > 0:
                            strength = intensity * falloff * 0.28
                            for c in range(3):
                                blended = result[py, px, c] * (1 - strength) + self.config.color_frost[c] * strength
                                result[py, px, c] = int(np.clip(blended, 0, 255))
        
        return result
    
    def _add_shards(self, canvas: np.ndarray, shards: List[dict], t: float, expand: int) -> np.ndarray:
        result = canvas.copy()
        h, w = canvas.shape[:2]
        phase = t * 2 * np.pi
        
        for shard in shards:
            orbit_angle = phase * shard['orbit_speed'] + shard['phase']
            sx = shard['base_x'] + expand + np.cos(orbit_angle) * shard['orbit_radius']
            sy = shard['base_y'] + expand + np.sin(orbit_angle) * shard['orbit_radius'] * 0.5
            sy += np.sin(phase * 2 + shard['phase']) * 1.8
            
            sx, sy = int(sx), int(sy)
            size = int(shard['size'])
            
            if 0 <= sx < w and 0 <= sy < h:
                for i in range(-size, size + 1):
                    width = size - abs(i)
                    for j in range(-width, width + 1):
                        px, py = sx + j, sy + i
                        if 0 <= px < w and 0 <= py < h:
                            d = (abs(i) + abs(j)) / (size + 0.1)
                            alpha = int(190 * (1 - d * 0.45) * self.config.frost_intensity)
                            color = color_lerp_hsv(self.config.color_ice, self.config.color_frost, d)
                            
                            if alpha > result[py, px, 3] * 0.35:
                                result[py, px, :3] = color
                                result[py, px, 3] = max(result[py, px, 3], alpha)
        
        return result
    
    def _add_sparkles(self, canvas: np.ndarray, sparkles: List[dict], t: float, expand: int) -> np.ndarray:
        result = canvas.copy()
        h, w = canvas.shape[:2]
        
        for sparkle in sparkles:
            twinkle = np.sin(t * 2 * np.pi * sparkle['speed'] + sparkle['phase'])
            brightness = max(0, twinkle) ** 1.5  # Sharper twinkle
            
            if brightness > 0.15:
                sx = int(sparkle['x'] + expand)
                sy = int(sparkle['y'] + expand)
                
                if 0 <= sx < w and 0 <= sy < h:
                    alpha = int(255 * brightness * self.config.frost_intensity)
                    size = int(sparkle['size'] * (0.5 + brightness * 0.5))
                    
                    for d in range(-size, size + 1):
                        falloff = 1 - abs(d) / (size + 0.5)
                        a = int(alpha * falloff)
                        
                        if 0 <= sx + d < w and a > result[sy, sx + d, 3] * 0.4:
                            result[sy, sx + d, :3] = self.config.color_white
                            result[sy, sx + d, 3] = max(result[sy, sx + d, 3], a)
                        
                        if 0 <= sy + d < h and a > result[sy + d, sx, 3] * 0.4:
                            result[sy + d, sx, :3] = self.config.color_white
                            result[sy + d, sx, 3] = max(result[sy + d, sx, 3], a)
        
        return result
    
    def _apply_ice_tint(self, canvas: np.ndarray, visible_mask: np.ndarray) -> np.ndarray:
        result = canvas.copy()
        if not visible_mask.any():
            return result
        
        tint = 0.1 * self.config.frost_intensity
        
        for c in range(3):
            channel = result[visible_mask, c].astype(np.float32)
            tinted = channel * (1 - tint) + self.config.color_frost[c] * tint
            result[visible_mask, c] = np.clip(tinted, 0, 255).astype(np.uint8)
        
        return result
