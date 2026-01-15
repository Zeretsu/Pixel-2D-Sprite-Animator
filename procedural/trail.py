"""
Trail Effects - Afterimages, motion trails, and ghost effects.

Creates visual trails following sprite movement or simulated motion.
Supports: afterimages, ghosts, echo trails, ribbon trails, smear frames.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum, auto

from .base import BaseEffect, EffectConfig, Easing


class TrailStyle(Enum):
    """Different trail rendering styles."""
    AFTERIMAGE = auto()    # Fading copies behind sprite
    GHOST = auto()         # Semi-transparent duplicates
    ECHO = auto()          # Outlined echoes
    RIBBON = auto()        # Smooth ribbon trail
    SMEAR = auto()         # Smear frame effect
    STROBOSCOPIC = auto()  # Multiple solid copies


@dataclass
class TrailConfig(EffectConfig):
    """Configuration for trail effects."""
    style: str = "afterimage"  # Trail style
    trail_count: int = 5  # Number of trail copies
    spacing: float = 0.15  # Spacing between trails (as fraction of frame)
    
    # Fading
    fade_start: float = 0.8  # Initial opacity of first trail
    fade_end: float = 0.1  # Final opacity of last trail
    fade_curve: str = "linear"  # "linear", "ease_out", "ease_in"
    
    # Movement simulation
    direction: float = 0.0  # Movement direction (degrees)
    speed: float = 3.0  # Movement speed (pixels per trail)
    follow_animation: bool = True  # Trail follows actual animation motion
    
    # Color effects
    color_shift: bool = False  # Shift hue along trail
    hue_shift_amount: float = 30.0  # Degrees of hue shift per trail
    tint_color: Optional[Tuple[int, int, int]] = None  # Tint trails with color
    desaturate: float = 0.0  # Desaturate trails (0-1)
    
    # Scale effects
    scale_trails: bool = False  # Scale trails
    scale_start: float = 1.0  # Scale of first trail
    scale_end: float = 0.8  # Scale of last trail
    
    # Ghost specific
    ghost_outline_only: bool = False  # For ghost style
    outline_thickness: int = 1
    
    # Smear specific
    smear_stretch: float = 2.0  # Stretch factor for smear
    
    # Ribbon specific
    ribbon_width: float = 1.0  # Width multiplier for ribbon
    
    # Advanced
    blend_mode: str = "normal"  # "normal", "add", "screen"
    only_moving_pixels: bool = False  # Only trail pixels that moved
    
    seed: Optional[int] = None


class TrailEffect(BaseEffect):
    """Trail and afterimage effect."""
    
    name = "trail"
    description = "Afterimages, ghosts, and motion trails"
    
    config_class = TrailConfig
    
    def __init__(self, config: TrailConfig):
        super().__init__(config)
        self._frame_history: List[np.ndarray] = []
        self._position_history: List[Tuple[float, float]] = []
        self._max_history = config.trail_count + 2
    
    def _get_sprite_center(self, image: np.ndarray) -> Tuple[float, float]:
        """Calculate sprite center from alpha."""
        if image.shape[2] == 4:
            mask = image[:, :, 3] > 128
        else:
            mask = np.any(image > 0, axis=2)
        
        if not np.any(mask):
            h, w = image.shape[:2]
            return w / 2, h / 2
        
        ys, xs = np.where(mask)
        return xs.mean(), ys.mean()
    
    def _calculate_fade(self, trail_idx: int) -> float:
        """Calculate opacity for trail at given index."""
        t = trail_idx / max(1, self.config.trail_count - 1)
        
        if self.config.fade_curve == "ease_out":
            t = 1 - (1 - t) ** 2
        elif self.config.fade_curve == "ease_in":
            t = t ** 2
        
        opacity = self.config.fade_start + (self.config.fade_end - self.config.fade_start) * t
        return max(0, min(1, opacity))
    
    def _shift_hue(self, image: np.ndarray, shift: float) -> np.ndarray:
        """Shift hue of image by given degrees."""
        result = image.astype(np.float32)
        
        # Simple hue rotation in RGB space
        # Convert shift to radians
        angle = np.radians(shift)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Hue rotation matrix
        r = result[:, :, 0]
        g = result[:, :, 1]
        b = result[:, :, 2]
        
        # Apply rotation in YIQ-like space
        y = 0.299 * r + 0.587 * g + 0.114 * b
        i = 0.596 * r - 0.274 * g - 0.322 * b
        q = 0.211 * r - 0.523 * g + 0.312 * b
        
        # Rotate I and Q
        new_i = i * cos_a - q * sin_a
        new_q = i * sin_a + q * cos_a
        
        # Convert back to RGB
        result[:, :, 0] = y + 0.956 * new_i + 0.621 * new_q
        result[:, :, 1] = y - 0.272 * new_i - 0.647 * new_q
        result[:, :, 2] = y - 1.106 * new_i + 1.703 * new_q
        
        return np.clip(result, 0, 255)
    
    def _desaturate(self, image: np.ndarray, amount: float) -> np.ndarray:
        """Desaturate image by given amount (0-1)."""
        if amount <= 0:
            return image
        
        result = image.astype(np.float32)
        gray = 0.299 * result[:, :, 0] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 2]
        
        for c in range(3):
            result[:, :, c] = result[:, :, c] * (1 - amount) + gray * amount
        
        return result
    
    def _apply_tint(self, image: np.ndarray, tint: Tuple[int, int, int]) -> np.ndarray:
        """Apply color tint to image."""
        result = image.astype(np.float32)
        
        for c in range(3):
            result[:, :, c] = result[:, :, c] * (tint[c] / 255.0)
        
        return result
    
    def _extract_outline(self, image: np.ndarray, thickness: int = 1) -> np.ndarray:
        """Extract outline from sprite."""
        h, w = image.shape[:2]
        
        if image.shape[2] == 4:
            mask = image[:, :, 3] > 128
        else:
            mask = np.any(image > 0, axis=2)
        
        # Find edges
        outline = np.zeros_like(mask)
        
        for dy in range(-thickness, thickness + 1):
            for dx in range(-thickness, thickness + 1):
                if dx == 0 and dy == 0:
                    continue
                
                shifted = np.zeros_like(mask)
                
                src_y0 = max(0, -dy)
                src_y1 = min(h, h - dy)
                src_x0 = max(0, -dx)
                src_x1 = min(w, w - dx)
                
                dst_y0 = max(0, dy)
                dst_y1 = min(h, h + dy)
                dst_x0 = max(0, dx)
                dst_x1 = min(w, w + dx)
                
                shifted[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
                
                # Edge is where mask differs from shifted
                outline |= (mask != shifted) & mask
        
        # Create outline image
        result = np.zeros_like(image)
        result[:, :, :3] = image[:, :, :3]
        
        if image.shape[2] == 4:
            result[:, :, 3] = outline.astype(np.uint8) * 255
        
        return result
    
    def _scale_image(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Scale image around its center."""
        if abs(scale - 1.0) < 0.01:
            return image
        
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        if new_h <= 0 or new_w <= 0:
            return np.zeros_like(image)
        
        # Create scaled image
        result = np.zeros_like(image)
        
        # Simple nearest-neighbor scaling
        yy, xx = np.mgrid[0:h, 0:w]
        
        # Map to source coordinates
        cy, cx = h / 2, w / 2
        src_x = (xx - cx) / scale + cx
        src_y = (yy - cy) / scale + cy
        
        # Sample
        valid = (src_x >= 0) & (src_x < w - 1) & (src_y >= 0) & (src_y < h - 1)
        
        src_x_int = np.clip(src_x.astype(np.int32), 0, w - 1)
        src_y_int = np.clip(src_y.astype(np.int32), 0, h - 1)
        
        for c in range(image.shape[2]):
            result[:, :, c] = np.where(valid, image[src_y_int, src_x_int, c], 0)
        
        return result
    
    def _offset_image(self, image: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """Offset image by given amount."""
        h, w = image.shape[:2]
        result = np.zeros_like(image)
        
        # Integer offsets
        idx, idy = int(dx), int(dy)
        
        src_x0 = max(0, -idx)
        src_x1 = min(w, w - idx)
        src_y0 = max(0, -idy)
        src_y1 = min(h, h - idy)
        
        dst_x0 = max(0, idx)
        dst_x1 = min(w, w + idx)
        dst_y0 = max(0, idy)
        dst_y1 = min(h, h + idy)
        
        if src_x1 > src_x0 and src_y1 > src_y0:
            result[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
        
        return result
    
    def _stretch_image(self, image: np.ndarray, direction: float, stretch: float) -> np.ndarray:
        """Stretch image in given direction."""
        if abs(stretch - 1.0) < 0.01:
            return image
        
        h, w = image.shape[:2]
        result = np.zeros_like(image, dtype=np.float32)
        
        rad = np.radians(direction)
        dx = np.cos(rad)
        dy = np.sin(rad)
        
        # Sample along stretch direction
        samples = int(abs(stretch - 1.0) * 10) + 1
        
        for i in range(samples):
            t = i / max(1, samples - 1)
            offset = (stretch - 1.0) * t
            
            ox = dx * offset * 5
            oy = dy * offset * 5
            
            shifted = self._offset_image(image, ox, oy)
            result += shifted.astype(np.float32) / samples
        
        return result.astype(np.uint8)
    
    def _blend_images(self, base: np.ndarray, overlay: np.ndarray, opacity: float) -> np.ndarray:
        """Blend overlay onto base with given opacity and blend mode."""
        if opacity <= 0:
            return base
        
        base_f = base.astype(np.float32)
        over_f = overlay.astype(np.float32)
        
        # Get alpha mask from overlay
        if overlay.shape[2] == 4:
            alpha = over_f[:, :, 3:4] / 255.0 * opacity
        else:
            alpha = (np.any(overlay > 0, axis=2, keepdims=True)).astype(np.float32) * opacity
        
        blend = self.config.blend_mode.lower()
        
        if blend == "add":
            result = base_f + over_f * alpha
        elif blend == "screen":
            result = 255 - (255 - base_f) * (255 - over_f * alpha) / 255
        else:  # normal
            result = base_f * (1 - alpha) + over_f * alpha
        
        # Handle alpha channel
        if base.shape[2] == 4:
            # Composite alpha
            base_alpha = base_f[:, :, 3:4] / 255.0
            over_alpha = alpha
            result_alpha = base_alpha + over_alpha * (1 - base_alpha)
            result[:, :, 3:4] = result_alpha * 255
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _render_afterimage(self, image: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        """Render afterimage style trails."""
        result = np.zeros_like(image, dtype=np.float32)
        
        # Calculate movement direction
        if self.config.follow_animation and len(self._position_history) >= 2:
            dx = self._position_history[-1][0] - self._position_history[0][0]
            dy = self._position_history[-1][1] - self._position_history[0][1]
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                direction = np.degrees(np.arctan2(dy, dx))
            else:
                direction = self.config.direction
        else:
            direction = self.config.direction
        
        rad = np.radians(direction + 180)  # Trails go opposite to movement
        
        # Render trails from back to front
        for i in range(self.config.trail_count - 1, -1, -1):
            opacity = self._calculate_fade(i)
            offset = (i + 1) * self.config.speed
            
            # Create trail copy
            trail = image.copy()
            
            # Apply color effects
            if self.config.color_shift:
                trail = self._shift_hue(trail, self.config.hue_shift_amount * i)
            
            if self.config.desaturate > 0:
                trail = self._desaturate(trail, self.config.desaturate * (i / self.config.trail_count))
            
            if self.config.tint_color:
                tint_amount = i / self.config.trail_count
                tinted = self._apply_tint(trail, self.config.tint_color)
                trail = trail * (1 - tint_amount) + tinted * tint_amount
            
            # Apply scale
            if self.config.scale_trails:
                t = i / max(1, self.config.trail_count - 1)
                scale = self.config.scale_start + (self.config.scale_end - self.config.scale_start) * t
                trail = self._scale_image(trail.astype(np.uint8), scale)
            
            # Offset trail
            ox = np.cos(rad) * offset
            oy = np.sin(rad) * offset
            trail = self._offset_image(trail.astype(np.uint8), ox, oy)
            
            # Blend trail
            result = self._blend_images(result.astype(np.uint8), trail, opacity).astype(np.float32)
        
        # Add current frame on top
        result = self._blend_images(result.astype(np.uint8), image, 1.0)
        
        return result.astype(np.uint8)
    
    def _render_ghost(self, image: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        """Render ghost style trails."""
        result = np.zeros_like(image, dtype=np.float32)
        
        rad = np.radians(self.config.direction + 180)
        
        for i in range(self.config.trail_count - 1, -1, -1):
            opacity = self._calculate_fade(i)
            offset = (i + 1) * self.config.speed
            
            # Create ghost copy
            if self.config.ghost_outline_only:
                ghost = self._extract_outline(image, self.config.outline_thickness)
            else:
                ghost = image.copy()
            
            # Apply color effects
            if self.config.color_shift:
                ghost = self._shift_hue(ghost, self.config.hue_shift_amount * i)
            
            # Make ghostly (desaturate and tint)
            ghost = self._desaturate(ghost, 0.5)
            
            if self.config.tint_color:
                ghost = self._apply_tint(ghost, self.config.tint_color)
            else:
                # Default ghostly blue tint
                ghost = self._apply_tint(ghost, (200, 220, 255))
            
            # Offset
            ox = np.cos(rad) * offset
            oy = np.sin(rad) * offset
            ghost = self._offset_image(ghost.astype(np.uint8), ox, oy)
            
            # Blend with additive for glowy effect
            old_blend = self.config.blend_mode
            self.config.blend_mode = "add"
            result = self._blend_images(result.astype(np.uint8), ghost, opacity * 0.7).astype(np.float32)
            self.config.blend_mode = old_blend
        
        # Add current frame
        result = self._blend_images(result.astype(np.uint8), image, 1.0)
        
        return result.astype(np.uint8)
    
    def _render_echo(self, image: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        """Render echo/outline style trails."""
        result = np.zeros_like(image, dtype=np.float32)
        
        rad = np.radians(self.config.direction + 180)
        
        for i in range(self.config.trail_count - 1, -1, -1):
            opacity = self._calculate_fade(i)
            offset = (i + 1) * self.config.speed
            
            # Extract outline
            echo = self._extract_outline(image, self.config.outline_thickness + i)
            
            # Color shift for rainbow effect
            if self.config.color_shift:
                echo = self._shift_hue(echo, self.config.hue_shift_amount * i)
            
            # Offset
            ox = np.cos(rad) * offset
            oy = np.sin(rad) * offset
            echo = self._offset_image(echo, ox, oy)
            
            # Blend
            result = self._blend_images(result.astype(np.uint8), echo, opacity).astype(np.float32)
        
        # Add current frame
        result = self._blend_images(result.astype(np.uint8), image, 1.0)
        
        return result.astype(np.uint8)
    
    def _render_smear(self, image: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        """Render smear frame effect."""
        result = image.copy().astype(np.float32)
        
        # Calculate stretch direction from animation
        if len(self._position_history) >= 2:
            dx = self._position_history[-1][0] - self._position_history[0][0]
            dy = self._position_history[-1][1] - self._position_history[0][1]
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                direction = np.degrees(np.arctan2(dy, dx))
            else:
                direction = self.config.direction
        else:
            direction = self.config.direction
        
        # Apply smear stretch
        smeared = self._stretch_image(image, direction, self.config.smear_stretch)
        
        # Blend smear behind current frame
        result = self._blend_images(smeared, image, 0.8)
        
        return result.astype(np.uint8)
    
    def _render_stroboscopic(self, image: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        """Render stroboscopic (multiple solid copies) effect."""
        result = np.zeros_like(image, dtype=np.float32)
        
        rad = np.radians(self.config.direction + 180)
        
        # Render multiple solid copies
        for i in range(self.config.trail_count - 1, -1, -1):
            offset = (i + 1) * self.config.speed * 1.5  # Larger spacing
            
            trail = image.copy()
            
            # Apply scale
            if self.config.scale_trails:
                t = i / max(1, self.config.trail_count - 1)
                scale = self.config.scale_start + (self.config.scale_end - self.config.scale_start) * t
                trail = self._scale_image(trail, scale)
            
            # Offset
            ox = np.cos(rad) * offset
            oy = np.sin(rad) * offset
            trail = self._offset_image(trail, ox, oy)
            
            # Solid blend (no fade)
            result = self._blend_images(result.astype(np.uint8), trail, 1.0).astype(np.float32)
        
        # Add current frame
        result = self._blend_images(result.astype(np.uint8), image, 1.0)
        
        return result.astype(np.uint8)
    
    def process_frame(self, image: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        """Apply trail effect to frame."""
        # Track position history
        center = self._get_sprite_center(image)
        self._position_history.append(center)
        if len(self._position_history) > self._max_history:
            self._position_history.pop(0)
        
        # Store frame history
        self._frame_history.append(image.copy())
        if len(self._frame_history) > self._max_history:
            self._frame_history.pop(0)
        
        # Render based on style
        style = self.config.style.lower()
        
        if style == "ghost":
            return self._render_ghost(image, frame_idx, total_frames)
        elif style == "echo":
            return self._render_echo(image, frame_idx, total_frames)
        elif style == "smear":
            return self._render_smear(image, frame_idx, total_frames)
        elif style == "stroboscopic":
            return self._render_stroboscopic(image, frame_idx, total_frames)
        else:  # afterimage (default)
            return self._render_afterimage(image, frame_idx, total_frames)
    
    def apply(self, sprite) -> list:
        """Apply trail effect to sprite and return animation frames."""
        from src.core import Sprite
        
        frames = []
        for i in range(self.config.frame_count):
            pixels = self.process_frame(sprite.pixels.copy(), i, self.config.frame_count)
            frame = Sprite(
                width=sprite.width,
                height=sprite.height,
                pixels=pixels,
                name=f"{sprite.name}_trail_{i}",
                source_path=sprite.source_path
            )
            frames.append(frame)
        return frames


# Additional trail-related effects

@dataclass
class RibbonTrailConfig(EffectConfig):
    """Configuration for ribbon trail effect."""
    width: float = 3.0
    length: int = 10
    color_start: Tuple[int, int, int, int] = (255, 200, 100, 255)
    color_end: Tuple[int, int, int, int] = (255, 100, 50, 0)
    taper: bool = True
    smooth: bool = True
    glow: bool = False
    seed: Optional[int] = None


class RibbonTrail(BaseEffect):
    """Ribbon/stream trail effect."""
    
    name = "ribbon_trail"
    description = "Smooth ribbon trail following movement"
    
    config_class = RibbonTrailConfig
    
    def __init__(self, config: RibbonTrailConfig):
        super().__init__(config)
        self._points: List[Tuple[float, float]] = []
    
    def _get_sprite_tip(self, image: np.ndarray) -> Tuple[float, float]:
        """Get a tip point from sprite (e.g., sword tip)."""
        if image.shape[2] == 4:
            mask = image[:, :, 3] > 128
        else:
            mask = np.any(image > 0, axis=2)
        
        if not np.any(mask):
            h, w = image.shape[:2]
            return w / 2, h / 2
        
        # Find rightmost/topmost point (good for swords, wands)
        ys, xs = np.where(mask)
        
        # Score points by distance from center-bottom
        h, w = image.shape[:2]
        scores = (xs - w/2)**2 + (h - ys)**2
        best_idx = np.argmax(scores)
        
        return xs[best_idx], ys[best_idx]
    
    def _draw_ribbon(self, canvas: np.ndarray, points: List[Tuple[float, float]]) -> None:
        """Draw smooth ribbon through points."""
        if len(points) < 2:
            return
        
        h, w = canvas.shape[:2]
        
        for i in range(len(points) - 1):
            t = i / max(1, len(points) - 2)
            
            # Interpolate color
            color = np.array([
                self.config.color_start[j] + (self.config.color_end[j] - self.config.color_start[j]) * t
                for j in range(4)
            ], dtype=np.float32)
            
            # Width taper
            if self.config.taper:
                width = self.config.width * (1 - t * 0.8)
            else:
                width = self.config.width
            
            # Draw segment
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            
            # Calculate perpendicular
            dx, dy = x1 - x0, y1 - y0
            length = np.sqrt(dx*dx + dy*dy) + 0.001
            nx, ny = -dy/length, dx/length
            
            # Draw thick line segment
            for offset in np.linspace(-width/2, width/2, int(width * 2) + 1):
                for step in np.linspace(0, 1, int(length) + 1):
                    px = x0 + dx * step + nx * offset
                    py = y0 + dy * step + ny * offset
                    
                    ix, iy = int(px), int(py)
                    if 0 <= ix < w and 0 <= iy < h:
                        alpha = color[3] / 255.0 * (1 - abs(offset) / (width/2 + 0.1))
                        
                        if self.config.glow:
                            for c in range(3):
                                canvas[iy, ix, c] = min(255, canvas[iy, ix, c] + color[c] * alpha)
                        else:
                            for c in range(3):
                                canvas[iy, ix, c] = canvas[iy, ix, c] * (1 - alpha) + color[c] * alpha
    
    def process_frame(self, image: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        """Apply ribbon trail effect."""
        # Track tip position
        tip = self._get_sprite_tip(image)
        self._points.append(tip)
        
        if len(self._points) > self.config.length:
            self._points.pop(0)
        
        # Draw ribbon
        result = image.astype(np.float32)
        
        if len(self._points) >= 2:
            self._draw_ribbon(result, self._points)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def apply(self, sprite) -> list:
        """Apply ribbon trail effect to sprite and return animation frames."""
        from src.core import Sprite
        
        frames = []
        for i in range(self.config.frame_count):
            pixels = self.process_frame(sprite.pixels.copy(), i, self.config.frame_count)
            frame = Sprite(
                width=sprite.width,
                height=sprite.height,
                pixels=pixels,
                name=f"{sprite.name}_ribbon_{i}",
                source_path=sprite.source_path
            )
            frames.append(frame)
        return frames
