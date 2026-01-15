"""
Motion Blur - Professional quality per-pixel motion blur.

Calculates pixel movement between frames and applies directional blur.
Supports: linear motion, radial motion, zoom blur, directional blur.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum, auto

from .base import BaseEffect, EffectConfig, Easing


class BlurType(Enum):
    """Types of motion blur."""
    LINEAR = auto()      # Standard directional blur
    RADIAL = auto()      # Spin/rotation blur from center
    ZOOM = auto()        # Zoom in/out blur from center
    DIRECTIONAL = auto() # Fixed direction blur
    MOTION = auto()      # Per-pixel motion-based blur


@dataclass 
class MotionBlurConfig(EffectConfig):
    """Configuration for motion blur effect."""
    blur_type: str = "radial"  # "linear", "radial", "zoom", "directional", "motion"
    intensity: float = 1.0  # Overall blur strength (0-2)
    samples: int = 32  # Number of blur samples (more = smoother)
    direction: float = 0.0  # Direction in degrees (for directional blur)
    falloff: str = "gaussian"  # "linear", "gaussian", "constant"
    
    # Motion tracking
    motion_scale: float = 1.0  # Scale factor for detected motion
    motion_threshold: float = 0.1  # Minimum motion to trigger blur
    
    # Radial/Zoom settings
    center_x: float = 0.5  # Center X (0-1, relative to image)
    center_y: float = 0.5  # Center Y (0-1, relative to image)
    auto_center: bool = True  # Auto-detect sprite center
    
    # Advanced
    preserve_alpha: bool = True
    blur_background: bool = False
    temporal_blend: float = 0.0  # Blend with previous frame (0-1)
    
    seed: Optional[int] = None


class MotionBlur(BaseEffect):
    """Motion blur effect with multiple blur modes."""
    
    name = "motion_blur"
    description = "Professional per-pixel motion blur with multiple modes"
    
    config_class = MotionBlurConfig
    
    def __init__(self, config: MotionBlurConfig):
        super().__init__(config)
        self._prev_frame: Optional[np.ndarray] = None
        self._motion_field: Optional[np.ndarray] = None
        self._sprite_center: Optional[Tuple[float, float]] = None
    
    def _find_sprite_center(self, image: np.ndarray) -> Tuple[float, float]:
        """Find the center of the sprite based on non-transparent pixels."""
        h, w = image.shape[:2]
        
        if image.shape[2] == 4:
            alpha = image[:, :, 3]
            mask = alpha > 10
        else:
            mask = np.any(image > 0, axis=2)
        
        if not mask.any():
            return w / 2, h / 2
        
        # Find centroid of visible pixels
        yy, xx = np.where(mask)
        cx = np.mean(xx)
        cy = np.mean(yy)
        
        return cx, cy
        self._motion_field: Optional[np.ndarray] = None
    
    def _calculate_motion_field(self, current: np.ndarray, previous: np.ndarray) -> np.ndarray:
        """Calculate per-pixel motion vectors between frames."""
        h, w = current.shape[:2]
        
        # Convert to grayscale for motion detection
        if current.shape[2] >= 3:
            curr_gray = 0.299 * current[:, :, 0] + 0.587 * current[:, :, 1] + 0.114 * current[:, :, 2]
            prev_gray = 0.299 * previous[:, :, 0] + 0.587 * previous[:, :, 1] + 0.114 * previous[:, :, 2]
        else:
            curr_gray = current[:, :, 0].astype(np.float32)
            prev_gray = previous[:, :, 0].astype(np.float32)
        
        # Simple block matching for motion estimation
        motion = np.zeros((h, w, 2), dtype=np.float32)
        
        block_size = 4
        search_range = 4
        
        for by in range(0, h - block_size, block_size):
            for bx in range(0, w - block_size, block_size):
                curr_block = curr_gray[by:by+block_size, bx:bx+block_size]
                
                best_dx, best_dy = 0, 0
                best_diff = float('inf')
                
                for dy in range(-search_range, search_range + 1):
                    for dx in range(-search_range, search_range + 1):
                        py, px = by + dy, bx + dx
                        if py < 0 or py + block_size > h or px < 0 or px + block_size > w:
                            continue
                        
                        prev_block = prev_gray[py:py+block_size, px:px+block_size]
                        diff = np.sum(np.abs(curr_block - prev_block))
                        
                        if diff < best_diff:
                            best_diff = diff
                            best_dx, best_dy = dx, dy
                
                # Store motion for this block
                motion[by:by+block_size, bx:bx+block_size, 0] = best_dx
                motion[by:by+block_size, bx:bx+block_size, 1] = best_dy
        
        return motion * self.config.motion_scale
    
    def _apply_gaussian_weights(self, samples: int) -> np.ndarray:
        """Generate Gaussian weights for blur samples."""
        if samples == 1:
            return np.array([1.0])
        
        # Gaussian distribution centered at 0
        x = np.linspace(-2, 2, samples)
        weights = np.exp(-x * x / 2)
        return weights / weights.sum()
    
    def _apply_linear_weights(self, samples: int) -> np.ndarray:
        """Generate linear falloff weights."""
        if samples == 1:
            return np.array([1.0])
        
        weights = np.linspace(1, 0, samples)
        return weights / weights.sum()
    
    def _get_weights(self, samples: int) -> np.ndarray:
        """Get blur weights based on falloff type."""
        if self.config.falloff == "gaussian":
            return self._apply_gaussian_weights(samples)
        elif self.config.falloff == "linear":
            return self._apply_linear_weights(samples)
        else:  # constant
            return np.ones(samples) / samples
    
    def _blur_directional(self, image: np.ndarray, angle: float, strength: float) -> np.ndarray:
        """Apply directional motion blur with high quality."""
        h, w = image.shape[:2]
        result = np.zeros_like(image, dtype=np.float64)
        
        # Direction vector - scale for visible blur
        rad = np.radians(angle)
        dx = np.cos(rad) * strength * 1.5  # Increased visibility
        dy = np.sin(rad) * strength * 1.5
        
        samples = max(8, self.config.samples)
        weights = self._get_weights(samples)
        
        # Create coordinate grids once
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        
        for i, weight in enumerate(weights):
            t = (i / (samples - 1) - 0.5) * 2 if samples > 1 else 0
            offset_x = dx * t
            offset_y = dy * t
            
            # Sample at offset position
            sampled = self._bilinear_sample(image, xx + offset_x, yy + offset_y)
            result += sampled * weight
        
        return result.astype(np.float32)
    
    def _blur_radial(self, image: np.ndarray, strength: float, cx: float = None, cy: float = None) -> np.ndarray:
        """Apply radial (spin) motion blur with high quality."""
        h, w = image.shape[:2]
        result = np.zeros_like(image, dtype=np.float64)
        
        # Use provided center or config center
        if cx is None:
            cx = w * self.config.center_x
        if cy is None:
            cy = h * self.config.center_y
        
        # More samples = smoother blur
        samples = max(8, self.config.samples)
        weights = self._get_weights(samples)
        
        # Max rotation angle - scale properly for visible effect
        # Higher intensity = more rotation spread
        max_angle = strength * 25  # degrees (was 10, now more visible)
        
        # Create coordinate grids once
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        
        # Precompute distance from center for falloff
        dx = xx - cx
        dy = yy - cy
        dist = np.sqrt(dx*dx + dy*dy)
        max_dist = max(np.sqrt(cx*cx + cy*cy), 
                       np.sqrt((w-cx)**2 + (h-cy)**2))
        dist_factor = np.clip(dist / (max_dist + 1), 0, 1)
        
        # Apply rotation samples with distance-based angle scaling
        for i, weight in enumerate(weights):
            # t goes from -1 to 1 across samples
            t = (i / (samples - 1) - 0.5) * 2 if samples > 1 else 0
            
            # Base angle for this sample
            base_angle = np.radians(max_angle * t)
            
            # Scale angle by distance from center (more blur at edges)
            angle = base_angle * (0.3 + 0.7 * dist_factor)
            
            # Rotate coordinates around center
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            
            new_x = cx + dx * cos_a - dy * sin_a
            new_y = cy + dx * sin_a + dy * cos_a
            
            # Sample with bilinear interpolation
            sampled = self._bilinear_sample(image, new_x, new_y)
            result += sampled * weight
        
        return result.astype(np.float32)
    
    def _blur_zoom(self, image: np.ndarray, strength: float, cx: float = None, cy: float = None) -> np.ndarray:
        """Apply zoom motion blur with high quality."""
        h, w = image.shape[:2]
        result = np.zeros_like(image, dtype=np.float64)
        
        # Use provided center or config center
        if cx is None:
            cx = w * self.config.center_x
        if cy is None:
            cy = h * self.config.center_y
        
        samples = max(8, self.config.samples)
        weights = self._get_weights(samples)
        
        # Scale range - more visible effect
        scale_range = strength * 0.2  # was 0.1
        
        # Create coordinate grids once
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        dx = xx - cx
        dy = yy - cy
        
        for i, weight in enumerate(weights):
            t = (i / (samples - 1) - 0.5) * 2 if samples > 1 else 0
            scale = 1.0 + scale_range * t
            
            if abs(scale) < 0.001:
                scale = 0.001
            
            # Scale coordinates from center
            new_x = cx + dx / scale
            new_y = cy + dy / scale
            
            sampled = self._bilinear_sample(image, new_x, new_y)
            result += sampled * weight
        
        return result.astype(np.float32)
    
    def _blur_motion_based(self, image: np.ndarray, motion_field: np.ndarray) -> np.ndarray:
        """Apply motion-based blur using motion vectors."""
        h, w = image.shape[:2]
        result = np.zeros_like(image, dtype=np.float32)
        
        samples = max(3, self.config.samples)
        weights = self._get_weights(samples)
        
        # Pre-compute sample positions
        yy, xx = np.mgrid[0:h, 0:w]
        
        for i, weight in enumerate(weights):
            t = (i / (samples - 1) - 0.5) * 2 if samples > 1 else 0
            
            # Calculate sample coordinates
            sample_x = xx + motion_field[:, :, 0] * t * self.config.intensity
            sample_y = yy + motion_field[:, :, 1] * t * self.config.intensity
            
            # Bilinear sample
            sampled = self._bilinear_sample(image, sample_x, sample_y)
            result += sampled * weight
        
        return result
    
    def _sample_at_offset(self, image: np.ndarray, ox: float, oy: float) -> np.ndarray:
        """Sample image at offset position with bilinear interpolation."""
        h, w = image.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        return self._bilinear_sample(image, xx + ox, yy + oy)
    
    def _bilinear_sample(self, image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinear sampling of image at fractional coordinates."""
        h, w = image.shape[:2]
        
        # Clamp coordinates
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
        
        # Integer and fractional parts
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        
        fx = (x - x0)[:, :, np.newaxis]
        fy = (y - y0)[:, :, np.newaxis]
        
        # Bilinear interpolation
        result = (
            image[y0, x0] * (1 - fx) * (1 - fy) +
            image[y0, x1] * fx * (1 - fy) +
            image[y1, x0] * (1 - fx) * fy +
            image[y1, x1] * fx * fy
        )
        
        return result.astype(np.float32)
    
    def _rotate_around_center(self, image: np.ndarray, cx: int, cy: int, angle: float) -> np.ndarray:
        """Rotate image around a center point."""
        h, w = image.shape[:2]
        
        # Create rotation matrix
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Coordinate grids
        yy, xx = np.mgrid[0:h, 0:w]
        
        # Translate to center, rotate, translate back
        dx = xx - cx
        dy = yy - cy
        
        new_x = dx * cos_a - dy * sin_a + cx
        new_y = dx * sin_a + dy * cos_a + cy
        
        return self._bilinear_sample(image, new_x, new_y)
    
    def _scale_from_center(self, image: np.ndarray, cx: float, cy: float, scale: float) -> np.ndarray:
        """Scale image from a center point."""
        h, w = image.shape[:2]
        
        yy, xx = np.mgrid[0:h, 0:w]
        
        # Scale from center
        new_x = cx + (xx - cx) / scale
        new_y = cy + (yy - cy) / scale
        
        return self._bilinear_sample(image, new_x, new_y)
    
    def process_frame(self, image: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        """Apply motion blur to a frame with animation."""
        h, w = image.shape[:2]
        original = image.astype(np.float64)
        
        # Get sprite mask and alpha
        if image.shape[2] == 4:
            sprite_mask = image[:, :, 3] > 10
            alpha = image[:, :, 3:4].astype(np.float64)
        else:
            sprite_mask = np.any(image > 0, axis=2)
            alpha = np.ones((h, w, 1), dtype=np.float64) * 255
        
        # Auto-detect sprite center if enabled
        if self.config.auto_center and self._sprite_center is None:
            self._sprite_center = self._find_sprite_center(image)
        
        cx, cy = self._sprite_center if self._sprite_center else (w * self.config.center_x, h * self.config.center_y)
        
        blur_type = self.config.blur_type.lower()
        
        # Animate intensity based on frame position for visual interest
        # Creates a pulse effect: starts subtle, peaks in middle, ends subtle
        t = frame_idx / max(1, total_frames - 1)
        animation_curve = np.sin(t * np.pi)  # 0 -> 1 -> 0
        
        # Base strength with animation
        base_strength = self.config.intensity * 8  # Increased base strength
        strength = base_strength * (0.4 + 0.6 * animation_curve)  # Always at least 40% blur
        
        # Apply blur based on type
        if blur_type == "directional":
            # Animate direction slightly for more dynamic feel
            angle_offset = np.sin(t * np.pi * 2) * 15  # ±15 degree oscillation
            blurred = self._blur_directional(original, self.config.direction + angle_offset, strength)
        
        elif blur_type == "radial":
            # Oscillating radial blur - spins one way then the other
            direction = 1 if (frame_idx % 2 == 0) else -1
            blurred = self._blur_radial(original, strength * direction * (0.5 + animation_curve), cx, cy)
        
        elif blur_type == "zoom":
            # Pulsing zoom blur
            blurred = self._blur_zoom(original, strength * animation_curve, cx, cy)
        
        elif blur_type == "motion" and self._prev_frame is not None:
            # Calculate motion field from previous frame
            self._motion_field = self._calculate_motion_field(image, self._prev_frame)
            
            # Apply threshold
            motion_mag = np.sqrt(self._motion_field[:, :, 0]**2 + self._motion_field[:, :, 1]**2)
            self._motion_field[motion_mag < self.config.motion_threshold] = 0
            
            blurred = self._blur_motion_based(original, self._motion_field)
        
        else:
            # Linear blur - directional with animated angle
            t_cycle = (frame_idx / max(1, total_frames - 1)) * 2 * np.pi
            angle = self.config.direction + np.sin(t_cycle) * 45
            blurred = self._blur_directional(original, angle, strength)
        
        # Store frame for next iteration
        self._prev_frame = image.copy()
        
        # Handle alpha channel properly
        if self.config.preserve_alpha and image.shape[2] == 4:
            # Keep original alpha, but blur the RGB slightly beyond sprite bounds
            # for a nice feathered blur effect
            blurred[:, :, 3:4] = alpha
        
        # Blend blurred result with original based on sprite mask
        if not self.config.blur_background:
            result = original.copy()
            # Expand mask slightly for blur feathering (numpy-only dilation)
            expanded_mask = sprite_mask.copy()
            for _ in range(2):
                padded = np.pad(expanded_mask, 1, mode='constant', constant_values=False)
                expanded_mask = (
                    padded[:-2, 1:-1] | padded[2:, 1:-1] |  # vertical
                    padded[1:-1, :-2] | padded[1:-1, 2:] |  # horizontal
                    padded[1:-1, 1:-1]  # center
                )
            
            mask_3d = expanded_mask[:, :, np.newaxis]
            result = np.where(mask_3d, blurred, original)
        else:
            result = blurred
        
        # Temporal blend for smoother animation
        if self.config.temporal_blend > 0 and self._prev_frame is not None:
            blend = self.config.temporal_blend
            result = result * (1 - blend) + self._prev_frame.astype(np.float64) * blend
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def apply(self, sprite) -> list:
        """Apply motion blur effect to sprite and return animation frames."""
        from src.core import Sprite
        
        frames = []
        for i in range(self.config.frame_count):
            pixels = self.process_frame(sprite.pixels.copy(), i, self.config.frame_count)
            frame = Sprite(
                width=sprite.width,
                height=sprite.height,
                pixels=pixels,
                name=f"{sprite.name}_motion_blur_{i}",
                source_path=sprite.source_path
            )
            frames.append(frame)
        return frames


# Additional specialized motion blur effects

@dataclass
class SpeedLinesConfig(EffectConfig):
    """Configuration for speed lines effect."""
    intensity: float = 0.7
    line_count: int = 8
    line_length: float = 0.3
    direction: float = 0.0  # degrees
    color: Tuple[int, int, int, int] = (255, 255, 255, 200)
    fade_in: bool = True
    only_on_sprite: bool = True
    seed: Optional[int] = None


class SpeedLines(BaseEffect):
    """Speed lines effect for fast motion emphasis."""
    
    name = "speed_lines"
    description = "Manga/anime style speed lines for motion emphasis"
    
    config_class = SpeedLinesConfig
    
    def __init__(self, config: SpeedLinesConfig):
        super().__init__(config)
        self.rng = np.random.default_rng(config.seed)
    
    def process_frame(self, image: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        h, w = image.shape[:2]
        result = image.astype(np.float32)
        
        # Animation progress
        t = frame_idx / max(1, total_frames - 1)
        
        # Direction
        rad = np.radians(self.config.direction)
        dx = np.cos(rad)
        dy = np.sin(rad)
        
        # Get sprite bounds
        if image.shape[2] == 4:
            mask = image[:, :, 3] > 128
        else:
            mask = np.any(image > 0, axis=2)
        
        if not np.any(mask):
            return image
        
        ys, xs = np.where(mask)
        cx, cy = xs.mean(), ys.mean()
        
        # Draw speed lines
        color = np.array(self.config.color, dtype=np.float32)
        
        for i in range(self.config.line_count):
            # Random position along perpendicular axis
            perp_offset = (self.rng.random() - 0.5) * h * 0.8
            
            # Line start and end
            line_len = w * self.config.line_length
            
            start_x = cx - dx * line_len / 2 - dy * perp_offset
            start_y = cy - dy * line_len / 2 + dx * perp_offset
            
            end_x = start_x + dx * line_len
            end_y = start_y + dy * line_len
            
            # Animate line growth
            if self.config.fade_in:
                current_len = line_len * t
                end_x = start_x + dx * current_len
                end_y = start_y + dy * current_len
            
            # Draw line with anti-aliasing
            self._draw_line(result, start_x, start_y, end_x, end_y, 
                           color * self.config.intensity, 1.5)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _draw_line(self, canvas: np.ndarray, x0: float, y0: float, 
                   x1: float, y1: float, color: np.ndarray, thickness: float) -> None:
        """Draw anti-aliased line on canvas."""
        h, w = canvas.shape[:2]
        
        # Bresenham with anti-aliasing
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        length = max(dx, dy)
        
        if length == 0:
            return
        
        for i in range(int(length) + 1):
            t = i / length if length > 0 else 0
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            
            # Draw with soft falloff
            for oy in range(-int(thickness), int(thickness) + 1):
                for ox in range(-int(thickness), int(thickness) + 1):
                    px, py = int(x + ox), int(y + oy)
                    if 0 <= px < w and 0 <= py < h:
                        dist = np.sqrt(ox*ox + oy*oy)
                        alpha = max(0, 1 - dist / (thickness + 0.5))
                        
                        for c in range(min(3, canvas.shape[2])):
                            canvas[py, px, c] = canvas[py, px, c] * (1 - alpha) + color[c] * alpha
    
    def apply(self, sprite) -> list:
        """Apply speed lines effect to sprite and return animation frames."""
        from src.core import Sprite
        
        frames = []
        for i in range(self.config.frame_count):
            pixels = self.process_frame(sprite.pixels.copy(), i, self.config.frame_count)
            frame = Sprite(
                width=sprite.width,
                height=sprite.height,
                pixels=pixels,
                name=f"{sprite.name}_speed_lines_{i}",
                source_path=sprite.source_path
            )
            frames.append(frame)
        return frames
