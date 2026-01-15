"""
Wing Flap Effect - Animate wings flapping up and down

Perfect for bats, birds, butterflies, dragons, etc.
Automatically detects left/right wing regions and animates them.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from .base import BaseEffect, EffectConfig, PixelMath
from ..core.parser import Sprite


@dataclass
class FlapConfig(EffectConfig):
    """Configuration for wing flap animation."""
    flap_angle: float = 25.0  # Max rotation angle in degrees
    flap_speed: float = 1.0   # Flaps per animation cycle
    wing_split: float = 0.35  # How far from center wings start (0-0.5)
    body_width: float = 0.3   # Width of body region that stays still (0-1)
    vertical_bob: float = 0.15  # Body bobs up/down with flaps
    asymmetric: float = 0.0   # Phase offset between wings (0=sync, 0.5=alternate)
    ease_flap: bool = True    # Use easing for smoother motion
    squash_wings: bool = True # Slight squash at flap extremes
    pivot_y: float = 0.5      # Vertical pivot point (0=top, 1=bottom)


class FlapEffect(BaseEffect):
    """Animate sprite with wing flapping motion - FAST vectorized version."""
    
    name = "flap"
    description = "Wing flapping animation for bats, birds, butterflies"
    config_class = FlapConfig
    
    def __init__(self, config: Optional[FlapConfig] = None):
        if config is None:
            config = FlapConfig()
        super().__init__(config)
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        
        # Find sprite bounds and center
        visible_mask = original[:, :, 3] > 10
        if not visible_mask.any():
            return [sprite]
        
        ys, xs = np.where(visible_mask)
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        sprite_w = max_x - min_x + 1
        sprite_h = max_y - min_y + 1
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # Define wing regions
        body_half = self.config.body_width * sprite_w / 2
        left_wing_end = center_x - body_half
        right_wing_start = center_x + body_half
        
        # Pivot points for wings
        left_pivot_x = left_wing_end
        right_pivot_x = right_wing_start
        pivot_y = min_y + sprite_h * self.config.pivot_y
        
        # Padding for wing movement
        pad = int(sprite_h * 0.4) + 5
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            phase = t * 2 * np.pi * self.config.flap_speed
            
            # Flap angles
            flap_t = np.sin(phase)
            left_angle = np.radians(self.config.flap_angle * flap_t)
            right_angle = np.radians(-self.config.flap_angle * np.sin(phase + self.config.asymmetric * np.pi))
            
            # Body bob
            bob_offset = -self.config.vertical_bob * sprite_h * flap_t * 0.5
            
            # Create output canvas
            canvas_h = h + pad * 2
            canvas_w = w + pad * 2
            canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
            
            # Create coordinate grids
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            
            # Output coordinates (start with identity + padding + bob)
            out_x = x_coords.astype(np.float32) + pad
            out_y = y_coords.astype(np.float32) + pad + bob_offset
            
            # Left wing rotation
            left_mask = (x_coords < left_wing_end) & visible_mask
            if left_mask.any():
                dx = x_coords[left_mask] - left_pivot_x
                dy = y_coords[left_mask] - pivot_y
                
                cos_a, sin_a = np.cos(left_angle), np.sin(left_angle)
                new_dx = dx * cos_a - dy * sin_a
                new_dy = dx * sin_a + dy * cos_a
                
                # Squash at extremes
                if self.config.squash_wings:
                    squash = 1.0 - 0.12 * abs(flap_t)
                    new_dx *= squash
                
                out_x[left_mask] = left_pivot_x + new_dx + pad
                out_y[left_mask] = pivot_y + new_dy + pad + bob_offset
            
            # Right wing rotation
            right_mask = (x_coords > right_wing_start) & visible_mask
            if right_mask.any():
                dx = x_coords[right_mask] - right_pivot_x
                dy = y_coords[right_mask] - pivot_y
                
                cos_a, sin_a = np.cos(right_angle), np.sin(right_angle)
                new_dx = dx * cos_a - dy * sin_a
                new_dy = dx * sin_a + dy * cos_a
                
                if self.config.squash_wings:
                    squash = 1.0 - 0.12 * abs(np.sin(phase + self.config.asymmetric * np.pi))
                    new_dx *= squash
                
                out_x[right_mask] = right_pivot_x + new_dx + pad
                out_y[right_mask] = pivot_y + new_dy + pad + bob_offset
            
            # Draw all visible pixels
            for y in range(h):
                for x in range(w):
                    if original[y, x, 3] < 10:
                        continue
                    
                    ox, oy = int(out_x[y, x]), int(out_y[y, x])
                    if 0 <= ox < canvas_w and 0 <= oy < canvas_h:
                        if original[y, x, 3] > canvas[oy, ox, 3]:
                            canvas[oy, ox] = original[y, x]
            
            # Crop to content
            content_mask = canvas[:, :, 3] > 0
            if content_mask.any():
                content_ys, content_xs = np.where(content_mask)
                crop_min_x = max(0, content_xs.min() - 2)
                crop_max_x = min(canvas_w, content_xs.max() + 3)
                crop_min_y = max(0, content_ys.min() - 2)
                crop_max_y = min(canvas_h, content_ys.max() + 3)
                final = canvas[crop_min_y:crop_max_y, crop_min_x:crop_max_x]
            else:
                final = canvas[pad:pad+h, pad:pad+w]
            
            frame = Sprite(
                width=final.shape[1], height=final.shape[0],
                pixels=final, name=f"{sprite.name}_flap_{i}",
                source_path=sprite.source_path
            )
            frames.append(frame)
        
        return frames


class HoverFlapEffect(BaseEffect):
    """Wing flapping with hovering motion."""
    
    name = "hover_flap"
    description = "Hovering wing flap with figure-8 motion"
    config_class = FlapConfig
    
    def __init__(self, config: Optional[FlapConfig] = None):
        if config is None:
            config = FlapConfig(flap_angle=20.0, vertical_bob=0.08, flap_speed=1.5)
        super().__init__(config)
        self._base_flap = FlapEffect(config)
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        return self._base_flap.apply(sprite)


class GlideFlapEffect(BaseEffect):
    """Gliding with occasional flaps."""
    
    name = "glide_flap"
    description = "Gliding with occasional wing flaps"
    config_class = FlapConfig
    
    def __init__(self, config: Optional[FlapConfig] = None):
        if config is None:
            config = FlapConfig(flap_angle=35.0, vertical_bob=0.05, flap_speed=0.5)
        super().__init__(config)
        self._base_flap = FlapEffect(config)
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        return self._base_flap.apply(sprite)
