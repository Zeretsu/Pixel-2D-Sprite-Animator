"""
Stretch Effect - Cartoon squash and stretch
Impact and jump animations
"""

import numpy as np
from typing import List, Optional
from .base import BaseEffect, EffectConfig, Easing, PixelMath
from ..core.parser import Sprite


class StretchEffect(BaseEffect):
    """Creates cartoon squash and stretch animation"""
    
    name = "stretch"
    description = "Cartoon squash and stretch for impacts/jumps"
    
    def __init__(self, config: Optional[EffectConfig] = None):
        super().__init__(config)
        
        self.stretch_amount = self.config.extra.get('amount', 0.3)
        self.mode = self.config.extra.get('mode', 'bounce')  # 'bounce', 'impact', 'jump', 'cycle'
        self.anchor = self.config.extra.get('anchor', 'bottom')  # 'bottom', 'center', 'top'
        self.preserve_volume = self.config.extra.get('preserve_volume', True)
    
    def apply(self, sprite: Sprite) -> List[Sprite]:
        frames = []
        h, w = sprite.height, sprite.width
        original = sprite.pixels.copy()
        
        # Find anchor point
        anchor_y = self._get_anchor_y(original, h)
        cx = w / 2.0
        
        # Pre-compute coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        for i in range(self.config.frame_count):
            t = i / self.config.frame_count
            
            # Calculate stretch factors
            scale_x, scale_y = self._calculate_stretch(t)
            
            frame_pixels = self._apply_stretch(
                original, scale_x, scale_y, cx, anchor_y, x_coords, y_coords
            )
            
            frames.append(self._create_frame(sprite, frame_pixels))
        
        return frames
    
    def _get_anchor_y(self, pixels: np.ndarray, h: int) -> float:
        """Find anchor point based on setting"""
        if pixels.shape[2] == 4:
            mask = pixels[:, :, 3] > 0
        else:
            mask = np.any(pixels > 0, axis=2)
        
        rows = np.any(mask, axis=1)
        if not np.any(rows):
            return h / 2
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        
        if self.anchor == 'bottom':
            return float(y_max)
        elif self.anchor == 'top':
            return float(y_min)
        else:  # center
            return (y_min + y_max) / 2.0
    
    def _calculate_stretch(self, t: float) -> tuple:
        """Calculate X and Y scale factors based on mode"""
        amount = self.stretch_amount * self.config.intensity
        
        if self.mode == 'bounce':
            # Bouncing ball style: squash at bottom, stretch at top
            bounce = np.abs(np.sin(t * 2 * np.pi))
            
            # Squash when near ground (low bounce value)
            if bounce < 0.3:
                squash = (0.3 - bounce) / 0.3
                scale_y = 1.0 - amount * squash
            # Stretch when moving fast (mid bounce)
            elif bounce > 0.7:
                stretch = (bounce - 0.7) / 0.3
                scale_y = 1.0 + amount * 0.5 * stretch
            else:
                scale_y = 1.0
            
        elif self.mode == 'impact':
            # Single impact: squash then recover
            if t < 0.2:
                # Quick squash
                scale_y = 1.0 - amount * Easing.ease_out_quad(t / 0.2)
            elif t < 0.5:
                # Overshoot recovery
                local_t = (t - 0.2) / 0.3
                scale_y = (1.0 - amount) + amount * 1.2 * Easing.ease_out_elastic(local_t)
            else:
                # Settle to normal
                local_t = (t - 0.5) / 0.5
                scale_y = 1.0 + 0.2 * amount * (1 - Easing.ease_out_quad(local_t))
            
        elif self.mode == 'jump':
            # Jump animation: crouch, stretch up, squash land
            if t < 0.2:
                # Anticipation crouch
                scale_y = 1.0 - amount * 0.5 * Easing.ease_in_quad(t / 0.2)
            elif t < 0.4:
                # Launch stretch
                local_t = (t - 0.2) / 0.2
                scale_y = (1.0 - amount * 0.5) + amount * Easing.ease_out_quad(local_t)
            elif t < 0.7:
                # Air time (slight stretch)
                scale_y = 1.0 + amount * 0.3
            elif t < 0.85:
                # Landing squash
                local_t = (t - 0.7) / 0.15
                scale_y = 1.0 + amount * 0.3 - amount * 0.8 * Easing.ease_in_quad(local_t)
            else:
                # Recovery
                local_t = (t - 0.85) / 0.15
                scale_y = 1.0 - amount * 0.5 + amount * 0.5 * Easing.ease_out_quad(local_t)
            
        else:  # cycle
            # Continuous squash/stretch cycle
            phase = t * 2 * np.pi
            scale_y = 1.0 + amount * np.sin(phase)
        
        # Volume preservation: if Y shrinks, X grows
        if self.preserve_volume:
            scale_x = 1.0 / scale_y
        else:
            scale_x = 1.0
        
        return scale_x, scale_y
    
    def _apply_stretch(
        self,
        pixels: np.ndarray,
        scale_x: float,
        scale_y: float,
        cx: float,
        anchor_y: float,
        x_coords: np.ndarray,
        y_coords: np.ndarray
    ) -> np.ndarray:
        """Apply stretch transformation"""
        # Inverse transform: where to sample from
        src_x = cx + (x_coords - cx) / scale_x
        src_y = anchor_y + (y_coords - anchor_y) / scale_y
        
        return PixelMath.bilinear_sample(pixels, src_x, src_y, gamma_correct=True)
