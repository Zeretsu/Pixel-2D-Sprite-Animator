"""
Shape Analyzer - Analyze sprite shape for effect detection
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ShapeMetrics:
    """Shape analysis metrics"""
    width: int
    height: int
    aspect_ratio: float
    fill_ratio: float  # How much of bounding box is filled
    center_of_mass: Tuple[float, float]
    is_tall: bool
    is_wide: bool
    is_round: bool
    symmetry_h: float  # Horizontal symmetry (0-1)
    symmetry_v: float  # Vertical symmetry (0-1)


class ShapeAnalyzer:
    """Analyzes sprite shape to suggest appropriate effects"""
    
    def __init__(self, pixels: np.ndarray):
        self.pixels = pixels
        self._mask = None
        self._metrics = None
        self._contour = None
    
    @property
    def mask(self) -> np.ndarray:
        """Get binary mask of non-transparent pixels"""
        if self._mask is None:
            if self.pixels.shape[2] == 4:
                self._mask = self.pixels[:, :, 3] > 0
            else:
                # Assume no transparency
                self._mask = np.ones(self.pixels.shape[:2], dtype=bool)
        return self._mask
    
    @property
    def metrics(self) -> ShapeMetrics:
        """Get cached shape metrics"""
        if self._metrics is None:
            self._metrics = self._calculate_metrics()
        return self._metrics
    
    def _calculate_metrics(self) -> ShapeMetrics:
        """Calculate shape metrics"""
        mask = self.mask
        
        # Get bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return ShapeMetrics(
                width=0, height=0, aspect_ratio=1,
                fill_ratio=0, center_of_mass=(0, 0),
                is_tall=False, is_wide=False, is_round=False,
                symmetry_h=0, symmetry_v=0
            )
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        
        # Aspect ratio
        aspect_ratio = width / height if height > 0 else 1
        
        # Fill ratio (how much of bounding box is filled)
        bbox_area = width * height
        filled_area = np.sum(mask[y_min:y_max+1, x_min:x_max+1])
        fill_ratio = filled_area / bbox_area if bbox_area > 0 else 0
        
        # Center of mass
        y_coords, x_coords = np.where(mask)
        if len(x_coords) > 0:
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
        else:
            center_x = center_y = 0
        
        # Shape classifications
        is_tall = aspect_ratio < 0.7
        is_wide = aspect_ratio > 1.4
        is_round = 0.8 <= aspect_ratio <= 1.2 and fill_ratio > 0.6
        
        # Symmetry
        symmetry_h = self._calculate_symmetry(mask, 'horizontal')
        symmetry_v = self._calculate_symmetry(mask, 'vertical')
        
        return ShapeMetrics(
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            fill_ratio=fill_ratio,
            center_of_mass=(center_x, center_y),
            is_tall=is_tall,
            is_wide=is_wide,
            is_round=is_round,
            symmetry_h=symmetry_h,
            symmetry_v=symmetry_v
        )
    
    def _calculate_symmetry(self, mask: np.ndarray, axis: str) -> float:
        """Calculate symmetry score (0-1)"""
        if axis == 'horizontal':
            flipped = np.flip(mask, axis=1)
        else:
            flipped = np.flip(mask, axis=0)
        
        # Compare original with flipped
        total = np.sum(mask) + np.sum(flipped)
        if total == 0:
            return 1.0
        
        matching = np.sum(mask & flipped) * 2
        return matching / total
    
    def get_edge_softness(self) -> float:
        """Analyze how soft/fuzzy the edges are (0=sharp, 1=soft)"""
        if self.pixels.shape[2] != 4:
            return 0  # No alpha = sharp edges
        
        alpha = self.pixels[:, :, 3]
        
        # Count partially transparent pixels
        partial = np.sum((alpha > 0) & (alpha < 255))
        total = np.sum(alpha > 0)
        
        if total == 0:
            return 0
        
        return partial / total
    
    def score_float(self) -> float:
        """Score likelihood of floating/bobbing animation (0-1)"""
        m = self.metrics
        
        # Round objects float well
        round_score = 0.5 if m.is_round else 0
        
        # High fill ratio suggests solid object
        solid_score = m.fill_ratio * 0.3
        
        # Symmetric objects look better floating
        sym_score = (m.symmetry_h + m.symmetry_v) / 2 * 0.2
        
        return min(1.0, round_score + solid_score + sym_score)
    
    def score_sway(self) -> float:
        """Score likelihood of swaying animation (0-1)"""
        m = self.metrics
        
        # Tall objects sway (trees, grass, candles)
        tall_score = 0.6 if m.is_tall else 0
        
        # Low symmetry is OK for sway
        asym_bonus = (1 - m.symmetry_h) * 0.2
        
        return min(1.0, tall_score + asym_bonus)
    
    def score_pulse(self) -> float:
        """Score likelihood of pulsing animation (0-1)"""
        m = self.metrics
        
        # Round objects pulse well
        round_score = 0.5 if m.is_round else 0.2
        
        # High symmetry looks better pulsing
        sym_score = (m.symmetry_h + m.symmetry_v) / 2 * 0.3
        
        # Good fill ratio
        fill_score = m.fill_ratio * 0.2
        
        return min(1.0, round_score + sym_score + fill_score)
    
    def score_spin(self) -> float:
        """Score likelihood of spinning animation (0-1)"""
        m = self.metrics
        
        # Round AND symmetric objects spin well
        if m.is_round and m.symmetry_h > 0.7 and m.symmetry_v > 0.7:
            return 0.8
        elif m.is_round:
            return 0.5
        return 0.1
    
    def score_wave(self) -> float:
        """Score likelihood of wave animation (0-1)"""
        m = self.metrics
        
        # Wide objects wave well (water, banners)
        wide_score = 0.5 if m.is_wide else 0
        
        # Low vertical symmetry suggests wave potential
        asym_bonus = (1 - m.symmetry_v) * 0.3
        
        return min(1.0, wide_score + asym_bonus)
    
    def get_all_scores(self) -> Dict[str, float]:
        """Get all shape-based effect scores"""
        return {
            'float': self.score_float(),
            'sway': self.score_sway(),
            'pulse': self.score_pulse(),
            'spin': self.score_spin(),
            'wave': self.score_wave(),
        }
