"""
Edge Analyzer - Analyze sprite edges for effect detection
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class EdgeMetrics:
    """Edge analysis metrics"""
    edge_pixel_count: int
    total_pixel_count: int
    edge_ratio: float
    softness: float  # 0=hard edges, 1=soft/gradient edges
    complexity: float  # How complex/jagged the edge is
    top_softness: float
    bottom_softness: float
    left_softness: float
    right_softness: float


class EdgeAnalyzer:
    """Analyzes sprite edges to suggest appropriate effects"""
    
    def __init__(self, pixels: np.ndarray):
        self.pixels = pixels
        self._metrics = None
        self._edge_mask = None
    
    @property
    def edge_mask(self) -> np.ndarray:
        """Get mask of edge pixels"""
        if self._edge_mask is None:
            self._edge_mask = self._detect_edges()
        return self._edge_mask
    
    @property
    def metrics(self) -> EdgeMetrics:
        """Get cached edge metrics"""
        if self._metrics is None:
            self._metrics = self._calculate_metrics()
        return self._metrics
    
    def _get_alpha(self) -> np.ndarray:
        """Get alpha channel or create from non-black pixels"""
        if self.pixels.shape[2] == 4:
            return self.pixels[:, :, 3]
        else:
            # Treat non-black as opaque
            return np.any(self.pixels > 0, axis=2).astype(np.uint8) * 255
    
    def _detect_edges(self) -> np.ndarray:
        """Detect edge pixels using alpha transitions"""
        alpha = self._get_alpha()
        
        # Simple edge detection: pixel is edge if adjacent to transparent
        edges = np.zeros_like(alpha, dtype=bool)
        
        # Check all 4 neighbors
        edges[1:, :] |= (alpha[1:, :] > 0) & (alpha[:-1, :] == 0)
        edges[:-1, :] |= (alpha[:-1, :] > 0) & (alpha[1:, :] == 0)
        edges[:, 1:] |= (alpha[:, 1:] > 0) & (alpha[:, :-1] == 0)
        edges[:, :-1] |= (alpha[:, :-1] > 0) & (alpha[:, 1:] == 0)
        
        return edges
    
    def _calculate_metrics(self) -> EdgeMetrics:
        """Calculate edge metrics"""
        alpha = self._get_alpha()
        edges = self.edge_mask
        
        edge_count = np.sum(edges)
        total_count = np.sum(alpha > 0)
        
        edge_ratio = edge_count / total_count if total_count > 0 else 0
        
        # Softness: how many edge pixels have partial alpha
        if self.pixels.shape[2] == 4 and edge_count > 0:
            edge_alpha = alpha[edges]
            partial = np.sum((edge_alpha > 0) & (edge_alpha < 255))
            softness = partial / edge_count
        else:
            softness = 0
        
        # Complexity: edge perimeter vs area ratio
        # Higher = more jagged/complex edges
        if total_count > 0:
            expected_perimeter = 4 * np.sqrt(total_count)  # For a square
            complexity = edge_count / expected_perimeter if expected_perimeter > 0 else 1
        else:
            complexity = 0
        
        # Regional softness
        h, w = alpha.shape
        regions = {
            'top': alpha[:h//3, :],
            'bottom': alpha[2*h//3:, :],
            'left': alpha[:, :w//3],
            'right': alpha[:, 2*w//3:]
        }
        
        regional_softness = {}
        for name, region in regions.items():
            partial = np.sum((region > 0) & (region < 255))
            total = np.sum(region > 0)
            regional_softness[name] = partial / total if total > 0 else 0
        
        return EdgeMetrics(
            edge_pixel_count=edge_count,
            total_pixel_count=total_count,
            edge_ratio=edge_ratio,
            softness=softness,
            complexity=complexity,
            top_softness=regional_softness['top'],
            bottom_softness=regional_softness['bottom'],
            left_softness=regional_softness['left'],
            right_softness=regional_softness['right']
        )
    
    def score_flame(self) -> float:
        """Score flame based on edges - flames have soft tops"""
        m = self.metrics
        
        # Flames have soft tops, harder bottoms
        top_bottom_diff = m.top_softness - m.bottom_softness
        
        # Some overall softness
        softness_score = m.softness * 0.3
        
        # Complex edges suggest flickering potential
        complexity_score = min(m.complexity, 1.0) * 0.2
        
        return min(1.0, max(0, top_bottom_diff * 0.5 + softness_score + complexity_score))
    
    def score_smoke(self) -> float:
        """Score smoke/cloud - very soft overall edges"""
        m = self.metrics
        
        # Smoke has very soft edges all around
        return m.softness
    
    def score_solid(self) -> float:
        """Score solid object - hard pixel edges"""
        m = self.metrics
        
        # Solid objects have crisp edges
        return 1.0 - m.softness
    
    def score_wobble(self) -> float:
        """Score wobble potential - based on edge complexity"""
        m = self.metrics
        
        # Complex edges wobble interestingly
        # Hard edges wobble better than soft
        hard_edge = 1.0 - m.softness
        complexity = min(m.complexity, 1.5) / 1.5
        
        return hard_edge * 0.5 + complexity * 0.5
    
    def get_all_scores(self) -> Dict[str, float]:
        """Get all edge-based effect scores"""
        return {
            'flame_edge': self.score_flame(),
            'smoke': self.score_smoke(),
            'solid': self.score_solid(),
            'wobble': self.score_wobble(),
        }
