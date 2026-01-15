"""
Color Analyzer - Analyze sprite colors for effect detection
"""

import numpy as np
from typing import Dict, List, Tuple
from ..core.utils import ColorUtils


class ColorAnalyzer:
    """Analyzes sprite colors to suggest appropriate effects"""
    
    def __init__(self, pixels: np.ndarray):
        self.pixels = pixels
        self._distribution = None
        self._dominant_colors = None
    
    @property
    def distribution(self) -> Dict[str, float]:
        """Get cached color distribution"""
        if self._distribution is None:
            self._distribution = ColorUtils.get_color_distribution(self.pixels)
        return self._distribution
    
    @property
    def dominant_colors(self) -> List[Tuple[int, int, int, int]]:
        """Get dominant colors in the sprite"""
        if self._dominant_colors is None:
            self._dominant_colors = self._calculate_dominant_colors()
        return self._dominant_colors
    
    def _calculate_dominant_colors(self, n: int = 5) -> List[Tuple[int, int, int, int]]:
        """Calculate the N most common colors"""
        flat = self.pixels.reshape(-1, self.pixels.shape[2])
        
        # Filter transparent
        if self.pixels.shape[2] == 4:
            visible = flat[flat[:, 3] > 0]
        else:
            visible = flat
        
        if len(visible) == 0:
            return []
        
        # Simple color bucketing (reduce precision for grouping)
        quantized = (visible // 16) * 16
        
        # Count unique colors
        unique, counts = np.unique(quantized, axis=0, return_counts=True)
        
        # Sort by count
        sorted_idx = np.argsort(-counts)
        
        return [tuple(unique[i]) for i in sorted_idx[:n]]
    
    def score_flame(self) -> float:
        """Score likelihood of fire effect (0-1)"""
        dist = self.distribution
        
        # Fire colors
        warm = dist.get('red', 0) + dist.get('orange', 0) + dist.get('yellow', 0)
        
        # Check for gradient (fire has light to dark gradient)
        has_light = dist.get('yellow', 0) > 0.05 or dist.get('white', 0) > 0.05
        has_dark = dist.get('red', 0) > 0.1 or dist.get('black', 0) > 0.05
        
        gradient_bonus = 0.2 if (has_light and has_dark) else 0
        
        return min(1.0, warm + gradient_bonus)
    
    def score_water(self) -> float:
        """Score likelihood of water effect (0-1)"""
        dist = self.distribution
        
        # Water colors
        cool = dist.get('blue', 0) + dist.get('cyan', 0)
        
        # Bonus for having white highlights (reflections)
        reflection_bonus = 0.1 if dist.get('white', 0) > 0.02 else 0
        
        return min(1.0, cool + reflection_bonus)
    
    def score_magic(self) -> float:
        """Score likelihood of magic/sparkle effect (0-1)"""
        dist = self.distribution
        
        # Magic colors
        magic = (
            dist.get('purple', 0) * 1.5 +
            dist.get('magenta', 0) * 1.3 +
            dist.get('cyan', 0) * 0.8 +
            dist.get('white', 0) * 0.5
        )
        
        return min(1.0, magic)
    
    def score_void(self) -> float:
        """Score likelihood of void/dark portal effect (0-1)"""
        dist = self.distribution
        
        # Void is dark with accent colors
        dark = dist.get('black', 0) + dist.get('purple', 0) * 0.5
        
        # Check for glow accent
        has_glow = (
            dist.get('purple', 0) > 0.1 or 
            dist.get('magenta', 0) > 0.1 or
            dist.get('cyan', 0) > 0.1
        )
        
        glow_bonus = 0.3 if has_glow else 0
        
        return min(1.0, dark * 0.7 + glow_bonus)
    
    def score_nature(self) -> float:
        """Score likelihood of nature/sway effect (0-1)"""
        dist = self.distribution
        
        # Nature colors
        nature = dist.get('green', 0) + dist.get('yellow', 0) * 0.3
        
        # Bonus for brown (wood, earth)
        # Brown is usually dark orange/red
        brown_like = dist.get('orange', 0) * 0.3
        
        return min(1.0, nature + brown_like)
    
    def get_warmth(self) -> float:
        """Get overall warmth of the sprite (-1 cold to 1 warm)"""
        dist = self.distribution
        
        warm = dist.get('red', 0) + dist.get('orange', 0) + dist.get('yellow', 0)
        cool = dist.get('blue', 0) + dist.get('cyan', 0) + dist.get('purple', 0)
        
        if warm + cool == 0:
            return 0
        
        return (warm - cool) / (warm + cool)
    
    def get_brightness(self) -> float:
        """Get average brightness of the sprite (0-1)"""
        flat = self.pixels.reshape(-1, self.pixels.shape[2])
        
        # Filter transparent
        if self.pixels.shape[2] == 4:
            visible = flat[flat[:, 3] > 0]
        else:
            visible = flat
        
        if len(visible) == 0:
            return 0.5
        
        # Calculate perceived brightness
        brightness = (
            0.299 * visible[:, 0] + 
            0.587 * visible[:, 1] + 
            0.114 * visible[:, 2]
        ) / 255
        
        return float(np.mean(brightness))
    
    def get_all_scores(self) -> Dict[str, float]:
        """Get all effect scores"""
        return {
            'flame': self.score_flame(),
            'water': self.score_water(),
            'magic': self.score_magic(),
            'void': self.score_void(),
            'nature': self.score_nature(),
        }
