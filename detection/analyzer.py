"""
Enhanced Sprite Analyzer - Better auto-detection with improved scoring
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .color import ColorAnalyzer
from .shape import ShapeAnalyzer
from .edges import EdgeAnalyzer
from ..core.parser import Sprite


@dataclass
class EffectSuggestion:
    """A suggested effect with confidence score"""
    effect: str
    confidence: float
    reasons: List[str]


@dataclass
class AnalysisResult:
    """Complete analysis result"""
    suggestions: List[EffectSuggestion]
    color_scores: Dict[str, float]
    shape_scores: Dict[str, float]
    edge_scores: Dict[str, float]
    sprite_type: str  # 'fire', 'water', 'object', 'plant', 'magic', 'unknown'
    
    @property
    def best_effect(self) -> Optional[str]:
        """Get the highest confidence effect"""
        if not self.suggestions:
            return None
        return self.suggestions[0].effect
    
    @property
    def best_confidence(self) -> float:
        """Get the highest confidence score"""
        if not self.suggestions:
            return 0.0
        return self.suggestions[0].confidence


class SpriteAnalyzer:
    """
    Enhanced analyzer that combines color, shape, and edge analysis
    with better heuristics and layer name detection.
    """
    
    # Effect weights - refined for better detection
    EFFECT_WEIGHTS = {
        'flame': {
            'color': [('flame', 0.45)],
            'shape': [('sway', 0.15)],
            'edge': [('flame_edge', 0.25)],
            'bonus': {
                'warm_dominant': 0.15,
            }
        },
        'water': {
            'color': [('water', 0.45)],
            'shape': [('wave', 0.25)],
            'edge': [('solid', 0.15)],
            'bonus': {
                'cool_dominant': 0.15,
            }
        },
        'sparkle': {
            'color': [('magic', 0.5)],
            'shape': [('pulse', 0.2)],
            'edge': [('solid', 0.15)],
            'bonus': {
                'has_white': 0.15,
            }
        },
        'void': {
            'color': [('void', 0.55)],
            'shape': [('pulse', 0.15)],
            'edge': [('smoke', 0.15)],
            'bonus': {
                'mostly_dark': 0.15,
            }
        },
        'float': {
            'color': [],  # No specific color
            'shape': [('float', 0.5), ('pulse', 0.15)],
            'edge': [('solid', 0.25)],
            'bonus': {
                'symmetric': 0.1,
            }
        },
        'sway': {
            'color': [('nature', 0.25)],
            'shape': [('sway', 0.45)],
            'edge': [('solid', 0.15)],
            'bonus': {
                'tall_sprite': 0.15,
            }
        },
        'pulse': {
            'color': [('magic', 0.25)],
            'shape': [('pulse', 0.4), ('float', 0.15)],
            'edge': [('solid', 0.1)],
            'bonus': {
                'symmetric': 0.1,
            }
        },
        'smoke': {
            'color': [],
            'shape': [],
            'edge': [('smoke', 0.5)],
            'bonus': {
                'soft_edges': 0.3,
            }
        },
        'wobble': {
            'color': [],
            'shape': [('float', 0.2), ('pulse', 0.2)],
            'edge': [('wobble', 0.4)],
            'bonus': {
                'solid_fill': 0.2,
            }
        },
    }
    
    # Keywords in layer names that hint at effects
    LAYER_HINTS = {
        'flame': ['fire', 'flame', 'torch', 'candle', 'burn', 'ember', 'blaze'],
        'water': ['water', 'wave', 'ocean', 'sea', 'pool', 'liquid', 'ripple'],
        'sparkle': ['sparkle', 'magic', 'gem', 'crystal', 'star', 'glitter', 'shine'],
        'void': ['void', 'portal', 'dark', 'shadow', 'hole', 'abyss'],
        'float': ['float', 'hover', 'item', 'pickup', 'coin', 'orb'],
        'sway': ['plant', 'tree', 'grass', 'leaf', 'flower', 'vine', 'banner', 'flag'],
        'pulse': ['glow', 'pulse', 'heart', 'orb', 'energy', 'power'],
        'smoke': ['smoke', 'cloud', 'fog', 'steam', 'mist', 'dust'],
        'wobble': ['slime', 'jelly', 'blob', 'goo', 'bounce', 'elastic'],
    }
    
    def __init__(self, sprite: Sprite):
        self.sprite = sprite
        self.color_analyzer = ColorAnalyzer(sprite.pixels)
        self.shape_analyzer = ShapeAnalyzer(sprite.pixels)
        self.edge_analyzer = EdgeAnalyzer(sprite.pixels)
        
        # Check layer names for hints
        self.layer_hints = self._check_layer_hints()
    
    @classmethod
    def from_pixels(cls, pixels: np.ndarray) -> 'SpriteAnalyzer':
        """Create analyzer from pixel array"""
        from ..core.parser import SpriteParser
        sprite = SpriteParser.from_array(pixels)
        return cls(sprite)
    
    def _check_layer_hints(self) -> Dict[str, float]:
        """Check layer names for effect hints"""
        hints = {effect: 0.0 for effect in self.LAYER_HINTS}
        
        # Check sprite name and layer names
        names_to_check = [self.sprite.name.lower()]
        for layer in self.sprite.layers:
            names_to_check.append(layer.name.lower())
        
        for name in names_to_check:
            for effect, keywords in self.LAYER_HINTS.items():
                for keyword in keywords:
                    if keyword in name:
                        hints[effect] = max(hints[effect], 0.3)  # Significant boost
        
        return hints
    
    def _calculate_bonuses(self) -> Dict[str, float]:
        """Calculate bonus scores based on sprite characteristics"""
        bonuses = {}
        
        # Get basic metrics
        shape_metrics = self.shape_analyzer.metrics
        edge_metrics = self.edge_analyzer.metrics
        color_dist = self.color_analyzer.distribution
        
        # Warm vs cool dominant
        warm = color_dist.get('red', 0) + color_dist.get('orange', 0) + color_dist.get('yellow', 0)
        cool = color_dist.get('blue', 0) + color_dist.get('cyan', 0)
        bonuses['warm_dominant'] = 1.0 if warm > cool + 0.2 else 0
        bonuses['cool_dominant'] = 1.0 if cool > warm + 0.2 else 0
        
        # White highlights
        bonuses['has_white'] = 1.0 if color_dist.get('white', 0) > 0.05 else 0
        
        # Darkness
        dark = color_dist.get('black', 0) + color_dist.get('dark_gray', 0)
        bonuses['mostly_dark'] = 1.0 if dark > 0.4 else dark
        
        # Shape
        bonuses['tall_sprite'] = 1.0 if shape_metrics.is_tall else 0
        bonuses['symmetric'] = (shape_metrics.symmetry_h + shape_metrics.symmetry_v) / 2
        bonuses['solid_fill'] = shape_metrics.fill_ratio
        
        # Edges
        bonuses['soft_edges'] = edge_metrics.softness
        
        return bonuses
    
    def _detect_sprite_type(self) -> str:
        """Detect the general type of sprite"""
        color_scores = self.color_analyzer.get_all_scores()
        
        # Find dominant type
        if color_scores.get('flame', 0) > 0.5:
            return 'fire'
        elif color_scores.get('water', 0) > 0.5:
            return 'water'
        elif color_scores.get('magic', 0) > 0.4:
            return 'magic'
        elif color_scores.get('void', 0) > 0.4:
            return 'void'
        elif color_scores.get('nature', 0) > 0.4:
            return 'plant'
        else:
            return 'object'
    
    def analyze(self) -> AnalysisResult:
        """Perform full analysis and return suggestions"""
        # Get all individual scores
        color_scores = self.color_analyzer.get_all_scores()
        shape_scores = self.shape_analyzer.get_all_scores()
        edge_scores = self.edge_analyzer.get_all_scores()
        
        # Calculate bonuses
        bonuses = self._calculate_bonuses()
        
        # Detect sprite type
        sprite_type = self._detect_sprite_type()
        
        # Calculate combined scores for each effect
        effect_scores = {}
        effect_reasons = {}
        
        for effect, weights in self.EFFECT_WEIGHTS.items():
            score = 0.0
            reasons = []
            
            # Color contributions
            for color_key, weight in weights.get('color', []):
                if color_key in color_scores:
                    color_score = color_scores[color_key]
                    score += color_score * weight
                    if color_score > 0.25:
                        reasons.append(f"Color: {color_key} ({color_score:.0%})")
            
            # Shape contributions
            for shape_key, weight in weights.get('shape', []):
                if shape_key in shape_scores:
                    shape_score = shape_scores[shape_key]
                    score += shape_score * weight
                    if shape_score > 0.3:
                        reasons.append(f"Shape: {shape_key} ({shape_score:.0%})")
            
            # Edge contributions
            for edge_key, weight in weights.get('edge', []):
                if edge_key in edge_scores:
                    edge_score = edge_scores[edge_key]
                    score += edge_score * weight
                    if edge_score > 0.3:
                        reasons.append(f"Edge: {edge_key} ({edge_score:.0%})")
            
            # Bonus contributions
            for bonus_key, weight in weights.get('bonus', {}).items():
                if bonus_key in bonuses:
                    bonus_val = bonuses[bonus_key]
                    score += bonus_val * weight
                    if bonus_val > 0.5:
                        reasons.append(f"Bonus: {bonus_key}")
            
            # Layer name hints
            if effect in self.layer_hints and self.layer_hints[effect] > 0:
                score += self.layer_hints[effect]
                reasons.append(f"Name hint detected")
            
            effect_scores[effect] = score
            effect_reasons[effect] = reasons
        
        # Sort by score
        sorted_effects = sorted(effect_scores.items(), key=lambda x: -x[1])
        
        # Normalize scores relative to best
        max_score = sorted_effects[0][1] if sorted_effects else 1.0
        
        # Create suggestions
        suggestions = []
        for effect, score in sorted_effects:
            if score > 0.08:  # Lower threshold for more suggestions
                # Normalize confidence
                confidence = min(1.0, score / max(max_score, 0.5))
                
                suggestions.append(EffectSuggestion(
                    effect=effect,
                    confidence=confidence,
                    reasons=effect_reasons[effect]
                ))
        
        # Ensure we have at least one suggestion
        if not suggestions:
            suggestions.append(EffectSuggestion(
                effect='float',
                confidence=0.3,
                reasons=['Default fallback for unrecognized sprites']
            ))
        
        return AnalysisResult(
            suggestions=suggestions,
            color_scores=color_scores,
            shape_scores=shape_scores,
            edge_scores=edge_scores,
            sprite_type=sprite_type
        )
    
    def get_best_effect(self) -> Tuple[str, float]:
        """Quick method to get just the best effect"""
        result = self.analyze()
        return (result.best_effect or 'float', result.best_confidence)
    
    def explain(self) -> str:
        """Get a human-readable explanation of the analysis"""
        result = self.analyze()
        
        lines = [
            "=" * 50,
            "SPRITE ANALYSIS RESULTS",
            "=" * 50,
            f"Sprite: {self.sprite.name}",
            f"Size: {self.sprite.width}x{self.sprite.height}",
            f"Type: {result.sprite_type.upper()}",
            ""
        ]
        
        if result.suggestions:
            lines.append("RECOMMENDED EFFECTS:")
            lines.append("-" * 30)
            for i, suggestion in enumerate(result.suggestions[:5], 1):
                conf_bar = "█" * int(suggestion.confidence * 10) + "░" * (10 - int(suggestion.confidence * 10))
                lines.append(f"  {i}. {suggestion.effect.upper():12} [{conf_bar}] {suggestion.confidence:.0%}")
                for reason in suggestion.reasons[:3]:
                    lines.append(f"       • {reason}")
            lines.append("")
        
        # Detailed scores (abbreviated)
        lines.append("ANALYSIS DETAILS:")
        lines.append("-" * 30)
        
        lines.append("  Color signals:")
        for k, v in sorted(result.color_scores.items(), key=lambda x: -x[1])[:4]:
            if v > 0.1:
                lines.append(f"    {k}: {v:.0%}")
        
        lines.append("  Shape signals:")
        for k, v in sorted(result.shape_scores.items(), key=lambda x: -x[1])[:3]:
            if v > 0.2:
                lines.append(f"    {k}: {v:.0%}")
        
        lines.append("  Edge signals:")
        for k, v in sorted(result.edge_scores.items(), key=lambda x: -x[1])[:3]:
            if v > 0.2:
                lines.append(f"    {k}: {v:.0%}")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)


def detect_effect(sprite: Sprite) -> str:
    """Convenience function to detect effect for a sprite"""
    analyzer = SpriteAnalyzer(sprite)
    effect, _ = analyzer.get_best_effect()
    return effect
