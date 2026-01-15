"""
Detection System - Auto-detect best animation for sprites
"""

from .analyzer import SpriteAnalyzer
from .color import ColorAnalyzer
from .shape import ShapeAnalyzer
from .edges import EdgeAnalyzer
from .semantic import (
    # Enums
    SpriteCategory, SpriteType,
    # Feature data
    SpriteFeatures, EffectConfig,
    # Core detector
    SemanticDetector,
    # Effect configs
    EFFECT_CONFIGS,
    # Convenience functions
    detect_sprite_type,
    get_recommended_effects,
    analyze_sprite,
)

__all__ = [
    'SpriteAnalyzer', 'ColorAnalyzer', 'ShapeAnalyzer', 'EdgeAnalyzer',
    # Semantic detection
    'SpriteCategory', 'SpriteType',
    'SpriteFeatures', 'EffectConfig',
    'SemanticDetector',
    'EFFECT_CONFIGS',
    'detect_sprite_type',
    'get_recommended_effects',
    'analyze_sprite',
]
