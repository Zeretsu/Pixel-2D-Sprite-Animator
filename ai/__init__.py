"""
AI-powered prompt interpretation and ML optimization for sprite animation
"""

from .prompt_interpreter import PromptInterpreter, AnimationIntent
from .ml_optimizer import (
    # Data structures
    SpriteFeatures, AnimationParams, FeedbackEntry,
    # Feature extraction
    FeatureExtractor,
    # Neural network
    SimpleNN,
    # Predictor
    ParameterPredictor,
    # Recommender
    EffectRecommender,
    # Convenience functions
    get_predictor, get_recommender,
    predict_params, recommend_effects,
    record_feedback, get_learning_stats,
    extract_features,
)

__all__ = [
    'PromptInterpreter', 'AnimationIntent',
    # ML Optimizer
    'SpriteFeatures', 'AnimationParams', 'FeedbackEntry',
    'FeatureExtractor',
    'SimpleNN',
    'ParameterPredictor',
    'EffectRecommender',
    'get_predictor', 'get_recommender',
    'predict_params', 'recommend_effects',
    'record_feedback', 'get_learning_stats',
    'extract_features',
]
