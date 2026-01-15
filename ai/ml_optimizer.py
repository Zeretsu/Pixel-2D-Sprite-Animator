"""
Machine Learning Enhancement - Adaptive parameter optimization
Learns from usage patterns to suggest optimal animation parameters
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SpriteFeatures:
    """Extracted features from a sprite for ML prediction"""
    
    width: int = 0
    height: int = 0
    pixel_count: int = 0          # Non-transparent pixels
    coverage: float = 0.0         # Ratio of non-transparent to total
    avg_luminance: float = 0.0    # Average brightness
    luminance_std: float = 0.0    # Brightness variation
    color_count: int = 0          # Unique colors
    edge_ratio: float = 0.0       # Edge pixels / total pixels
    aspect_ratio: float = 1.0     # width / height
    complexity: float = 0.0       # Estimated visual complexity
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for ML"""
        return np.array([
            self.width / 128.0,           # Normalize to typical sprite size
            self.height / 128.0,
            self.pixel_count / 16384.0,   # Normalize to 128x128
            self.coverage,
            self.avg_luminance,
            self.luminance_std,
            min(self.color_count / 64.0, 1.0),  # Cap at 64 colors
            self.edge_ratio,
            self.aspect_ratio,
            self.complexity,
        ], dtype=np.float32)
    
    @classmethod
    def from_vector(cls, vec: np.ndarray) -> 'SpriteFeatures':
        """Create from feature vector"""
        return cls(
            width=int(vec[0] * 128),
            height=int(vec[1] * 128),
            pixel_count=int(vec[2] * 16384),
            coverage=float(vec[3]),
            avg_luminance=float(vec[4]),
            luminance_std=float(vec[5]),
            color_count=int(vec[6] * 64),
            edge_ratio=float(vec[7]),
            aspect_ratio=float(vec[8]),
            complexity=float(vec[9]),
        )


@dataclass
class AnimationParams:
    """Animation parameters to predict/optimize"""
    
    effect: str = "float"
    frames: int = 8
    intensity: float = 1.0
    speed: float = 1.0
    
    def to_vector(self) -> np.ndarray:
        """Convert to parameter vector"""
        return np.array([
            self.frames / 24.0,      # Normalize (typical max 24)
            self.intensity / 2.0,    # Normalize (max 2.0)
            self.speed / 2.0,        # Normalize (max 2.0)
        ], dtype=np.float32)
    
    @classmethod
    def from_vector(cls, vec: np.ndarray, effect: str = "float") -> 'AnimationParams':
        """Create from parameter vector"""
        return cls(
            effect=effect,
            frames=max(4, min(24, int(vec[0] * 24))),
            intensity=max(0.1, min(2.0, float(vec[1] * 2.0))),
            speed=max(0.25, min(2.0, float(vec[2] * 2.0))),
        )


@dataclass
class FeedbackEntry:
    """User feedback on animation quality"""
    
    sprite_hash: str              # Hash of sprite for identification
    features: SpriteFeatures      # Sprite features at time of animation
    predicted: AnimationParams    # What we predicted
    actual: AnimationParams       # What user actually used
    timestamp: str = ""           # When feedback was recorded
    rating: Optional[float] = None  # Optional 1-5 rating
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


# ============================================================================
# Feature Extraction
# ============================================================================

class FeatureExtractor:
    """Extract ML features from sprites"""
    
    # Luminance coefficients (Rec. 709)
    LUMA_R = 0.2126
    LUMA_G = 0.7152
    LUMA_B = 0.0722
    
    @classmethod
    def extract(cls, sprite: np.ndarray) -> SpriteFeatures:
        """
        Extract features from sprite for ML prediction.
        
        Args:
            sprite: RGBA numpy array
            
        Returns:
            SpriteFeatures with extracted values
        """
        h, w = sprite.shape[:2]
        
        # Alpha mask
        alpha = sprite[:, :, 3] if sprite.shape[2] == 4 else np.ones((h, w), dtype=np.uint8) * 255
        visible_mask = alpha > 10
        pixel_count = visible_mask.sum()
        
        features = SpriteFeatures(
            width=w,
            height=h,
            pixel_count=int(pixel_count),
            coverage=pixel_count / (w * h) if w * h > 0 else 0,
            aspect_ratio=w / h if h > 0 else 1.0,
        )
        
        if pixel_count == 0:
            return features
        
        # Luminance statistics
        rgb = sprite[:, :, :3].astype(np.float32) / 255.0
        luminance = (
            cls.LUMA_R * rgb[:, :, 0] +
            cls.LUMA_G * rgb[:, :, 1] +
            cls.LUMA_B * rgb[:, :, 2]
        )
        
        vis_lum = luminance[visible_mask]
        features.avg_luminance = float(vis_lum.mean())
        features.luminance_std = float(vis_lum.std())
        
        # Color count (quantized for efficiency)
        vis_colors = sprite[visible_mask, :3]
        quantized = (vis_colors // 16).astype(np.uint32)
        color_keys = quantized[:, 0] * 256 + quantized[:, 1] * 16 + quantized[:, 2]
        features.color_count = len(np.unique(color_keys))
        
        # Edge detection
        edge_mask = cls._find_edges(visible_mask)
        features.edge_ratio = edge_mask.sum() / pixel_count if pixel_count > 0 else 0
        
        # Complexity estimate (combines several factors)
        features.complexity = cls._estimate_complexity(
            features.color_count,
            features.luminance_std,
            features.edge_ratio,
            features.coverage
        )
        
        return features
    
    @classmethod
    def _find_edges(cls, mask: np.ndarray) -> np.ndarray:
        """Find edge pixels in mask"""
        h, w = mask.shape
        edge = np.zeros_like(mask)
        
        # Check 4 neighbors (faster than 8)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = np.zeros_like(mask)
            
            src_y = slice(max(0, -dy), min(h, h - dy))
            src_x = slice(max(0, -dx), min(w, w - dx))
            dst_y = slice(max(0, dy), min(h, h + dy))
            dst_x = slice(max(0, dx), min(w, w + dx))
            
            shifted[dst_y, dst_x] = mask[src_y, src_x]
            edge |= mask & ~shifted
        
        return edge
    
    @classmethod
    def _estimate_complexity(
        cls,
        color_count: int,
        lum_std: float,
        edge_ratio: float,
        coverage: float
    ) -> float:
        """Estimate visual complexity 0-1"""
        # Weighted combination of factors
        color_score = min(color_count / 32.0, 1.0)  # More colors = more complex
        contrast_score = min(lum_std * 4, 1.0)      # More contrast = more complex
        detail_score = min(edge_ratio * 5, 1.0)    # More edges = more complex
        size_score = min(coverage * 2, 1.0)        # Larger sprites = more complex
        
        return (
            0.3 * color_score +
            0.2 * contrast_score +
            0.3 * detail_score +
            0.2 * size_score
        )
    
    @classmethod
    def compute_hash(cls, sprite: np.ndarray) -> str:
        """Compute hash for sprite identification"""
        # Use downsampled version for faster hashing
        h, w = sprite.shape[:2]
        step = max(1, min(h, w) // 16)
        sampled = sprite[::step, ::step, :].tobytes()
        return hashlib.md5(sampled).hexdigest()[:16]


# ============================================================================
# Simple Neural Network (numpy-only)
# ============================================================================

class SimpleNN:
    """
    Minimal neural network for parameter prediction.
    Uses only numpy, no external ML libraries required.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        learning_rate: float = 0.01
    ):
        self.learning_rate = learning_rate
        
        # Initialize weights with Xavier initialization
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            
            self.weights.append(np.random.randn(fan_in, fan_out).astype(np.float32) * scale)
            self.biases.append(np.zeros(fan_out, dtype=np.float32))
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation"""
        return np.maximum(0, x)
    
    def _relu_grad(self, x: np.ndarray) -> np.ndarray:
        """ReLU gradient"""
        return (x > 0).astype(np.float32)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation for output"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        self._activations = [x]
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            
            if i < len(self.weights) - 1:
                x = self._relu(x)
            else:
                x = self._sigmoid(x)  # Output in [0, 1]
            
            self._activations.append(x)
        
        return x
    
    def backward(self, target: np.ndarray) -> float:
        """Backward pass with gradient descent"""
        output = self._activations[-1]
        
        # MSE loss gradient
        error = output - target
        loss = np.mean(error ** 2)
        
        # Backpropagation
        delta = error * output * (1 - output)  # Sigmoid gradient
        
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient for weights and biases
            prev_activation = self._activations[i]
            if prev_activation.ndim == 1:
                prev_activation = prev_activation.reshape(1, -1)
            if delta.ndim == 1:
                delta = delta.reshape(1, -1)
            
            w_grad = prev_activation.T @ delta
            b_grad = delta.sum(axis=0)
            
            # Update weights
            self.weights[i] -= self.learning_rate * w_grad.squeeze()
            self.biases[i] -= self.learning_rate * b_grad.squeeze()
            
            if i > 0:
                # Propagate error
                delta = (delta @ self.weights[i].T) * self._relu_grad(self._activations[i])
        
        return loss
    
    def train(self, x: np.ndarray, target: np.ndarray) -> float:
        """Single training step"""
        self.forward(x)
        return self.backward(target)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict without storing activations for backprop"""
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = self._relu(x)
            else:
                x = self._sigmoid(x)
        return x
    
    def save(self, path: Path) -> None:
        """Save model weights"""
        data = {
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'learning_rate': self.learning_rate,
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: Path) -> 'SimpleNN':
        """Load model weights"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        weights = [np.array(w, dtype=np.float32) for w in data['weights']]
        biases = [np.array(b, dtype=np.float32) for b in data['biases']]
        
        # Reconstruct layer sizes
        layer_sizes = [weights[0].shape[0]]
        for w in weights:
            layer_sizes.append(w.shape[1])
        
        model = cls(
            input_size=layer_sizes[0],
            hidden_sizes=layer_sizes[1:-1],
            output_size=layer_sizes[-1],
            learning_rate=data.get('learning_rate', 0.01)
        )
        model.weights = weights
        model.biases = biases
        
        return model


# ============================================================================
# Parameter Predictor
# ============================================================================

class ParameterPredictor:
    """
    Predicts optimal animation parameters using learned models.
    One model per effect type, trained on user feedback.
    """
    
    # Default parameters per effect (used before training)
    DEFAULTS = {
        'flame': {'frames': 8, 'intensity': 1.0, 'speed': 1.0},
        'water': {'frames': 12, 'intensity': 0.8, 'speed': 0.8},
        'float': {'frames': 16, 'intensity': 0.6, 'speed': 0.5},
        'sparkle': {'frames': 8, 'intensity': 1.2, 'speed': 1.0},
        'sway': {'frames': 12, 'intensity': 0.8, 'speed': 0.6},
        'pulse': {'frames': 8, 'intensity': 1.0, 'speed': 1.0},
        'smoke': {'frames': 16, 'intensity': 0.7, 'speed': 0.4},
        'wobble': {'frames': 8, 'intensity': 0.8, 'speed': 1.0},
        'glitch': {'frames': 6, 'intensity': 1.5, 'speed': 1.5},
        'shake': {'frames': 4, 'intensity': 1.0, 'speed': 2.0},
        'bounce': {'frames': 12, 'intensity': 1.0, 'speed': 1.0},
        'flicker': {'frames': 6, 'intensity': 1.2, 'speed': 1.5},
        'glow': {'frames': 16, 'intensity': 1.0, 'speed': 0.6},
        'dissolve': {'frames': 16, 'intensity': 1.0, 'speed': 0.8},
        'rainbow': {'frames': 24, 'intensity': 1.0, 'speed': 0.5},
        'spin': {'frames': 12, 'intensity': 1.0, 'speed': 1.0},
        'melt': {'frames': 16, 'intensity': 1.0, 'speed': 0.6},
        'electric': {'frames': 6, 'intensity': 1.5, 'speed': 1.5},
        'levitate': {'frames': 16, 'intensity': 0.8, 'speed': 0.5},
    }
    
    # Heuristic adjustments based on sprite features
    FEATURE_ADJUSTMENTS = {
        # Large sprites need more frames for smooth animation
        'size_frame_scale': 0.5,      # +50% frames for large sprites
        # Complex sprites benefit from lower intensity
        'complexity_intensity_scale': -0.3,
        # Small sprites can use faster animations
        'size_speed_scale': -0.3,
    }
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize predictor.
        
        Args:
            data_dir: Directory for storing models and feedback data
        """
        self.data_dir = data_dir or Path.home() / '.sprite-animator'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, SimpleNN] = {}
        self.feedback_history: List[FeedbackEntry] = []
        self.training_samples: Dict[str, int] = {}  # Samples per effect
        
        self._load_data()
    
    def predict(
        self,
        sprite: np.ndarray,
        effect: str,
        use_ml: bool = True
    ) -> AnimationParams:
        """
        Predict optimal parameters for sprite and effect.
        
        Args:
            sprite: RGBA sprite array
            effect: Effect name
            use_ml: Whether to use trained model (if available)
            
        Returns:
            AnimationParams with predicted values
        """
        features = FeatureExtractor.extract(sprite)
        
        # Start with defaults
        defaults = self.DEFAULTS.get(effect, self.DEFAULTS['float'])
        params = AnimationParams(
            effect=effect,
            frames=defaults['frames'],
            intensity=defaults['intensity'],
            speed=defaults['speed'],
        )
        
        # Apply heuristic adjustments
        params = self._apply_heuristics(params, features)
        
        # Apply ML model if trained
        if use_ml and effect in self.models:
            model = self.models[effect]
            
            # Only trust ML if we have enough training data
            if self.training_samples.get(effect, 0) >= 5:
                feature_vec = features.to_vector()
                prediction = model.predict(feature_vec)
                ml_params = AnimationParams.from_vector(prediction, effect)
                
                # Blend ML prediction with heuristics (more ML weight with more data)
                samples = self.training_samples.get(effect, 0)
                ml_weight = min(0.8, samples / 20.0)  # Max 80% ML weight
                
                params.frames = int(
                    (1 - ml_weight) * params.frames + 
                    ml_weight * ml_params.frames
                )
                params.intensity = (
                    (1 - ml_weight) * params.intensity + 
                    ml_weight * ml_params.intensity
                )
                params.speed = (
                    (1 - ml_weight) * params.speed + 
                    ml_weight * ml_params.speed
                )
        
        return params
    
    def _apply_heuristics(
        self,
        params: AnimationParams,
        features: SpriteFeatures
    ) -> AnimationParams:
        """Apply rule-based adjustments"""
        # Size-based adjustments
        size_factor = (features.width * features.height) / (64 * 64)  # Relative to 64x64
        
        if size_factor > 2:  # Large sprite
            params.frames = int(params.frames * (1 + self.FEATURE_ADJUSTMENTS['size_frame_scale'] * min(size_factor - 1, 2)))
            params.speed *= (1 + self.FEATURE_ADJUSTMENTS['size_speed_scale'] * min(size_factor - 1, 1))
        elif size_factor < 0.5:  # Small sprite
            params.frames = max(4, int(params.frames * 0.8))
            params.speed *= 1.2
        
        # Complexity-based adjustments
        if features.complexity > 0.7:  # Complex sprite
            params.intensity *= (1 + self.FEATURE_ADJUSTMENTS['complexity_intensity_scale'])
        
        # Ensure bounds
        params.frames = max(4, min(32, params.frames))
        params.intensity = max(0.1, min(2.0, params.intensity))
        params.speed = max(0.25, min(2.0, params.speed))
        
        return params
    
    def record_feedback(
        self,
        sprite: np.ndarray,
        predicted: AnimationParams,
        actual: AnimationParams,
        rating: Optional[float] = None
    ) -> None:
        """
        Record user feedback for learning.
        
        Args:
            sprite: The sprite that was animated
            predicted: Parameters we predicted
            actual: Parameters user actually used
            rating: Optional quality rating 1-5
        """
        features = FeatureExtractor.extract(sprite)
        sprite_hash = FeatureExtractor.compute_hash(sprite)
        
        entry = FeedbackEntry(
            sprite_hash=sprite_hash,
            features=features,
            predicted=predicted,
            actual=actual,
            rating=rating,
        )
        
        self.feedback_history.append(entry)
        
        # Train model with this feedback
        self._train_on_feedback(entry)
        
        # Save periodically
        if len(self.feedback_history) % 5 == 0:
            self._save_data()
    
    def _train_on_feedback(self, entry: FeedbackEntry) -> None:
        """Train model on a single feedback entry"""
        effect = entry.actual.effect
        
        # Create model if needed
        if effect not in self.models:
            self.models[effect] = SimpleNN(
                input_size=10,     # Feature vector size
                hidden_sizes=[16, 8],
                output_size=3,     # frames, intensity, speed
                learning_rate=0.05
            )
            self.training_samples[effect] = 0
        
        model = self.models[effect]
        
        # Train on the feedback
        x = entry.features.to_vector()
        y = entry.actual.to_vector()
        
        # Multiple passes for better learning
        for _ in range(10):
            model.train(x, y)
        
        self.training_samples[effect] = self.training_samples.get(effect, 0) + 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            'total_feedback': len(self.feedback_history),
            'samples_per_effect': dict(self.training_samples),
            'trained_effects': list(self.models.keys()),
            'average_adjustments': self._calculate_avg_adjustments(),
        }
    
    def _calculate_avg_adjustments(self) -> Dict[str, Dict[str, float]]:
        """Calculate average parameter adjustments per effect"""
        adjustments: Dict[str, Dict[str, List[float]]] = {}
        
        for entry in self.feedback_history:
            effect = entry.actual.effect
            if effect not in adjustments:
                adjustments[effect] = {'frames': [], 'intensity': [], 'speed': []}
            
            adjustments[effect]['frames'].append(
                entry.actual.frames - entry.predicted.frames
            )
            adjustments[effect]['intensity'].append(
                entry.actual.intensity - entry.predicted.intensity
            )
            adjustments[effect]['speed'].append(
                entry.actual.speed - entry.predicted.speed
            )
        
        # Convert to averages
        result = {}
        for effect, adj in adjustments.items():
            result[effect] = {
                'frames': np.mean(adj['frames']) if adj['frames'] else 0,
                'intensity': np.mean(adj['intensity']) if adj['intensity'] else 0,
                'speed': np.mean(adj['speed']) if adj['speed'] else 0,
            }
        
        return result
    
    def _save_data(self) -> None:
        """Save models and feedback history"""
        # Save models
        models_dir = self.data_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        
        for effect, model in self.models.items():
            model.save(models_dir / f'{effect}.json')
        
        # Save feedback history
        history_path = self.data_dir / 'feedback_history.json'
        history_data = []
        for entry in self.feedback_history[-1000:]:  # Keep last 1000
            history_data.append({
                'sprite_hash': entry.sprite_hash,
                'features': asdict(entry.features),
                'predicted': asdict(entry.predicted),
                'actual': asdict(entry.actual),
                'timestamp': entry.timestamp,
                'rating': entry.rating,
            })
        
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        # Save training samples count
        samples_path = self.data_dir / 'training_samples.json'
        with open(samples_path, 'w') as f:
            json.dump(self.training_samples, f)
    
    def _load_data(self) -> None:
        """Load saved models and history"""
        # Load models
        models_dir = self.data_dir / 'models'
        if models_dir.exists():
            for model_path in models_dir.glob('*.json'):
                effect = model_path.stem
                try:
                    self.models[effect] = SimpleNN.load(model_path)
                except Exception:
                    pass  # Skip corrupted models
        
        # Load feedback history
        history_path = self.data_dir / 'feedback_history.json'
        if history_path.exists():
            try:
                with open(history_path, 'r') as f:
                    history_data = json.load(f)
                
                for item in history_data:
                    entry = FeedbackEntry(
                        sprite_hash=item['sprite_hash'],
                        features=SpriteFeatures(**item['features']),
                        predicted=AnimationParams(**item['predicted']),
                        actual=AnimationParams(**item['actual']),
                        timestamp=item.get('timestamp', ''),
                        rating=item.get('rating'),
                    )
                    self.feedback_history.append(entry)
            except Exception:
                pass  # Start fresh if corrupted
        
        # Load training samples
        samples_path = self.data_dir / 'training_samples.json'
        if samples_path.exists():
            try:
                with open(samples_path, 'r') as f:
                    self.training_samples = json.load(f)
            except Exception:
                pass


# ============================================================================
# Effect Recommender
# ============================================================================

class EffectRecommender:
    """
    Recommends effects based on sprite characteristics.
    Uses both heuristics and learned preferences.
    """
    
    # Effect suitability scores based on sprite features
    EFFECT_PROFILES = {
        'flame': {
            'luminance_range': (0.4, 0.8),
            'color_preference': 'warm',
            'complexity_range': (0.3, 0.7),
        },
        'water': {
            'luminance_range': (0.3, 0.7),
            'color_preference': 'cool',
            'complexity_range': (0.2, 0.6),
        },
        'sparkle': {
            'luminance_range': (0.5, 1.0),
            'color_preference': 'bright',
            'complexity_range': (0.4, 0.8),
        },
        'glow': {
            'luminance_range': (0.5, 0.9),
            'color_preference': 'any',
            'complexity_range': (0.2, 0.8),
        },
        'float': {
            'luminance_range': (0.0, 1.0),
            'color_preference': 'any',
            'complexity_range': (0.0, 1.0),
        },
        'wobble': {
            'luminance_range': (0.0, 1.0),
            'color_preference': 'any',
            'complexity_range': (0.3, 0.8),
        },
        'pulse': {
            'luminance_range': (0.3, 0.8),
            'color_preference': 'any',
            'complexity_range': (0.2, 0.7),
        },
        'glitch': {
            'luminance_range': (0.0, 1.0),
            'color_preference': 'any',
            'complexity_range': (0.5, 1.0),
        },
    }
    
    def __init__(self, predictor: Optional[ParameterPredictor] = None):
        self.predictor = predictor
        self.effect_usage: Dict[str, int] = {}
    
    def recommend(
        self,
        sprite: np.ndarray,
        top_k: int = 3
    ) -> List[Tuple[str, float, AnimationParams]]:
        """
        Recommend top effects for a sprite.
        
        Args:
            sprite: RGBA sprite array
            top_k: Number of recommendations
            
        Returns:
            List of (effect_name, score, suggested_params)
        """
        features = FeatureExtractor.extract(sprite)
        
        # Score each effect
        scores = {}
        for effect in self.EFFECT_PROFILES:
            scores[effect] = self._score_effect(effect, features)
        
        # Sort by score
        sorted_effects = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top k with parameters
        recommendations = []
        for effect, score in sorted_effects[:top_k]:
            if self.predictor:
                params = self.predictor.predict(sprite, effect)
            else:
                defaults = ParameterPredictor.DEFAULTS.get(effect, {'frames': 8, 'intensity': 1.0, 'speed': 1.0})
                params = AnimationParams(effect=effect, **defaults)
            
            recommendations.append((effect, score, params))
        
        return recommendations
    
    def _score_effect(self, effect: str, features: SpriteFeatures) -> float:
        """Score how suitable an effect is for given features"""
        profile = self.EFFECT_PROFILES.get(effect)
        if not profile:
            return 0.5
        
        score = 1.0
        
        # Luminance fit
        lum_min, lum_max = profile['luminance_range']
        if lum_min <= features.avg_luminance <= lum_max:
            score *= 1.2
        elif features.avg_luminance < lum_min - 0.2 or features.avg_luminance > lum_max + 0.2:
            score *= 0.6
        
        # Complexity fit
        comp_min, comp_max = profile['complexity_range']
        if comp_min <= features.complexity <= comp_max:
            score *= 1.1
        
        # Usage frequency bonus (prefer effects user hasn't tried)
        usage = self.effect_usage.get(effect, 0)
        if usage == 0:
            score *= 1.1
        
        return score
    
    def record_usage(self, effect: str) -> None:
        """Record that an effect was used"""
        self.effect_usage[effect] = self.effect_usage.get(effect, 0) + 1


# ============================================================================
# Global Instance & Convenience Functions
# ============================================================================

# Global predictor instance
_predictor: Optional[ParameterPredictor] = None
_recommender: Optional[EffectRecommender] = None


def get_predictor() -> ParameterPredictor:
    """Get or create global predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = ParameterPredictor()
    return _predictor


def get_recommender() -> EffectRecommender:
    """Get or create global recommender instance"""
    global _recommender
    if _recommender is None:
        _recommender = EffectRecommender(get_predictor())
    return _recommender


def predict_params(sprite: np.ndarray, effect: str) -> AnimationParams:
    """Predict optimal parameters for sprite and effect"""
    return get_predictor().predict(sprite, effect)


def recommend_effects(sprite: np.ndarray, top_k: int = 3) -> List[Tuple[str, float, AnimationParams]]:
    """Get effect recommendations for sprite"""
    return get_recommender().recommend(sprite, top_k)


def record_feedback(
    sprite: np.ndarray,
    predicted: AnimationParams,
    actual: AnimationParams,
    rating: Optional[float] = None
) -> None:
    """Record user feedback for learning"""
    get_predictor().record_feedback(sprite, predicted, actual, rating)


def get_learning_stats() -> Dict[str, Any]:
    """Get ML learning statistics"""
    return get_predictor().get_statistics()


def extract_features(sprite: np.ndarray) -> SpriteFeatures:
    """Extract ML features from sprite"""
    return FeatureExtractor.extract(sprite)
