"""
Animation Curves & Keyframe System - Precise animation control.

Provides professional-grade animation curves, keyframe interpolation,
and timeline-based animation control for any effect parameter.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Callable, Any, Union
from enum import Enum, auto
import json


class InterpolationType(Enum):
    """Types of keyframe interpolation."""
    CONSTANT = auto()    # No interpolation, jump to value
    LINEAR = auto()      # Linear interpolation
    EASE_IN = auto()     # Slow start
    EASE_OUT = auto()    # Slow end
    EASE_IN_OUT = auto() # Slow start and end
    BEZIER = auto()      # Custom bezier curve
    BOUNCE = auto()      # Bouncy interpolation
    ELASTIC = auto()     # Elastic spring interpolation
    BACK = auto()        # Overshoot interpolation
    STEPS = auto()       # Stepped interpolation


@dataclass
class BezierHandle:
    """Bezier curve control handle."""
    x: float  # Time offset (0-1 range relative to segment)
    y: float  # Value offset (0-1 range relative to segment)


@dataclass
class Keyframe:
    """A single keyframe in an animation curve."""
    time: float  # Time position (0-1 or frame number)
    value: float  # Value at this keyframe
    interpolation: InterpolationType = InterpolationType.EASE_IN_OUT
    
    # Bezier handles (for BEZIER interpolation)
    handle_in: Optional[BezierHandle] = None
    handle_out: Optional[BezierHandle] = None
    
    # For STEPS interpolation
    step_count: int = 4
    
    def __post_init__(self):
        # Default bezier handles
        if self.handle_in is None:
            self.handle_in = BezierHandle(-0.25, 0)
        if self.handle_out is None:
            self.handle_out = BezierHandle(0.25, 0)


class AnimationCurve:
    """
    Animation curve with keyframes and interpolation.
    
    Provides smooth interpolation between keyframes with support for
    multiple easing functions and bezier curves.
    """
    
    def __init__(self, keyframes: Optional[List[Keyframe]] = None):
        self.keyframes: List[Keyframe] = keyframes or []
        self._sort_keyframes()
    
    def _sort_keyframes(self) -> None:
        """Sort keyframes by time."""
        self.keyframes.sort(key=lambda k: k.time)
    
    def add_keyframe(self, time: float, value: float, 
                     interpolation: InterpolationType = InterpolationType.EASE_IN_OUT) -> 'AnimationCurve':
        """Add a keyframe to the curve."""
        self.keyframes.append(Keyframe(time, value, interpolation))
        self._sort_keyframes()
        return self
    
    def remove_keyframe(self, index: int) -> 'AnimationCurve':
        """Remove keyframe at index."""
        if 0 <= index < len(self.keyframes):
            self.keyframes.pop(index)
        return self
    
    def clear(self) -> 'AnimationCurve':
        """Remove all keyframes."""
        self.keyframes.clear()
        return self
    
    @staticmethod
    def _ease_in(t: float) -> float:
        """Quadratic ease in."""
        return t * t
    
    @staticmethod
    def _ease_out(t: float) -> float:
        """Quadratic ease out."""
        return 1 - (1 - t) * (1 - t)
    
    @staticmethod
    def _ease_in_out(t: float) -> float:
        """Quadratic ease in/out."""
        if t < 0.5:
            return 2 * t * t
        return 1 - (-2 * t + 2) ** 2 / 2
    
    @staticmethod
    def _bounce(t: float) -> float:
        """Bounce easing."""
        if t < 1 / 2.75:
            return 7.5625 * t * t
        elif t < 2 / 2.75:
            t -= 1.5 / 2.75
            return 7.5625 * t * t + 0.75
        elif t < 2.5 / 2.75:
            t -= 2.25 / 2.75
            return 7.5625 * t * t + 0.9375
        else:
            t -= 2.625 / 2.75
            return 7.5625 * t * t + 0.984375
    
    @staticmethod
    def _elastic(t: float) -> float:
        """Elastic easing."""
        if t == 0 or t == 1:
            return t
        p = 0.3
        s = p / 4
        return np.power(2, -10 * t) * np.sin((t - s) * (2 * np.pi) / p) + 1
    
    @staticmethod
    def _back(t: float) -> float:
        """Back/overshoot easing."""
        c1 = 1.70158
        c3 = c1 + 1
        return 1 + c3 * np.power(t - 1, 3) + c1 * np.power(t - 1, 2)
    
    def _bezier(self, t: float, k0: Keyframe, k1: Keyframe) -> float:
        """Cubic bezier interpolation."""
        # Control points
        p0 = (k0.time, k0.value)
        p3 = (k1.time, k1.value)
        
        # Handle positions
        p1 = (k0.time + k0.handle_out.x * (k1.time - k0.time),
              k0.value + k0.handle_out.y * (k1.value - k0.value))
        p2 = (k1.time + k1.handle_in.x * (k1.time - k0.time),
              k1.value + k1.handle_in.y * (k1.value - k0.value))
        
        # De Casteljau's algorithm for cubic bezier
        # First, we need to find t for given x (time)
        # For simplicity, we'll use the parameter directly
        
        one_minus_t = 1 - t
        
        # Cubic bezier formula
        value = (one_minus_t ** 3 * p0[1] +
                3 * one_minus_t ** 2 * t * p1[1] +
                3 * one_minus_t * t ** 2 * p2[1] +
                t ** 3 * p3[1])
        
        return value
    
    def _steps(self, t: float, k0: Keyframe, k1: Keyframe) -> float:
        """Stepped interpolation."""
        step = int(t * k1.step_count)
        return k0.value + (k1.value - k0.value) * (step / k1.step_count)
    
    def _interpolate(self, t: float, k0: Keyframe, k1: Keyframe) -> float:
        """Interpolate between two keyframes."""
        interp = k1.interpolation
        
        if interp == InterpolationType.CONSTANT:
            return k0.value
        
        elif interp == InterpolationType.LINEAR:
            return k0.value + (k1.value - k0.value) * t
        
        elif interp == InterpolationType.EASE_IN:
            eased_t = self._ease_in(t)
            return k0.value + (k1.value - k0.value) * eased_t
        
        elif interp == InterpolationType.EASE_OUT:
            eased_t = self._ease_out(t)
            return k0.value + (k1.value - k0.value) * eased_t
        
        elif interp == InterpolationType.EASE_IN_OUT:
            eased_t = self._ease_in_out(t)
            return k0.value + (k1.value - k0.value) * eased_t
        
        elif interp == InterpolationType.BEZIER:
            return self._bezier(t, k0, k1)
        
        elif interp == InterpolationType.BOUNCE:
            eased_t = self._bounce(t)
            return k0.value + (k1.value - k0.value) * eased_t
        
        elif interp == InterpolationType.ELASTIC:
            eased_t = self._elastic(t)
            return k0.value + (k1.value - k0.value) * eased_t
        
        elif interp == InterpolationType.BACK:
            eased_t = self._back(t)
            return k0.value + (k1.value - k0.value) * eased_t
        
        elif interp == InterpolationType.STEPS:
            return self._steps(t, k0, k1)
        
        # Default to linear
        return k0.value + (k1.value - k0.value) * t
    
    def evaluate(self, time: float) -> float:
        """
        Evaluate the curve at given time.
        
        Args:
            time: Time position (0-1 for normalized, or frame number)
            
        Returns:
            Interpolated value at the given time
        """
        if not self.keyframes:
            return 0.0
        
        if len(self.keyframes) == 1:
            return self.keyframes[0].value
        
        # Before first keyframe
        if time <= self.keyframes[0].time:
            return self.keyframes[0].value
        
        # After last keyframe
        if time >= self.keyframes[-1].time:
            return self.keyframes[-1].value
        
        # Find surrounding keyframes
        for i in range(len(self.keyframes) - 1):
            k0 = self.keyframes[i]
            k1 = self.keyframes[i + 1]
            
            if k0.time <= time <= k1.time:
                # Normalize time to 0-1 range within this segment
                segment_duration = k1.time - k0.time
                if segment_duration <= 0:
                    return k0.value
                
                local_t = (time - k0.time) / segment_duration
                return self._interpolate(local_t, k0, k1)
        
        return self.keyframes[-1].value
    
    def evaluate_normalized(self, t: float) -> float:
        """
        Evaluate curve with normalized time (0-1).
        Automatically maps to keyframe time range.
        """
        if not self.keyframes:
            return 0.0
        
        start_time = self.keyframes[0].time
        end_time = self.keyframes[-1].time
        
        actual_time = start_time + (end_time - start_time) * t
        return self.evaluate(actual_time)
    
    def to_dict(self) -> Dict:
        """Serialize curve to dictionary."""
        return {
            "keyframes": [
                {
                    "time": k.time,
                    "value": k.value,
                    "interpolation": k.interpolation.name,
                    "handle_in": {"x": k.handle_in.x, "y": k.handle_in.y} if k.handle_in else None,
                    "handle_out": {"x": k.handle_out.x, "y": k.handle_out.y} if k.handle_out else None,
                    "step_count": k.step_count
                }
                for k in self.keyframes
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AnimationCurve':
        """Deserialize curve from dictionary."""
        keyframes = []
        for kf_data in data.get("keyframes", []):
            handle_in = None
            handle_out = None
            
            if kf_data.get("handle_in"):
                handle_in = BezierHandle(kf_data["handle_in"]["x"], kf_data["handle_in"]["y"])
            if kf_data.get("handle_out"):
                handle_out = BezierHandle(kf_data["handle_out"]["x"], kf_data["handle_out"]["y"])
            
            keyframe = Keyframe(
                time=kf_data["time"],
                value=kf_data["value"],
                interpolation=InterpolationType[kf_data.get("interpolation", "EASE_IN_OUT")],
                handle_in=handle_in,
                handle_out=handle_out,
                step_count=kf_data.get("step_count", 4)
            )
            keyframes.append(keyframe)
        
        return cls(keyframes)


# Preset curves for common animations
class PresetCurves:
    """Collection of commonly used animation curves."""
    
    @staticmethod
    def linear() -> AnimationCurve:
        """Simple linear interpolation 0 to 1."""
        return AnimationCurve([
            Keyframe(0, 0, InterpolationType.LINEAR),
            Keyframe(1, 1, InterpolationType.LINEAR)
        ])
    
    @staticmethod
    def ease_in_out() -> AnimationCurve:
        """Standard ease in/out curve."""
        return AnimationCurve([
            Keyframe(0, 0, InterpolationType.EASE_IN_OUT),
            Keyframe(1, 1, InterpolationType.EASE_IN_OUT)
        ])
    
    @staticmethod
    def bounce_settle() -> AnimationCurve:
        """Bounce and settle animation."""
        return AnimationCurve([
            Keyframe(0, 0, InterpolationType.EASE_OUT),
            Keyframe(0.4, 1.2, InterpolationType.EASE_IN_OUT),
            Keyframe(0.6, 0.9, InterpolationType.EASE_IN_OUT),
            Keyframe(0.8, 1.05, InterpolationType.EASE_IN_OUT),
            Keyframe(1, 1, InterpolationType.EASE_IN_OUT)
        ])
    
    @staticmethod
    def overshoot() -> AnimationCurve:
        """Overshoot then settle."""
        return AnimationCurve([
            Keyframe(0, 0, InterpolationType.EASE_OUT),
            Keyframe(0.7, 1.15, InterpolationType.EASE_IN_OUT),
            Keyframe(1, 1, InterpolationType.EASE_IN_OUT)
        ])
    
    @staticmethod
    def anticipation() -> AnimationCurve:
        """Pull back before moving forward."""
        return AnimationCurve([
            Keyframe(0, 0, InterpolationType.EASE_OUT),
            Keyframe(0.2, -0.1, InterpolationType.EASE_IN_OUT),
            Keyframe(1, 1, InterpolationType.EASE_OUT)
        ])
    
    @staticmethod
    def anticipation_overshoot() -> AnimationCurve:
        """Pull back, overshoot, then settle."""
        return AnimationCurve([
            Keyframe(0, 0, InterpolationType.EASE_OUT),
            Keyframe(0.15, -0.08, InterpolationType.EASE_IN_OUT),
            Keyframe(0.75, 1.12, InterpolationType.EASE_IN_OUT),
            Keyframe(1, 1, InterpolationType.EASE_IN_OUT)
        ])
    
    @staticmethod
    def elastic() -> AnimationCurve:
        """Elastic spring animation."""
        return AnimationCurve([
            Keyframe(0, 0, InterpolationType.LINEAR),
            Keyframe(1, 1, InterpolationType.ELASTIC)
        ])
    
    @staticmethod
    def heartbeat() -> AnimationCurve:
        """Heartbeat/pulse pattern."""
        return AnimationCurve([
            Keyframe(0, 1, InterpolationType.EASE_IN_OUT),
            Keyframe(0.15, 1.3, InterpolationType.EASE_OUT),
            Keyframe(0.3, 1, InterpolationType.EASE_IN),
            Keyframe(0.45, 1.15, InterpolationType.EASE_OUT),
            Keyframe(0.6, 1, InterpolationType.EASE_IN_OUT),
            Keyframe(1, 1, InterpolationType.EASE_IN_OUT)
        ])
    
    @staticmethod
    def breathing() -> AnimationCurve:
        """Slow breathing rhythm."""
        return AnimationCurve([
            Keyframe(0, 1, InterpolationType.EASE_IN_OUT),
            Keyframe(0.4, 1.08, InterpolationType.EASE_IN_OUT),
            Keyframe(0.5, 1.1, InterpolationType.EASE_IN_OUT),
            Keyframe(0.9, 0.95, InterpolationType.EASE_IN_OUT),
            Keyframe(1, 1, InterpolationType.EASE_IN_OUT)
        ])
    
    @staticmethod
    def squash_stretch() -> AnimationCurve:
        """Classic squash and stretch."""
        return AnimationCurve([
            Keyframe(0, 1, InterpolationType.EASE_IN),
            Keyframe(0.25, 0.7, InterpolationType.EASE_OUT),  # Squash
            Keyframe(0.5, 1.3, InterpolationType.EASE_IN_OUT),  # Stretch
            Keyframe(0.75, 0.9, InterpolationType.EASE_IN_OUT),
            Keyframe(1, 1, InterpolationType.EASE_OUT)
        ])
    
    @staticmethod
    def wobble() -> AnimationCurve:
        """Wobble/jiggle motion."""
        return AnimationCurve([
            Keyframe(0, 0, InterpolationType.LINEAR),
            Keyframe(0.2, 1.2, InterpolationType.EASE_OUT),
            Keyframe(0.4, -0.8, InterpolationType.EASE_IN_OUT),
            Keyframe(0.6, 0.5, InterpolationType.EASE_IN_OUT),
            Keyframe(0.8, -0.2, InterpolationType.EASE_IN_OUT),
            Keyframe(1, 0, InterpolationType.EASE_IN_OUT)
        ])


@dataclass
class KeyframeAnimationConfig:
    """Configuration for keyframe-driven animation."""
    # Parameter curves (parameter_name -> AnimationCurve)
    curves: Dict[str, AnimationCurve] = field(default_factory=dict)
    
    # Timing
    duration_frames: int = 12
    loop: bool = True
    ping_pong: bool = False  # Reverse on loop
    
    # Default interpolation for new keyframes
    default_interpolation: InterpolationType = InterpolationType.EASE_IN_OUT


class KeyframeAnimator:
    """
    Keyframe animator that applies curves to effect parameters.
    
    Usage:
        animator = KeyframeAnimator()
        animator.add_curve("intensity", PresetCurves.bounce_settle())
        animator.add_curve("rotation", AnimationCurve([
            Keyframe(0, 0),
            Keyframe(0.5, 180),
            Keyframe(1, 360)
        ]))
        
        for frame in range(12):
            params = animator.get_values(frame, 12)
            # params = {"intensity": 0.8, "rotation": 90.0}
    """
    
    def __init__(self, config: Optional[KeyframeAnimationConfig] = None):
        self.config = config or KeyframeAnimationConfig()
        self.curves: Dict[str, AnimationCurve] = {}
    
    def add_curve(self, parameter: str, curve: AnimationCurve) -> 'KeyframeAnimator':
        """Add an animation curve for a parameter."""
        self.curves[parameter] = curve
        return self
    
    def remove_curve(self, parameter: str) -> 'KeyframeAnimator':
        """Remove animation curve for a parameter."""
        self.curves.pop(parameter, None)
        return self
    
    def get_value(self, parameter: str, frame_idx: int, total_frames: int) -> Optional[float]:
        """Get interpolated value for a parameter at given frame."""
        if parameter not in self.curves:
            return None
        
        # Calculate normalized time
        t = frame_idx / max(1, total_frames - 1)
        
        # Handle ping-pong
        if self.config.ping_pong and self.config.loop:
            # Double the period, reverse in second half
            if t > 0.5:
                t = 1 - (t - 0.5) * 2
            else:
                t = t * 2
        
        return self.curves[parameter].evaluate_normalized(t)
    
    def get_values(self, frame_idx: int, total_frames: int) -> Dict[str, float]:
        """Get all parameter values for given frame."""
        result = {}
        for parameter in self.curves:
            value = self.get_value(parameter, frame_idx, total_frames)
            if value is not None:
                result[parameter] = value
        return result
    
    def to_dict(self) -> Dict:
        """Serialize animator to dictionary."""
        return {
            "config": {
                "duration_frames": self.config.duration_frames,
                "loop": self.config.loop,
                "ping_pong": self.config.ping_pong,
                "default_interpolation": self.config.default_interpolation.name
            },
            "curves": {
                name: curve.to_dict()
                for name, curve in self.curves.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'KeyframeAnimator':
        """Deserialize animator from dictionary."""
        config_data = data.get("config", {})
        config = KeyframeAnimationConfig(
            duration_frames=config_data.get("duration_frames", 12),
            loop=config_data.get("loop", True),
            ping_pong=config_data.get("ping_pong", False),
            default_interpolation=InterpolationType[config_data.get("default_interpolation", "EASE_IN_OUT")]
        )
        
        animator = cls(config)
        
        for name, curve_data in data.get("curves", {}).items():
            animator.add_curve(name, AnimationCurve.from_dict(curve_data))
        
        return animator
    
    def save(self, filepath: str) -> None:
        """Save animator to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'KeyframeAnimator':
        """Load animator from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# Effect wrapper that uses keyframe animation

from .base import BaseEffect, EffectConfig

@dataclass
class KeyframeEffectConfig(EffectConfig):
    """Configuration for keyframe-driven effect."""
    base_effect: str = "wobble"  # Base effect to animate
    curve_preset: str = "ease_in_out"  # Preset curve to use
    parameter: str = "intensity"  # Parameter to animate
    value_min: float = 0.0  # Minimum value
    value_max: float = 1.0  # Maximum value
    
    # Additional curves as JSON
    additional_curves: Optional[str] = None
    
    seed: Optional[int] = None


class KeyframeEffect(BaseEffect):
    """Effect wrapper that applies keyframe animation to parameters."""
    
    name = "keyframe"
    description = "Apply keyframe animation curves to any effect parameter"
    
    config_class = KeyframeEffectConfig
    
    def __init__(self, config: KeyframeEffectConfig):
        super().__init__(config)
        self._base_effect: Optional[BaseEffect] = None
        self._animator = KeyframeAnimator()
        self._setup_animator()
    
    def _setup_animator(self) -> None:
        """Set up animator with configured curves."""
        # Get preset curve
        preset_map = {
            "linear": PresetCurves.linear,
            "ease_in_out": PresetCurves.ease_in_out,
            "bounce": PresetCurves.bounce_settle,
            "overshoot": PresetCurves.overshoot,
            "anticipation": PresetCurves.anticipation,
            "elastic": PresetCurves.elastic,
            "heartbeat": PresetCurves.heartbeat,
            "breathing": PresetCurves.breathing,
            "squash_stretch": PresetCurves.squash_stretch,
            "wobble": PresetCurves.wobble,
        }
        
        preset_func = preset_map.get(self.config.curve_preset.lower(), PresetCurves.ease_in_out)
        self._animator.add_curve(self.config.parameter, preset_func())
        
        # Parse additional curves from JSON
        if self.config.additional_curves:
            try:
                additional = json.loads(self.config.additional_curves)
                for param, curve_data in additional.items():
                    self._animator.add_curve(param, AnimationCurve.from_dict(curve_data))
            except (json.JSONDecodeError, KeyError):
                pass
    
    def _get_base_effect(self) -> Optional[BaseEffect]:
        """Get or create base effect instance."""
        if self._base_effect is None:
            from . import EFFECTS
            from .base import EffectConfig
            effect_class = EFFECTS.get(self.config.base_effect)
            if effect_class:
                # Create with default config - handle missing config_class
                if hasattr(effect_class, 'config_class') and effect_class.config_class:
                    config = effect_class.config_class()
                else:
                    config = EffectConfig()
                self._base_effect = effect_class(config)
        return self._base_effect
    
    def process_frame(self, image: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        """Process frame with keyframe-animated parameters."""
        base = self._get_base_effect()
        if base is None:
            return image
        
        # Get animated parameter values
        values = self._animator.get_values(frame_idx, total_frames)
        
        # Map to actual value range
        for param, normalized_value in values.items():
            actual_value = self.config.value_min + (self.config.value_max - self.config.value_min) * normalized_value
            
            # Apply to base effect config
            if hasattr(base.config, param):
                setattr(base.config, param, actual_value)
        
        # Process with base effect
        return base.process_frame(image, frame_idx, total_frames)
    
    def apply(self, sprite) -> list:
        """Apply keyframe effect to sprite and return animation frames."""
        from src.core import Sprite
        
        frames = []
        for i in range(self.config.frame_count):
            pixels = self.process_frame(sprite.pixels.copy(), i, self.config.frame_count)
            frame = Sprite(
                width=sprite.width,
                height=sprite.height,
                pixels=pixels,
                name=f"{sprite.name}_keyframe_{i}",
                source_path=sprite.source_path
            )
            frames.append(frame)
        return frames


# Utility functions for creating common animation patterns

def create_attack_animation() -> KeyframeAnimator:
    """Create attack animation timing (anticipation -> strike -> recovery)."""
    animator = KeyframeAnimator()
    
    # Position/movement curve
    animator.add_curve("position", AnimationCurve([
        Keyframe(0, 0, InterpolationType.EASE_IN),  # Start
        Keyframe(0.2, -0.1, InterpolationType.EASE_IN_OUT),  # Pull back
        Keyframe(0.4, 1.2, InterpolationType.EASE_OUT),  # Strike
        Keyframe(0.6, 1.0, InterpolationType.EASE_IN_OUT),  # Impact
        Keyframe(1, 0, InterpolationType.EASE_IN_OUT)  # Recovery
    ]))
    
    # Blur/motion intensity
    animator.add_curve("blur", AnimationCurve([
        Keyframe(0, 0, InterpolationType.LINEAR),
        Keyframe(0.3, 0.2, InterpolationType.LINEAR),
        Keyframe(0.4, 1.0, InterpolationType.EASE_OUT),  # Max blur at strike
        Keyframe(0.5, 0.5, InterpolationType.EASE_IN_OUT),
        Keyframe(1, 0, InterpolationType.EASE_IN_OUT)
    ]))
    
    return animator


def create_jump_animation() -> KeyframeAnimator:
    """Create jump animation timing (crouch -> jump -> fall -> land)."""
    animator = KeyframeAnimator()
    
    # Height curve
    animator.add_curve("height", AnimationCurve([
        Keyframe(0, 0, InterpolationType.EASE_IN),  # Ground
        Keyframe(0.15, -0.1, InterpolationType.EASE_OUT),  # Crouch
        Keyframe(0.5, 1.0, InterpolationType.EASE_OUT),  # Peak
        Keyframe(0.85, 0, InterpolationType.EASE_IN),  # Land
        Keyframe(0.95, -0.05, InterpolationType.EASE_OUT),  # Compress
        Keyframe(1, 0, InterpolationType.EASE_OUT)  # Recover
    ]))
    
    # Squash/stretch
    animator.add_curve("scale_y", AnimationCurve([
        Keyframe(0, 1, InterpolationType.EASE_IN),
        Keyframe(0.15, 0.7, InterpolationType.EASE_OUT),  # Squash for crouch
        Keyframe(0.3, 1.3, InterpolationType.EASE_OUT),  # Stretch for jump
        Keyframe(0.5, 1.1, InterpolationType.EASE_IN_OUT),  # Slight stretch at peak
        Keyframe(0.85, 0.75, InterpolationType.EASE_IN),  # Squash for land
        Keyframe(1, 1, InterpolationType.EASE_OUT)  # Normal
    ]))
    
    return animator


def create_idle_animation() -> KeyframeAnimator:
    """Create idle breathing animation."""
    animator = KeyframeAnimator()
    
    animator.add_curve("scale", PresetCurves.breathing())
    animator.add_curve("offset_y", AnimationCurve([
        Keyframe(0, 0, InterpolationType.EASE_IN_OUT),
        Keyframe(0.5, -2, InterpolationType.EASE_IN_OUT),  # Slight rise
        Keyframe(1, 0, InterpolationType.EASE_IN_OUT)
    ]))
    
    return animator
