"""
Advanced Easing & Timing Control

Professional easing functions for pixel-perfect animations.
Inspired by the animation quality of Celeste, Hollow Knight, and classic Disney principles.

Easing Types:
- Linear: Constant speed (robotic)
- Quad/Cubic/Quart/Quint: Polynomial curves (smooth acceleration)
- Sine: Sinusoidal curves (gentle, natural)
- Expo: Exponential curves (dramatic)
- Circ: Circular curves (sharp)
- Back: Overshoot with anticipation (bouncy start/end)
- Elastic: Spring-like oscillation (wobbly)
- Bounce: Ball-bounce physics (playful)

Each type has three variants:
- ease_in: Starts slow, accelerates
- ease_out: Starts fast, decelerates  
- ease_in_out: Slow-fast-slow (most natural for UI)
"""

import numpy as np
from typing import Callable, List, Tuple, Union
from dataclasses import dataclass


# =============================================================================
# Core Easing Functions
# =============================================================================

def linear(t: float) -> float:
    """No easing - constant velocity"""
    return t


# -----------------------------------------------------------------------------
# Polynomial Easing (Quad, Cubic, Quart, Quint)
# -----------------------------------------------------------------------------

def ease_in_quad(t: float) -> float:
    """Quadratic ease in - accelerating from zero"""
    return t * t


def ease_out_quad(t: float) -> float:
    """Quadratic ease out - decelerating to zero"""
    return t * (2 - t)


def ease_in_out_quad(t: float) -> float:
    """Quadratic ease in/out - acceleration until halfway, then deceleration"""
    if t < 0.5:
        return 2 * t * t
    return -1 + (4 - 2 * t) * t


def ease_in_cubic(t: float) -> float:
    """Cubic ease in - accelerating from zero"""
    return t * t * t


def ease_out_cubic(t: float) -> float:
    """Cubic ease out - decelerating to zero"""
    t1 = t - 1
    return t1 * t1 * t1 + 1


def ease_in_out_cubic(t: float) -> float:
    """Cubic ease in/out - smooth S-curve, most versatile"""
    if t < 0.5:
        return 4 * t * t * t
    t1 = 2 * t - 2
    return 0.5 * t1 * t1 * t1 + 1


def ease_in_quart(t: float) -> float:
    """Quartic ease in"""
    return t * t * t * t


def ease_out_quart(t: float) -> float:
    """Quartic ease out"""
    t1 = t - 1
    return 1 - t1 * t1 * t1 * t1


def ease_in_out_quart(t: float) -> float:
    """Quartic ease in/out"""
    if t < 0.5:
        return 8 * t * t * t * t
    t1 = t - 1
    return 1 - 8 * t1 * t1 * t1 * t1


def ease_in_quint(t: float) -> float:
    """Quintic ease in"""
    return t * t * t * t * t


def ease_out_quint(t: float) -> float:
    """Quintic ease out"""
    t1 = t - 1
    return 1 + t1 * t1 * t1 * t1 * t1


def ease_in_out_quint(t: float) -> float:
    """Quintic ease in/out"""
    if t < 0.5:
        return 16 * t * t * t * t * t
    t1 = 2 * t - 2
    return 0.5 * t1 * t1 * t1 * t1 * t1 + 1


# -----------------------------------------------------------------------------
# Sinusoidal Easing
# -----------------------------------------------------------------------------

def ease_in_sine(t: float) -> float:
    """Sinusoidal ease in - gentle start"""
    return 1 - np.cos(t * np.pi / 2)


def ease_out_sine(t: float) -> float:
    """Sinusoidal ease out - gentle end"""
    return np.sin(t * np.pi / 2)


def ease_in_out_sine(t: float) -> float:
    """Sinusoidal ease in/out - very smooth, subtle"""
    return 0.5 * (1 - np.cos(np.pi * t))


# -----------------------------------------------------------------------------
# Exponential Easing
# -----------------------------------------------------------------------------

def ease_in_expo(t: float) -> float:
    """Exponential ease in - dramatic acceleration"""
    if t == 0:
        return 0
    return 2 ** (10 * (t - 1))


def ease_out_expo(t: float) -> float:
    """Exponential ease out - dramatic deceleration"""
    if t == 1:
        return 1
    return 1 - 2 ** (-10 * t)


def ease_in_out_expo(t: float) -> float:
    """Exponential ease in/out - very dramatic"""
    if t == 0:
        return 0
    if t == 1:
        return 1
    if t < 0.5:
        return 0.5 * 2 ** (20 * t - 10)
    return 1 - 0.5 * 2 ** (-20 * t + 10)


# -----------------------------------------------------------------------------
# Circular Easing
# -----------------------------------------------------------------------------

def ease_in_circ(t: float) -> float:
    """Circular ease in - sharp acceleration"""
    return 1 - np.sqrt(1 - t * t)


def ease_out_circ(t: float) -> float:
    """Circular ease out - sharp deceleration"""
    t1 = t - 1
    return np.sqrt(1 - t1 * t1)


def ease_in_out_circ(t: float) -> float:
    """Circular ease in/out"""
    if t < 0.5:
        return 0.5 * (1 - np.sqrt(1 - 4 * t * t))
    t1 = 2 * t - 2
    return 0.5 * (np.sqrt(1 - t1 * t1) + 1)


# -----------------------------------------------------------------------------
# Back Easing (Overshoot/Anticipation)
# -----------------------------------------------------------------------------

def ease_in_back(t: float, overshoot: float = 1.70158) -> float:
    """
    Back ease in - slight overshoot at the beginning (anticipation)
    
    Args:
        t: Progress 0-1
        overshoot: Controls overshoot amount (default 1.70158 = 10% overshoot)
    """
    return t * t * ((overshoot + 1) * t - overshoot)


def ease_out_back(t: float, overshoot: float = 1.70158) -> float:
    """
    Back ease out - slight overshoot at the end (follow-through)
    
    Great for UI elements snapping into place with a bit of bounce.
    """
    t1 = t - 1
    return t1 * t1 * ((overshoot + 1) * t1 + overshoot) + 1


def ease_in_out_back(t: float, overshoot: float = 1.70158) -> float:
    """
    Back ease in/out - anticipation and follow-through
    
    Perfect for dramatic UI transitions.
    """
    s = overshoot * 1.525
    if t < 0.5:
        return 0.5 * (4 * t * t * ((s + 1) * 2 * t - s))
    t1 = 2 * t - 2
    return 0.5 * (t1 * t1 * ((s + 1) * t1 + s) + 2)


# -----------------------------------------------------------------------------
# Elastic Easing (Spring/Wobbly)
# -----------------------------------------------------------------------------

def ease_in_elastic(t: float, amplitude: float = 1.0, period: float = 0.3) -> float:
    """
    Elastic ease in - spring-like at the start
    
    Args:
        t: Progress 0-1
        amplitude: Overshoot amplitude (1.0 = normal)
        period: Oscillation period (smaller = more oscillations)
    """
    if t == 0 or t == 1:
        return t
    
    s = period / (2 * np.pi) * np.arcsin(1 / amplitude) if amplitude >= 1 else period / 4
    t1 = t - 1
    return -(amplitude * 2 ** (10 * t1) * np.sin((t1 - s) * (2 * np.pi) / period))


def ease_out_elastic(t: float, amplitude: float = 1.0, period: float = 0.3) -> float:
    """
    Elastic ease out - spring-like at the end (bouncy landing)
    
    Perfect for items appearing, notifications, game pickups.
    """
    if t == 0 or t == 1:
        return t
    
    s = period / (2 * np.pi) * np.arcsin(1 / amplitude) if amplitude >= 1 else period / 4
    return amplitude * 2 ** (-10 * t) * np.sin((t - s) * (2 * np.pi) / period) + 1


def ease_in_out_elastic(t: float, amplitude: float = 1.0, period: float = 0.45) -> float:
    """
    Elastic ease in/out - spring-like at both ends
    
    Great for dramatic emphasis, magic effects.
    """
    if t == 0 or t == 1:
        return t
    
    s = period / (2 * np.pi) * np.arcsin(1 / amplitude) if amplitude >= 1 else period / 4
    
    if t < 0.5:
        t1 = 2 * t - 1
        return -0.5 * amplitude * 2 ** (10 * t1) * np.sin((t1 - s) * (2 * np.pi) / period)
    
    t1 = 2 * t - 1
    return 0.5 * amplitude * 2 ** (-10 * t1) * np.sin((t1 - s) * (2 * np.pi) / period) + 1


# -----------------------------------------------------------------------------
# Bounce Easing (Ball Physics)
# -----------------------------------------------------------------------------

def ease_out_bounce(t: float) -> float:
    """
    Bounce ease out - ball bouncing to rest
    
    Perfect for: landing animations, playful UI, game collectibles
    """
    if t < 1 / 2.75:
        return 7.5625 * t * t
    elif t < 2 / 2.75:
        t1 = t - 1.5 / 2.75
        return 7.5625 * t1 * t1 + 0.75
    elif t < 2.5 / 2.75:
        t1 = t - 2.25 / 2.75
        return 7.5625 * t1 * t1 + 0.9375
    else:
        t1 = t - 2.625 / 2.75
        return 7.5625 * t1 * t1 + 0.984375


def ease_in_bounce(t: float) -> float:
    """Bounce ease in - reverse bounce"""
    return 1 - ease_out_bounce(1 - t)


def ease_in_out_bounce(t: float) -> float:
    """Bounce ease in/out"""
    if t < 0.5:
        return 0.5 * ease_in_bounce(2 * t)
    return 0.5 * ease_out_bounce(2 * t - 1) + 0.5


# =============================================================================
# Custom Bezier Curves
# =============================================================================

@dataclass
class BezierCurve:
    """
    Cubic Bezier easing curve (like CSS cubic-bezier)
    
    Control points: P0=(0,0), P1=(x1,y1), P2=(x2,y2), P3=(1,1)
    
    Common presets:
        ease:        (0.25, 0.1, 0.25, 1.0)
        ease-in:     (0.42, 0, 1.0, 1.0)
        ease-out:    (0, 0, 0.58, 1.0)
        ease-in-out: (0.42, 0, 0.58, 1.0)
    """
    x1: float
    y1: float
    x2: float
    y2: float
    
    def __call__(self, t: float) -> float:
        """Evaluate bezier at time t"""
        return self._sample_curve_y(self._solve_curve_x(t))
    
    def _solve_curve_x(self, x: float, epsilon: float = 1e-6) -> float:
        """Newton-Raphson to find t for given x"""
        t = x
        for _ in range(8):
            x_at_t = self._sample_curve_x(t) - x
            if abs(x_at_t) < epsilon:
                return t
            d = self._sample_curve_x_derivative(t)
            if abs(d) < epsilon:
                break
            t -= x_at_t / d
        
        # Fallback: bisection
        t0, t1 = 0.0, 1.0
        t = x
        while t0 < t1:
            x_at_t = self._sample_curve_x(t)
            if abs(x_at_t - x) < epsilon:
                return t
            if x > x_at_t:
                t0 = t
            else:
                t1 = t
            t = (t0 + t1) / 2
        return t
    
    def _sample_curve_x(self, t: float) -> float:
        return ((1 - 3 * self.x2 + 3 * self.x1) * t + (3 * self.x2 - 6 * self.x1)) * t * t + 3 * self.x1 * t
    
    def _sample_curve_y(self, t: float) -> float:
        return ((1 - 3 * self.y2 + 3 * self.y1) * t + (3 * self.y2 - 6 * self.y1)) * t * t + 3 * self.y1 * t
    
    def _sample_curve_x_derivative(self, t: float) -> float:
        return (3 * (1 - 3 * self.x2 + 3 * self.x1) * t + 2 * (3 * self.x2 - 6 * self.x1)) * t + 3 * self.x1


def custom_bezier(x1: float, y1: float, x2: float, y2: float) -> Callable[[float], float]:
    """
    Create a custom cubic bezier easing function.
    
    Args:
        x1, y1: First control point (affects curve start)
        x2, y2: Second control point (affects curve end)
    
    Returns:
        Easing function that takes t (0-1) and returns eased value
    
    Example:
        # CSS-like "ease" curve
        ease = custom_bezier(0.25, 0.1, 0.25, 1.0)
        
        # Snappy animation
        snappy = custom_bezier(0.4, 0.0, 0.2, 1.0)
    """
    curve = BezierCurve(x1, y1, x2, y2)
    return curve


# =============================================================================
# Easing Presets (Named Curves)
# =============================================================================

# CSS standard curves
CSS_EASE = custom_bezier(0.25, 0.1, 0.25, 1.0)
CSS_EASE_IN = custom_bezier(0.42, 0, 1.0, 1.0)
CSS_EASE_OUT = custom_bezier(0, 0, 0.58, 1.0)
CSS_EASE_IN_OUT = custom_bezier(0.42, 0, 0.58, 1.0)

# Material Design curves
MATERIAL_STANDARD = custom_bezier(0.4, 0.0, 0.2, 1.0)      # Most common
MATERIAL_DECELERATE = custom_bezier(0.0, 0.0, 0.2, 1.0)    # Entering elements
MATERIAL_ACCELERATE = custom_bezier(0.4, 0.0, 1.0, 1.0)    # Exiting elements

# Game-feel curves (inspired by Celeste, Hollow Knight)
GAME_SNAPPY = custom_bezier(0.5, 0.0, 0.1, 1.0)            # Quick, responsive
GAME_HEAVY = custom_bezier(0.7, 0.0, 0.3, 1.0)             # Weighty objects
GAME_FLOATY = custom_bezier(0.2, 0.0, 0.4, 1.0)            # Light, airy


# =============================================================================
# Easing Registry & Utilities
# =============================================================================

EASING_FUNCTIONS = {
    # Linear
    'linear': linear,
    
    # Polynomial
    'ease_in_quad': ease_in_quad,
    'ease_out_quad': ease_out_quad,
    'ease_in_out_quad': ease_in_out_quad,
    'ease_in_cubic': ease_in_cubic,
    'ease_out_cubic': ease_out_cubic,
    'ease_in_out_cubic': ease_in_out_cubic,
    'ease_in_quart': ease_in_quart,
    'ease_out_quart': ease_out_quart,
    'ease_in_out_quart': ease_in_out_quart,
    'ease_in_quint': ease_in_quint,
    'ease_out_quint': ease_out_quint,
    'ease_in_out_quint': ease_in_out_quint,
    
    # Sinusoidal
    'ease_in_sine': ease_in_sine,
    'ease_out_sine': ease_out_sine,
    'ease_in_out_sine': ease_in_out_sine,
    
    # Exponential
    'ease_in_expo': ease_in_expo,
    'ease_out_expo': ease_out_expo,
    'ease_in_out_expo': ease_in_out_expo,
    
    # Circular
    'ease_in_circ': ease_in_circ,
    'ease_out_circ': ease_out_circ,
    'ease_in_out_circ': ease_in_out_circ,
    
    # Back (overshoot)
    'ease_in_back': ease_in_back,
    'ease_out_back': ease_out_back,
    'ease_in_out_back': ease_in_out_back,
    
    # Elastic (spring)
    'ease_in_elastic': ease_in_elastic,
    'ease_out_elastic': ease_out_elastic,
    'ease_in_out_elastic': ease_in_out_elastic,
    
    # Bounce
    'ease_in_bounce': ease_in_bounce,
    'ease_out_bounce': ease_out_bounce,
    'ease_in_out_bounce': ease_in_out_bounce,
}


def get_easing(name: str) -> Callable[[float], float]:
    """
    Get an easing function by name.
    
    Args:
        name: Easing function name (e.g., 'ease_out_elastic')
    
    Returns:
        Easing function
    
    Raises:
        ValueError: If easing name not found
    """
    if name not in EASING_FUNCTIONS:
        available = ', '.join(sorted(EASING_FUNCTIONS.keys()))
        raise ValueError(f"Unknown easing '{name}'. Available: {available}")
    return EASING_FUNCTIONS[name]


def ease(t: float, easing: Union[str, Callable[[float], float]] = 'linear') -> float:
    """
    Apply easing to a value.
    
    Args:
        t: Progress value 0-1
        easing: Easing function name or callable
    
    Returns:
        Eased value
    """
    t = np.clip(t, 0.0, 1.0)
    
    if isinstance(easing, str):
        easing = get_easing(easing)
    
    return easing(t)


def ease_range(
    t: float,
    start: float,
    end: float,
    easing: Union[str, Callable[[float], float]] = 'linear'
) -> float:
    """
    Interpolate between start and end with easing.
    
    Args:
        t: Progress 0-1
        start: Start value
        end: End value
        easing: Easing function
    
    Returns:
        Interpolated value
    """
    eased_t = ease(t, easing)
    return start + (end - start) * eased_t


def generate_easing_curve(
    easing: Union[str, Callable[[float], float]],
    samples: int = 100
) -> np.ndarray:
    """
    Generate an array of eased values for visualization or caching.
    
    Args:
        easing: Easing function
        samples: Number of samples
    
    Returns:
        Array of shape (samples,) with eased values
    """
    if isinstance(easing, str):
        easing = get_easing(easing)
    
    t_values = np.linspace(0, 1, samples)
    return np.array([easing(t) for t in t_values])


# =============================================================================
# Animation Timing Utilities
# =============================================================================

def chain_easings(
    t: float,
    segments: List[Tuple[float, str]]
) -> float:
    """
    Chain multiple easing functions together.
    
    Args:
        t: Overall progress 0-1
        segments: List of (duration_fraction, easing_name) tuples
                  Durations should sum to 1.0
    
    Example:
        # Quick start, slow middle, quick end
        chain_easings(t, [
            (0.2, 'ease_out_quad'),   # 20% duration
            (0.6, 'linear'),          # 60% duration
            (0.2, 'ease_in_quad'),    # 20% duration
        ])
    """
    t = np.clip(t, 0.0, 1.0)
    
    accumulated = 0.0
    for duration, easing_name in segments:
        if t <= accumulated + duration:
            local_t = (t - accumulated) / duration
            return accumulated + duration * ease(local_t, easing_name)
        accumulated += duration
    
    return 1.0


def ping_pong(t: float, easing: Union[str, Callable[[float], float]] = 'linear') -> float:
    """
    Ease from 0→1→0 over t=0→1.
    
    Useful for looping animations that return to start.
    """
    if t < 0.5:
        return ease(t * 2, easing)
    return ease(2 - t * 2, easing)


def repeat(t: float, count: int, easing: Union[str, Callable[[float], float]] = 'linear') -> float:
    """
    Repeat an easing curve multiple times.
    
    Args:
        t: Progress 0-1
        count: Number of repetitions
        easing: Easing function
    """
    local_t = (t * count) % 1.0
    return ease(local_t, easing)


def delay_start(t: float, delay: float, easing: Union[str, Callable[[float], float]] = 'linear') -> float:
    """
    Add a delay before easing starts.
    
    Args:
        t: Progress 0-1
        delay: Delay fraction (0-1)
        easing: Easing function
    """
    if t < delay:
        return 0.0
    adjusted_t = (t - delay) / (1 - delay)
    return ease(adjusted_t, easing)


# =============================================================================
# Smoothstep Variants (for procedural animation)
# =============================================================================

def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """
    Hermite smoothstep - standard smooth interpolation.
    
    Clamps x to [edge0, edge1] and returns smooth 0-1 value.
    """
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-10), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(edge0: float, edge1: float, x: float) -> float:
    """
    Ken Perlin's improved smoothstep - zero 1st and 2nd derivative at edges.
    
    Even smoother than smoothstep, better for animation.
    """
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-10), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def inverse_smoothstep(y: float) -> float:
    """
    Inverse of smoothstep - useful for timing adjustments.
    """
    return 0.5 - np.sin(np.arcsin(1.0 - 2.0 * y) / 3.0)
