"""
Anticipation & Overshoot - Disney Animation Principles

Classic animation principles for lifelike motion:
- Anticipation: Wind-up before main action (crouch before jump)
- Overshoot: Go past target before settling (bounce on landing)
- Squash & Stretch: Deform during motion
- Slow In/Out: Ease at start and end

These principles make motion feel alive rather than robotic.

Example motion timeline:
    frame 0: y=0   (start)
    frame 1: y=2   (squash down - ANTICIPATION)
    frame 2: y=-10 (jump!)
    frame 3: y=-12 (OVERSHOOT)
    frame 4: y=-10 (settle)
"""

import numpy as np
from typing import Callable, Tuple, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# Animation Timing Curves
# =============================================================================

@dataclass
class AnimationTiming:
    """
    Complete animation timing with anticipation, action, and overshoot.
    
    Timeline breakdown:
    |-- anticipation --|-- action --|-- overshoot --|-- settle --|
    0                  t1           t2              t3            1
    
    Example:
        timing = AnimationTiming(
            anticipation=0.2,  # 20% of time for windup
            overshoot=0.15,    # 15% overshoot amount
            settle=0.1         # 10% of time to settle
        )
        
        for t in range(frames):
            progress = t / frames
            value = timing.evaluate(progress, start=0, end=100)
    """
    anticipation: float = 0.0       # Duration of anticipation (0-0.4)
    anticipation_scale: float = 0.2  # How far to wind up (fraction of motion)
    overshoot: float = 0.0          # Overshoot amount (fraction of motion)
    settle: float = 0.0             # Duration of settle phase (0-0.3)
    
    # Easing for each phase
    anticipation_ease: str = "ease_out_quad"
    action_ease: str = "ease_out_cubic"
    overshoot_ease: str = "ease_out_elastic"
    settle_ease: str = "ease_out_quad"
    
    def evaluate(
        self,
        t: float,
        start: float = 0.0,
        end: float = 1.0
    ) -> float:
        """
        Evaluate animation value at time t.
        
        Args:
            t: Progress (0-1)
            start: Start value
            end: End value
        
        Returns:
            Animated value with anticipation/overshoot
        """
        t = np.clip(t, 0.0, 1.0)
        motion_range = end - start
        
        # Calculate phase boundaries
        t_anticipation_end = self.anticipation
        t_action_end = 1.0 - self.settle
        
        if t < t_anticipation_end and self.anticipation > 0:
            # Anticipation phase - move opposite to intended direction
            phase_t = t / self.anticipation
            eased = _apply_ease(phase_t, self.anticipation_ease)
            
            # Wind up (move backward)
            return start - motion_range * self.anticipation_scale * eased
        
        elif t < t_action_end:
            # Main action phase
            if self.anticipation > 0:
                phase_t = (t - t_anticipation_end) / (t_action_end - t_anticipation_end)
            else:
                phase_t = t / t_action_end
            
            eased = _apply_ease(phase_t, self.action_ease)
            
            # Move from start (or anticipation position) to overshoot position
            if self.overshoot > 0:
                target = end + motion_range * self.overshoot
            else:
                target = end
            
            start_pos = start - motion_range * self.anticipation_scale if self.anticipation > 0 else start
            return start_pos + (target - start_pos) * eased
        
        else:
            # Settle phase
            if self.settle > 0:
                phase_t = (t - t_action_end) / self.settle
                eased = _apply_ease(phase_t, self.settle_ease)
                
                # Settle from overshoot to final position
                overshoot_pos = end + motion_range * self.overshoot
                return overshoot_pos + (end - overshoot_pos) * eased
            else:
                return end
    
    def evaluate_2d(
        self,
        t: float,
        start: Tuple[float, float],
        end: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Evaluate for 2D motion"""
        return (
            self.evaluate(t, start[0], end[0]),
            self.evaluate(t, start[1], end[1])
        )


def _apply_ease(t: float, ease_name: str) -> float:
    """Apply named easing function"""
    t = np.clip(t, 0.0, 1.0)
    
    if ease_name == "linear":
        return t
    elif ease_name == "ease_in_quad":
        return t * t
    elif ease_name == "ease_out_quad":
        return t * (2 - t)
    elif ease_name == "ease_in_out_quad":
        return 2 * t * t if t < 0.5 else -1 + (4 - 2 * t) * t
    elif ease_name == "ease_in_cubic":
        return t * t * t
    elif ease_name == "ease_out_cubic":
        t1 = t - 1
        return t1 * t1 * t1 + 1
    elif ease_name == "ease_in_out_cubic":
        if t < 0.5:
            return 4 * t * t * t
        t1 = 2 * t - 2
        return 0.5 * t1 * t1 * t1 + 1
    elif ease_name == "ease_out_elastic":
        if t == 0 or t == 1:
            return t
        return np.sin(-13 * np.pi / 2 * (t + 1)) * (2 ** (-10 * t)) + 1
    elif ease_name == "ease_out_back":
        c = 1.70158
        t1 = t - 1
        return t1 * t1 * ((c + 1) * t1 + c) + 1
    elif ease_name == "ease_out_bounce":
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
    else:
        return t


# =============================================================================
# Squash & Stretch
# =============================================================================

@dataclass
class SquashStretch:
    """
    Squash and stretch deformation for dynamic motion.
    
    Objects compress when accelerating/landing and stretch when moving fast.
    Volume should be preserved (wider = shorter, taller = thinner).
    
    Example:
        ss = SquashStretch(intensity=0.3)
        
        # During fast upward motion
        scale_x, scale_y = ss.from_velocity(vx=0, vy=-10)
        # Returns (0.85, 1.15) - stretched tall and thin
        
        # During landing impact
        scale_x, scale_y = ss.from_impact(impact_force=5)
        # Returns (1.3, 0.7) - squashed wide and flat
    """
    intensity: float = 0.3          # Overall effect strength (0-1)
    preserve_volume: bool = True    # Keep area constant
    max_squash: float = 0.5         # Maximum squash (0.5 = 50% height)
    max_stretch: float = 1.5        # Maximum stretch (1.5 = 150% height)
    recovery_speed: float = 0.3     # How fast to return to normal
    
    # Current state
    _current_scale: Tuple[float, float] = field(default=(1.0, 1.0), repr=False)
    
    def from_velocity(
        self,
        vx: float,
        vy: float,
        velocity_scale: float = 0.1
    ) -> Tuple[float, float]:
        """
        Calculate squash/stretch from velocity.
        
        Moving objects stretch in direction of motion.
        
        Args:
            vx, vy: Velocity components
            velocity_scale: How much velocity affects stretch
        
        Returns:
            (scale_x, scale_y) tuple
        """
        speed = np.sqrt(vx * vx + vy * vy)
        
        if speed < 0.1:
            return (1.0, 1.0)
        
        # Stretch amount based on speed
        stretch = 1.0 + speed * velocity_scale * self.intensity
        stretch = min(stretch, self.max_stretch)
        
        # Direction of stretch
        if abs(vy) > abs(vx):
            # Vertical motion - stretch vertically, squash horizontally
            scale_y = stretch
            scale_x = 1.0 / stretch if self.preserve_volume else 1.0
        else:
            # Horizontal motion - stretch horizontally
            scale_x = stretch
            scale_y = 1.0 / stretch if self.preserve_volume else 1.0
        
        return (scale_x, scale_y)
    
    def from_impact(
        self,
        impact_force: float,
        direction: str = "down"
    ) -> Tuple[float, float]:
        """
        Calculate squash from impact/landing.
        
        Args:
            impact_force: Impact strength (0-10+)
            direction: Impact direction ("down", "up", "left", "right")
        
        Returns:
            (scale_x, scale_y) tuple
        """
        # Squash amount
        squash = 1.0 - impact_force * 0.1 * self.intensity
        squash = max(squash, self.max_squash)
        
        # Calculate scales based on direction
        if direction in ("down", "up"):
            scale_y = squash
            scale_x = 1.0 / squash if self.preserve_volume else 1.0
        else:
            scale_x = squash
            scale_y = 1.0 / squash if self.preserve_volume else 1.0
        
        return (scale_x, scale_y)
    
    def from_acceleration(
        self,
        ax: float,
        ay: float,
        accel_scale: float = 0.05
    ) -> Tuple[float, float]:
        """
        Calculate squash/stretch from acceleration.
        
        Objects squash in direction of acceleration (anticipation).
        """
        accel = np.sqrt(ax * ax + ay * ay)
        
        if accel < 0.1:
            return (1.0, 1.0)
        
        # Squash in direction of acceleration
        squash = 1.0 - accel * accel_scale * self.intensity
        squash = max(squash, self.max_squash)
        
        if abs(ay) > abs(ax):
            scale_y = squash
            scale_x = 1.0 / squash if self.preserve_volume else 1.0
        else:
            scale_x = squash
            scale_y = 1.0 / squash if self.preserve_volume else 1.0
        
        return (scale_x, scale_y)
    
    def blend_to_normal(
        self,
        current: Tuple[float, float],
        t: float
    ) -> Tuple[float, float]:
        """Blend current scale back towards (1, 1)"""
        blend = t * self.recovery_speed
        return (
            current[0] + (1.0 - current[0]) * blend,
            current[1] + (1.0 - current[1]) * blend
        )


# =============================================================================
# Complete Animation Curve with All Principles
# =============================================================================

class AnimationPrinciples(Enum):
    """Animation principle presets"""
    NONE = "none"
    SUBTLE = "subtle"       # Minimal, professional
    STANDARD = "standard"   # Good default
    EXAGGERATED = "exaggerated"  # Cartoony
    BOUNCY = "bouncy"       # Playful
    SNAPPY = "snappy"       # Quick, game-feel


@dataclass
class AnimationCurve:
    """
    Complete animation curve with all Disney principles.
    
    Combines:
    - Anticipation (wind-up)
    - Follow-through (overshoot + settle)
    - Slow in/out (easing)
    - Squash & stretch
    
    Example:
        curve = AnimationCurve.from_preset(AnimationPrinciples.BOUNCY)
        
        for frame in range(total_frames):
            t = frame / total_frames
            
            # Get position with anticipation/overshoot
            y = curve.evaluate_position(t, start_y=0, end_y=-20)
            
            # Get squash/stretch scale
            scale_x, scale_y = curve.evaluate_scale(t)
            
            render_sprite(y=y, scale_x=scale_x, scale_y=scale_y)
    """
    # Timing
    anticipation: float = 0.15
    anticipation_scale: float = 0.15
    overshoot: float = 0.1
    settle: float = 0.1
    
    # Squash/stretch
    squash_on_anticipation: float = 0.1
    stretch_on_motion: float = 0.15
    squash_on_land: float = 0.2
    
    # Easing
    ease_anticipation: str = "ease_out_quad"
    ease_action: str = "ease_out_cubic"
    ease_settle: str = "ease_out_quad"
    
    # Internal state
    _timing: AnimationTiming = field(default=None, repr=False)
    _squash_stretch: SquashStretch = field(default=None, repr=False)
    
    def __post_init__(self):
        self._timing = AnimationTiming(
            anticipation=self.anticipation,
            anticipation_scale=self.anticipation_scale,
            overshoot=self.overshoot,
            settle=self.settle,
            anticipation_ease=self.ease_anticipation,
            action_ease=self.ease_action,
            settle_ease=self.ease_settle
        )
        self._squash_stretch = SquashStretch(
            intensity=max(self.squash_on_anticipation, self.stretch_on_motion, self.squash_on_land)
        )
    
    @classmethod
    def from_preset(cls, preset: AnimationPrinciples) -> 'AnimationCurve':
        """Create curve from preset"""
        if preset == AnimationPrinciples.NONE:
            return cls(
                anticipation=0, anticipation_scale=0,
                overshoot=0, settle=0,
                squash_on_anticipation=0, stretch_on_motion=0, squash_on_land=0
            )
        elif preset == AnimationPrinciples.SUBTLE:
            return cls(
                anticipation=0.08, anticipation_scale=0.05,
                overshoot=0.03, settle=0.05,
                squash_on_anticipation=0.03, stretch_on_motion=0.05, squash_on_land=0.05
            )
        elif preset == AnimationPrinciples.STANDARD:
            return cls(
                anticipation=0.15, anticipation_scale=0.12,
                overshoot=0.08, settle=0.1,
                squash_on_anticipation=0.08, stretch_on_motion=0.1, squash_on_land=0.12
            )
        elif preset == AnimationPrinciples.EXAGGERATED:
            return cls(
                anticipation=0.25, anticipation_scale=0.25,
                overshoot=0.2, settle=0.15,
                squash_on_anticipation=0.2, stretch_on_motion=0.25, squash_on_land=0.3,
                ease_settle="ease_out_bounce"
            )
        elif preset == AnimationPrinciples.BOUNCY:
            return cls(
                anticipation=0.12, anticipation_scale=0.15,
                overshoot=0.15, settle=0.12,
                squash_on_anticipation=0.1, stretch_on_motion=0.15, squash_on_land=0.25,
                ease_settle="ease_out_elastic"
            )
        elif preset == AnimationPrinciples.SNAPPY:
            return cls(
                anticipation=0.05, anticipation_scale=0.08,
                overshoot=0.05, settle=0.05,
                squash_on_anticipation=0.05, stretch_on_motion=0.08, squash_on_land=0.1,
                ease_action="ease_out_back"
            )
        else:
            return cls()
    
    @classmethod
    def from_flags(
        cls,
        anticipation: float = 0.0,
        overshoot: float = 0.0,
        squash_stretch: float = 0.0
    ) -> 'AnimationCurve':
        """
        Create curve from CLI-style flags.
        
        Args:
            anticipation: Windup amount (0-0.4), e.g., 0.2 = 20%
            overshoot: Overshoot amount (0-0.3), e.g., 0.15 = 15%
            squash_stretch: S&S intensity (0-0.5)
        """
        return cls(
            anticipation=min(anticipation, 0.4),
            anticipation_scale=anticipation * 0.8,
            overshoot=min(overshoot, 0.3),
            settle=overshoot * 0.8,
            squash_on_anticipation=squash_stretch * 0.6,
            stretch_on_motion=squash_stretch,
            squash_on_land=squash_stretch * 1.2
        )
    
    def evaluate_position(
        self,
        t: float,
        start: float = 0.0,
        end: float = 1.0
    ) -> float:
        """Get position at time t with anticipation/overshoot"""
        return self._timing.evaluate(t, start, end)
    
    def evaluate_position_2d(
        self,
        t: float,
        start: Tuple[float, float],
        end: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Get 2D position at time t"""
        return self._timing.evaluate_2d(t, start, end)
    
    def evaluate_scale(self, t: float) -> Tuple[float, float]:
        """
        Get squash/stretch scale at time t.
        
        Returns:
            (scale_x, scale_y) for sprite deformation
        """
        t_anticipation_end = self.anticipation
        t_action_end = 1.0 - self.settle
        
        if t < t_anticipation_end and self.anticipation > 0:
            # Anticipation - squash in direction of upcoming motion
            phase_t = t / self.anticipation
            squash = 1.0 - self.squash_on_anticipation * _apply_ease(phase_t, "ease_out_quad")
            return (1.0 / squash, squash)  # Squash vertically
        
        elif t < t_action_end:
            # Main motion - stretch in direction of motion
            if self.anticipation > 0:
                phase_t = (t - t_anticipation_end) / (t_action_end - t_anticipation_end)
            else:
                phase_t = t / t_action_end
            
            # Peak stretch at middle of motion
            stretch_curve = np.sin(phase_t * np.pi)
            stretch = 1.0 + self.stretch_on_motion * stretch_curve
            return (1.0 / stretch, stretch)  # Stretch vertically
        
        else:
            # Settle - squash on landing, then recover
            if self.settle > 0:
                phase_t = (t - t_action_end) / self.settle
                
                # Quick squash then recover
                if phase_t < 0.3:
                    # Impact squash
                    squash_t = phase_t / 0.3
                    squash = 1.0 - self.squash_on_land * _apply_ease(squash_t, "ease_out_quad")
                else:
                    # Recovery
                    recover_t = (phase_t - 0.3) / 0.7
                    squash = (1.0 - self.squash_on_land) + self.squash_on_land * _apply_ease(recover_t, "ease_out_elastic")
                
                return (1.0 / squash, squash)
            
            return (1.0, 1.0)
    
    def generate_keyframes(
        self,
        start: float,
        end: float,
        num_frames: int
    ) -> List[Tuple[float, float, float]]:
        """
        Generate keyframe data for the animation.
        
        Returns:
            List of (position, scale_x, scale_y) tuples
        """
        keyframes = []
        for i in range(num_frames):
            t = i / max(1, num_frames - 1)
            pos = self.evaluate_position(t, start, end)
            scale_x, scale_y = self.evaluate_scale(t)
            keyframes.append((pos, scale_x, scale_y))
        return keyframes


# =============================================================================
# Helper Functions for Effect Integration
# =============================================================================

def apply_anticipation_overshoot(
    values: np.ndarray,
    anticipation: float = 0.2,
    overshoot: float = 0.15
) -> np.ndarray:
    """
    Apply anticipation and overshoot to an array of animation values.
    
    Args:
        values: Array of animation values (e.g., positions over time)
        anticipation: Anticipation amount (0-0.4)
        overshoot: Overshoot amount (0-0.3)
    
    Returns:
        Modified values with anticipation/overshoot
    """
    if len(values) < 3:
        return values
    
    curve = AnimationCurve.from_flags(anticipation=anticipation, overshoot=overshoot)
    
    result = np.zeros_like(values)
    start_val = values[0]
    end_val = values[-1]
    
    for i in range(len(values)):
        t = i / (len(values) - 1)
        result[i] = curve.evaluate_position(t, start_val, end_val)
    
    return result


def create_bounce_animation(
    start_y: float,
    peak_y: float,
    num_frames: int,
    anticipation: float = 0.2,
    overshoot: float = 0.15,
    squash_stretch: float = 0.2
) -> List[Tuple[float, float, float]]:
    """
    Create a complete bounce animation with all principles.
    
    Args:
        start_y: Starting Y position
        peak_y: Peak Y position (usually negative for upward)
        num_frames: Total frames
        anticipation: Wind-up amount
        overshoot: Overshoot amount  
        squash_stretch: Deformation intensity
    
    Returns:
        List of (y_position, scale_x, scale_y) per frame
    
    Example:
        # Bounce from y=0 to y=-20 (upward) over 16 frames
        frames = create_bounce_animation(
            start_y=0, peak_y=-20, num_frames=16,
            anticipation=0.2, overshoot=0.15
        )
        
        for y, sx, sy in frames:
            render_sprite(y=y, scale_x=sx, scale_y=sy)
    """
    curve = AnimationCurve.from_flags(
        anticipation=anticipation,
        overshoot=overshoot,
        squash_stretch=squash_stretch
    )
    
    return curve.generate_keyframes(start_y, peak_y, num_frames)


def get_squash_stretch_for_velocity(
    vx: float,
    vy: float,
    intensity: float = 0.2
) -> Tuple[float, float]:
    """
    Quick helper to get squash/stretch from velocity.
    
    Args:
        vx, vy: Velocity components
        intensity: Effect strength (0-0.5)
    
    Returns:
        (scale_x, scale_y) tuple
    """
    ss = SquashStretch(intensity=intensity)
    return ss.from_velocity(vx, vy)


# =============================================================================
# Frame-by-Frame Animation Builder
# =============================================================================

class AnimationBuilder:
    """
    Build frame-by-frame animations with anticipation/overshoot.
    
    Example:
        builder = AnimationBuilder(num_frames=16)
        builder.set_anticipation(0.2)
        builder.set_overshoot(0.15)
        builder.set_squash_stretch(0.2)
        
        # Define keyframes
        builder.add_movement(start=(0, 0), end=(0, -20))  # Jump up
        
        # Generate all frame data
        frames = builder.build()
        
        for frame in frames:
            x, y = frame['position']
            sx, sy = frame['scale']
            render_sprite(x=x, y=y, scale_x=sx, scale_y=sy)
    """
    
    def __init__(self, num_frames: int = 8):
        self.num_frames = num_frames
        self.anticipation = 0.0
        self.overshoot = 0.0
        self.squash_stretch = 0.0
        
        self.start_pos = (0.0, 0.0)
        self.end_pos = (0.0, 0.0)
        
        self._curve: Optional[AnimationCurve] = None
    
    def set_anticipation(self, amount: float) -> 'AnimationBuilder':
        """Set anticipation amount (0-0.4)"""
        self.anticipation = np.clip(amount, 0, 0.4)
        return self
    
    def set_overshoot(self, amount: float) -> 'AnimationBuilder':
        """Set overshoot amount (0-0.3)"""
        self.overshoot = np.clip(amount, 0, 0.3)
        return self
    
    def set_squash_stretch(self, amount: float) -> 'AnimationBuilder':
        """Set squash/stretch intensity (0-0.5)"""
        self.squash_stretch = np.clip(amount, 0, 0.5)
        return self
    
    def add_movement(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float]
    ) -> 'AnimationBuilder':
        """Define start and end positions"""
        self.start_pos = start
        self.end_pos = end
        return self
    
    def use_preset(self, preset: AnimationPrinciples) -> 'AnimationBuilder':
        """Use a preset configuration"""
        self._curve = AnimationCurve.from_preset(preset)
        return self
    
    def build(self) -> List[dict]:
        """
        Build all frame data.
        
        Returns:
            List of frame dictionaries with 'position', 'scale', 't' keys
        """
        if self._curve is None:
            self._curve = AnimationCurve.from_flags(
                anticipation=self.anticipation,
                overshoot=self.overshoot,
                squash_stretch=self.squash_stretch
            )
        
        frames = []
        for i in range(self.num_frames):
            t = i / max(1, self.num_frames - 1)
            
            pos = self._curve.evaluate_position_2d(t, self.start_pos, self.end_pos)
            scale = self._curve.evaluate_scale(t)
            
            frames.append({
                't': t,
                'frame': i,
                'position': pos,
                'scale': scale
            })
        
        return frames
