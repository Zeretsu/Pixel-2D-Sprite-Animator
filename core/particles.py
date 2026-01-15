"""
Professional Particle System

Game-quality particle effects with proper physics and visual curves.

Features:
- Emission shapes: point, cone, sphere, line, ring, box
- Velocity curves: bezier paths for acceleration
- Size over lifetime: shrink/grow particles
- Color over lifetime: gradient ramps integration
- Collision: bounce off sprite bounds
- Turbulence: Perlin noise for organic movement
- Burst/continuous emission modes

Examples:
- Sparks: cone emission, upward acceleration, shrink, orange→red→fade
- Fire: cone up, grow then shrink, fire ramp colors, turbulence
- Magic: sphere emission, spiral paths, rainbow color shift
- Rain: line emission, gravity, splash on collision
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import math


# =============================================================================
# Constants
# =============================================================================

GAMMA = 2.2
INV_GAMMA = 1.0 / GAMMA

# Perlin noise permutation table
_PERM = np.array([
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
    18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
    250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
    189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
    172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
    228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
], dtype=np.int32)

_PERM = np.concatenate([_PERM, _PERM])  # Double for wraparound


# =============================================================================
# Noise Functions
# =============================================================================

def _fade(t: np.ndarray) -> np.ndarray:
    """Perlin fade function: 6t^5 - 15t^4 + 10t^3"""
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation"""
    return a + t * (b - a)


def _grad1d(hash_val: int, x: float) -> float:
    """1D gradient"""
    return x if (hash_val & 1) == 0 else -x


def _grad2d(hash_val: int, x: float, y: float) -> float:
    """2D gradient"""
    h = hash_val & 3
    if h == 0:
        return x + y
    elif h == 1:
        return -x + y
    elif h == 2:
        return x - y
    else:
        return -x - y


def perlin_noise_1d(x: float) -> float:
    """1D Perlin noise, returns value in [-1, 1]"""
    xi = int(np.floor(x)) & 255
    xf = x - np.floor(x)
    u = _fade(np.array([xf]))[0]
    
    a = _PERM[xi]
    b = _PERM[xi + 1]
    
    return _lerp(_grad1d(a, xf), _grad1d(b, xf - 1), u)


def perlin_noise_2d(x: float, y: float) -> float:
    """2D Perlin noise, returns value in [-1, 1]"""
    xi = int(np.floor(x)) & 255
    yi = int(np.floor(y)) & 255
    xf = x - np.floor(x)
    yf = y - np.floor(y)
    
    u = _fade(np.array([xf]))[0]
    v = _fade(np.array([yf]))[0]
    
    aa = _PERM[_PERM[xi] + yi]
    ab = _PERM[_PERM[xi] + yi + 1]
    ba = _PERM[_PERM[xi + 1] + yi]
    bb = _PERM[_PERM[xi + 1] + yi + 1]
    
    x1 = _lerp(_grad2d(aa, xf, yf), _grad2d(ba, xf - 1, yf), u)
    x2 = _lerp(_grad2d(ab, xf, yf - 1), _grad2d(bb, xf - 1, yf - 1), u)
    
    return _lerp(x1, x2, v)


def fbm_noise_2d(x: float, y: float, octaves: int = 4, lacunarity: float = 2.0, gain: float = 0.5) -> float:
    """Fractal Brownian Motion - layered Perlin noise"""
    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    
    for _ in range(octaves):
        value += amplitude * perlin_noise_2d(x * frequency, y * frequency)
        amplitude *= gain
        frequency *= lacunarity
    
    return value


def curl_noise_2d(x: float, y: float, epsilon: float = 0.0001) -> Tuple[float, float]:
    """
    2D curl noise for divergence-free flow (smoke, fluid).
    Returns velocity vector.
    """
    # Compute gradient via finite differences
    dx = (perlin_noise_2d(x + epsilon, y) - perlin_noise_2d(x - epsilon, y)) / (2 * epsilon)
    dy = (perlin_noise_2d(x, y + epsilon) - perlin_noise_2d(x, y - epsilon)) / (2 * epsilon)
    
    # Curl in 2D: rotate gradient 90 degrees
    return (dy, -dx)


# =============================================================================
# Bezier Curves for Velocity/Size/etc
# =============================================================================

@dataclass
class BezierPath:
    """
    Cubic bezier curve for smooth value changes over lifetime.
    
    Points are (time, value) pairs where time is 0-1.
    """
    control_points: List[Tuple[float, float]]
    
    def sample(self, t: float) -> float:
        """Sample curve at time t (0-1)"""
        t = np.clip(t, 0.0, 1.0)
        
        if len(self.control_points) < 2:
            return self.control_points[0][1] if self.control_points else 0.0
        
        if len(self.control_points) == 2:
            # Linear
            p0, p1 = self.control_points
            return p0[1] + (p1[1] - p0[1]) * t
        
        # Find segment
        for i in range(len(self.control_points) - 1):
            p0 = self.control_points[i]
            p1 = self.control_points[i + 1]
            
            if t <= p1[0] or i == len(self.control_points) - 2:
                # Normalize t to this segment
                segment_t = (t - p0[0]) / (p1[0] - p0[0] + 1e-10)
                segment_t = np.clip(segment_t, 0.0, 1.0)
                
                # Simple cubic ease
                ease_t = segment_t * segment_t * (3 - 2 * segment_t)
                return p0[1] + (p1[1] - p0[1]) * ease_t
        
        return self.control_points[-1][1]
    
    @classmethod
    def linear(cls, start: float, end: float) -> 'BezierPath':
        """Linear interpolation from start to end"""
        return cls([(0.0, start), (1.0, end)])
    
    @classmethod
    def ease_out(cls, start: float, end: float) -> 'BezierPath':
        """Fast start, slow end"""
        return cls([(0.0, start), (0.3, end * 0.8 + start * 0.2), (1.0, end)])
    
    @classmethod
    def ease_in(cls, start: float, end: float) -> 'BezierPath':
        """Slow start, fast end"""
        return cls([(0.0, start), (0.7, start * 0.8 + end * 0.2), (1.0, end)])
    
    @classmethod
    def ease_in_out(cls, start: float, end: float) -> 'BezierPath':
        """Slow start and end"""
        mid = (start + end) / 2
        return cls([(0.0, start), (0.5, mid), (1.0, end)])
    
    @classmethod
    def pulse(cls, low: float, high: float, peak_time: float = 0.3) -> 'BezierPath':
        """Spike up then down"""
        return cls([(0.0, low), (peak_time, high), (1.0, low)])


# Preset paths
SIZE_SHRINK = BezierPath.linear(1.0, 0.0)
SIZE_GROW_SHRINK = BezierPath([(0.0, 0.0), (0.3, 1.0), (1.0, 0.0)])
SIZE_GROW = BezierPath.linear(0.5, 1.5)
SIZE_CONSTANT = BezierPath.linear(1.0, 1.0)

ALPHA_FADE = BezierPath.linear(1.0, 0.0)
ALPHA_FADE_LATE = BezierPath([(0.0, 1.0), (0.7, 1.0), (1.0, 0.0)])
ALPHA_FLASH_FADE = BezierPath([(0.0, 0.0), (0.1, 1.0), (1.0, 0.0)])

SPEED_CONSTANT = BezierPath.linear(1.0, 1.0)
SPEED_DECELERATE = BezierPath.ease_out(1.0, 0.0)
SPEED_ACCELERATE = BezierPath.ease_in(0.0, 1.0)
SPEED_BURST = BezierPath([(0.0, 2.0), (0.2, 1.0), (1.0, 0.5)])


# =============================================================================
# Emission Shapes
# =============================================================================

class EmissionShape(Enum):
    """Shapes for particle emission"""
    POINT = auto()      # Single point
    CONE = auto()       # Cone with angle spread
    SPHERE = auto()     # Sphere surface
    CIRCLE = auto()     # Circle edge (2D sphere)
    LINE = auto()       # Line segment
    RING = auto()       # Ring/donut shape
    BOX = auto()        # Rectangle area
    DISC = auto()       # Filled circle


@dataclass
class EmissionConfig:
    """Configuration for particle emission shape"""
    shape: EmissionShape = EmissionShape.POINT
    
    # Position offset from emitter
    offset: Tuple[float, float] = (0.0, 0.0)
    
    # Shape parameters
    angle: float = 0.0          # Direction angle (radians)
    spread: float = 0.5         # Cone spread (radians), or box/line size
    radius: float = 5.0         # For sphere/circle/ring
    inner_radius: float = 3.0   # For ring (donut hole)
    width: float = 10.0         # For line/box
    height: float = 10.0        # For box
    
    # Emission from edge vs volume
    surface_only: bool = True   # True = emit from surface, False = fill volume
    
    def get_position_and_velocity(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Get random position and initial velocity direction based on shape.
        
        Returns:
            (position, velocity_direction) - velocity is normalized
        """
        ox, oy = self.offset
        
        if self.shape == EmissionShape.POINT:
            pos = (ox, oy)
            # Random direction within spread angle
            angle = self.angle + (np.random.random() - 0.5) * self.spread
            vel = (np.cos(angle), np.sin(angle))
            
        elif self.shape == EmissionShape.CONE:
            pos = (ox, oy)
            angle = self.angle + (np.random.random() - 0.5) * self.spread
            vel = (np.cos(angle), np.sin(angle))
            
        elif self.shape == EmissionShape.SPHERE or self.shape == EmissionShape.CIRCLE:
            angle = np.random.random() * 2 * np.pi
            if self.surface_only:
                r = self.radius
            else:
                r = np.sqrt(np.random.random()) * self.radius
            pos = (ox + np.cos(angle) * r, oy + np.sin(angle) * r)
            # Velocity outward from center
            vel = (np.cos(angle), np.sin(angle))
            
        elif self.shape == EmissionShape.DISC:
            angle = np.random.random() * 2 * np.pi
            r = np.sqrt(np.random.random()) * self.radius
            pos = (ox + np.cos(angle) * r, oy + np.sin(angle) * r)
            vel_angle = self.angle + (np.random.random() - 0.5) * self.spread
            vel = (np.cos(vel_angle), np.sin(vel_angle))
            
        elif self.shape == EmissionShape.LINE:
            t = np.random.random() - 0.5
            # Line perpendicular to angle
            perp_angle = self.angle + np.pi / 2
            pos = (
                ox + np.cos(perp_angle) * t * self.width,
                oy + np.sin(perp_angle) * t * self.width
            )
            vel_angle = self.angle + (np.random.random() - 0.5) * self.spread
            vel = (np.cos(vel_angle), np.sin(vel_angle))
            
        elif self.shape == EmissionShape.RING:
            angle = np.random.random() * 2 * np.pi
            if self.surface_only:
                r = self.radius
            else:
                # Between inner and outer radius
                r = self.inner_radius + np.random.random() * (self.radius - self.inner_radius)
            pos = (ox + np.cos(angle) * r, oy + np.sin(angle) * r)
            vel = (np.cos(angle), np.sin(angle))
            
        elif self.shape == EmissionShape.BOX:
            if self.surface_only:
                # Emit from edges
                edge = np.random.randint(4)
                t = np.random.random()
                if edge == 0:  # Top
                    pos = (ox + (t - 0.5) * self.width, oy - self.height / 2)
                elif edge == 1:  # Bottom
                    pos = (ox + (t - 0.5) * self.width, oy + self.height / 2)
                elif edge == 2:  # Left
                    pos = (ox - self.width / 2, oy + (t - 0.5) * self.height)
                else:  # Right
                    pos = (ox + self.width / 2, oy + (t - 0.5) * self.height)
            else:
                pos = (
                    ox + (np.random.random() - 0.5) * self.width,
                    oy + (np.random.random() - 0.5) * self.height
                )
            vel_angle = self.angle + (np.random.random() - 0.5) * self.spread
            vel = (np.cos(vel_angle), np.sin(vel_angle))
            
        else:
            pos = (ox, oy)
            vel = (0.0, -1.0)
        
        return pos, vel


# =============================================================================
# Particle Class
# =============================================================================

@dataclass
class Particle:
    """Individual particle with full physics state"""
    # Position and velocity
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    
    # Lifetime
    age: float = 0.0
    lifetime: float = 1.0
    
    # Visual properties
    size: float = 1.0
    base_size: float = 1.0
    rotation: float = 0.0
    rotation_speed: float = 0.0
    
    # Color (RGBA 0-255)
    r: int = 255
    g: int = 255
    b: int = 255
    a: int = 255
    
    # Noise offset for unique turbulence per particle
    noise_offset: float = 0.0
    
    # Custom data
    custom: dict = field(default_factory=dict)
    
    @property
    def alive(self) -> bool:
        return self.age < self.lifetime
    
    @property
    def normalized_age(self) -> float:
        """Age as 0-1 fraction of lifetime"""
        return min(self.age / self.lifetime, 1.0) if self.lifetime > 0 else 1.0
    
    @property
    def color(self) -> Tuple[int, int, int, int]:
        return (self.r, self.g, self.b, self.a)
    
    @color.setter
    def color(self, rgba: Tuple[int, int, int, int]):
        self.r, self.g, self.b, self.a = rgba


# =============================================================================
# Color Gradient for Particles
# =============================================================================

@dataclass
class ColorGradient:
    """
    Color gradient for particle color over lifetime.
    
    Integrates with palette.py ramps.
    """
    colors: List[Tuple[float, Tuple[int, int, int, int]]]  # (time, RGBA)
    
    def sample(self, t: float) -> Tuple[int, int, int, int]:
        """Sample color at time t (0-1)"""
        t = np.clip(t, 0.0, 1.0)
        
        if not self.colors:
            return (255, 255, 255, 255)
        
        if len(self.colors) == 1:
            return self.colors[0][1]
        
        # Find segment
        for i in range(len(self.colors) - 1):
            t1, c1 = self.colors[i]
            t2, c2 = self.colors[i + 1]
            
            if t <= t2 or i == len(self.colors) - 2:
                # Interpolate
                local_t = (t - t1) / (t2 - t1 + 1e-10)
                local_t = np.clip(local_t, 0.0, 1.0)
                
                return (
                    int(c1[0] + (c2[0] - c1[0]) * local_t),
                    int(c1[1] + (c2[1] - c1[1]) * local_t),
                    int(c1[2] + (c2[2] - c1[2]) * local_t),
                    int(c1[3] + (c2[3] - c1[3]) * local_t)
                )
        
        return self.colors[-1][1]
    
    @classmethod
    def from_ramp(cls, colors: List[Tuple[int, int, int]], alpha_curve: BezierPath = None) -> 'ColorGradient':
        """Create gradient from RGB color list (like from palette.py)"""
        if alpha_curve is None:
            alpha_curve = ALPHA_FADE
        
        gradient_colors = []
        n = len(colors)
        for i, rgb in enumerate(colors):
            t = i / (n - 1) if n > 1 else 0.0
            alpha = int(alpha_curve.sample(t) * 255)
            gradient_colors.append((t, (rgb[0], rgb[1], rgb[2], alpha)))
        
        return cls(gradient_colors)
    
    @classmethod
    def solid(cls, r: int, g: int, b: int, alpha_curve: BezierPath = None) -> 'ColorGradient':
        """Solid color with optional alpha fade"""
        if alpha_curve is None:
            alpha_curve = ALPHA_FADE
        
        return cls([
            (0.0, (r, g, b, int(alpha_curve.sample(0.0) * 255))),
            (1.0, (r, g, b, int(alpha_curve.sample(1.0) * 255)))
        ])


# Preset gradients
GRADIENT_FIRE = ColorGradient([
    (0.0, (255, 200, 100, 255)),   # Bright yellow
    (0.3, (255, 150, 50, 255)),    # Orange
    (0.6, (255, 80, 20, 200)),     # Red-orange
    (1.0, (100, 30, 10, 0))        # Dark red, faded
])

GRADIENT_SPARK = ColorGradient([
    (0.0, (255, 255, 200, 255)),   # White-yellow
    (0.2, (255, 200, 50, 255)),    # Yellow
    (0.5, (255, 100, 20, 200)),    # Orange
    (1.0, (150, 50, 10, 0))        # Dark orange, faded
])

GRADIENT_SMOKE = ColorGradient([
    (0.0, (200, 200, 200, 200)),   # Light gray
    (0.5, (120, 120, 120, 150)),   # Mid gray
    (1.0, (60, 60, 60, 0))         # Dark gray, faded
])

GRADIENT_MAGIC = ColorGradient([
    (0.0, (255, 200, 255, 255)),   # Light pink
    (0.3, (200, 100, 255, 255)),   # Purple
    (0.7, (100, 50, 200, 150)),    # Dark purple
    (1.0, (50, 20, 100, 0))        # Deep purple, faded
])

GRADIENT_WATER = ColorGradient([
    (0.0, (200, 240, 255, 220)),   # Light blue
    (0.5, (100, 180, 255, 180)),   # Blue
    (1.0, (50, 100, 200, 0))       # Dark blue, faded
])

GRADIENT_ELECTRIC = ColorGradient([
    (0.0, (255, 255, 255, 255)),   # White
    (0.2, (200, 220, 255, 255)),   # Light blue
    (0.5, (100, 150, 255, 200)),   # Blue
    (1.0, (50, 80, 200, 0))        # Dark blue, faded
])


# =============================================================================
# Particle Emitter
# =============================================================================

@dataclass
class ParticleEmitter:
    """
    Full-featured particle emitter.
    
    Example:
        emitter = ParticleEmitter(
            emission=EmissionConfig(shape=EmissionShape.CONE, angle=-np.pi/2, spread=0.5),
            speed=50.0,
            lifetime=1.5,
            size_over_lifetime=SIZE_SHRINK,
            color_gradient=GRADIENT_FIRE,
            gravity=(0, -50),  # Sparks rise
            turbulence=0.3
        )
        
        emitter.emit(10)  # Burst of 10 particles
        
        for frame in range(frames):
            emitter.update(dt)
            emitter.render(canvas)
    """
    # Emission configuration
    emission: EmissionConfig = field(default_factory=EmissionConfig)
    
    # Position (can be updated for moving emitters)
    x: float = 0.0
    y: float = 0.0
    
    # Particle properties
    speed: float = 50.0
    speed_variance: float = 0.2      # ±20% speed variation
    lifetime: float = 1.0
    lifetime_variance: float = 0.2
    size: float = 2.0
    size_variance: float = 0.3
    
    # Curves over lifetime
    size_over_lifetime: BezierPath = field(default_factory=lambda: SIZE_SHRINK)
    speed_over_lifetime: BezierPath = field(default_factory=lambda: SPEED_CONSTANT)
    alpha_over_lifetime: BezierPath = field(default_factory=lambda: ALPHA_FADE)
    
    # Color
    color_gradient: ColorGradient = field(default_factory=lambda: GRADIENT_FIRE)
    
    # Physics
    gravity: Tuple[float, float] = (0.0, 0.0)
    drag: float = 0.0                # Air resistance (0-1)
    
    # Turbulence (Perlin noise)
    turbulence: float = 0.0          # Turbulence strength
    turbulence_scale: float = 0.1    # Noise scale (smaller = smoother)
    turbulence_speed: float = 1.0    # How fast noise changes
    
    # Rotation
    rotation_speed: float = 0.0
    rotation_variance: float = 0.0
    align_to_velocity: bool = False  # Point in direction of movement
    
    # Collision
    collision_enabled: bool = False
    collision_bounds: Tuple[float, float, float, float] = (0, 0, 100, 100)  # (x, y, w, h)
    collision_bounce: float = 0.5    # Velocity retained after bounce
    collision_friction: float = 0.9  # Horizontal velocity retained
    
    # Continuous emission
    emission_rate: float = 0.0       # Particles per second (0 = manual burst only)
    _emission_accumulator: float = field(default=0.0, repr=False)
    
    # Particle storage
    particles: List[Particle] = field(default_factory=list, repr=False)
    max_particles: int = 500
    
    # Time tracking
    _time: float = field(default=0.0, repr=False)
    
    def emit(self, count: int = 1):
        """Emit a burst of particles"""
        for _ in range(count):
            if len(self.particles) >= self.max_particles:
                break
            
            # Get position and direction from emission shape
            pos, vel_dir = self.emission.get_position_and_velocity()
            
            # Apply variance to properties
            speed = self.speed * (1 + (np.random.random() - 0.5) * 2 * self.speed_variance)
            lifetime = self.lifetime * (1 + (np.random.random() - 0.5) * 2 * self.lifetime_variance)
            size = self.size * (1 + (np.random.random() - 0.5) * 2 * self.size_variance)
            
            rot_speed = self.rotation_speed
            if self.rotation_variance > 0:
                rot_speed += (np.random.random() - 0.5) * 2 * self.rotation_variance
            
            particle = Particle(
                x=self.x + pos[0],
                y=self.y + pos[1],
                vx=vel_dir[0] * speed,
                vy=vel_dir[1] * speed,
                lifetime=max(0.01, lifetime),
                base_size=size,
                size=size,
                rotation=np.random.random() * 2 * np.pi if rot_speed != 0 else 0,
                rotation_speed=rot_speed,
                noise_offset=np.random.random() * 1000
            )
            
            # Initial color
            color = self.color_gradient.sample(0.0)
            particle.color = color
            
            self.particles.append(particle)
    
    def update(self, dt: float):
        """Update all particles"""
        self._time += dt
        
        # Continuous emission
        if self.emission_rate > 0:
            self._emission_accumulator += self.emission_rate * dt
            while self._emission_accumulator >= 1.0:
                self.emit(1)
                self._emission_accumulator -= 1.0
        
        # Update each particle
        alive_particles = []
        
        for p in self.particles:
            p.age += dt
            
            if not p.alive:
                continue
            
            t = p.normalized_age
            
            # Speed curve
            speed_mult = self.speed_over_lifetime.sample(t)
            
            # Apply gravity
            p.vx += self.gravity[0] * dt
            p.vy += self.gravity[1] * dt
            
            # Apply drag
            if self.drag > 0:
                drag_factor = 1.0 - self.drag * dt
                p.vx *= drag_factor
                p.vy *= drag_factor
            
            # Apply turbulence
            if self.turbulence > 0:
                noise_x = p.noise_offset + p.x * self.turbulence_scale
                noise_y = p.noise_offset * 0.7 + p.y * self.turbulence_scale
                noise_t = self._time * self.turbulence_speed
                
                # Use curl noise for divergence-free flow
                turb_vx, turb_vy = curl_noise_2d(noise_x + noise_t, noise_y + noise_t)
                
                p.vx += turb_vx * self.turbulence * dt * 100
                p.vy += turb_vy * self.turbulence * dt * 100
            
            # Move particle
            p.x += p.vx * speed_mult * dt
            p.y += p.vy * speed_mult * dt
            
            # Collision
            if self.collision_enabled:
                bx, by, bw, bh = self.collision_bounds
                
                # Bottom collision
                if p.y > by + bh and p.vy > 0:
                    p.y = by + bh
                    p.vy = -p.vy * self.collision_bounce
                    p.vx *= self.collision_friction
                
                # Top collision
                if p.y < by and p.vy < 0:
                    p.y = by
                    p.vy = -p.vy * self.collision_bounce
                    p.vx *= self.collision_friction
                
                # Right collision
                if p.x > bx + bw and p.vx > 0:
                    p.x = bx + bw
                    p.vx = -p.vx * self.collision_bounce
                    p.vy *= self.collision_friction
                
                # Left collision
                if p.x < bx and p.vx < 0:
                    p.x = bx
                    p.vx = -p.vx * self.collision_bounce
                    p.vy *= self.collision_friction
            
            # Rotation
            if self.align_to_velocity:
                p.rotation = np.arctan2(p.vy, p.vx)
            else:
                p.rotation += p.rotation_speed * dt
            
            # Size over lifetime
            p.size = p.base_size * self.size_over_lifetime.sample(t)
            
            # Color over lifetime
            color = self.color_gradient.sample(t)
            # Apply alpha curve on top
            alpha_mult = self.alpha_over_lifetime.sample(t)
            p.color = (color[0], color[1], color[2], int(color[3] * alpha_mult))
            
            alive_particles.append(p)
        
        self.particles = alive_particles
    
    def render(
        self,
        canvas: np.ndarray,
        offset: Tuple[int, int] = (0, 0),
        blend_mode: str = "additive"
    ) -> np.ndarray:
        """
        Render particles onto canvas.
        
        Args:
            canvas: RGBA image to render onto
            offset: Render offset
            blend_mode: "additive", "alpha", "multiply"
        
        Returns:
            Modified canvas
        """
        result = canvas.copy()
        h, w = canvas.shape[:2]
        ox, oy = offset
        
        for p in self.particles:
            if p.a <= 0 or p.size <= 0:
                continue
            
            # Particle center in canvas coords
            px = int(p.x + ox)
            py = int(p.y + oy)
            
            # Particle radius
            radius = max(1, int(p.size / 2))
            
            # Bounds check
            if px + radius < 0 or px - radius >= w:
                continue
            if py + radius < 0 or py - radius >= h:
                continue
            
            # Draw particle (simple circle)
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx * dx + dy * dy <= radius * radius:
                        x = px + dx
                        y = py + dy
                        
                        if 0 <= x < w and 0 <= y < h:
                            # Distance-based falloff
                            dist = np.sqrt(dx * dx + dy * dy)
                            falloff = 1.0 - dist / (radius + 0.5)
                            falloff = max(0, falloff)
                            
                            # Particle color with falloff
                            pr = p.r
                            pg = p.g
                            pb = p.b
                            pa = int(p.a * falloff)
                            
                            if pa <= 0:
                                continue
                            
                            # Blend
                            if blend_mode == "additive":
                                # Additive blending
                                src_alpha = pa / 255.0
                                result[y, x, 0] = min(255, result[y, x, 0] + int(pr * src_alpha))
                                result[y, x, 1] = min(255, result[y, x, 1] + int(pg * src_alpha))
                                result[y, x, 2] = min(255, result[y, x, 2] + int(pb * src_alpha))
                            
                            elif blend_mode == "alpha":
                                # Standard alpha blend
                                src_alpha = pa / 255.0
                                dst_alpha = result[y, x, 3] / 255.0
                                out_alpha = src_alpha + dst_alpha * (1 - src_alpha)
                                
                                if out_alpha > 0:
                                    result[y, x, 0] = int((pr * src_alpha + result[y, x, 0] * dst_alpha * (1 - src_alpha)) / out_alpha)
                                    result[y, x, 1] = int((pg * src_alpha + result[y, x, 1] * dst_alpha * (1 - src_alpha)) / out_alpha)
                                    result[y, x, 2] = int((pb * src_alpha + result[y, x, 2] * dst_alpha * (1 - src_alpha)) / out_alpha)
                                    result[y, x, 3] = int(out_alpha * 255)
                            
                            elif blend_mode == "multiply":
                                # Multiply blend
                                src_alpha = pa / 255.0
                                result[y, x, 0] = int(result[y, x, 0] * (1 - src_alpha + src_alpha * pr / 255))
                                result[y, x, 1] = int(result[y, x, 1] * (1 - src_alpha + src_alpha * pg / 255))
                                result[y, x, 2] = int(result[y, x, 2] * (1 - src_alpha + src_alpha * pb / 255))
        
        return result
    
    def clear(self):
        """Remove all particles"""
        self.particles.clear()
    
    @property
    def particle_count(self) -> int:
        return len(self.particles)


# =============================================================================
# Preset Emitters
# =============================================================================

def create_spark_emitter(
    x: float = 0,
    y: float = 0,
    angle: float = -np.pi / 2,  # Up
    spread: float = 0.8,
    speed: float = 80.0
) -> ParticleEmitter:
    """
    Sparks from metal grinding/impacts.
    Accelerate upward, shrink and fade.
    """
    return ParticleEmitter(
        emission=EmissionConfig(
            shape=EmissionShape.CONE,
            angle=angle,
            spread=spread
        ),
        x=x, y=y,
        speed=speed,
        speed_variance=0.4,
        lifetime=0.8,
        lifetime_variance=0.3,
        size=2.5,
        size_variance=0.5,
        size_over_lifetime=SIZE_SHRINK,
        color_gradient=GRADIENT_SPARK,
        gravity=(0, -100),  # Rise up (sparks are hot)
        drag=0.3,
        turbulence=0.15
    )


def create_fire_emitter(
    x: float = 0,
    y: float = 0,
    spread: float = 0.4,
    size: float = 4.0
) -> ParticleEmitter:
    """
    Fire/flame particles.
    Rise up, grow then shrink, fire colors.
    """
    return ParticleEmitter(
        emission=EmissionConfig(
            shape=EmissionShape.CONE,
            angle=-np.pi / 2,  # Up
            spread=spread
        ),
        x=x, y=y,
        speed=30.0,
        speed_variance=0.3,
        lifetime=1.2,
        lifetime_variance=0.3,
        size=size,
        size_variance=0.4,
        size_over_lifetime=SIZE_GROW_SHRINK,
        color_gradient=GRADIENT_FIRE,
        gravity=(0, -20),
        drag=0.2,
        turbulence=0.4,
        turbulence_scale=0.08
    )


def create_smoke_emitter(
    x: float = 0,
    y: float = 0,
    size: float = 6.0
) -> ParticleEmitter:
    """
    Smoke particles.
    Rise slowly, grow and fade.
    """
    return ParticleEmitter(
        emission=EmissionConfig(
            shape=EmissionShape.DISC,
            radius=3.0,
            angle=-np.pi / 2,
            spread=0.3
        ),
        x=x, y=y,
        speed=15.0,
        speed_variance=0.3,
        lifetime=2.0,
        lifetime_variance=0.4,
        size=size,
        size_variance=0.3,
        size_over_lifetime=SIZE_GROW,
        alpha_over_lifetime=ALPHA_FADE_LATE,
        color_gradient=GRADIENT_SMOKE,
        gravity=(0, -5),
        drag=0.4,
        turbulence=0.5,
        turbulence_scale=0.05
    )


def create_magic_emitter(
    x: float = 0,
    y: float = 0,
    radius: float = 10.0
) -> ParticleEmitter:
    """
    Magic/sparkle particles.
    Emit from sphere, spiral outward.
    """
    return ParticleEmitter(
        emission=EmissionConfig(
            shape=EmissionShape.SPHERE,
            radius=radius,
            surface_only=False
        ),
        x=x, y=y,
        speed=20.0,
        speed_variance=0.5,
        lifetime=1.5,
        lifetime_variance=0.3,
        size=2.0,
        size_variance=0.4,
        size_over_lifetime=BezierPath([(0.0, 0.5), (0.3, 1.2), (1.0, 0.0)]),
        color_gradient=GRADIENT_MAGIC,
        gravity=(0, 0),
        turbulence=0.6,
        turbulence_scale=0.1,
        rotation_speed=2.0,
        rotation_variance=1.0
    )


def create_rain_emitter(
    x: float = 0,
    y: float = 0,
    width: float = 100.0,
    speed: float = 200.0
) -> ParticleEmitter:
    """
    Rain particles with collision.
    Fall fast, bounce on ground.
    """
    return ParticleEmitter(
        emission=EmissionConfig(
            shape=EmissionShape.LINE,
            width=width,
            angle=np.pi / 2 + 0.1,  # Slightly angled down
            spread=0.05
        ),
        x=x, y=y,
        speed=speed,
        speed_variance=0.2,
        lifetime=2.0,
        size=1.5,
        size_variance=0.3,
        size_over_lifetime=SIZE_CONSTANT,
        color_gradient=GRADIENT_WATER,
        gravity=(0, 300),
        collision_enabled=True,
        collision_bounce=0.3,
        collision_friction=0.5,
        align_to_velocity=True
    )


def create_explosion_emitter(
    x: float = 0,
    y: float = 0,
    size: float = 3.0,
    count: int = 30
) -> ParticleEmitter:
    """
    Explosion burst.
    Emit all at once from center, decelerate.
    """
    emitter = ParticleEmitter(
        emission=EmissionConfig(
            shape=EmissionShape.SPHERE,
            radius=2.0,
            surface_only=False
        ),
        x=x, y=y,
        speed=100.0,
        speed_variance=0.5,
        lifetime=0.8,
        lifetime_variance=0.2,
        size=size,
        size_variance=0.4,
        size_over_lifetime=SIZE_GROW_SHRINK,
        speed_over_lifetime=SPEED_DECELERATE,
        color_gradient=ColorGradient([
            (0.0, (255, 255, 200, 255)),
            (0.3, (255, 150, 50, 255)),
            (0.6, (200, 80, 20, 200)),
            (1.0, (80, 30, 10, 0))
        ]),
        gravity=(0, 50),
        drag=0.5
    )
    
    emitter.emit(count)
    return emitter


def create_electric_emitter(
    x: float = 0,
    y: float = 0,
    radius: float = 15.0
) -> ParticleEmitter:
    """
    Electric/lightning particles.
    Erratic movement with turbulence.
    """
    return ParticleEmitter(
        emission=EmissionConfig(
            shape=EmissionShape.RING,
            radius=radius,
            inner_radius=radius * 0.5
        ),
        x=x, y=y,
        speed=60.0,
        speed_variance=0.6,
        lifetime=0.5,
        lifetime_variance=0.4,
        size=1.5,
        size_variance=0.5,
        size_over_lifetime=ALPHA_FLASH_FADE,  # Flash effect
        color_gradient=GRADIENT_ELECTRIC,
        turbulence=0.8,
        turbulence_scale=0.2,
        turbulence_speed=3.0
    )


# =============================================================================
# Particle System Manager
# =============================================================================

class ParticleSystem:
    """
    Manages multiple emitters and handles rendering.
    
    Example:
        system = ParticleSystem()
        system.add_emitter("fire", create_fire_emitter(x=50, y=80))
        system.add_emitter("sparks", create_spark_emitter(x=50, y=80))
        
        # Enable continuous emission
        system.emitters["fire"].emission_rate = 20
        
        for frame in range(frames):
            system.update(dt)
            canvas = system.render(canvas)
    """
    
    def __init__(self):
        self.emitters: dict[str, ParticleEmitter] = {}
        self._time = 0.0
    
    def add_emitter(self, name: str, emitter: ParticleEmitter):
        """Add an emitter to the system"""
        self.emitters[name] = emitter
    
    def remove_emitter(self, name: str):
        """Remove an emitter"""
        if name in self.emitters:
            del self.emitters[name]
    
    def emit(self, name: str, count: int = 1):
        """Emit particles from a specific emitter"""
        if name in self.emitters:
            self.emitters[name].emit(count)
    
    def emit_all(self, count: int = 1):
        """Emit particles from all emitters"""
        for emitter in self.emitters.values():
            emitter.emit(count)
    
    def update(self, dt: float):
        """Update all emitters"""
        self._time += dt
        for emitter in self.emitters.values():
            emitter.update(dt)
    
    def render(
        self,
        canvas: np.ndarray,
        offset: Tuple[int, int] = (0, 0),
        blend_mode: str = "additive"
    ) -> np.ndarray:
        """Render all emitters to canvas"""
        result = canvas.copy()
        
        for emitter in self.emitters.values():
            result = emitter.render(result, offset, blend_mode)
        
        return result
    
    def clear(self, name: str = None):
        """Clear particles from one or all emitters"""
        if name:
            if name in self.emitters:
                self.emitters[name].clear()
        else:
            for emitter in self.emitters.values():
                emitter.clear()
    
    @property
    def total_particles(self) -> int:
        """Total particles across all emitters"""
        return sum(e.particle_count for e in self.emitters.values())
