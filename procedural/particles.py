"""
Particle System - Advanced particle emitter and simulation system.

Supports multiple particle types: sparks, dust, magic, fire, smoke, bubbles, etc.
Features: gravity, wind, turbulence, attraction, collision, color gradients, size curves.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Callable
from enum import Enum, auto

from .base import BaseEffect, EffectConfig, Easing


class ParticleType(Enum):
    """Built-in particle types with preset behaviors."""
    SPARK = auto()
    DUST = auto()
    MAGIC = auto()
    FIRE = auto()
    SMOKE = auto()
    BUBBLE = auto()
    STAR = auto()
    SNOW = auto()
    RAIN = auto()
    LEAF = auto()
    DEBRIS = auto()
    ENERGY = auto()
    CUSTOM = auto()


class EmitterShape(Enum):
    """Shape of particle emission area."""
    POINT = auto()
    LINE = auto()
    CIRCLE = auto()
    RECTANGLE = auto()
    SPRITE_EDGE = auto()
    SPRITE_SURFACE = auto()


@dataclass
class ParticleState:
    """State of a single particle."""
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    size: float
    rotation: float
    rotation_speed: float
    color: np.ndarray  # RGBA
    alpha: float
    
    @property
    def life_ratio(self) -> float:
        """Get normalized life (0 = just born, 1 = about to die)."""
        return 1.0 - (self.life / self.max_life) if self.max_life > 0 else 1.0


@dataclass
class ParticleConfig(EffectConfig):
    """Configuration for particle system."""
    # Emitter settings
    particle_type: ParticleType = ParticleType.SPARK
    emitter_shape: EmitterShape = EmitterShape.SPRITE_EDGE
    emission_rate: float = 10.0  # Particles per frame
    burst_count: int = 0  # One-time burst (0 = continuous)
    
    # Particle lifetime
    lifetime_min: float = 0.5
    lifetime_max: float = 1.5
    
    # Initial velocity
    speed_min: float = 1.0
    speed_max: float = 3.0
    direction_min: float = 0.0  # Degrees
    direction_max: float = 360.0
    spread: float = 45.0  # Cone spread in degrees
    
    # Size
    size_min: float = 1.0
    size_max: float = 3.0
    size_over_life: str = "fade"  # "constant", "fade", "grow", "pulse"
    
    # Physics
    gravity: float = 0.0
    wind_x: float = 0.0
    wind_y: float = 0.0
    drag: float = 0.02
    turbulence: float = 0.0
    
    # Rotation
    rotation_min: float = 0.0
    rotation_max: float = 0.0
    rotation_speed_min: float = 0.0
    rotation_speed_max: float = 0.0
    
    # Color
    color_start: Tuple[int, int, int, int] = (255, 255, 255, 255)
    color_end: Tuple[int, int, int, int] = (255, 255, 255, 0)
    use_sprite_colors: bool = False
    color_variation: float = 0.0
    
    # Advanced
    attract_to_center: float = 0.0  # Attraction force to sprite center
    collision_bounce: float = 0.0  # Bounce off sprite pixels
    blend_mode: str = "add"  # "add", "normal", "screen"
    
    # Shape
    particle_shape: str = "circle"  # "circle", "square", "star", "line"
    
    seed: Optional[int] = None


class Particle:
    """A single particle with physics simulation."""
    
    def __init__(self, state: ParticleState):
        self.state = state
        self.alive = True
    
    def update(self, dt: float, config: ParticleConfig, 
               sprite_center: Tuple[float, float],
               sprite_mask: Optional[np.ndarray] = None) -> None:
        """Update particle physics."""
        if not self.alive:
            return
        
        # Age the particle
        self.state.life -= dt
        if self.state.life <= 0:
            self.alive = False
            return
        
        # Apply forces
        # Gravity
        self.state.vy += config.gravity * dt
        
        # Wind
        self.state.vx += config.wind_x * dt
        self.state.vy += config.wind_y * dt
        
        # Drag
        self.state.vx *= (1.0 - config.drag)
        self.state.vy *= (1.0 - config.drag)
        
        # Turbulence
        if config.turbulence > 0:
            self.state.vx += (np.random.random() - 0.5) * config.turbulence
            self.state.vy += (np.random.random() - 0.5) * config.turbulence
        
        # Attraction to center
        if config.attract_to_center != 0:
            dx = sprite_center[0] - self.state.x
            dy = sprite_center[1] - self.state.y
            dist = np.sqrt(dx * dx + dy * dy) + 0.1
            force = config.attract_to_center / dist
            self.state.vx += (dx / dist) * force * dt
            self.state.vy += (dy / dist) * force * dt
        
        # Update position
        self.state.x += self.state.vx
        self.state.y += self.state.vy
        
        # Update rotation
        self.state.rotation += self.state.rotation_speed * dt
        
        # Collision with sprite
        if config.collision_bounce > 0 and sprite_mask is not None:
            ix, iy = int(self.state.x), int(self.state.y)
            h, w = sprite_mask.shape
            if 0 <= ix < w and 0 <= iy < h:
                if sprite_mask[iy, ix]:
                    self.state.vx *= -config.collision_bounce
                    self.state.vy *= -config.collision_bounce
                    self.state.x += self.state.vx * 2
                    self.state.y += self.state.vy * 2
    
    def get_render_state(self, config: ParticleConfig) -> Tuple[float, float, float, np.ndarray, float]:
        """Get current render state (x, y, size, color, rotation)."""
        life_ratio = self.state.life_ratio
        
        # Size over life
        if config.size_over_life == "fade":
            size = self.state.size * (1.0 - life_ratio)
        elif config.size_over_life == "grow":
            size = self.state.size * (0.2 + life_ratio * 0.8)
        elif config.size_over_life == "pulse":
            size = self.state.size * (0.5 + 0.5 * np.sin(life_ratio * np.pi * 4))
        else:
            size = self.state.size
        
        # Color interpolation
        t = life_ratio
        color = np.array([
            config.color_start[0] + (config.color_end[0] - config.color_start[0]) * t,
            config.color_start[1] + (config.color_end[1] - config.color_start[1]) * t,
            config.color_start[2] + (config.color_end[2] - config.color_start[2]) * t,
            config.color_start[3] + (config.color_end[3] - config.color_start[3]) * t,
        ], dtype=np.float32)
        
        # Apply individual color variation
        if config.color_variation > 0:
            color[:3] = color[:3] * self.state.color[:3]
        
        return self.state.x, self.state.y, size, color, self.state.rotation


class ParticleEmitter:
    """Particle emitter that spawns and manages particles."""
    
    def __init__(self, config: ParticleConfig, sprite_shape: Tuple[int, int],
                 sprite_mask: Optional[np.ndarray] = None,
                 sprite_colors: Optional[np.ndarray] = None):
        self.config = config
        self.sprite_shape = sprite_shape  # (height, width)
        self.sprite_mask = sprite_mask
        self.sprite_colors = sprite_colors
        self.particles: List[Particle] = []
        self.emission_accumulator = 0.0
        self.rng = np.random.default_rng(config.seed)
        
        # Pre-calculate emission points for sprite-based emitters
        self._edge_points: List[Tuple[int, int]] = []
        self._surface_points: List[Tuple[int, int]] = []
        if sprite_mask is not None:
            self._calculate_emission_points()
        
        # Burst at start if configured
        if config.burst_count > 0:
            for _ in range(config.burst_count):
                self._spawn_particle()
    
    def _calculate_emission_points(self) -> None:
        """Pre-calculate valid emission points from sprite."""
        h, w = self.sprite_mask.shape
        
        # Find edge pixels
        for y in range(h):
            for x in range(w):
                if self.sprite_mask[y, x]:
                    self._surface_points.append((x, y))
                    # Check if edge
                    is_edge = False
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if ny < 0 or ny >= h or nx < 0 or nx >= w:
                            is_edge = True
                        elif not self.sprite_mask[ny, nx]:
                            is_edge = True
                    if is_edge:
                        self._edge_points.append((x, y))
    
    def _get_emission_position(self) -> Tuple[float, float]:
        """Get a position to spawn a particle."""
        h, w = self.sprite_shape
        shape = self.config.emitter_shape
        
        if shape == EmitterShape.POINT:
            return w / 2, h / 2
        
        elif shape == EmitterShape.LINE:
            return self.rng.random() * w, h / 2
        
        elif shape == EmitterShape.CIRCLE:
            angle = self.rng.random() * 2 * np.pi
            radius = min(w, h) / 3
            return w / 2 + np.cos(angle) * radius, h / 2 + np.sin(angle) * radius
        
        elif shape == EmitterShape.RECTANGLE:
            return self.rng.random() * w, self.rng.random() * h
        
        elif shape == EmitterShape.SPRITE_EDGE:
            if self._edge_points:
                return self._edge_points[self.rng.integers(len(self._edge_points))]
            return w / 2, h / 2
        
        elif shape == EmitterShape.SPRITE_SURFACE:
            if self._surface_points:
                return self._surface_points[self.rng.integers(len(self._surface_points))]
            return w / 2, h / 2
        
        return w / 2, h / 2
    
    def _get_initial_velocity(self) -> Tuple[float, float]:
        """Calculate initial velocity for a new particle."""
        speed = self.rng.uniform(self.config.speed_min, self.config.speed_max)
        
        # Direction with spread
        base_dir = (self.config.direction_min + self.config.direction_max) / 2
        spread = self.config.spread / 2
        direction = np.radians(base_dir + self.rng.uniform(-spread, spread))
        
        return np.cos(direction) * speed, np.sin(direction) * speed
    
    def _get_particle_color(self, x: float, y: float) -> np.ndarray:
        """Get color for particle, optionally from sprite."""
        if self.config.use_sprite_colors and self.sprite_colors is not None:
            ix, iy = int(x), int(y)
            h, w = self.sprite_colors.shape[:2]
            if 0 <= ix < w and 0 <= iy < h:
                return self.sprite_colors[iy, ix, :3].astype(np.float32) / 255.0
        
        # Color variation
        if self.config.color_variation > 0:
            var = self.config.color_variation
            return np.array([
                1.0 + (self.rng.random() - 0.5) * var,
                1.0 + (self.rng.random() - 0.5) * var,
                1.0 + (self.rng.random() - 0.5) * var,
            ], dtype=np.float32)
        
        return np.ones(3, dtype=np.float32)
    
    def _spawn_particle(self) -> None:
        """Spawn a new particle."""
        x, y = self._get_emission_position()
        vx, vy = self._get_initial_velocity()
        
        state = ParticleState(
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            life=self.rng.uniform(self.config.lifetime_min, self.config.lifetime_max),
            max_life=self.config.lifetime_max,
            size=self.rng.uniform(self.config.size_min, self.config.size_max),
            rotation=self.rng.uniform(self.config.rotation_min, self.config.rotation_max),
            rotation_speed=self.rng.uniform(self.config.rotation_speed_min, self.config.rotation_speed_max),
            color=self._get_particle_color(x, y),
            alpha=1.0
        )
        
        self.particles.append(Particle(state))
    
    def update(self, dt: float = 1.0) -> None:
        """Update all particles and spawn new ones."""
        h, w = self.sprite_shape
        center = (w / 2, h / 2)
        
        # Update existing particles
        for particle in self.particles:
            particle.update(dt, self.config, center, self.sprite_mask)
        
        # Remove dead particles
        self.particles = [p for p in self.particles if p.alive]
        
        # Spawn new particles (continuous emission)
        if self.config.burst_count == 0:
            self.emission_accumulator += self.config.emission_rate * dt
            while self.emission_accumulator >= 1.0:
                self._spawn_particle()
                self.emission_accumulator -= 1.0
    
    def render(self, canvas: np.ndarray) -> np.ndarray:
        """Render all particles onto the canvas."""
        result = canvas.astype(np.float32)
        h, w = canvas.shape[:2]
        
        for particle in self.particles:
            if not particle.alive:
                continue
            
            x, y, size, color, rotation = particle.get_render_state(self.config)
            
            # Skip if outside bounds
            if x < -size or x >= w + size or y < -size or y >= h + size:
                continue
            
            # Render particle shape
            self._render_particle(result, x, y, size, color, rotation)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _render_particle(self, canvas: np.ndarray, x: float, y: float, 
                         size: float, color: np.ndarray, rotation: float) -> None:
        """Render a single particle."""
        h, w = canvas.shape[:2]
        shape = self.config.particle_shape
        blend = self.config.blend_mode
        
        # Calculate bounds
        half = int(np.ceil(size / 2)) + 1
        x0 = max(0, int(x - half))
        x1 = min(w, int(x + half + 1))
        y0 = max(0, int(y - half))
        y1 = min(h, int(y + half + 1))
        
        if x0 >= x1 or y0 >= y1:
            return
        
        # Create coordinate grids
        yy, xx = np.mgrid[y0:y1, x0:x1]
        xx = xx.astype(np.float32)
        yy = yy.astype(np.float32)
        
        # Calculate distance/shape mask
        if shape == "circle":
            dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
            mask = np.maximum(0, 1.0 - dist / (size / 2 + 0.5))
        
        elif shape == "square":
            dist_x = np.abs(xx - x)
            dist_y = np.abs(yy - y)
            dist = np.maximum(dist_x, dist_y)
            mask = (dist <= size / 2).astype(np.float32)
        
        elif shape == "star":
            # Simple 4-point star
            dx = xx - x
            dy = yy - y
            # Rotate
            c, s = np.cos(rotation), np.sin(rotation)
            rx = dx * c - dy * s
            ry = dx * s + dy * c
            # Star shape
            r = np.sqrt(rx ** 2 + ry ** 2)
            angle = np.arctan2(ry, rx)
            star_r = size / 2 * (0.5 + 0.5 * np.abs(np.cos(angle * 2)))
            mask = np.maximum(0, 1.0 - r / (star_r + 0.5))
        
        elif shape == "line":
            # Oriented line
            dx = xx - x
            dy = yy - y
            c, s = np.cos(rotation), np.sin(rotation)
            # Distance along line and perpendicular
            along = dx * c + dy * s
            perp = np.abs(-dx * s + dy * c)
            mask = ((np.abs(along) <= size / 2) & (perp <= 1.0)).astype(np.float32)
        
        else:
            dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
            mask = np.maximum(0, 1.0 - dist / (size / 2 + 0.5))
        
        # Apply alpha from color
        mask = mask * (color[3] / 255.0)
        
        # Blend onto canvas
        for c_idx in range(3):
            src_color = color[c_idx]
            if blend == "add":
                canvas[y0:y1, x0:x1, c_idx] += src_color * mask
            elif blend == "screen":
                dst = canvas[y0:y1, x0:x1, c_idx]
                canvas[y0:y1, x0:x1, c_idx] = 255 - (255 - dst) * (1 - src_color / 255 * mask)
            else:  # normal
                dst = canvas[y0:y1, x0:x1, c_idx]
                canvas[y0:y1, x0:x1, c_idx] = dst * (1 - mask) + src_color * mask


# Preset configurations for different particle types
PARTICLE_PRESETS: Dict[ParticleType, Dict] = {
    ParticleType.SPARK: {
        "emission_rate": 8.0,
        "lifetime_min": 0.3,
        "lifetime_max": 0.8,
        "speed_min": 2.0,
        "speed_max": 5.0,
        "size_min": 1.0,
        "size_max": 2.0,
        "gravity": 0.3,
        "color_start": (255, 220, 100, 255),
        "color_end": (255, 100, 0, 0),
        "particle_shape": "circle",
        "blend_mode": "add",
    },
    ParticleType.DUST: {
        "emission_rate": 3.0,
        "lifetime_min": 1.0,
        "lifetime_max": 2.0,
        "speed_min": 0.2,
        "speed_max": 0.8,
        "size_min": 1.0,
        "size_max": 3.0,
        "gravity": -0.05,
        "turbulence": 0.3,
        "color_start": (200, 180, 150, 150),
        "color_end": (200, 180, 150, 0),
        "particle_shape": "circle",
        "blend_mode": "normal",
    },
    ParticleType.MAGIC: {
        "emission_rate": 12.0,
        "lifetime_min": 0.5,
        "lifetime_max": 1.2,
        "speed_min": 0.5,
        "speed_max": 2.0,
        "size_min": 1.0,
        "size_max": 4.0,
        "size_over_life": "pulse",
        "gravity": -0.1,
        "turbulence": 0.5,
        "color_start": (150, 200, 255, 255),
        "color_end": (200, 100, 255, 0),
        "particle_shape": "star",
        "blend_mode": "add",
    },
    ParticleType.FIRE: {
        "emission_rate": 15.0,
        "lifetime_min": 0.3,
        "lifetime_max": 0.7,
        "speed_min": 1.0,
        "speed_max": 3.0,
        "direction_min": 250,
        "direction_max": 290,
        "spread": 30,
        "size_min": 2.0,
        "size_max": 5.0,
        "size_over_life": "grow",
        "gravity": -0.5,
        "color_start": (255, 200, 50, 255),
        "color_end": (255, 50, 0, 0),
        "particle_shape": "circle",
        "blend_mode": "add",
    },
    ParticleType.SMOKE: {
        "emission_rate": 5.0,
        "lifetime_min": 1.5,
        "lifetime_max": 3.0,
        "speed_min": 0.3,
        "speed_max": 1.0,
        "direction_min": 250,
        "direction_max": 290,
        "size_min": 3.0,
        "size_max": 8.0,
        "size_over_life": "grow",
        "gravity": -0.15,
        "turbulence": 0.4,
        "drag": 0.05,
        "color_start": (100, 100, 100, 180),
        "color_end": (80, 80, 80, 0),
        "particle_shape": "circle",
        "blend_mode": "normal",
    },
    ParticleType.BUBBLE: {
        "emission_rate": 4.0,
        "lifetime_min": 1.0,
        "lifetime_max": 2.5,
        "speed_min": 0.5,
        "speed_max": 1.5,
        "direction_min": 250,
        "direction_max": 290,
        "size_min": 2.0,
        "size_max": 5.0,
        "gravity": -0.2,
        "turbulence": 0.3,
        "color_start": (200, 230, 255, 200),
        "color_end": (255, 255, 255, 0),
        "particle_shape": "circle",
        "blend_mode": "screen",
    },
    ParticleType.STAR: {
        "emission_rate": 6.0,
        "lifetime_min": 0.5,
        "lifetime_max": 1.5,
        "speed_min": 0.2,
        "speed_max": 1.0,
        "size_min": 2.0,
        "size_max": 4.0,
        "size_over_life": "pulse",
        "rotation_speed_min": -2.0,
        "rotation_speed_max": 2.0,
        "color_start": (255, 255, 200, 255),
        "color_end": (255, 255, 100, 0),
        "particle_shape": "star",
        "blend_mode": "add",
    },
    ParticleType.SNOW: {
        "emission_rate": 8.0,
        "emitter_shape": EmitterShape.LINE,
        "lifetime_min": 2.0,
        "lifetime_max": 4.0,
        "speed_min": 0.5,
        "speed_max": 1.5,
        "direction_min": 80,
        "direction_max": 100,
        "size_min": 1.0,
        "size_max": 3.0,
        "gravity": 0.1,
        "wind_x": 0.2,
        "turbulence": 0.2,
        "color_start": (255, 255, 255, 255),
        "color_end": (255, 255, 255, 100),
        "particle_shape": "circle",
        "blend_mode": "normal",
    },
    ParticleType.RAIN: {
        "emission_rate": 20.0,
        "emitter_shape": EmitterShape.LINE,
        "lifetime_min": 0.5,
        "lifetime_max": 1.0,
        "speed_min": 5.0,
        "speed_max": 8.0,
        "direction_min": 85,
        "direction_max": 95,
        "size_min": 1.0,
        "size_max": 2.0,
        "gravity": 0.5,
        "color_start": (180, 200, 255, 200),
        "color_end": (180, 200, 255, 50),
        "particle_shape": "line",
        "rotation_min": 1.5,
        "rotation_max": 1.6,
        "blend_mode": "add",
    },
    ParticleType.LEAF: {
        "emission_rate": 3.0,
        "lifetime_min": 2.0,
        "lifetime_max": 4.0,
        "speed_min": 0.5,
        "speed_max": 1.5,
        "size_min": 2.0,
        "size_max": 4.0,
        "gravity": 0.1,
        "wind_x": 0.3,
        "turbulence": 0.4,
        "rotation_speed_min": -1.0,
        "rotation_speed_max": 1.0,
        "color_start": (100, 180, 60, 255),
        "color_end": (150, 120, 50, 100),
        "particle_shape": "square",
        "blend_mode": "normal",
    },
    ParticleType.DEBRIS: {
        "emission_rate": 10.0,
        "burst_count": 20,
        "lifetime_min": 0.5,
        "lifetime_max": 1.5,
        "speed_min": 2.0,
        "speed_max": 6.0,
        "size_min": 1.0,
        "size_max": 3.0,
        "gravity": 0.4,
        "drag": 0.03,
        "rotation_speed_min": -3.0,
        "rotation_speed_max": 3.0,
        "color_start": (180, 160, 140, 255),
        "color_end": (120, 100, 80, 100),
        "particle_shape": "square",
        "blend_mode": "normal",
    },
    ParticleType.ENERGY: {
        "emission_rate": 15.0,
        "emitter_shape": EmitterShape.SPRITE_SURFACE,
        "lifetime_min": 0.2,
        "lifetime_max": 0.5,
        "speed_min": 0.5,
        "speed_max": 2.0,
        "size_min": 1.0,
        "size_max": 3.0,
        "size_over_life": "fade",
        "attract_to_center": -0.5,
        "color_start": (100, 200, 255, 255),
        "color_end": (200, 100, 255, 0),
        "particle_shape": "circle",
        "blend_mode": "add",
    },
}


@dataclass
class ParticleEffectConfig(EffectConfig):
    """Configuration for particle effect wrapper."""
    particle_type: str = "spark"  # Name of ParticleType
    emission_rate: float = 10.0
    lifetime: float = 1.0
    speed: float = 2.0
    size: float = 2.0
    gravity: float = 0.0
    turbulence: float = 0.0
    color_start: Tuple[int, int, int, int] = (255, 255, 255, 255)
    color_end: Tuple[int, int, int, int] = (255, 255, 255, 0)
    emitter_shape: str = "sprite_edge"
    blend_mode: str = "add"
    use_sprite_colors: bool = False
    burst: int = 0
    seed: Optional[int] = None


class ParticleEffect(BaseEffect):
    """Procedural effect wrapper for particle system."""
    
    name = "particles"
    description = "Advanced particle emitter system with multiple types"
    
    config_class = ParticleEffectConfig
    
    def __init__(self, config: ParticleEffectConfig):
        super().__init__(config)
        self.emitter: Optional[ParticleEmitter] = None
        self._frame_count = 0
    
    def _get_particle_type(self) -> ParticleType:
        """Convert string to ParticleType enum."""
        type_map = {
            "spark": ParticleType.SPARK,
            "dust": ParticleType.DUST,
            "magic": ParticleType.MAGIC,
            "fire": ParticleType.FIRE,
            "smoke": ParticleType.SMOKE,
            "bubble": ParticleType.BUBBLE,
            "star": ParticleType.STAR,
            "snow": ParticleType.SNOW,
            "rain": ParticleType.RAIN,
            "leaf": ParticleType.LEAF,
            "debris": ParticleType.DEBRIS,
            "energy": ParticleType.ENERGY,
        }
        return type_map.get(self.config.particle_type.lower(), ParticleType.SPARK)
    
    def _get_emitter_shape(self) -> EmitterShape:
        """Convert string to EmitterShape enum."""
        shape_map = {
            "point": EmitterShape.POINT,
            "line": EmitterShape.LINE,
            "circle": EmitterShape.CIRCLE,
            "rectangle": EmitterShape.RECTANGLE,
            "sprite_edge": EmitterShape.SPRITE_EDGE,
            "sprite_surface": EmitterShape.SPRITE_SURFACE,
        }
        return shape_map.get(self.config.emitter_shape.lower(), EmitterShape.SPRITE_EDGE)
    
    def _create_particle_config(self, particle_type: ParticleType) -> ParticleConfig:
        """Create particle config from effect config and presets."""
        # Start with preset
        preset = PARTICLE_PRESETS.get(particle_type, {}).copy()
        
        # Override with user config
        config = ParticleConfig(
            particle_type=particle_type,
            emitter_shape=preset.get("emitter_shape", self._get_emitter_shape()),
            emission_rate=self.config.emission_rate,
            burst_count=self.config.burst,
            lifetime_min=self.config.lifetime * 0.5,
            lifetime_max=self.config.lifetime * 1.5,
            speed_min=self.config.speed * 0.5,
            speed_max=self.config.speed * 1.5,
            direction_min=preset.get("direction_min", 0),
            direction_max=preset.get("direction_max", 360),
            spread=preset.get("spread", 45),
            size_min=self.config.size * 0.5,
            size_max=self.config.size * 1.5,
            size_over_life=preset.get("size_over_life", "fade"),
            gravity=self.config.gravity if self.config.gravity != 0 else preset.get("gravity", 0),
            turbulence=self.config.turbulence if self.config.turbulence != 0 else preset.get("turbulence", 0),
            drag=preset.get("drag", 0.02),
            rotation_min=preset.get("rotation_min", 0),
            rotation_max=preset.get("rotation_max", 0),
            rotation_speed_min=preset.get("rotation_speed_min", 0),
            rotation_speed_max=preset.get("rotation_speed_max", 0),
            color_start=self.config.color_start,
            color_end=self.config.color_end,
            use_sprite_colors=self.config.use_sprite_colors,
            blend_mode=self.config.blend_mode,
            particle_shape=preset.get("particle_shape", "circle"),
            attract_to_center=preset.get("attract_to_center", 0),
            seed=self.config.seed,
        )
        
        return config
    
    def process_frame(self, image: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        """Process a single frame with particle effects."""
        # Initialize emitter on first frame
        if self.emitter is None or frame_idx == 0:
            h, w = image.shape[:2]
            
            # Create sprite mask
            if image.shape[2] == 4:
                sprite_mask = image[:, :, 3] > 128
            else:
                sprite_mask = np.any(image > 0, axis=2)
            
            particle_type = self._get_particle_type()
            particle_config = self._create_particle_config(particle_type)
            
            self.emitter = ParticleEmitter(
                config=particle_config,
                sprite_shape=(h, w),
                sprite_mask=sprite_mask,
                sprite_colors=image
            )
            self._frame_count = 0
        
        # Update emitter multiple times to "catch up" to current frame
        steps_needed = frame_idx - self._frame_count
        for _ in range(max(1, steps_needed)):
            self.emitter.update(dt=1.0)
        self._frame_count = frame_idx
        
        # Render particles
        result = self.emitter.render(image.copy())
        
        return result
    
    def apply(self, sprite) -> list:
        """Apply particle effect to sprite and return animation frames."""
        from src.core import Sprite
        
        frames = []
        for i in range(self.config.frame_count):
            pixels = self.process_frame(sprite.pixels.copy(), i, self.config.frame_count)
            frame = Sprite(
                width=sprite.width,
                height=sprite.height,
                pixels=pixels,
                name=f"{sprite.name}_particles_{i}",
                source_path=sprite.source_path
            )
            frames.append(frame)
        return frames
