"""
Secondary Motion & Follow-Through Physics

Professional animations have layers of motion:
- Primary: Main sprite movement
- Secondary: Trailing parts (hair, cloth, tails) 
- Tertiary: Particles, after-effects

This module provides:
- Bone/attachment system for dangling parts
- Spring physics for cloth/hair simulation
- Particle inheritance (particles follow sprite motion with lag)

Inspired by Disney's 12 principles of animation:
- Follow-through and overlapping action
- Secondary action
- Slow in and slow out
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum


# =============================================================================
# Vector Utilities
# =============================================================================

@dataclass
class Vec2:
    """2D vector with physics operations"""
    x: float = 0.0
    y: float = 0.0
    
    def __add__(self, other: 'Vec2') -> 'Vec2':
        return Vec2(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Vec2') -> 'Vec2':
        return Vec2(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Vec2':
        return Vec2(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar: float) -> 'Vec2':
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: float) -> 'Vec2':
        return Vec2(self.x / scalar, self.y / scalar) if scalar != 0 else Vec2()
    
    def __neg__(self) -> 'Vec2':
        return Vec2(-self.x, -self.y)
    
    @property
    def length(self) -> float:
        return np.sqrt(self.x * self.x + self.y * self.y)
    
    @property
    def length_squared(self) -> float:
        return self.x * self.x + self.y * self.y
    
    def normalized(self) -> 'Vec2':
        l = self.length
        return Vec2(self.x / l, self.y / l) if l > 1e-10 else Vec2()
    
    def dot(self, other: 'Vec2') -> float:
        return self.x * other.x + self.y * other.y
    
    def cross(self, other: 'Vec2') -> float:
        """2D cross product (returns scalar)"""
        return self.x * other.y - self.y * other.x
    
    def rotate(self, angle: float) -> 'Vec2':
        """Rotate by angle (radians)"""
        c, s = np.cos(angle), np.sin(angle)
        return Vec2(self.x * c - self.y * s, self.x * s + self.y * c)
    
    def lerp(self, other: 'Vec2', t: float) -> 'Vec2':
        return Vec2(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t
        )
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def to_int_tuple(self) -> Tuple[int, int]:
        return (int(round(self.x)), int(round(self.y)))
    
    @staticmethod
    def from_angle(angle: float, length: float = 1.0) -> 'Vec2':
        return Vec2(np.cos(angle) * length, np.sin(angle) * length)


# =============================================================================
# Spring Physics
# =============================================================================

@dataclass
class SpringConfig:
    """Configuration for spring physics behavior"""
    stiffness: float = 180.0      # Spring constant (higher = snappier)
    damping: float = 12.0         # Damping (higher = less oscillation)
    mass: float = 1.0             # Mass (higher = slower response)
    rest_length: float = 0.0      # Rest length for distance springs
    
    # Presets
    @classmethod
    def gentle(cls) -> 'SpringConfig':
        """Slow, gentle motion (floating, ambient)"""
        return cls(stiffness=80, damping=8, mass=1.5)
    
    @classmethod
    def bouncy(cls) -> 'SpringConfig':
        """Bouncy, elastic motion (jelly, rubber)"""
        return cls(stiffness=300, damping=10, mass=1.0)
    
    @classmethod
    def snappy(cls) -> 'SpringConfig':
        """Quick, responsive motion (UI, game feel)"""
        return cls(stiffness=400, damping=26, mass=1.0)
    
    @classmethod
    def wobbly(cls) -> 'SpringConfig':
        """Wobbly, underdamped motion (hair, cloth)"""
        return cls(stiffness=150, damping=6, mass=0.8)
    
    @classmethod
    def heavy(cls) -> 'SpringConfig':
        """Heavy, slow motion (chains, ropes)"""
        return cls(stiffness=100, damping=15, mass=2.0)


class Spring:
    """
    Critically-damped spring for smooth follow-through motion.
    
    Uses velocity Verlet integration for stable physics.
    """
    
    def __init__(self, config: SpringConfig = None):
        self.config = config or SpringConfig()
        self.position = Vec2()
        self.velocity = Vec2()
        self.target = Vec2()
    
    def set_position(self, x: float, y: float):
        """Set current position (teleport)"""
        self.position = Vec2(x, y)
        self.velocity = Vec2()
    
    def set_target(self, x: float, y: float):
        """Set target position to spring towards"""
        self.target = Vec2(x, y)
    
    def update(self, dt: float) -> Vec2:
        """
        Update spring physics and return new position.
        
        Args:
            dt: Delta time in seconds
        
        Returns:
            Current position after physics update
        """
        # Spring force: F = -k * (x - target)
        displacement = self.position - self.target
        spring_force = displacement * (-self.config.stiffness)
        
        # Damping force: F = -c * v
        damping_force = self.velocity * (-self.config.damping)
        
        # Total acceleration: a = F / m
        acceleration = (spring_force + damping_force) / self.config.mass
        
        # Velocity Verlet integration (more stable than Euler)
        self.velocity = self.velocity + acceleration * dt
        self.position = self.position + self.velocity * dt
        
        return self.position
    
    def get_position(self) -> Vec2:
        return self.position
    
    def is_settled(self, threshold: float = 0.01) -> bool:
        """Check if spring has settled (position near target, low velocity)"""
        dist = (self.position - self.target).length
        speed = self.velocity.length
        return dist < threshold and speed < threshold


class Spring1D:
    """1D spring for single-axis motion"""
    
    def __init__(self, config: SpringConfig = None):
        self.config = config or SpringConfig()
        self.position = 0.0
        self.velocity = 0.0
        self.target = 0.0
    
    def set_position(self, value: float):
        self.position = value
        self.velocity = 0.0
    
    def set_target(self, value: float):
        self.target = value
    
    def update(self, dt: float) -> float:
        displacement = self.position - self.target
        spring_force = -self.config.stiffness * displacement
        damping_force = -self.config.damping * self.velocity
        acceleration = (spring_force + damping_force) / self.config.mass
        
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        
        return self.position


# =============================================================================
# Bone/Attachment System
# =============================================================================

@dataclass
class Bone:
    """
    A bone in a skeletal hierarchy for secondary motion.
    
    Bones can be attached to:
    - Parent bones (for chains like hair, tails)
    - Sprite anchor points (for dangling attachments)
    """
    name: str
    length: float
    angle: float = 0.0              # Local angle relative to parent
    
    # Physics properties
    spring_config: SpringConfig = field(default_factory=SpringConfig.wobbly)
    gravity: float = 50.0           # Downward force
    wind_factor: float = 1.0        # How much wind affects this bone
    
    # State
    world_position: Vec2 = field(default_factory=Vec2)
    world_angle: float = 0.0
    angular_velocity: float = 0.0
    
    # Hierarchy
    parent: Optional['Bone'] = None
    children: List['Bone'] = field(default_factory=list)


class Skeleton:
    """
    Skeletal system for secondary motion on dangling parts.
    
    Use for: hair, tails, capes, chains, tentacles, ribbons
    
    Example:
        skeleton = Skeleton()
        
        # Create a 3-bone hair chain
        skeleton.add_chain(
            "hair",
            anchor=(16, 8),  # Attach point on sprite
            bone_count=3,
            bone_length=4,
            spring=SpringConfig.wobbly()
        )
        
        # Each frame, update with sprite motion
        skeleton.update(dt, sprite_velocity, wind)
        
        # Get bone positions for rendering
        positions = skeleton.get_chain_positions("hair")
    """
    
    def __init__(self):
        self.roots: Dict[str, Bone] = {}
        self.all_bones: List[Bone] = []
        self.anchors: Dict[str, Vec2] = {}
    
    def add_bone(
        self,
        name: str,
        length: float,
        parent: Optional[Bone] = None,
        angle: float = np.pi / 2,  # Default: pointing down
        spring_config: SpringConfig = None,
        gravity: float = 50.0,
        wind_factor: float = 1.0
    ) -> Bone:
        """Add a single bone to the skeleton"""
        bone = Bone(
            name=name,
            length=length,
            angle=angle,
            spring_config=spring_config or SpringConfig.wobbly(),
            gravity=gravity,
            wind_factor=wind_factor,
            parent=parent
        )
        
        if parent:
            parent.children.append(bone)
        else:
            self.roots[name] = bone
        
        self.all_bones.append(bone)
        return bone
    
    def add_chain(
        self,
        name: str,
        anchor: Tuple[float, float],
        bone_count: int,
        bone_length: float,
        spring_config: SpringConfig = None,
        gravity: float = 50.0,
        wind_factor: float = 1.0,
        stiffness_falloff: float = 0.7  # Each bone gets softer
    ) -> List[Bone]:
        """
        Create a chain of bones (for hair, tails, etc.)
        
        Args:
            name: Base name for chain (bones named "{name}_0", "{name}_1", etc.)
            anchor: Attachment point on sprite
            bone_count: Number of bones in chain
            bone_length: Length of each bone
            spring_config: Base spring configuration
            gravity: Gravity strength
            wind_factor: Wind influence
            stiffness_falloff: How much softer each successive bone gets
        
        Returns:
            List of bones in the chain
        """
        self.anchors[name] = Vec2(anchor[0], anchor[1])
        
        config = spring_config or SpringConfig.wobbly()
        bones = []
        parent = None
        
        for i in range(bone_count):
            # Each bone gets progressively softer
            falloff = stiffness_falloff ** i
            bone_config = SpringConfig(
                stiffness=config.stiffness * falloff,
                damping=config.damping * falloff,
                mass=config.mass
            )
            
            bone = self.add_bone(
                name=f"{name}_{i}",
                length=bone_length,
                parent=parent,
                spring_config=bone_config,
                gravity=gravity * (1 + i * 0.2),  # More gravity at tips
                wind_factor=wind_factor * (1 + i * 0.3)  # More wind at tips
            )
            bones.append(bone)
            parent = bone
        
        return bones
    
    def update(
        self,
        dt: float,
        sprite_velocity: Vec2 = None,
        sprite_offset: Vec2 = None,
        wind: Vec2 = None
    ):
        """
        Update all bone physics.
        
        Args:
            dt: Delta time in seconds
            sprite_velocity: Velocity of the main sprite (for momentum transfer)
            sprite_offset: Current offset of sprite from rest position
            wind: Wind force vector
        """
        sprite_velocity = sprite_velocity or Vec2()
        sprite_offset = sprite_offset or Vec2()
        wind = wind or Vec2()
        
        # Update anchors based on sprite offset
        for name, anchor in self.anchors.items():
            if name in self.roots or any(b.name.startswith(name + "_") for b in self.all_bones):
                pass  # Anchor positions are relative to sprite
        
        # Update each bone chain
        for root_name, root in self.roots.items():
            self._update_bone_recursive(
                root,
                self.anchors.get(root_name.split('_')[0], Vec2()) + sprite_offset,
                0.0,
                sprite_velocity,
                wind,
                dt
            )
    
    def _update_bone_recursive(
        self,
        bone: Bone,
        parent_end: Vec2,
        parent_angle: float,
        sprite_velocity: Vec2,
        wind: Vec2,
        dt: float
    ):
        """Recursively update bone and children"""
        # Target angle based on parent + rest angle
        target_angle = parent_angle + bone.angle
        
        # Apply forces to angular velocity
        # 1. Spring force towards rest angle
        angle_diff = self._normalize_angle(target_angle - bone.world_angle)
        spring_torque = bone.spring_config.stiffness * angle_diff
        
        # 2. Damping
        damping_torque = -bone.spring_config.damping * bone.angular_velocity
        
        # 3. Gravity (always pulls down)
        gravity_torque = bone.gravity * np.sin(bone.world_angle)
        
        # 4. Wind
        wind_torque = (wind.x * np.cos(bone.world_angle) + 
                       wind.y * np.sin(bone.world_angle)) * bone.wind_factor
        
        # 5. Momentum from sprite motion (inertia)
        inertia_torque = -(sprite_velocity.x * np.cos(bone.world_angle) +
                          sprite_velocity.y * np.sin(bone.world_angle)) * 0.5
        
        # Total angular acceleration
        total_torque = spring_torque + damping_torque - gravity_torque + wind_torque + inertia_torque
        angular_accel = total_torque / bone.spring_config.mass
        
        # Integrate
        bone.angular_velocity += angular_accel * dt
        bone.world_angle += bone.angular_velocity * dt
        
        # Clamp angular velocity to prevent instability
        max_angular_vel = 20.0
        bone.angular_velocity = np.clip(bone.angular_velocity, -max_angular_vel, max_angular_vel)
        
        # Calculate world position (end of bone)
        bone.world_position = parent_end + Vec2.from_angle(bone.world_angle, bone.length)
        
        # Update children
        for child in bone.children:
            self._update_bone_recursive(
                child,
                bone.world_position,
                bone.world_angle,
                sprite_velocity,
                wind,
                dt
            )
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def get_chain_positions(self, chain_name: str) -> List[Vec2]:
        """Get world positions of all bones in a chain"""
        positions = []
        
        # Find anchor
        anchor = self.anchors.get(chain_name, Vec2())
        positions.append(anchor)
        
        # Find chain bones
        for bone in self.all_bones:
            if bone.name.startswith(chain_name + "_"):
                positions.append(bone.world_position)
        
        return positions
    
    def get_bone(self, name: str) -> Optional[Bone]:
        """Get a bone by name"""
        for bone in self.all_bones:
            if bone.name == name:
                return bone
        return None


# =============================================================================
# Particle Follow System
# =============================================================================

@dataclass
class TrailingParticle:
    """A particle that follows the sprite with lag"""
    position: Vec2
    velocity: Vec2
    target_offset: Vec2      # Offset from sprite center
    spring: Spring
    lifetime: float
    max_lifetime: float
    color: Tuple[int, int, int]
    size: float
    
    # Visual properties
    fade_in: float = 0.1     # Fraction of lifetime for fade in
    fade_out: float = 0.3    # Fraction of lifetime for fade out
    
    @property
    def age(self) -> float:
        return 1.0 - (self.lifetime / self.max_lifetime)
    
    @property
    def alpha(self) -> float:
        age = self.age
        if age < self.fade_in:
            return age / self.fade_in
        elif age > (1.0 - self.fade_out):
            return (1.0 - age) / self.fade_out
        return 1.0


class ParticleFollowSystem:
    """
    System for particles that follow sprite motion with lag.
    
    Creates trailing effects like:
    - Ember trails behind flames
    - Magic sparkles following a wand
    - Dust clouds behind moving objects
    
    Example:
        particles = ParticleFollowSystem(
            spring_config=SpringConfig.gentle(),
            spawn_rate=10,  # particles per second
            lifetime=0.5
        )
        
        # Each frame
        particles.set_sprite_position(sprite_x, sprite_y)
        particles.update(dt)
        
        # Render
        for p in particles.get_particles():
            draw_particle(p.position, p.color, p.alpha)
    """
    
    def __init__(
        self,
        spring_config: SpringConfig = None,
        spawn_rate: float = 10.0,
        lifetime: float = 0.5,
        spawn_radius: float = 2.0,
        colors: List[Tuple[int, int, int]] = None,
        size_range: Tuple[float, float] = (1.0, 2.0),
        inherit_velocity: float = 0.3,  # How much sprite velocity affects particles
        gravity: float = 0.0,
        seed: int = None
    ):
        self.spring_config = spring_config or SpringConfig.gentle()
        self.spawn_rate = spawn_rate
        self.lifetime = lifetime
        self.spawn_radius = spawn_radius
        self.colors = colors or [(255, 200, 100), (255, 150, 50), (255, 100, 30)]
        self.size_range = size_range
        self.inherit_velocity = inherit_velocity
        self.gravity = gravity
        
        self.rng = np.random.default_rng(seed)
        
        self.particles: List[TrailingParticle] = []
        self.sprite_position = Vec2()
        self.sprite_velocity = Vec2()
        self.spawn_accumulator = 0.0
    
    def set_sprite_position(self, x: float, y: float):
        """Update sprite position (call each frame)"""
        new_pos = Vec2(x, y)
        self.sprite_velocity = new_pos - self.sprite_position
        self.sprite_position = new_pos
    
    def update(self, dt: float):
        """Update particle system"""
        # Spawn new particles
        self.spawn_accumulator += dt * self.spawn_rate
        while self.spawn_accumulator >= 1.0:
            self._spawn_particle()
            self.spawn_accumulator -= 1.0
        
        # Update existing particles
        dead_particles = []
        for particle in self.particles:
            # Update spring target (particle tries to stay at offset from sprite)
            target = self.sprite_position + particle.target_offset
            particle.spring.set_target(target.x, target.y)
            
            # Update spring physics
            particle.spring.update(dt)
            particle.position = particle.spring.get_position()
            
            # Apply gravity
            if self.gravity != 0:
                particle.velocity = particle.velocity + Vec2(0, self.gravity * dt)
                particle.position = particle.position + particle.velocity * dt
            
            # Update lifetime
            particle.lifetime -= dt
            if particle.lifetime <= 0:
                dead_particles.append(particle)
        
        # Remove dead particles
        for p in dead_particles:
            self.particles.remove(p)
    
    def _spawn_particle(self):
        """Spawn a new trailing particle"""
        # Random offset from sprite center
        angle = self.rng.random() * 2 * np.pi
        radius = self.rng.random() * self.spawn_radius
        offset = Vec2.from_angle(angle, radius)
        
        # Initial position with some lag
        initial_pos = self.sprite_position + offset - self.sprite_velocity * self.inherit_velocity
        
        # Create spring
        spring = Spring(self.spring_config)
        spring.set_position(initial_pos.x, initial_pos.y)
        spring.set_target(self.sprite_position.x + offset.x, self.sprite_position.y + offset.y)
        
        # Random properties
        color = self.colors[self.rng.integers(0, len(self.colors))]
        size = self.rng.uniform(self.size_range[0], self.size_range[1])
        
        particle = TrailingParticle(
            position=initial_pos,
            velocity=self.sprite_velocity * self.inherit_velocity,
            target_offset=offset,
            spring=spring,
            lifetime=self.lifetime * (0.8 + self.rng.random() * 0.4),
            max_lifetime=self.lifetime,
            color=color,
            size=size
        )
        
        self.particles.append(particle)
    
    def get_particles(self) -> List[TrailingParticle]:
        """Get all active particles for rendering"""
        return self.particles
    
    def clear(self):
        """Remove all particles"""
        self.particles.clear()


# =============================================================================
# Motion Lag System (Simple Follow-Through)
# =============================================================================

class MotionLag:
    """
    Simple motion lag for follow-through effects.
    
    Tracks a target position with springy delay.
    Perfect for secondary elements that should follow primary motion.
    
    Example:
        # Create lag for a trailing element
        shadow_lag = MotionLag(delay_frames=3, spring=SpringConfig.gentle())
        
        # Each frame
        shadow_lag.push(sprite_x, sprite_y)
        shadow_x, shadow_y = shadow_lag.get_position()
    """
    
    def __init__(
        self,
        delay_frames: int = 3,
        spring_config: SpringConfig = None,
        use_spring: bool = True
    ):
        self.delay_frames = delay_frames
        self.history: List[Vec2] = []
        
        self.use_spring = use_spring
        if use_spring:
            self.spring = Spring(spring_config or SpringConfig.gentle())
        else:
            self.spring = None
        
        self.current_position = Vec2()
    
    def push(self, x: float, y: float):
        """Add new position to history"""
        self.history.append(Vec2(x, y))
        
        # Keep only needed history
        if len(self.history) > self.delay_frames + 1:
            self.history.pop(0)
        
        # Update spring target if using spring
        if self.use_spring and len(self.history) > self.delay_frames:
            target = self.history[-self.delay_frames - 1]
            self.spring.set_target(target.x, target.y)
    
    def update(self, dt: float):
        """Update spring physics (call each frame)"""
        if self.use_spring:
            self.current_position = self.spring.update(dt)
        elif len(self.history) > self.delay_frames:
            self.current_position = self.history[-self.delay_frames - 1]
    
    def get_position(self) -> Tuple[float, float]:
        """Get current lagged position"""
        return self.current_position.to_tuple()
    
    def get_offset(self, current_x: float, current_y: float) -> Tuple[float, float]:
        """Get offset from current position to lagged position"""
        return (
            self.current_position.x - current_x,
            self.current_position.y - current_y
        )


# =============================================================================
# Verlet Rope/Chain Physics
# =============================================================================

@dataclass 
class VerletPoint:
    """A point in a Verlet integration chain"""
    position: Vec2
    old_position: Vec2
    pinned: bool = False
    mass: float = 1.0


class VerletChain:
    """
    Verlet integration chain for realistic rope/cloth physics.
    
    Better for:
    - Ropes and chains with many segments
    - Cloth edges
    - Realistic dangling physics
    
    Example:
        chain = VerletChain(
            start=(16, 8),
            end=(16, 24),
            segments=8,
            gravity=100
        )
        
        # Pin the start point
        chain.pin_start()
        
        # Each frame
        chain.update(dt)
        positions = chain.get_positions()
    """
    
    def __init__(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        segments: int = 5,
        gravity: float = 100.0,
        damping: float = 0.98,
        iterations: int = 3
    ):
        self.gravity = gravity
        self.damping = damping
        self.iterations = iterations
        
        # Create points along the chain
        self.points: List[VerletPoint] = []
        for i in range(segments + 1):
            t = i / segments
            pos = Vec2(
                start[0] + (end[0] - start[0]) * t,
                start[1] + (end[1] - start[1]) * t
            )
            self.points.append(VerletPoint(
                position=pos,
                old_position=Vec2(pos.x, pos.y)
            ))
        
        # Calculate rest lengths between points
        self.rest_lengths: List[float] = []
        for i in range(len(self.points) - 1):
            length = (self.points[i+1].position - self.points[i].position).length
            self.rest_lengths.append(length)
    
    def pin_start(self):
        """Pin the first point in place"""
        if self.points:
            self.points[0].pinned = True
    
    def pin_end(self):
        """Pin the last point in place"""
        if self.points:
            self.points[-1].pinned = True
    
    def set_start(self, x: float, y: float):
        """Move the start point (for attached chains)"""
        if self.points:
            self.points[0].position = Vec2(x, y)
            if self.points[0].pinned:
                self.points[0].old_position = Vec2(x, y)
    
    def update(self, dt: float, wind: Vec2 = None):
        """Update chain physics"""
        wind = wind or Vec2()
        
        # Apply forces (Verlet integration)
        for point in self.points:
            if point.pinned:
                continue
            
            # Calculate velocity from position difference
            velocity = point.position - point.old_position
            
            # Store old position
            point.old_position = Vec2(point.position.x, point.position.y)
            
            # Apply gravity and wind
            acceleration = Vec2(wind.x, self.gravity + wind.y)
            
            # Update position with damping
            point.position = point.position + velocity * self.damping + acceleration * dt * dt
        
        # Constraint solving (maintain distances)
        for _ in range(self.iterations):
            for i in range(len(self.points) - 1):
                p1, p2 = self.points[i], self.points[i + 1]
                
                delta = p2.position - p1.position
                distance = delta.length
                if distance < 1e-10:
                    continue
                
                diff = (distance - self.rest_lengths[i]) / distance
                
                if not p1.pinned and not p2.pinned:
                    p1.position = p1.position + delta * (0.5 * diff)
                    p2.position = p2.position - delta * (0.5 * diff)
                elif p1.pinned:
                    p2.position = p2.position - delta * diff
                elif p2.pinned:
                    p1.position = p1.position + delta * diff
    
    def get_positions(self) -> List[Tuple[float, float]]:
        """Get all point positions"""
        return [p.position.to_tuple() for p in self.points]
    
    def get_positions_vec(self) -> List[Vec2]:
        """Get all point positions as Vec2"""
        return [p.position for p in self.points]


# =============================================================================
# Secondary Motion Manager
# =============================================================================

class SecondaryMotionManager:
    """
    Central manager for all secondary motion effects.
    
    Coordinates:
    - Skeletal chains (hair, tails)
    - Trailing particles
    - Motion lag effects
    - Verlet chains
    
    Example:
        motion = SecondaryMotionManager()
        
        # Add a hair chain
        motion.add_skeleton_chain("hair", anchor=(16, 4), bones=3, length=3)
        
        # Add ember particles
        motion.add_particle_trail("embers", colors=[(255, 200, 100)], spawn_rate=5)
        
        # Each frame
        motion.update(dt, sprite_position, sprite_velocity)
        
        # Get render data
        hair_positions = motion.get_chain_positions("hair")
        particles = motion.get_particles("embers")
    """
    
    def __init__(self):
        self.skeleton = Skeleton()
        self.particle_systems: Dict[str, ParticleFollowSystem] = {}
        self.motion_lags: Dict[str, MotionLag] = {}
        self.verlet_chains: Dict[str, VerletChain] = {}
        
        self.sprite_position = Vec2()
        self.sprite_velocity = Vec2()
    
    def add_skeleton_chain(
        self,
        name: str,
        anchor: Tuple[float, float],
        bones: int = 3,
        length: float = 4.0,
        spring: SpringConfig = None,
        gravity: float = 50.0
    ):
        """Add a skeletal chain for dangling parts"""
        self.skeleton.add_chain(
            name=name,
            anchor=anchor,
            bone_count=bones,
            bone_length=length,
            spring_config=spring or SpringConfig.wobbly(),
            gravity=gravity
        )
    
    def add_particle_trail(
        self,
        name: str,
        colors: List[Tuple[int, int, int]] = None,
        spawn_rate: float = 10.0,
        lifetime: float = 0.5,
        spring: SpringConfig = None
    ):
        """Add a trailing particle system"""
        self.particle_systems[name] = ParticleFollowSystem(
            spring_config=spring or SpringConfig.gentle(),
            spawn_rate=spawn_rate,
            lifetime=lifetime,
            colors=colors or [(255, 255, 255)]
        )
    
    def add_motion_lag(
        self,
        name: str,
        delay_frames: int = 3,
        spring: SpringConfig = None
    ):
        """Add a motion lag tracker"""
        self.motion_lags[name] = MotionLag(
            delay_frames=delay_frames,
            spring_config=spring or SpringConfig.gentle()
        )
    
    def add_verlet_chain(
        self,
        name: str,
        start: Tuple[float, float],
        end: Tuple[float, float],
        segments: int = 5,
        gravity: float = 100.0,
        pin_start: bool = True
    ):
        """Add a Verlet physics chain"""
        chain = VerletChain(start, end, segments, gravity)
        if pin_start:
            chain.pin_start()
        self.verlet_chains[name] = chain
    
    def update(
        self,
        dt: float,
        sprite_x: float,
        sprite_y: float,
        wind: Vec2 = None
    ):
        """Update all secondary motion systems"""
        # Calculate velocity
        new_pos = Vec2(sprite_x, sprite_y)
        if dt > 0:
            self.sprite_velocity = (new_pos - self.sprite_position) / dt
        self.sprite_position = new_pos
        
        wind = wind or Vec2()
        sprite_offset = Vec2()  # Could be calculated from animation
        
        # Update skeleton
        self.skeleton.update(dt, self.sprite_velocity, sprite_offset, wind)
        
        # Update particle systems
        for ps in self.particle_systems.values():
            ps.set_sprite_position(sprite_x, sprite_y)
            ps.update(dt)
        
        # Update motion lags
        for lag in self.motion_lags.values():
            lag.push(sprite_x, sprite_y)
            lag.update(dt)
        
        # Update Verlet chains
        for name, chain in self.verlet_chains.items():
            anchor = self.skeleton.anchors.get(name)
            if anchor:
                chain.set_start(anchor.x + sprite_x, anchor.y + sprite_y)
            chain.update(dt, wind)
    
    def get_chain_positions(self, name: str) -> List[Tuple[float, float]]:
        """Get positions of a skeletal chain"""
        positions = self.skeleton.get_chain_positions(name)
        return [p.to_tuple() for p in positions]
    
    def get_particles(self, name: str) -> List[TrailingParticle]:
        """Get particles from a particle system"""
        if name in self.particle_systems:
            return self.particle_systems[name].get_particles()
        return []
    
    def get_lag_position(self, name: str) -> Tuple[float, float]:
        """Get position from a motion lag tracker"""
        if name in self.motion_lags:
            return self.motion_lags[name].get_position()
        return (0, 0)
    
    def get_verlet_positions(self, name: str) -> List[Tuple[float, float]]:
        """Get positions of a Verlet chain"""
        if name in self.verlet_chains:
            return self.verlet_chains[name].get_positions()
        return []
