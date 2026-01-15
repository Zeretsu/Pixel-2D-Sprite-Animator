"""
Context-Aware Semantic Detection

Intelligently identifies sprite types based on multiple visual cues:
- Color patterns (fire colors, water blues, etc.)
- Shape characteristics (vertical, horizontal, round)
- Mass distribution (top-heavy, bottom-heavy, centered)
- Edge patterns (wavy, sharp, soft)
- Internal features (bubbles, flames, ripples)

Then applies contextual sub-effects appropriate for each type.

Examples:
- Torch: flame rises upward, flickers
- Campfire: flame spreads wide, ember particles
- Potion: bubbles inside container, liquid wobbles
- Water surface: horizontal ripples, reflections
- Crystal: inner glow pulses, outer sparkles
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum, auto


# =============================================================================
# Sprite Type Taxonomy
# =============================================================================

class SpriteCategory(Enum):
    """High-level sprite categories"""
    FIRE = auto()
    WATER = auto()
    MAGIC = auto()
    PLANT = auto()
    OBJECT = auto()
    CREATURE = auto()
    EFFECT = auto()
    UNKNOWN = auto()


class SpriteType(Enum):
    """Specific sprite types with contextual behaviors"""
    # Fire types
    TORCH = "torch"
    CAMPFIRE = "campfire"
    CANDLE = "candle"
    BRAZIER = "brazier"
    FIRE_GENERIC = "fire_generic"
    EMBER = "ember"
    EXPLOSION = "explosion"
    
    # Water types
    WATER_SURFACE = "water_surface"
    WATERFALL = "waterfall"
    PUDDLE = "puddle"
    FOUNTAIN = "fountain"
    RAIN = "rain"
    WATER_GENERIC = "water_generic"
    
    # Container/Potion types
    POTION = "potion"
    FLASK = "flask"
    CAULDRON = "cauldron"
    BARREL = "barrel"
    CHEST = "chest"
    
    # Magic types
    CRYSTAL = "crystal"
    GEM = "gem"
    ORB = "orb"
    PORTAL = "portal"
    MAGIC_CIRCLE = "magic_circle"
    RUNE = "rune"
    SPARKLE = "sparkle"
    
    # Plant types
    TREE = "tree"
    GRASS = "grass"
    FLOWER = "flower"
    VINE = "vine"
    MUSHROOM = "mushroom"
    
    # Object types
    COIN = "coin"
    KEY = "key"
    WEAPON = "weapon"
    BANNER = "banner"
    LANTERN = "lantern"
    
    # Creature types
    SLIME = "slime"
    GHOST = "ghost"
    WISP = "wisp"
    
    # Effect types
    SMOKE = "smoke"
    CLOUD = "cloud"
    DUST = "dust"
    
    # Fallback
    UNKNOWN = "unknown"


# =============================================================================
# Effect Configurations for Each Type
# =============================================================================

@dataclass
class EffectConfig:
    """Configuration for effects to apply to a sprite type"""
    primary_effect: str
    effect_params: Dict[str, Any] = field(default_factory=dict)
    secondary_effects: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)
    particle_config: Optional[Dict[str, Any]] = None
    palette_config: Optional[Dict[str, Any]] = None
    description: str = ""


# Effect configurations for each sprite type
EFFECT_CONFIGS: Dict[SpriteType, EffectConfig] = {
    # Fire types - flames rise and flicker
    SpriteType.TORCH: EffectConfig(
        primary_effect="flame",
        effect_params={
            "intensity": 0.7,
            "speed": 1.2,
            "direction": "up",
            "spread": 0.3,
            "height_bias": 0.8,  # Flames concentrated at top
        },
        secondary_effects=[
            ("flicker", {"intensity": 0.2, "speed": 8}),
            ("glow", {"radius": 2, "intensity": 0.4}),
        ],
        particle_config={
            "type": "spark",
            "rate": 3,
            "direction": "up",
            "speed": 40,
            "lifetime": 0.6,
        },
        palette_config={"ramp": "fire", "cycle_speed": 2.0},
        description="Upward flame with flickering and sparks"
    ),
    
    SpriteType.CAMPFIRE: EffectConfig(
        primary_effect="flame",
        effect_params={
            "intensity": 0.8,
            "speed": 0.9,
            "direction": "up",
            "spread": 0.6,  # Wider spread
            "height_bias": 0.5,  # More spread out
        },
        secondary_effects=[
            ("flicker", {"intensity": 0.3, "speed": 6}),
            ("glow", {"radius": 4, "intensity": 0.5}),
        ],
        particle_config={
            "type": "ember",
            "rate": 8,
            "direction": "up",
            "spread": 0.8,
            "speed": 25,
            "lifetime": 1.2,
        },
        palette_config={"ramp": "ember", "cycle_speed": 1.5},
        description="Wide spreading flame with embers"
    ),
    
    SpriteType.CANDLE: EffectConfig(
        primary_effect="flame",
        effect_params={
            "intensity": 0.4,
            "speed": 1.5,
            "direction": "up",
            "spread": 0.15,  # Narrow
            "height_bias": 0.9,
        },
        secondary_effects=[
            ("flicker", {"intensity": 0.15, "speed": 10}),
            ("glow", {"radius": 1, "intensity": 0.3}),
        ],
        description="Small, delicate flame"
    ),
    
    SpriteType.BRAZIER: EffectConfig(
        primary_effect="flame",
        effect_params={
            "intensity": 0.9,
            "speed": 1.0,
            "direction": "up",
            "spread": 0.5,
            "height_bias": 0.6,
        },
        secondary_effects=[
            ("flicker", {"intensity": 0.25, "speed": 7}),
            ("glow", {"radius": 3, "intensity": 0.6}),
        ],
        particle_config={
            "type": "spark",
            "rate": 5,
            "direction": "up",
            "speed": 50,
        },
        description="Contained fire with strong glow"
    ),
    
    # Water types
    SpriteType.WATER_SURFACE: EffectConfig(
        primary_effect="water",
        effect_params={
            "wave_direction": "horizontal",
            "wave_speed": 1.0,
            "wave_height": 1.5,
            "ripple_density": 0.5,
        },
        secondary_effects=[
            ("shimmer", {"intensity": 0.3, "speed": 2}),
        ],
        palette_config={"ramp": "water", "cycle_speed": 0.8},
        description="Horizontal ripples with shimmer"
    ),
    
    SpriteType.WATERFALL: EffectConfig(
        primary_effect="water",
        effect_params={
            "wave_direction": "vertical",
            "wave_speed": 2.0,
            "flow_direction": "down",
        },
        secondary_effects=[
            ("mist", {"intensity": 0.4, "spread": 0.3}),
        ],
        particle_config={
            "type": "splash",
            "rate": 10,
            "direction": "down",
            "speed": 80,
        },
        description="Downward flowing water with mist"
    ),
    
    SpriteType.PUDDLE: EffectConfig(
        primary_effect="water",
        effect_params={
            "wave_direction": "radial",
            "wave_speed": 0.5,
            "wave_height": 0.5,
        },
        secondary_effects=[
            ("reflect", {"intensity": 0.5}),
        ],
        description="Gentle radial ripples"
    ),
    
    SpriteType.FOUNTAIN: EffectConfig(
        primary_effect="water",
        effect_params={
            "wave_direction": "up",
            "wave_speed": 1.5,
        },
        particle_config={
            "type": "water_drop",
            "rate": 15,
            "direction": "up",
            "gravity": 150,
            "spread": 0.4,
        },
        description="Upward water jets with droplets"
    ),
    
    # Potion/Container types
    SpriteType.POTION: EffectConfig(
        primary_effect="liquid",
        effect_params={
            "bubble_rate": 0.3,
            "wobble_amount": 0.15,
            "fill_level": 0.7,
        },
        secondary_effects=[
            ("bubble", {"size": 1, "rate": 2, "speed": 15}),
            ("glow", {"radius": 1, "intensity": 0.2}),
        ],
        palette_config={"ramp": "magic", "cycle_speed": 1.0},
        description="Bubbling liquid in container"
    ),
    
    SpriteType.CAULDRON: EffectConfig(
        primary_effect="liquid",
        effect_params={
            "bubble_rate": 0.6,
            "wobble_amount": 0.1,
            "steam": True,
        },
        secondary_effects=[
            ("bubble", {"size": 2, "rate": 4, "speed": 20}),
            ("steam", {"intensity": 0.4, "rise_speed": 30}),
        ],
        particle_config={
            "type": "bubble",
            "rate": 3,
            "direction": "up",
            "speed": 20,
        },
        description="Bubbling cauldron with steam"
    ),
    
    # Magic types
    SpriteType.CRYSTAL: EffectConfig(
        primary_effect="pulse",
        effect_params={
            "pulse_speed": 0.8,
            "pulse_intensity": 0.4,
            "inner_glow": True,
        },
        secondary_effects=[
            ("sparkle", {"rate": 0.3, "intensity": 0.6}),
            ("glow", {"radius": 2, "intensity": 0.3}),
        ],
        palette_config={"ramp": "magic", "intensity_pulse": True},
        description="Pulsing inner glow with sparkles"
    ),
    
    SpriteType.ORB: EffectConfig(
        primary_effect="pulse",
        effect_params={
            "pulse_speed": 1.2,
            "pulse_intensity": 0.5,
            "radial": True,
        },
        secondary_effects=[
            ("float", {"height": 2, "speed": 0.5}),
            ("glow", {"radius": 3, "intensity": 0.5}),
        ],
        description="Floating orb with radial pulse"
    ),
    
    SpriteType.PORTAL: EffectConfig(
        primary_effect="void",
        effect_params={
            "swirl_speed": 1.5,
            "pull_strength": 0.3,
        },
        secondary_effects=[
            ("distort", {"intensity": 0.2}),
            ("glow", {"radius": 4, "intensity": 0.6}),
        ],
        particle_config={
            "type": "magic",
            "rate": 8,
            "direction": "inward",
            "speed": 40,
        },
        description="Swirling void with particle pull"
    ),
    
    # Plant types
    SpriteType.TREE: EffectConfig(
        primary_effect="sway",
        effect_params={
            "sway_amount": 0.1,
            "sway_speed": 0.3,
            "top_weight": 0.8,  # More sway at top
        },
        secondary_effects=[
            ("rustle", {"intensity": 0.2}),
        ],
        description="Gentle swaying, more at top"
    ),
    
    SpriteType.GRASS: EffectConfig(
        primary_effect="sway",
        effect_params={
            "sway_amount": 0.2,
            "sway_speed": 0.8,
            "wave_effect": True,
        },
        description="Wave-like swaying motion"
    ),
    
    SpriteType.FLOWER: EffectConfig(
        primary_effect="sway",
        effect_params={
            "sway_amount": 0.15,
            "sway_speed": 0.5,
        },
        secondary_effects=[
            ("bob", {"height": 1, "speed": 0.3}),
        ],
        description="Gentle sway with subtle bob"
    ),
    
    # Object types
    SpriteType.COIN: EffectConfig(
        primary_effect="float",
        effect_params={
            "bob_height": 2,
            "bob_speed": 1.0,
        },
        secondary_effects=[
            ("spin", {"speed": 2.0, "axis": "vertical"}),
            ("sparkle", {"rate": 0.5, "intensity": 0.4}),
        ],
        description="Floating, spinning with sparkle"
    ),
    
    SpriteType.BANNER: EffectConfig(
        primary_effect="sway",
        effect_params={
            "sway_amount": 0.25,
            "sway_speed": 0.6,
            "wave_propagation": True,
        },
        description="Flag-like waving motion"
    ),
    
    SpriteType.LANTERN: EffectConfig(
        primary_effect="sway",
        effect_params={
            "sway_amount": 0.08,
            "sway_speed": 0.4,
        },
        secondary_effects=[
            ("flicker", {"intensity": 0.15, "speed": 8}),
            ("glow", {"radius": 3, "intensity": 0.4}),
        ],
        description="Gentle swing with inner flicker"
    ),
    
    # Creature types
    SpriteType.SLIME: EffectConfig(
        primary_effect="wobble",
        effect_params={
            "wobble_speed": 1.5,
            "wobble_amount": 0.2,
            "squash_stretch": True,
        },
        secondary_effects=[
            ("bounce", {"compression": 0.15}),
        ],
        description="Gelatinous wobbling"
    ),
    
    SpriteType.GHOST: EffectConfig(
        primary_effect="float",
        effect_params={
            "bob_height": 3,
            "bob_speed": 0.5,
        },
        secondary_effects=[
            ("fade", {"min_alpha": 0.5, "speed": 1.0}),
            ("distort", {"intensity": 0.1}),
        ],
        description="Floating with transparency pulse"
    ),
    
    SpriteType.WISP: EffectConfig(
        primary_effect="float",
        effect_params={
            "bob_height": 4,
            "bob_speed": 0.8,
            "drift": True,
        },
        secondary_effects=[
            ("glow", {"radius": 2, "intensity": 0.6}),
            ("pulse", {"speed": 1.5}),
        ],
        particle_config={
            "type": "sparkle",
            "rate": 2,
            "spread": 0.5,
        },
        description="Drifting with glow trail"
    ),
    
    # Effect types
    SpriteType.SMOKE: EffectConfig(
        primary_effect="smoke",
        effect_params={
            "rise_speed": 20,
            "spread": 0.4,
            "fade_speed": 0.8,
        },
        description="Rising, spreading smoke"
    ),
    
    SpriteType.CLOUD: EffectConfig(
        primary_effect="float",
        effect_params={
            "drift_speed": 5,
            "drift_direction": "horizontal",
        },
        secondary_effects=[
            ("morph", {"intensity": 0.1, "speed": 0.3}),
        ],
        description="Slow horizontal drift with morphing"
    ),
    
    # Fallback
    SpriteType.UNKNOWN: EffectConfig(
        primary_effect="pulse",
        effect_params={"intensity": 0.2},
        description="Generic subtle animation"
    ),
}


# =============================================================================
# Feature Detection
# =============================================================================

@dataclass
class SpriteFeatures:
    """Detected features of a sprite"""
    # Color features
    has_fire_colors: bool = False
    has_water_colors: bool = False
    has_magic_colors: bool = False
    has_nature_colors: bool = False
    has_metallic_colors: bool = False
    dominant_hue: str = "neutral"
    color_variance: float = 0.0
    
    # Shape features
    is_vertical: bool = False
    is_horizontal: bool = False
    is_round: bool = False
    is_symmetric: bool = False
    aspect_ratio: float = 1.0
    
    # Mass distribution
    is_top_heavy: bool = False
    is_bottom_heavy: bool = False
    center_of_mass_y: float = 0.5  # 0=top, 1=bottom
    
    # Edge features
    has_wavy_edges: bool = False
    has_sharp_edges: bool = False
    has_soft_edges: bool = False
    edge_complexity: float = 0.0
    
    # Internal features
    has_bubbles: bool = False
    has_container_shape: bool = False
    has_flame_shape: bool = False
    has_liquid_appearance: bool = False
    
    # Detected patterns
    detected_patterns: List[str] = field(default_factory=list)


class SemanticDetector:
    """
    Analyzes sprites to determine semantic type and appropriate effects.
    """
    
    # Color ranges for detection (HSV)
    FIRE_HUE_RANGE = (0, 40)      # Red to orange-yellow
    WATER_HUE_RANGE = (180, 220)  # Cyan to blue
    MAGIC_HUE_RANGE = (260, 320)  # Purple to magenta
    NATURE_HUE_RANGE = (80, 150)  # Yellow-green to green
    
    def __init__(self, pixels: np.ndarray):
        """
        Initialize detector with sprite pixels.
        
        Args:
            pixels: RGBA image array
        """
        self.pixels = pixels
        self.height, self.width = pixels.shape[:2]
        self._features: Optional[SpriteFeatures] = None
        self._mask: Optional[np.ndarray] = None
    
    @property
    def mask(self) -> np.ndarray:
        """Binary mask of visible pixels"""
        if self._mask is None:
            if self.pixels.shape[2] == 4:
                self._mask = self.pixels[:, :, 3] > 10
            else:
                self._mask = np.ones((self.height, self.width), dtype=bool)
        return self._mask
    
    @property
    def features(self) -> SpriteFeatures:
        """Get detected features (cached)"""
        if self._features is None:
            self._features = self._detect_features()
        return self._features
    
    def _detect_features(self) -> SpriteFeatures:
        """Analyze sprite and detect all features"""
        features = SpriteFeatures()
        
        # Color analysis
        self._analyze_colors(features)
        
        # Shape analysis
        self._analyze_shape(features)
        
        # Mass distribution
        self._analyze_mass_distribution(features)
        
        # Edge analysis
        self._analyze_edges(features)
        
        # Internal patterns
        self._analyze_internal_patterns(features)
        
        return features
    
    def _analyze_colors(self, features: SpriteFeatures):
        """Analyze color distribution"""
        if not np.any(self.mask):
            return
        
        # Get visible pixels
        visible = self.pixels[self.mask]
        
        # Convert to HSV for analysis
        r, g, b = visible[:, 0], visible[:, 1], visible[:, 2]
        
        # Normalize to 0-1
        r_f = r.astype(float) / 255
        g_f = g.astype(float) / 255
        b_f = b.astype(float) / 255
        
        # Calculate HSV
        max_c = np.maximum(np.maximum(r_f, g_f), b_f)
        min_c = np.minimum(np.minimum(r_f, g_f), b_f)
        diff = max_c - min_c
        
        # Hue calculation
        hue = np.zeros_like(r_f)
        mask_r = (max_c == r_f) & (diff > 0)
        mask_g = (max_c == g_f) & (diff > 0)
        mask_b = (max_c == b_f) & (diff > 0)
        
        hue[mask_r] = (60 * ((g_f[mask_r] - b_f[mask_r]) / diff[mask_r]) + 360) % 360
        hue[mask_g] = (60 * ((b_f[mask_g] - r_f[mask_g]) / diff[mask_g]) + 120) % 360
        hue[mask_b] = (60 * ((r_f[mask_b] - g_f[mask_b]) / diff[mask_b]) + 240) % 360
        
        saturation = np.where(max_c > 0, diff / max_c, 0)
        value = max_c
        
        # Filter to saturated colors only
        saturated_mask = saturation > 0.2
        if np.sum(saturated_mask) > 0:
            saturated_hues = hue[saturated_mask]
            
            # Check color ranges
            fire_count = np.sum((saturated_hues >= 0) & (saturated_hues <= 40) |
                              (saturated_hues >= 340))
            water_count = np.sum((saturated_hues >= 180) & (saturated_hues <= 220))
            magic_count = np.sum((saturated_hues >= 260) & (saturated_hues <= 320))
            nature_count = np.sum((saturated_hues >= 80) & (saturated_hues <= 150))
            
            total_saturated = len(saturated_hues)
            
            features.has_fire_colors = fire_count / total_saturated > 0.3
            features.has_water_colors = water_count / total_saturated > 0.3
            features.has_magic_colors = magic_count / total_saturated > 0.3
            features.has_nature_colors = nature_count / total_saturated > 0.3
            
            # Dominant hue
            avg_hue = np.mean(saturated_hues)
            if 0 <= avg_hue <= 40 or avg_hue >= 340:
                features.dominant_hue = "warm"
            elif 180 <= avg_hue <= 220:
                features.dominant_hue = "cool"
            elif 260 <= avg_hue <= 320:
                features.dominant_hue = "magic"
            elif 80 <= avg_hue <= 150:
                features.dominant_hue = "nature"
            
            # Color variance
            features.color_variance = np.std(saturated_hues) / 180
        
        # Check for metallic (high value, low saturation, gray tones)
        gray_mask = saturation < 0.15
        bright_gray = gray_mask & (value > 0.5)
        features.has_metallic_colors = np.sum(bright_gray) / len(visible) > 0.3
    
    def _analyze_shape(self, features: SpriteFeatures):
        """Analyze overall shape"""
        if not np.any(self.mask):
            return
        
        # Bounding box
        rows = np.any(self.mask, axis=1)
        cols = np.any(self.mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        
        features.aspect_ratio = width / height if height > 0 else 1
        features.is_vertical = features.aspect_ratio < 0.6
        features.is_horizontal = features.aspect_ratio > 1.6
        
        # Roundness (fill ratio in bounding box)
        bbox_area = width * height
        fill_area = np.sum(self.mask[y_min:y_max+1, x_min:x_max+1])
        fill_ratio = fill_area / bbox_area if bbox_area > 0 else 0
        
        features.is_round = (0.7 < features.aspect_ratio < 1.4) and fill_ratio > 0.6
        
        # Symmetry
        cropped = self.mask[y_min:y_max+1, x_min:x_max+1]
        h_flip = np.flip(cropped, axis=1)
        v_flip = np.flip(cropped, axis=0)
        
        h_sym = np.sum(cropped & h_flip) / np.sum(cropped | h_flip) if np.any(cropped) else 0
        v_sym = np.sum(cropped & v_flip) / np.sum(cropped | v_flip) if np.any(cropped) else 0
        
        features.is_symmetric = h_sym > 0.7 or v_sym > 0.7
    
    def _analyze_mass_distribution(self, features: SpriteFeatures):
        """Analyze where the mass is concentrated"""
        if not np.any(self.mask):
            return
        
        y_coords = np.where(self.mask)[0]
        
        if len(y_coords) == 0:
            return
        
        # Center of mass (normalized 0-1, 0=top, 1=bottom)
        com_y = np.mean(y_coords) / self.height
        features.center_of_mass_y = com_y
        
        # Top vs bottom half mass
        mid_y = self.height // 2
        top_mass = np.sum(self.mask[:mid_y, :])
        bottom_mass = np.sum(self.mask[mid_y:, :])
        total_mass = top_mass + bottom_mass
        
        if total_mass > 0:
            top_ratio = top_mass / total_mass
            features.is_top_heavy = top_ratio > 0.6
            features.is_bottom_heavy = top_ratio < 0.4
    
    def _analyze_edges(self, features: SpriteFeatures):
        """Analyze edge characteristics"""
        if not np.any(self.mask):
            return
        
        # Get edge pixels using simple gradient
        # Shift mask in all directions and compare
        padded = np.pad(self.mask, 1, mode='constant', constant_values=False)
        
        edges = (
            (padded[1:-1, 1:-1] != padded[:-2, 1:-1]) |  # Top
            (padded[1:-1, 1:-1] != padded[2:, 1:-1]) |   # Bottom
            (padded[1:-1, 1:-1] != padded[1:-1, :-2]) |  # Left
            (padded[1:-1, 1:-1] != padded[1:-1, 2:])     # Right
        ) & self.mask
        
        edge_coords = np.where(edges)
        
        if len(edge_coords[0]) < 4:
            return
        
        # Edge complexity (perimeter / sqrt(area))
        perimeter = len(edge_coords[0])
        area = np.sum(self.mask)
        features.edge_complexity = perimeter / np.sqrt(area) if area > 0 else 0
        
        # Analyze edge waviness by looking at x-coordinate variance along edge
        edge_y = edge_coords[0]
        edge_x = edge_coords[1]
        
        # Group by y and check x variance
        unique_y = np.unique(edge_y)
        x_variances = []
        for y in unique_y:
            x_at_y = edge_x[edge_y == y]
            if len(x_at_y) > 1:
                x_variances.append(np.std(x_at_y))
        
        if x_variances:
            avg_variance = np.mean(x_variances)
            features.has_wavy_edges = avg_variance > 2.0
        
        # Sharp vs soft edges (check alpha gradient)
        if self.pixels.shape[2] == 4:
            alpha = self.pixels[:, :, 3]
            partial_alpha = (alpha > 10) & (alpha < 245)
            partial_ratio = np.sum(partial_alpha) / np.sum(alpha > 10) if np.sum(alpha > 10) > 0 else 0
            features.has_soft_edges = partial_ratio > 0.15
            features.has_sharp_edges = partial_ratio < 0.05
    
    def _analyze_internal_patterns(self, features: SpriteFeatures):
        """Analyze internal patterns like bubbles, flames, etc."""
        if not np.any(self.mask):
            return
        
        # Container detection: look for U-shape or enclosed area
        # Check if bottom is more filled than middle
        h = self.height
        top_third = self.mask[:h//3, :]
        mid_third = self.mask[h//3:2*h//3, :]
        bot_third = self.mask[2*h//3:, :]
        
        top_fill = np.sum(top_third) / (top_third.size + 1)
        mid_fill = np.sum(mid_third) / (mid_third.size + 1)
        bot_fill = np.sum(bot_third) / (bot_third.size + 1)
        
        # Container: narrower at top, wider at bottom, or U-shaped
        features.has_container_shape = (
            bot_fill > mid_fill * 1.2 and
            mid_fill > 0.3 and
            features.is_vertical
        )
        
        # Flame shape: wider at bottom, narrow and irregular at top
        features.has_flame_shape = (
            features.is_vertical and
            features.is_top_heavy and
            features.has_fire_colors and
            not features.is_symmetric
        )
        
        # Liquid appearance: check for horizontal bands of similar color
        if features.has_water_colors or features.has_magic_colors:
            features.has_liquid_appearance = True
        
        # Bubble detection: look for small round holes in the interior
        # Simplified: check for internal alpha variations in containers
        if features.has_container_shape or features.has_liquid_appearance:
            interior = self.mask.copy()
            # Erode to get interior
            for _ in range(2):
                padded = np.pad(interior, 1, mode='constant', constant_values=False)
                interior = (
                    padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1] &
                    padded[1:-1, :-2] & padded[1:-1, 2:]
                )
            
            if np.any(interior):
                # Check for brightness variations (bubbles are lighter)
                if self.pixels.shape[2] >= 3:
                    brightness = (
                        self.pixels[:, :, 0].astype(float) * 0.299 +
                        self.pixels[:, :, 1].astype(float) * 0.587 +
                        self.pixels[:, :, 2].astype(float) * 0.114
                    )
                    interior_brightness = brightness[interior]
                    if len(interior_brightness) > 10:
                        brightness_var = np.std(interior_brightness)
                        features.has_bubbles = brightness_var > 30
    
    def detect_type(self) -> Tuple[SpriteType, float, List[str]]:
        """
        Detect the semantic type of the sprite.
        
        Returns:
            (SpriteType, confidence 0-1, list of reasons)
        """
        f = self.features
        
        candidates: List[Tuple[SpriteType, float, List[str]]] = []
        
        # Fire types
        if f.has_fire_colors:
            if f.is_vertical and f.has_flame_shape:
                if f.is_top_heavy:
                    confidence = 0.9 if f.center_of_mass_y < 0.4 else 0.7
                    candidates.append((SpriteType.TORCH, confidence, [
                        "Fire colors detected",
                        "Vertical shape",
                        "Top-heavy mass distribution",
                        "Flame-like irregular edges"
                    ]))
                else:
                    candidates.append((SpriteType.CAMPFIRE, 0.75, [
                        "Fire colors detected",
                        "Flame shape",
                        "Bottom-heavy mass distribution"
                    ]))
            elif f.is_vertical and not f.is_top_heavy:
                candidates.append((SpriteType.CAMPFIRE, 0.7, [
                    "Fire colors detected",
                    "Vertical but not top-heavy"
                ]))
            elif f.is_round:
                candidates.append((SpriteType.BRAZIER, 0.6, [
                    "Fire colors detected",
                    "Round/contained shape"
                ]))
            else:
                candidates.append((SpriteType.FIRE_GENERIC, 0.5, [
                    "Fire colors detected"
                ]))
        
        # Water types
        if f.has_water_colors:
            if f.is_horizontal:
                if f.has_wavy_edges:
                    candidates.append((SpriteType.WATER_SURFACE, 0.9, [
                        "Water colors detected",
                        "Horizontal shape",
                        "Wavy edges"
                    ]))
                else:
                    candidates.append((SpriteType.WATER_SURFACE, 0.7, [
                        "Water colors detected",
                        "Horizontal shape"
                    ]))
            elif f.is_vertical:
                candidates.append((SpriteType.WATERFALL, 0.7, [
                    "Water colors detected",
                    "Vertical shape (falling water)"
                ]))
            elif f.is_round:
                candidates.append((SpriteType.PUDDLE, 0.6, [
                    "Water colors detected",
                    "Round shape"
                ]))
            else:
                candidates.append((SpriteType.WATER_GENERIC, 0.5, [
                    "Water colors detected"
                ]))
        
        # Container/Potion types
        if f.has_container_shape:
            if f.has_magic_colors or f.has_bubbles:
                candidates.append((SpriteType.POTION, 0.85, [
                    "Container shape detected",
                    "Magic colors or bubbles inside"
                ]))
            elif f.has_liquid_appearance:
                candidates.append((SpriteType.FLASK, 0.7, [
                    "Container shape detected",
                    "Liquid appearance"
                ]))
            else:
                candidates.append((SpriteType.BARREL, 0.5, [
                    "Container shape detected"
                ]))
        
        # Magic types
        if f.has_magic_colors:
            if f.is_round and f.is_symmetric:
                candidates.append((SpriteType.ORB, 0.8, [
                    "Magic colors detected",
                    "Round and symmetric"
                ]))
            elif f.is_round:
                candidates.append((SpriteType.CRYSTAL, 0.7, [
                    "Magic colors detected",
                    "Round shape"
                ]))
            elif f.is_vertical:
                candidates.append((SpriteType.CRYSTAL, 0.65, [
                    "Magic colors detected",
                    "Vertical shape (crystal formation)"
                ]))
            else:
                candidates.append((SpriteType.SPARKLE, 0.5, [
                    "Magic colors detected"
                ]))
        
        # Nature types
        if f.has_nature_colors:
            if f.is_vertical:
                if f.aspect_ratio < 0.3:
                    candidates.append((SpriteType.TREE, 0.7, [
                        "Nature colors detected",
                        "Tall vertical shape"
                    ]))
                elif f.aspect_ratio < 0.5:
                    candidates.append((SpriteType.GRASS, 0.65, [
                        "Nature colors detected",
                        "Tall thin shape"
                    ]))
                else:
                    candidates.append((SpriteType.FLOWER, 0.6, [
                        "Nature colors detected",
                        "Vertical shape"
                    ]))
            else:
                candidates.append((SpriteType.GRASS, 0.5, [
                    "Nature colors detected"
                ]))
        
        # Object types based on shape
        if f.has_metallic_colors:
            if f.is_round and f.is_symmetric:
                candidates.append((SpriteType.COIN, 0.75, [
                    "Metallic colors detected",
                    "Round and symmetric"
                ]))
            else:
                candidates.append((SpriteType.KEY, 0.5, [
                    "Metallic colors detected"
                ]))
        
        # Creature types based on shape
        if f.is_round and f.has_soft_edges and not f.is_symmetric:
            candidates.append((SpriteType.SLIME, 0.6, [
                "Round shape",
                "Soft edges",
                "Asymmetric (organic)"
            ]))
        
        # Effect types
        if f.has_soft_edges and not f.has_fire_colors and not f.has_water_colors:
            if f.dominant_hue == "neutral":
                candidates.append((SpriteType.SMOKE, 0.6, [
                    "Soft edges",
                    "Neutral colors"
                ]))
        
        # Sort by confidence
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if candidates:
            return candidates[0]
        
        return (SpriteType.UNKNOWN, 0.3, ["No specific features detected"])
    
    def get_effect_config(self) -> EffectConfig:
        """Get the recommended effect configuration"""
        sprite_type, confidence, reasons = self.detect_type()
        return EFFECT_CONFIGS.get(sprite_type, EFFECT_CONFIGS[SpriteType.UNKNOWN])
    
    def get_full_analysis(self) -> Dict[str, Any]:
        """Get complete analysis results"""
        sprite_type, confidence, reasons = self.detect_type()
        
        return {
            "sprite_type": sprite_type.value,
            "category": self._get_category(sprite_type).name,
            "confidence": confidence,
            "reasons": reasons,
            "features": {
                "colors": {
                    "fire": self.features.has_fire_colors,
                    "water": self.features.has_water_colors,
                    "magic": self.features.has_magic_colors,
                    "nature": self.features.has_nature_colors,
                    "metallic": self.features.has_metallic_colors,
                    "dominant_hue": self.features.dominant_hue,
                },
                "shape": {
                    "vertical": self.features.is_vertical,
                    "horizontal": self.features.is_horizontal,
                    "round": self.features.is_round,
                    "symmetric": self.features.is_symmetric,
                    "aspect_ratio": round(self.features.aspect_ratio, 2),
                },
                "mass": {
                    "top_heavy": self.features.is_top_heavy,
                    "bottom_heavy": self.features.is_bottom_heavy,
                    "center_y": round(self.features.center_of_mass_y, 2),
                },
                "edges": {
                    "wavy": self.features.has_wavy_edges,
                    "sharp": self.features.has_sharp_edges,
                    "soft": self.features.has_soft_edges,
                },
                "internal": {
                    "bubbles": self.features.has_bubbles,
                    "container": self.features.has_container_shape,
                    "flame": self.features.has_flame_shape,
                    "liquid": self.features.has_liquid_appearance,
                },
            },
            "effect_config": EFFECT_CONFIGS.get(sprite_type, EFFECT_CONFIGS[SpriteType.UNKNOWN]),
        }
    
    def _get_category(self, sprite_type: SpriteType) -> SpriteCategory:
        """Get high-level category for a sprite type"""
        fire_types = {SpriteType.TORCH, SpriteType.CAMPFIRE, SpriteType.CANDLE,
                     SpriteType.BRAZIER, SpriteType.FIRE_GENERIC, SpriteType.EMBER}
        water_types = {SpriteType.WATER_SURFACE, SpriteType.WATERFALL, SpriteType.PUDDLE,
                      SpriteType.FOUNTAIN, SpriteType.RAIN, SpriteType.WATER_GENERIC}
        magic_types = {SpriteType.CRYSTAL, SpriteType.GEM, SpriteType.ORB, SpriteType.PORTAL,
                      SpriteType.MAGIC_CIRCLE, SpriteType.RUNE, SpriteType.SPARKLE,
                      SpriteType.POTION, SpriteType.FLASK, SpriteType.CAULDRON}
        plant_types = {SpriteType.TREE, SpriteType.GRASS, SpriteType.FLOWER,
                      SpriteType.VINE, SpriteType.MUSHROOM}
        creature_types = {SpriteType.SLIME, SpriteType.GHOST, SpriteType.WISP}
        effect_types = {SpriteType.SMOKE, SpriteType.CLOUD, SpriteType.DUST,
                       SpriteType.EXPLOSION}
        
        if sprite_type in fire_types:
            return SpriteCategory.FIRE
        elif sprite_type in water_types:
            return SpriteCategory.WATER
        elif sprite_type in magic_types:
            return SpriteCategory.MAGIC
        elif sprite_type in plant_types:
            return SpriteCategory.PLANT
        elif sprite_type in creature_types:
            return SpriteCategory.CREATURE
        elif sprite_type in effect_types:
            return SpriteCategory.EFFECT
        else:
            return SpriteCategory.OBJECT


# =============================================================================
# Convenience Functions
# =============================================================================

def detect_sprite_type(pixels: np.ndarray) -> Tuple[str, float, List[str]]:
    """
    Quick detection of sprite type.
    
    Returns:
        (type_name, confidence, reasons)
    """
    detector = SemanticDetector(pixels)
    sprite_type, confidence, reasons = detector.detect_type()
    return (sprite_type.value, confidence, reasons)


def get_recommended_effects(pixels: np.ndarray) -> Dict[str, Any]:
    """
    Get recommended effects for a sprite.
    
    Returns:
        Dictionary with primary_effect, effect_params, secondary_effects, etc.
    """
    detector = SemanticDetector(pixels)
    config = detector.get_effect_config()
    
    return {
        "primary_effect": config.primary_effect,
        "effect_params": config.effect_params,
        "secondary_effects": config.secondary_effects,
        "particle_config": config.particle_config,
        "palette_config": config.palette_config,
        "description": config.description,
    }


def analyze_sprite(pixels: np.ndarray) -> Dict[str, Any]:
    """
    Full semantic analysis of a sprite.
    
    Returns:
        Complete analysis with type, features, and effect recommendations.
    """
    detector = SemanticDetector(pixels)
    return detector.get_full_analysis()
