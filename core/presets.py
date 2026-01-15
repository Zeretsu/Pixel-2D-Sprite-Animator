"""
Effect Presets Library - Pre-configured animation settings
Allows users to apply complex effect combinations with a single flag
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import copy


# ============================================================================
# Preset Data Structures
# ============================================================================

@dataclass
class EffectPreset:
    """A single effect preset configuration"""
    
    name: str
    description: str = ""
    
    # Primary effect
    effect: str = "float"
    frames: int = 8
    intensity: float = 1.0
    speed: float = 1.0
    
    # Easing
    easing: str = "ease_in_out_quad"
    
    # Secondary effects (applied in order)
    secondary_effects: List[Dict[str, Any]] = field(default_factory=list)
    
    # Color settings
    color_ramp: Optional[List[str]] = None
    palette_cycle: bool = False
    hue_shift: float = 0.0
    
    # Animation principles
    anticipation: float = 0.0
    overshoot: float = 0.0
    squash_stretch: float = 0.0
    
    # Motion blur
    motion_blur: bool = False
    blur_samples: int = 4
    
    # Layer-specific (for decomposition)
    layer_effects: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Output settings
    format: str = "gif"
    quality: str = "high"
    loop: bool = True
    
    # Tags for organization
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization"""
        return {k: v for k, v in asdict(self).items() if v}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EffectPreset':
        """Create from dictionary"""
        # Handle backwards compatibility
        if 'secondary' in data and 'secondary_effects' not in data:
            data['secondary_effects'] = [{'effect': data.pop('secondary')}]
        
        # Filter to valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**filtered)


class PresetCategory(Enum):
    """Categories for organizing presets"""
    FIRE = "fire"
    WATER = "water"
    MAGIC = "magic"
    NATURE = "nature"
    TECH = "tech"
    UI = "ui"
    CHARACTER = "character"
    ENVIRONMENT = "environment"
    CUSTOM = "custom"


# ============================================================================
# Built-in Presets
# ============================================================================

BUILTIN_PRESETS: Dict[str, Dict[str, Any]] = {
    # ==================== FIRE ====================
    "torch_realistic": {
        "name": "torch_realistic",
        "description": "Realistic flickering torch flame",
        "effect": "flame",
        "frames": 16,
        "intensity": 1.2,
        "speed": 1.0,
        "easing": "ease_out_quad",
        "color_ramp": ["#FF4500", "#FF8C00", "#FFD700", "#FFFACD"],
        "motion_blur": True,
        "blur_samples": 3,
        "tags": ["fire", "light", "dungeon"],
    },
    
    "torch_subtle": {
        "name": "torch_subtle",
        "description": "Gentle torch flicker for ambient lighting",
        "effect": "flame",
        "frames": 12,
        "intensity": 0.6,
        "speed": 0.8,
        "easing": "ease_in_out_sine",
        "color_ramp": ["#FF6347", "#FFA07A", "#FFDAB9"],
        "tags": ["fire", "light", "subtle"],
    },
    
    "campfire": {
        "name": "campfire",
        "description": "Cozy campfire with smoke wisps",
        "effect": "flame",
        "frames": 20,
        "intensity": 1.0,
        "speed": 0.7,
        "easing": "ease_in_out_quad",
        "secondary_effects": [
            {"effect": "smoke", "intensity": 0.4, "speed": 0.3}
        ],
        "color_ramp": ["#8B0000", "#FF4500", "#FF8C00", "#FFD700"],
        "tags": ["fire", "outdoor", "cozy"],
    },
    
    "inferno": {
        "name": "inferno",
        "description": "Intense blazing fire",
        "effect": "flame",
        "frames": 12,
        "intensity": 2.0,
        "speed": 1.5,
        "easing": "ease_out_expo",
        "motion_blur": True,
        "blur_samples": 5,
        "color_ramp": ["#8B0000", "#FF0000", "#FF4500", "#FFFF00", "#FFFFFF"],
        "tags": ["fire", "intense", "combat"],
    },
    
    "candle": {
        "name": "candle",
        "description": "Gentle candle flame",
        "effect": "flame",
        "frames": 16,
        "intensity": 0.4,
        "speed": 0.6,
        "easing": "ease_in_out_sine",
        "secondary_effects": [
            {"effect": "glow", "intensity": 0.3}
        ],
        "color_ramp": ["#FF8C00", "#FFD700", "#FFFACD"],
        "tags": ["fire", "light", "indoor", "subtle"],
    },
    
    # ==================== WATER ====================
    "water_calm": {
        "name": "water_calm",
        "description": "Gentle water surface ripples",
        "effect": "water",
        "frames": 24,
        "intensity": 0.5,
        "speed": 0.4,
        "easing": "ease_in_out_sine",
        "color_ramp": ["#1E90FF", "#00BFFF", "#87CEEB"],
        "tags": ["water", "calm", "ambient"],
    },
    
    "water_flowing": {
        "name": "water_flowing",
        "description": "Flowing stream or river",
        "effect": "water",
        "frames": 16,
        "intensity": 1.0,
        "speed": 1.2,
        "easing": "ease_in_out_quad",
        "motion_blur": True,
        "tags": ["water", "river", "flow"],
    },
    
    "waterfall": {
        "name": "waterfall",
        "description": "Cascading waterfall with mist",
        "effect": "water",
        "frames": 12,
        "intensity": 1.5,
        "speed": 1.8,
        "easing": "ease_out_quad",
        "secondary_effects": [
            {"effect": "smoke", "intensity": 0.3, "speed": 0.5}
        ],
        "motion_blur": True,
        "blur_samples": 4,
        "tags": ["water", "intense", "nature"],
    },
    
    "puddle_drip": {
        "name": "puddle_drip",
        "description": "Puddle with occasional drip ripples",
        "effect": "water",
        "frames": 32,
        "intensity": 0.3,
        "speed": 0.3,
        "easing": "ease_out_elastic",
        "tags": ["water", "rain", "puddle"],
    },
    
    # ==================== MAGIC ====================
    "magic_crystal": {
        "name": "magic_crystal",
        "description": "Enchanted crystal with sparkles and glow",
        "effect": "sparkle",
        "frames": 24,
        "intensity": 1.0,
        "speed": 0.6,
        "easing": "ease_in_out_elastic",
        "secondary_effects": [
            {"effect": "glow", "intensity": 0.8, "speed": 0.4}
        ],
        "color_ramp": ["#9400D3", "#8A2BE2", "#DA70D6", "#FFFFFF"],
        "tags": ["magic", "crystal", "fantasy"],
    },
    
    "magic_orb": {
        "name": "magic_orb",
        "description": "Floating magic orb with pulsing energy",
        "effect": "pulse",
        "frames": 16,
        "intensity": 1.2,
        "speed": 0.8,
        "easing": "ease_in_out_sine",
        "secondary_effects": [
            {"effect": "sparkle", "intensity": 0.6},
            {"effect": "float", "intensity": 0.3}
        ],
        "color_ramp": ["#4169E1", "#00CED1", "#00FA9A"],
        "tags": ["magic", "orb", "float"],
    },
    
    "enchanted_glow": {
        "name": "enchanted_glow",
        "description": "Soft magical glow effect",
        "effect": "glow",
        "frames": 20,
        "intensity": 1.0,
        "speed": 0.5,
        "easing": "ease_in_out_quad",
        "color_ramp": ["#7B68EE", "#9370DB", "#E6E6FA"],
        "tags": ["magic", "glow", "enchant"],
    },
    
    "spell_cast": {
        "name": "spell_cast",
        "description": "Magic spell casting burst",
        "effect": "sparkle",
        "frames": 12,
        "intensity": 2.0,
        "speed": 1.5,
        "easing": "ease_out_expo",
        "anticipation": 0.2,
        "overshoot": 0.1,
        "secondary_effects": [
            {"effect": "glow", "intensity": 1.5}
        ],
        "motion_blur": True,
        "tags": ["magic", "spell", "combat"],
    },
    
    "fairy_dust": {
        "name": "fairy_dust",
        "description": "Twinkling fairy dust particles",
        "effect": "sparkle",
        "frames": 20,
        "intensity": 0.8,
        "speed": 0.6,
        "easing": "ease_in_out_sine",
        "secondary_effects": [
            {"effect": "float", "intensity": 0.2}
        ],
        "color_ramp": ["#FFD700", "#FAFAD2", "#FFFFFF"],
        "tags": ["magic", "fairy", "whimsical"],
    },
    
    "dark_magic": {
        "name": "dark_magic",
        "description": "Ominous dark magic energy",
        "effect": "pulse",
        "frames": 16,
        "intensity": 1.3,
        "speed": 0.7,
        "easing": "ease_in_out_cubic",
        "secondary_effects": [
            {"effect": "smoke", "intensity": 0.5}
        ],
        "color_ramp": ["#2F0047", "#4B0082", "#8B008B", "#9932CC"],
        "tags": ["magic", "dark", "ominous"],
    },
    
    # ==================== NATURE ====================
    "tree_sway": {
        "name": "tree_sway",
        "description": "Gentle tree swaying in breeze",
        "effect": "sway",
        "frames": 24,
        "intensity": 0.6,
        "speed": 0.4,
        "easing": "ease_in_out_sine",
        "tags": ["nature", "tree", "wind"],
    },
    
    "grass_wind": {
        "name": "grass_wind",
        "description": "Grass blowing in the wind",
        "effect": "sway",
        "frames": 16,
        "intensity": 0.8,
        "speed": 0.6,
        "easing": "ease_in_out_quad",
        "tags": ["nature", "grass", "wind"],
    },
    
    "flower_bloom": {
        "name": "flower_bloom",
        "description": "Flower gently blooming",
        "effect": "pulse",
        "frames": 32,
        "intensity": 0.5,
        "speed": 0.3,
        "easing": "ease_out_quad",
        "squash_stretch": 0.1,
        "tags": ["nature", "flower", "growth"],
    },
    
    "leaf_fall": {
        "name": "leaf_fall",
        "description": "Falling leaf with flutter",
        "effect": "float",
        "frames": 24,
        "intensity": 1.0,
        "speed": 0.5,
        "easing": "ease_in_out_sine",
        "secondary_effects": [
            {"effect": "sway", "intensity": 0.4}
        ],
        "tags": ["nature", "leaf", "fall", "autumn"],
    },
    
    # ==================== TECH ====================
    "glitch_subtle": {
        "name": "glitch_subtle",
        "description": "Subtle digital glitch",
        "effect": "glitch",
        "frames": 8,
        "intensity": 0.5,
        "speed": 2.0,
        "easing": "linear",
        "tags": ["tech", "glitch", "digital"],
    },
    
    "glitch_heavy": {
        "name": "glitch_heavy",
        "description": "Heavy digital corruption",
        "effect": "glitch",
        "frames": 6,
        "intensity": 2.0,
        "speed": 3.0,
        "easing": "linear",
        "motion_blur": True,
        "tags": ["tech", "glitch", "corrupt"],
    },
    
    "hologram": {
        "name": "hologram",
        "description": "Flickering hologram display",
        "effect": "flicker",
        "frames": 12,
        "intensity": 0.8,
        "speed": 1.5,
        "easing": "linear",
        "secondary_effects": [
            {"effect": "glitch", "intensity": 0.3}
        ],
        "color_ramp": ["#00FFFF", "#00CED1", "#40E0D0"],
        "tags": ["tech", "hologram", "sci-fi"],
    },
    
    "electric_spark": {
        "name": "electric_spark",
        "description": "Electric sparking and arcing",
        "effect": "electric",
        "frames": 8,
        "intensity": 1.5,
        "speed": 2.0,
        "easing": "ease_out_expo",
        "motion_blur": True,
        "color_ramp": ["#4169E1", "#00BFFF", "#FFFFFF"],
        "tags": ["tech", "electric", "energy"],
    },
    
    "neon_pulse": {
        "name": "neon_pulse",
        "description": "Neon sign pulsing glow",
        "effect": "glow",
        "frames": 16,
        "intensity": 1.2,
        "speed": 0.8,
        "easing": "ease_in_out_sine",
        "secondary_effects": [
            {"effect": "flicker", "intensity": 0.2}
        ],
        "color_ramp": ["#FF1493", "#FF69B4", "#FFB6C1"],
        "tags": ["tech", "neon", "cyberpunk"],
    },
    
    # ==================== UI ====================
    "button_hover": {
        "name": "button_hover",
        "description": "Button hover highlight effect",
        "effect": "glow",
        "frames": 8,
        "intensity": 0.5,
        "speed": 1.5,
        "easing": "ease_out_quad",
        "tags": ["ui", "button", "hover"],
    },
    
    "icon_bounce": {
        "name": "icon_bounce",
        "description": "Bouncy icon attention animation",
        "effect": "bounce",
        "frames": 12,
        "intensity": 0.8,
        "speed": 1.0,
        "easing": "ease_out_bounce",
        "squash_stretch": 0.2,
        "tags": ["ui", "icon", "attention"],
    },
    
    "loading_pulse": {
        "name": "loading_pulse",
        "description": "Loading indicator pulse",
        "effect": "pulse",
        "frames": 16,
        "intensity": 0.6,
        "speed": 1.0,
        "easing": "ease_in_out_sine",
        "tags": ["ui", "loading", "indicator"],
    },
    
    "notification_pop": {
        "name": "notification_pop",
        "description": "Notification popup animation",
        "effect": "bounce",
        "frames": 10,
        "intensity": 1.0,
        "speed": 1.5,
        "easing": "ease_out_back",
        "anticipation": 0.1,
        "overshoot": 0.2,
        "tags": ["ui", "notification", "popup"],
    },
    
    # ==================== CHARACTER ====================
    "idle_breathing": {
        "name": "idle_breathing",
        "description": "Subtle breathing idle animation",
        "effect": "pulse",
        "frames": 24,
        "intensity": 0.15,
        "speed": 0.4,
        "easing": "ease_in_out_sine",
        "squash_stretch": 0.05,
        "tags": ["character", "idle", "breathing"],
    },
    
    "idle_bob": {
        "name": "idle_bob",
        "description": "Gentle floating bob for characters",
        "effect": "float",
        "frames": 20,
        "intensity": 0.3,
        "speed": 0.5,
        "easing": "ease_in_out_sine",
        "tags": ["character", "idle", "float"],
    },
    
    "damage_shake": {
        "name": "damage_shake",
        "description": "Hit/damage reaction shake",
        "effect": "shake",
        "frames": 6,
        "intensity": 1.5,
        "speed": 3.0,
        "easing": "ease_out_expo",
        "tags": ["character", "damage", "combat"],
    },
    
    "death_dissolve": {
        "name": "death_dissolve",
        "description": "Character death dissolve effect",
        "effect": "dissolve",
        "frames": 24,
        "intensity": 1.0,
        "speed": 0.6,
        "easing": "ease_in_quad",
        "secondary_effects": [
            {"effect": "sparkle", "intensity": 0.5}
        ],
        "tags": ["character", "death", "dissolve"],
    },
    
    "power_up": {
        "name": "power_up",
        "description": "Character power-up glow",
        "effect": "glow",
        "frames": 16,
        "intensity": 1.5,
        "speed": 1.0,
        "easing": "ease_out_expo",
        "secondary_effects": [
            {"effect": "sparkle", "intensity": 0.8}
        ],
        "color_ramp": ["#FFD700", "#FFFF00", "#FFFFFF"],
        "tags": ["character", "power", "buff"],
    },
    
    # ==================== ENVIRONMENT ====================
    "lava_flow": {
        "name": "lava_flow",
        "description": "Flowing molten lava",
        "effect": "water",
        "frames": 16,
        "intensity": 0.8,
        "speed": 0.4,
        "easing": "ease_in_out_quad",
        "secondary_effects": [
            {"effect": "glow", "intensity": 0.6}
        ],
        "color_ramp": ["#8B0000", "#FF4500", "#FF8C00", "#FFD700"],
        "tags": ["environment", "lava", "hot"],
    },
    
    "poison_bubble": {
        "name": "poison_bubble",
        "description": "Toxic bubbling effect",
        "effect": "water",
        "frames": 20,
        "intensity": 0.7,
        "speed": 0.6,
        "easing": "ease_in_out_quad",
        "color_ramp": ["#228B22", "#32CD32", "#7FFF00"],
        "tags": ["environment", "poison", "swamp"],
    },
    
    "snow_fall": {
        "name": "snow_fall",
        "description": "Gentle falling snow",
        "effect": "float",
        "frames": 24,
        "intensity": 0.5,
        "speed": 0.3,
        "easing": "ease_in_out_sine",
        "secondary_effects": [
            {"effect": "sway", "intensity": 0.2}
        ],
        "tags": ["environment", "snow", "winter"],
    },
    
    "dust_motes": {
        "name": "dust_motes",
        "description": "Floating dust particles in light",
        "effect": "float",
        "frames": 32,
        "intensity": 0.3,
        "speed": 0.2,
        "easing": "ease_in_out_sine",
        "secondary_effects": [
            {"effect": "sparkle", "intensity": 0.2}
        ],
        "tags": ["environment", "dust", "ambient"],
    },
    
    "fog_drift": {
        "name": "fog_drift",
        "description": "Drifting fog or mist",
        "effect": "smoke",
        "frames": 32,
        "intensity": 0.5,
        "speed": 0.2,
        "easing": "ease_in_out_sine",
        "tags": ["environment", "fog", "atmosphere"],
    },
}


# ============================================================================
# Preset Manager
# ============================================================================

class PresetManager:
    """
    Manages loading, saving, and applying effect presets.
    """
    
    def __init__(self, user_presets_dir: Optional[Path] = None):
        """
        Initialize preset manager.
        
        Args:
            user_presets_dir: Directory for user presets (default: ~/.sprite-animator/presets)
        """
        self.user_presets_dir = user_presets_dir or Path.home() / '.sprite-animator' / 'presets'
        self.user_presets_dir.mkdir(parents=True, exist_ok=True)
        
        self._builtin: Dict[str, EffectPreset] = {}
        self._user: Dict[str, EffectPreset] = {}
        
        self._load_builtin_presets()
        self._load_user_presets()
    
    def _load_builtin_presets(self) -> None:
        """Load built-in presets"""
        for name, data in BUILTIN_PRESETS.items():
            self._builtin[name] = EffectPreset.from_dict(data)
    
    def _load_user_presets(self) -> None:
        """Load user-defined presets from YAML files"""
        for yaml_file in self.user_presets_dir.glob('*.yaml'):
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                
                if isinstance(data, dict):
                    if 'presets' in data:
                        # Multiple presets in one file
                        for name, preset_data in data['presets'].items():
                            preset_data['name'] = name
                            self._user[name] = EffectPreset.from_dict(preset_data)
                    else:
                        # Single preset
                        name = yaml_file.stem
                        data['name'] = name
                        self._user[name] = EffectPreset.from_dict(data)
            except Exception as e:
                print(f"Warning: Could not load preset file {yaml_file}: {e}")
    
    def get(self, name: str) -> Optional[EffectPreset]:
        """
        Get a preset by name.
        User presets override built-in presets with same name.
        """
        return self._user.get(name) or self._builtin.get(name)
    
    def exists(self, name: str) -> bool:
        """Check if preset exists"""
        return name in self._user or name in self._builtin
    
    def list_all(self) -> List[str]:
        """List all preset names"""
        all_names = set(self._builtin.keys()) | set(self._user.keys())
        return sorted(all_names)
    
    def list_by_tag(self, tag: str) -> List[str]:
        """List presets with a specific tag"""
        matches = []
        for name, preset in {**self._builtin, **self._user}.items():
            if tag.lower() in [t.lower() for t in preset.tags]:
                matches.append(name)
        return sorted(matches)
    
    def list_by_effect(self, effect: str) -> List[str]:
        """List presets using a specific effect"""
        matches = []
        for name, preset in {**self._builtin, **self._user}.items():
            if preset.effect == effect:
                matches.append(name)
        return sorted(matches)
    
    def list_tags(self) -> List[str]:
        """List all available tags"""
        tags = set()
        for preset in {**self._builtin, **self._user}.values():
            tags.update(preset.tags)
        return sorted(tags)
    
    def save_preset(self, preset: EffectPreset, filename: Optional[str] = None) -> Path:
        """
        Save a user preset to YAML file.
        
        Args:
            preset: The preset to save
            filename: Optional filename (default: preset.name.yaml)
            
        Returns:
            Path to saved file
        """
        filename = filename or f"{preset.name}.yaml"
        if not filename.endswith('.yaml'):
            filename += '.yaml'
        
        filepath = self.user_presets_dir / filename
        
        with open(filepath, 'w') as f:
            yaml.dump(preset.to_dict(), f, default_flow_style=False, sort_keys=False)
        
        # Reload user presets
        self._user[preset.name] = preset
        
        return filepath
    
    def delete_preset(self, name: str) -> bool:
        """
        Delete a user preset.
        
        Returns:
            True if deleted, False if not found or is builtin
        """
        if name not in self._user:
            return False
        
        # Find and delete file
        for yaml_file in self.user_presets_dir.glob('*.yaml'):
            if yaml_file.stem == name:
                yaml_file.unlink()
                break
        
        del self._user[name]
        return True
    
    def create_preset(
        self,
        name: str,
        effect: str,
        description: str = "",
        **kwargs
    ) -> EffectPreset:
        """
        Create a new preset.
        
        Args:
            name: Preset name
            effect: Primary effect
            description: Optional description
            **kwargs: Additional preset parameters
            
        Returns:
            Created EffectPreset
        """
        return EffectPreset(
            name=name,
            effect=effect,
            description=description,
            **kwargs
        )
    
    def get_preset_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a preset"""
        preset = self.get(name)
        if not preset:
            return None
        
        return {
            'name': preset.name,
            'description': preset.description,
            'effect': preset.effect,
            'frames': preset.frames,
            'intensity': preset.intensity,
            'speed': preset.speed,
            'easing': preset.easing,
            'secondary_effects': preset.secondary_effects,
            'color_ramp': preset.color_ramp,
            'tags': preset.tags,
            'is_builtin': name in self._builtin,
            'is_user': name in self._user,
        }
    
    def search(self, query: str) -> List[str]:
        """Search presets by name, description, or tags"""
        query = query.lower()
        matches = []
        
        for name, preset in {**self._builtin, **self._user}.items():
            if (query in name.lower() or
                query in preset.description.lower() or
                any(query in tag.lower() for tag in preset.tags)):
                matches.append(name)
        
        return sorted(matches)


# ============================================================================
# Preset Application
# ============================================================================

def apply_preset_to_args(preset: EffectPreset, args: Any) -> Any:
    """
    Apply preset settings to CLI argument namespace.
    
    Args:
        preset: The preset to apply
        args: argparse namespace
        
    Returns:
        Modified args namespace
    """
    # Core animation settings
    if not hasattr(args, 'effect') or args.effect is None:
        args.effect = preset.effect
    
    if not hasattr(args, 'frames') or args.frames == 8:  # Default
        args.frames = preset.frames
    
    if not hasattr(args, 'intensity') or args.intensity == 1.0:
        args.intensity = preset.intensity
    
    if not hasattr(args, 'speed') or args.speed == 1.0:
        args.speed = preset.speed
    
    # Quality and format
    if hasattr(args, 'quality') and preset.quality:
        args.quality = preset.quality
    
    if hasattr(args, 'format') and preset.format:
        args.format = preset.format
    
    # Motion blur
    if hasattr(args, 'motion_blur'):
        args.motion_blur = preset.motion_blur
    
    # Animation principles
    if hasattr(args, 'anticipation') and preset.anticipation > 0:
        args.anticipation = preset.anticipation
    
    if hasattr(args, 'overshoot') and preset.overshoot > 0:
        args.overshoot = preset.overshoot
    
    if hasattr(args, 'squash_stretch') and preset.squash_stretch > 0:
        args.squash_stretch = preset.squash_stretch
    
    # Store preset metadata for secondary effects
    args._preset = preset
    args._secondary_effects = preset.secondary_effects
    args._color_ramp = preset.color_ramp
    args._easing = preset.easing
    
    return args


# ============================================================================
# Global Instance & Convenience Functions
# ============================================================================

_manager: Optional[PresetManager] = None


def get_preset_manager() -> PresetManager:
    """Get or create global preset manager"""
    global _manager
    if _manager is None:
        _manager = PresetManager()
    return _manager


def get_preset(name: str) -> Optional[EffectPreset]:
    """Get a preset by name"""
    return get_preset_manager().get(name)


def list_presets(tag: Optional[str] = None, effect: Optional[str] = None) -> List[str]:
    """List available presets, optionally filtered"""
    manager = get_preset_manager()
    
    if tag:
        return manager.list_by_tag(tag)
    elif effect:
        return manager.list_by_effect(effect)
    else:
        return manager.list_all()


def save_preset(preset: EffectPreset) -> Path:
    """Save a user preset"""
    return get_preset_manager().save_preset(preset)


def search_presets(query: str) -> List[str]:
    """Search presets"""
    return get_preset_manager().search(query)


def preset_exists(name: str) -> bool:
    """Check if preset exists"""
    return get_preset_manager().exists(name)


def get_preset_count() -> Dict[str, int]:
    """Get count of presets by category"""
    manager = get_preset_manager()
    return {
        'builtin': len(manager._builtin),
        'user': len(manager._user),
        'total': len(manager.list_all()),
    }
