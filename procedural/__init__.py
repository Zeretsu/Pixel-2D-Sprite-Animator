"""
Procedural Effects - Algorithm-driven sprite animations
"""

from .base import BaseEffect, EffectConfig
from .flame import FlameEffect
from .water import WaterEffect
from .float_bob import FloatEffect
from .sparkle import SparkleEffect
from .sway import SwayEffect
from .pulse import PulseEffect
from .smoke import SmokeEffect
from .wobble import WobbleEffect
from .glitch import GlitchEffect
from .shake import ShakeEffect
from .bounce import BounceEffect
from .flicker import FlickerEffect
from .glow import GlowEffect
from .dissolve import DissolveEffect
from .rainbow import RainbowEffect
from .spin import SpinEffect
from .melt import MeltEffect
from .electric import ElectricEffect

# New effects
from .shadow import ShadowEffect
from .teleport import TeleportEffect
from .charge import ChargeEffect
from .damage import DamageEffect
from .freeze import FreezeEffect
from .poison import PoisonEffect
from .petrify import PetrifyEffect
from .hologram import HologramEffect
from .chromatic import ChromaticEffect
from .stretch import StretchEffect
from .ripple import RippleEffect
from .levitate import LevitateEffect
from .window_flicker import WindowFlickerEffect
from .color_pulse import ColorPulseEffect
from .sunlight import SunlightEffect

# Effect registry for easy access
EFFECTS = {
    'flame': FlameEffect,
    'fire': FlameEffect,  # Alias
    'water': WaterEffect,
    'wave': WaterEffect,  # Alias
    'float': FloatEffect,
    'bob': FloatEffect,  # Alias
    'sparkle': SparkleEffect,
    'magic': SparkleEffect,  # Alias
    'sway': SwayEffect,
    'pulse': PulseEffect,
    'smoke': SmokeEffect,
    'cloud': SmokeEffect,  # Alias
    'wobble': WobbleEffect,
    'jelly': WobbleEffect,  # Alias
    # New effects
    'glitch': GlitchEffect,
    'digital': GlitchEffect,  # Alias
    'shake': ShakeEffect,
    'vibrate': ShakeEffect,  # Alias
    'bounce': BounceEffect,
    'jump': BounceEffect,  # Alias
    'flicker': FlickerEffect,
    'strobe': FlickerEffect,  # Alias
    'glow': GlowEffect,
    'aura': GlowEffect,  # Alias
    'dissolve': DissolveEffect,
    'disintegrate': DissolveEffect,  # Alias
    'rainbow': RainbowEffect,
    'hue': RainbowEffect,  # Alias
    'spin': SpinEffect,
    'rotate': SpinEffect,  # Alias
    'melt': MeltEffect,
    'drip': MeltEffect,  # Alias
    'electric': ElectricEffect,
    'lightning': ElectricEffect,  # Alias
    'zap': ElectricEffect,  # Alias
    # New batch of effects
    'shadow': ShadowEffect,
    'afterimage': ShadowEffect,  # Alias
    'trail': ShadowEffect,  # Alias
    'teleport': TeleportEffect,
    'warp': TeleportEffect,  # Alias
    'materialize': TeleportEffect,  # Alias
    'charge': ChargeEffect,
    'power': ChargeEffect,  # Alias
    'energy': ChargeEffect,  # Alias
    'damage': DamageEffect,
    'hit': DamageEffect,  # Alias
    'hurt': DamageEffect,  # Alias
    'freeze': FreezeEffect,
    'ice': FreezeEffect,  # Alias
    'frozen': FreezeEffect,  # Alias
    'poison': PoisonEffect,
    'toxic': PoisonEffect,  # Alias
    'venom': PoisonEffect,  # Alias
    'petrify': PetrifyEffect,
    'stone': PetrifyEffect,  # Alias
    'statue': PetrifyEffect,  # Alias
    'hologram': HologramEffect,
    'holo': HologramEffect,  # Alias
    'projection': HologramEffect,  # Alias
    'chromatic': ChromaticEffect,
    'aberration': ChromaticEffect,  # Alias
    'rgb': ChromaticEffect,  # Alias
    'stretch': StretchEffect,
    'squash': StretchEffect,  # Alias
    'cartoon': StretchEffect,  # Alias
    'ripple': RippleEffect,
    'distort': RippleEffect,  # Alias
    'shockwave': RippleEffect,  # Alias
    'levitate': LevitateEffect,
    'hover': LevitateEffect,  # Alias
    'float_magic': LevitateEffect,  # Alias
    # Window/light effects
    'window_flicker': WindowFlickerEffect,
    'window': WindowFlickerEffect,  # Alias
    'light_dim': WindowFlickerEffect,  # Alias
    'indoor_light': WindowFlickerEffect,  # Alias
    # Color-targeted effects
    'color_pulse': ColorPulseEffect,
    'blue_pulse': ColorPulseEffect,  # Alias
    'magic_glow': ColorPulseEffect,  # Alias
    'shine': ColorPulseEffect,  # Alias
    # Sunlight effect
    'sunlight': SunlightEffect,
    'dappled_light': SunlightEffect,  # Alias
    'sun': SunlightEffect,  # Alias
}

# Advanced effects - imported after EFFECTS dict to avoid circular import
from .particles import ParticleEffect
from .motion_blur import MotionBlur, SpeedLines
from .trail import TrailEffect, RibbonTrail
from .keyframes import KeyframeEffect, AnimationCurve, KeyframeAnimator, PresetCurves
from .elements import FireElement, WaterElement, IceElement
from .flap import FlapEffect, HoverFlapEffect, GlideFlapEffect

# Register advanced effects
EFFECTS.update({
    'particles': ParticleEffect,
    'particle': ParticleEffect,  # Alias
    'emitter': ParticleEffect,  # Alias
    'motion_blur': MotionBlur,
    'blur': MotionBlur,  # Alias
    'speed_lines': SpeedLines,
    'speed': SpeedLines,  # Alias
    'motion_trail': TrailEffect,
    'afterimages': TrailEffect,  # Alias
    'ghost_trail': TrailEffect,  # Alias
    'ribbon_trail': RibbonTrail,
    'ribbon': RibbonTrail,  # Alias
    'keyframe': KeyframeEffect,
    'curves': KeyframeEffect,  # Alias
    # Elemental effects - ADD visuals to sprite
    'fire_element': FireElement,
    'fire_sword': FireElement,  # Alias
    'flame_element': FireElement,  # Alias
    'water_element': WaterElement,
    'water_sword': WaterElement,  # Alias
    'aqua': WaterElement,  # Alias
    'ice_element': IceElement,
    'ice_sword': IceElement,  # Alias
    'frost': IceElement,  # Alias
    # Wing flap effects
    'flap': FlapEffect,
    'wing': FlapEffect,  # Alias
    'wings': FlapEffect,  # Alias
    'hover_flap': HoverFlapEffect,
    'hover_fly': HoverFlapEffect,  # Alias
    'glide_flap': GlideFlapEffect,
    'glide': GlideFlapEffect,  # Alias
    'soar': GlideFlapEffect,  # Alias
})


def get_effect(name: str) -> type:
    """Get effect class by name"""
    name = name.lower()
    if name not in EFFECTS:
        raise ValueError(f"Unknown effect: {name}. Available: {list(set(EFFECTS.values()))}")
    return EFFECTS[name]


__all__ = [
    'BaseEffect',
    'EffectConfig',
    'FlameEffect',
    'WaterEffect', 
    'FloatEffect',
    'SparkleEffect',
    'SwayEffect',
    'PulseEffect',
    'SmokeEffect',
    'WobbleEffect',
    'GlitchEffect',
    'ShakeEffect',
    'BounceEffect',
    'FlickerEffect',
    'GlowEffect',
    'DissolveEffect',
    'RainbowEffect',
    'SpinEffect',
    'MeltEffect',
    'ElectricEffect',
    # New effects
    'ShadowEffect',
    'TeleportEffect',
    'ChargeEffect',
    'DamageEffect',
    'FreezeEffect',
    'PoisonEffect',
    'PetrifyEffect',
    'HologramEffect',
    'ChromaticEffect',
    'StretchEffect',
    'RippleEffect',
    'LevitateEffect',
    # Advanced effects
    'ParticleEffect',
    'MotionBlur',
    'SpeedLines',
    'TrailEffect',
    'RibbonTrail',
    'KeyframeEffect',
    'AnimationCurve',
    'KeyframeAnimator',
    'PresetCurves',
    # Elemental effects
    'FireElement',
    'WaterElement',
    'IceElement',
    # Window/light effects
    'WindowFlickerEffect',
    'EFFECTS',
    'get_effect',
]
