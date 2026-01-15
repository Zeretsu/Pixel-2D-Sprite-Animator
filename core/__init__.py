"""
Sprite Animator - Core Utilities
"""

from .parser import SpriteParser, Sprite
from .exporter import SpriteExporter
from .utils import ColorUtils, MathUtils
from .easing import (
    # Core easing functions
    linear, ease, ease_range, get_easing,
    # Polynomial
    ease_in_quad, ease_out_quad, ease_in_out_quad,
    ease_in_cubic, ease_out_cubic, ease_in_out_cubic,
    ease_in_quart, ease_out_quart, ease_in_out_quart,
    ease_in_quint, ease_out_quint, ease_in_out_quint,
    # Sinusoidal
    ease_in_sine, ease_out_sine, ease_in_out_sine,
    # Exponential
    ease_in_expo, ease_out_expo, ease_in_out_expo,
    # Circular
    ease_in_circ, ease_out_circ, ease_in_out_circ,
    # Back (overshoot)
    ease_in_back, ease_out_back, ease_in_out_back,
    # Elastic (spring)
    ease_in_elastic, ease_out_elastic, ease_in_out_elastic,
    # Bounce
    ease_in_bounce, ease_out_bounce, ease_in_out_bounce,
    # Custom bezier
    custom_bezier, BezierCurve,
    # Presets
    CSS_EASE, CSS_EASE_IN, CSS_EASE_OUT, CSS_EASE_IN_OUT,
    MATERIAL_STANDARD, MATERIAL_DECELERATE, MATERIAL_ACCELERATE,
    GAME_SNAPPY, GAME_HEAVY, GAME_FLOATY,
    # Utilities
    smoothstep, smootherstep, ping_pong, chain_easings,
    generate_easing_curve, EASING_FUNCTIONS,
)
from .physics import (
    # Vector
    Vec2,
    # Spring physics
    Spring, Spring1D, SpringConfig,
    # Skeletal system
    Bone, Skeleton,
    # Particles
    TrailingParticle, ParticleFollowSystem,
    # Motion lag
    MotionLag,
    # Verlet physics
    VerletPoint, VerletChain,
    # Manager
    SecondaryMotionManager,
)
from .motion_blur import (
    # Core motion blur
    motion_blur, BlurMode,
    # Ghost/trail
    ghost_trail,
    # Velocity blur
    velocity_blur,
    # Accumulation
    AccumulationBuffer,
    # Frame interpolation
    interpolate_frames,
    # Directional blurs
    radial_blur, rotational_blur, directional_blur,
    # Color space utilities
    to_linear, to_srgb, to_linear_premul, from_linear_premul,
)
from .anticipation import (
    # Core timing
    AnimationTiming, AnimationCurve,
    # Squash & stretch
    SquashStretch,
    # Presets
    AnimationPrinciples,
    # Builder
    AnimationBuilder,
    # Helper functions
    apply_anticipation_overshoot,
    create_bounce_animation,
    get_squash_stretch_for_velocity,
)
from .smear import (
    # Core smear
    add_smear_frame, SmearType, SmearConfig,
    # Generators
    generate_smear_sequence,
    create_attack_smear,
    create_impact_smear,
    create_dash_smear,
)
from .palette import (
    # Color space
    rgb_to_hsv, hsv_to_rgb, rgb_to_hsl, hsl_to_rgb,
    get_luminance, color_distance,
    # Ramps
    ColorRamp, get_ramp,
    # Presets
    FIRE_RAMP, EMBER_RAMP, WATER_RAMP, OCEAN_RAMP,
    LAVA_RAMP, MAGIC_RAMP, ELECTRIC_RAMP, ICE_RAMP,
    POISON_RAMP, GOLD_RAMP, GRAYSCALE_RAMP, HEAT_RAMP,
    RAMP_PRESETS,
    # Cycler
    PaletteCycler,
    # Ramper
    ColorRamper,
    # Hue shifting
    HueShifter,
    # Intensity pulsing
    IntensityPulser,
    # Palette utilities
    extract_palette, match_to_palette,
    # Convenience
    create_fire_cycler, create_water_cycler, create_magic_cycler,
    apply_fire_ramp, apply_water_ramp, shift_hue,
)
from .particles import (
    # Noise functions
    perlin_noise_1d, perlin_noise_2d, fbm_noise_2d, curl_noise_2d,
    # Bezier paths
    BezierPath,
    SIZE_SHRINK, SIZE_GROW_SHRINK, SIZE_GROW, SIZE_CONSTANT,
    ALPHA_FADE, ALPHA_FADE_LATE, ALPHA_FLASH_FADE,
    SPEED_CONSTANT, SPEED_DECELERATE, SPEED_ACCELERATE, SPEED_BURST,
    # Emission
    EmissionShape, EmissionConfig,
    # Particle
    Particle,
    # Color gradient
    ColorGradient,
    GRADIENT_FIRE, GRADIENT_SPARK, GRADIENT_SMOKE,
    GRADIENT_MAGIC, GRADIENT_WATER, GRADIENT_ELECTRIC,
    # Emitter
    ParticleEmitter,
    # Preset emitters
    create_spark_emitter, create_fire_emitter, create_smoke_emitter,
    create_magic_emitter, create_rain_emitter, create_explosion_emitter,
    create_electric_emitter,
    # System
    ParticleSystem,
)
from .onion_skin import (
    # Config
    OnionSkinConfig, OnionBlendMode, OnionTintMode,
    # Core functions
    create_onion_frame, create_onion_animation, create_onion_layers,
    # Export functions
    export_onion_frames, export_onion_spritesheet, export_onion_gif,
    # Convenience
    add_onion_skin, preview_onion_skin,
)
from .preview import (
    # Config
    PreviewConfig, ViewMode,
    # Window
    PreviewWindow,
    # Session
    PreviewSession,
    # Convenience
    preview_animation, preview_sprite, check_pygame_available,
)
from .sdf import (
    # SDF Generation
    generate_sdf, generate_sdf_from_edges, generate_multi_channel_sdf,
    # Configs
    GlowConfig, OutlineConfig,
    # SDF Effects
    sdf_glow, sdf_outline, sdf_dissolve, sdf_shadow,
    # SDF Animation
    sdf_reveal, sdf_pulse_glow, sdf_breathing,
    # SDF Utilities
    get_sdf_bounds, get_sprite_thickness, get_edge_pixels,
    dilate_sprite, erode_sprite,
    # Convenience
    add_glow, add_outline, add_shadow,
)
from .normal_map import (
    # Normal Map Loading/Generation
    load_normal_map, encode_normal_map,
    generate_normal_map_from_height, generate_normal_map_from_sprite,
    create_flat_normal_map,
    # Lighting
    LightSource, apply_directional_light,
    apply_rim_light, apply_ambient_occlusion,
    # Animated Lighting
    animate_light_direction, animate_light_color, animate_light_pulse,
    # Effects
    apply_normal_mapped_glow, create_lit_animation,
    # Convenience
    quick_light, quick_gem_lighting, quick_metal_lighting,
)
from .decompose import (
    # Layer Types
    LayerType, SpriteLayer, DecomposedSprite,
    # Decomposer
    SpriteDecomposer,
    # Layer Animation Helpers
    animate_layer, pulse_highlights, shift_shadows,
    color_cycle_layer, independent_layer_animation,
    # Convenience
    quick_decompose, quick_glow_animation, quick_shadow_dance,
)
from .presets import (
    # Data structures
    EffectPreset, PresetCategory,
    # Manager
    PresetManager,
    # Application
    apply_preset_to_args,
    # Convenience
    get_preset_manager, get_preset, list_presets,
    save_preset, search_presets, preset_exists, get_preset_count,
    # Built-in presets dict
    BUILTIN_PRESETS,
)

__all__ = [
    'SpriteParser', 'Sprite', 'SpriteExporter', 'ColorUtils', 'MathUtils',
    # Easing
    'linear', 'ease', 'ease_range', 'get_easing',
    'ease_in_quad', 'ease_out_quad', 'ease_in_out_quad',
    'ease_in_cubic', 'ease_out_cubic', 'ease_in_out_cubic',
    'ease_in_quart', 'ease_out_quart', 'ease_in_out_quart',
    'ease_in_quint', 'ease_out_quint', 'ease_in_out_quint',
    'ease_in_sine', 'ease_out_sine', 'ease_in_out_sine',
    'ease_in_expo', 'ease_out_expo', 'ease_in_out_expo',
    'ease_in_circ', 'ease_out_circ', 'ease_in_out_circ',
    'ease_in_back', 'ease_out_back', 'ease_in_out_back',
    'ease_in_elastic', 'ease_out_elastic', 'ease_in_out_elastic',
    'ease_in_bounce', 'ease_out_bounce', 'ease_in_out_bounce',
    'custom_bezier', 'BezierCurve',
    'CSS_EASE', 'CSS_EASE_IN', 'CSS_EASE_OUT', 'CSS_EASE_IN_OUT',
    'MATERIAL_STANDARD', 'MATERIAL_DECELERATE', 'MATERIAL_ACCELERATE',
    'GAME_SNAPPY', 'GAME_HEAVY', 'GAME_FLOATY',
    'smoothstep', 'smootherstep', 'ping_pong', 'chain_easings',
    'generate_easing_curve', 'EASING_FUNCTIONS',
    # Physics
    'Vec2',
    'Spring', 'Spring1D', 'SpringConfig',
    'Bone', 'Skeleton',
    'TrailingParticle', 'ParticleFollowSystem',
    'MotionLag',
    'VerletPoint', 'VerletChain',
    'SecondaryMotionManager',
    # Motion Blur
    'motion_blur', 'BlurMode',
    'ghost_trail',
    'velocity_blur',
    'AccumulationBuffer',
    'interpolate_frames',
    'radial_blur', 'rotational_blur', 'directional_blur',
    'to_linear', 'to_srgb', 'to_linear_premul', 'from_linear_premul',
    # Anticipation & Overshoot
    'AnimationTiming', 'AnimationCurve',
    'SquashStretch',
    'AnimationPrinciples',
    'AnimationBuilder',
    'apply_anticipation_overshoot',
    'create_bounce_animation',
    'get_squash_stretch_for_velocity',
    # Smear Frames
    'add_smear_frame', 'SmearType', 'SmearConfig',
    'generate_smear_sequence',
    'create_attack_smear',
    'create_impact_smear',
    'create_dash_smear',
    # Palette Cycling & Color Ramping
    'rgb_to_hsv', 'hsv_to_rgb', 'rgb_to_hsl', 'hsl_to_rgb',
    'get_luminance', 'color_distance',
    'ColorRamp', 'get_ramp',
    'FIRE_RAMP', 'EMBER_RAMP', 'WATER_RAMP', 'OCEAN_RAMP',
    'LAVA_RAMP', 'MAGIC_RAMP', 'ELECTRIC_RAMP', 'ICE_RAMP',
    'POISON_RAMP', 'GOLD_RAMP', 'GRAYSCALE_RAMP', 'HEAT_RAMP',
    'RAMP_PRESETS',
    'PaletteCycler',
    'ColorRamper',
    'HueShifter',
    'IntensityPulser',
    'extract_palette', 'match_to_palette',
    'create_fire_cycler', 'create_water_cycler', 'create_magic_cycler',
    'apply_fire_ramp', 'apply_water_ramp', 'shift_hue',
    # Particle System
    'perlin_noise_1d', 'perlin_noise_2d', 'fbm_noise_2d', 'curl_noise_2d',
    'BezierPath',
    'SIZE_SHRINK', 'SIZE_GROW_SHRINK', 'SIZE_GROW', 'SIZE_CONSTANT',
    'ALPHA_FADE', 'ALPHA_FADE_LATE', 'ALPHA_FLASH_FADE',
    'SPEED_CONSTANT', 'SPEED_DECELERATE', 'SPEED_ACCELERATE', 'SPEED_BURST',
    'EmissionShape', 'EmissionConfig',
    'Particle',
    'ColorGradient',
    'GRADIENT_FIRE', 'GRADIENT_SPARK', 'GRADIENT_SMOKE',
    'GRADIENT_MAGIC', 'GRADIENT_WATER', 'GRADIENT_ELECTRIC',
    'ParticleEmitter',
    'create_spark_emitter', 'create_fire_emitter', 'create_smoke_emitter',
    'create_magic_emitter', 'create_rain_emitter', 'create_explosion_emitter',
    'create_electric_emitter',
    'ParticleSystem',
    # Onion Skinning
    'OnionSkinConfig', 'OnionBlendMode', 'OnionTintMode',
    'create_onion_frame', 'create_onion_animation', 'create_onion_layers',
    'export_onion_frames', 'export_onion_spritesheet', 'export_onion_gif',
    'add_onion_skin', 'preview_onion_skin',
    # Real-Time Preview
    'PreviewConfig', 'ViewMode',
    'PreviewWindow',
    'PreviewSession',
    'preview_animation', 'preview_sprite', 'check_pygame_available',
    # Signed Distance Fields
    'generate_sdf', 'generate_sdf_from_edges', 'generate_multi_channel_sdf',
    'GlowConfig', 'OutlineConfig',
    'sdf_glow', 'sdf_outline', 'sdf_dissolve', 'sdf_shadow',
    'sdf_reveal', 'sdf_pulse_glow', 'sdf_breathing',
    'get_sdf_bounds', 'get_sprite_thickness', 'get_edge_pixels',
    'dilate_sprite', 'erode_sprite',
    'add_glow', 'add_outline', 'add_shadow',
    # Normal Maps
    'load_normal_map', 'encode_normal_map',
    'generate_normal_map_from_height', 'generate_normal_map_from_sprite',
    'create_flat_normal_map',
    'LightSource', 'apply_directional_light',
    'apply_rim_light', 'apply_ambient_occlusion',
    'animate_light_direction', 'animate_light_color', 'animate_light_pulse',
    'apply_normal_mapped_glow', 'create_lit_animation',
    'quick_light', 'quick_gem_lighting', 'quick_metal_lighting',
    # Sprite Decomposition
    'LayerType', 'SpriteLayer', 'DecomposedSprite',
    'SpriteDecomposer',
    'animate_layer', 'pulse_highlights', 'shift_shadows',
    'color_cycle_layer', 'independent_layer_animation',
    'quick_decompose', 'quick_glow_animation', 'quick_shadow_dance',
    # Effect Presets
    'EffectPreset', 'PresetCategory',
    'PresetManager',
    'apply_preset_to_args',
    'get_preset_manager', 'get_preset', 'list_presets',
    'save_preset', 'search_presets', 'preset_exists', 'get_preset_count',
    'BUILTIN_PRESETS',
]
