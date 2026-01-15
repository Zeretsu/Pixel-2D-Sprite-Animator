"""
Sprite Animator - Smart procedural animation for pixel art sprites
"""

from .core import SpriteParser, Sprite, SpriteExporter, ColorUtils, MathUtils
from .detection import SpriteAnalyzer
from .procedural import EFFECTS, get_effect

__version__ = "0.1.0"
__all__ = [
    'SpriteParser',
    'Sprite', 
    'SpriteExporter',
    'SpriteAnalyzer',
    'EFFECTS',
    'get_effect',
    'animate',
    'analyze',
]


def analyze(image_path: str) -> dict:
    """
    Analyze a sprite and get animation suggestions.
    
    Args:
        image_path: Path to the sprite image
        
    Returns:
        Dictionary with analysis results and suggestions
    """
    sprite = SpriteParser.parse(image_path)
    analyzer = SpriteAnalyzer(sprite)
    result = analyzer.analyze()
    
    return {
        'best_effect': result.best_effect,
        'confidence': result.best_confidence,
        'suggestions': [
            {
                'effect': s.effect,
                'confidence': s.confidence,
                'reasons': s.reasons
            }
            for s in result.suggestions
        ],
        'explanation': analyzer.explain()
    }


def animate(
    image_path: str,
    effect: str = None,
    output_path: str = None,
    frames: int = 8,
    format: str = 'gif',
    intensity: float = 1.0,
    speed: float = 1.0,
    extra: dict = None,
    remove_background: bool = True,
    bg_tolerance: int = 15,
    outline: float = 0,
    outline_color: tuple = (0, 0, 0),
    **effect_kwargs
):
    """
    Animate a sprite with auto-detection or specified effect.
    
    Args:
        image_path: Path to the sprite image
        effect: Effect name (auto-detected if None)
        output_path: Output path (auto-generated if None)
        frames: Number of animation frames
        format: Output format ('gif', 'spritesheet', 'frames')
        intensity: Effect intensity (0.0-2.0)
        speed: Animation speed multiplier
        extra: Additional effect-specific parameters (quality, motion_blur, etc.)
        remove_background: Auto-remove background from edges (default: True)
        bg_tolerance: Background color tolerance 0-255 (default: 15, lower=more conservative)
        outline: Add outline with this thickness (0 = no outline)
        outline_color: RGB tuple for outline color (default: black)
        **effect_kwargs: Additional effect parameters
        
    Returns:
        Path to the output file(s)
    """
    from pathlib import Path
    from .procedural.base import EffectConfig
    from .core.sdf import add_outline
    
    # Load sprite with optional background removal
    sprite = SpriteParser.parse(image_path, remove_background=remove_background, bg_tolerance=bg_tolerance)
    
    # Auto-detect effect if not specified
    if effect is None:
        analyzer = SpriteAnalyzer(sprite)
        effect, confidence = analyzer.get_best_effect()
        print(f"Auto-detected effect: {effect} ({confidence:.0%} confidence)")
    
    # Get effect class
    EffectClass = get_effect(effect)
    
    # Merge extra config with effect_kwargs
    merged_extra = {}
    if extra:
        merged_extra.update(extra)
    if effect_kwargs:
        merged_extra.update(effect_kwargs)
    
    # Create config
    config = EffectConfig(
        frame_count=frames,
        intensity=intensity,
        speed=speed,
        extra=merged_extra
    )
    
    # Apply effect
    effect_instance = EffectClass(config)
    animation_frames = effect_instance.apply(sprite)
    
    # Add outline to each frame if requested
    if outline > 0:
        for frame in animation_frames:
            frame.pixels = add_outline(frame.pixels, thickness=outline, color=outline_color)
    
    # Generate output path if not specified
    if output_path is None:
        input_path = Path(image_path)
        output_path = input_path.parent / f"{input_path.stem}_{effect}.{format}"
    
    # Export
    if format == 'gif':
        return SpriteExporter.to_gif(animation_frames, output_path)
    elif format == 'spritesheet':
        path, meta = SpriteExporter.to_spritesheet(animation_frames, output_path)
        return path
    elif format == 'frames':
        return SpriteExporter.to_frames(animation_frames, output_path)
    else:
        raise ValueError(f"Unknown format: {format}")
