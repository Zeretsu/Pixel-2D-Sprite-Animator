#!/usr/bin/env python
"""
Sprite Animator CLI - Smart procedural animation for pixel art sprites

Usage:
    python main.py <input_image> [options]
    
Examples:
    python main.py torch.png                    # Auto-detect and animate
    python main.py torch.png --effect flame     # Use specific effect
    python main.py water.png --frames 12        # Custom frame count
    python main.py gem.png --format spritesheet # Output as spritesheet
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Smart procedural animation for pixel art sprites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Effects:
  Basic Effects:
  flame     - Fire/flickering with upward drift
  water     - Wave distortion and ripples
  float     - Gentle up-and-down bobbing
  sparkle   - Magic glitter particles
  sway      - Side-to-side swaying (for plants, candles)
  pulse     - Breathing/scaling animation
  smoke     - Soft drifting for clouds/smoke
  wobble    - Jelly-like elastic deformation
  glitch    - Digital corruption/RGB split
  shake     - Screen shake/vibration
  bounce    - Bouncing with squash/stretch
  flicker   - Light flicker/strobe
  glow      - Pulsing aura/glow
  dissolve  - Particle dissolve/materialize
  rainbow   - Color cycling/hue shift
  spin      - Rotation animation
  melt      - Melting/dripping
  electric  - Lightning/electricity
  
  Status Effects:
  shadow    - Shadow/afterimage trail
  teleport  - Warp/materialize effect
  charge    - Power/energy charging
  damage    - Hit/hurt flash effect
  freeze    - Ice/frozen effect
  poison    - Toxic/venom dripping
  petrify   - Stone/statue transformation
  hologram  - Holographic projection
  chromatic - RGB aberration/split
  stretch   - Squash and stretch cartoon
  ripple    - Shockwave distortion
  levitate  - Magical floating hover
  
  Advanced Effects:
  particles     - Customizable particle systems
  motion_blur   - Blur effects (linear, radial, zoom)
  speed_lines   - Anime-style speed/action lines
  motion_trail  - Afterimage/ghost trails
  ribbon_trail  - Flowing ribbon trails
  keyframe      - Custom keyframe animations
  
  Elemental Effects (add visuals to sprite):
  fire_element  - Flames, embers, glow emanating from sprite
  water_element - Ripples, droplets, flowing water
  ice_element   - Frost, crystals, sparkles
  
  Creature Effects:
  flap          - Wing flapping for bats, birds, butterflies
  hover_flap    - Hovering with figure-8 wing motion
  glide_flap    - Gliding with occasional flaps

Examples:
  %(prog)s torch.png                         # Auto-detect best effect
  %(prog)s torch.png --effect flame          # Force flame effect
  %(prog)s gem.png --effect sparkle --frames 12
  %(prog)s logo.png --effect glitch          # Digital glitch
  %(prog)s tree.png --analyze                # Just show analysis
  %(prog)s --list-presets                    # Show all presets
  %(prog)s --preset-info torch_realistic     # Show preset details
        """
    )
    
    parser.add_argument(
        'input',
        type=str,
        nargs='?',  # Make optional for --list-presets and --preset-info
        default=None,
        help='Input sprite image (PNG, GIF, etc.)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output path (auto-generated if not specified)'
    )
    
    parser.add_argument(
        '-e', '--effect',
        type=str,
        default=None,
        choices=[
            # Basic effects
            'flame', 'water', 'float', 'sparkle', 'sway', 'pulse', 'smoke', 'wobble',
            'glitch', 'shake', 'bounce', 'flicker', 'glow', 'dissolve', 'rainbow',
            'spin', 'melt', 'electric',
            # Status effects
            'shadow', 'teleport', 'charge', 'damage', 'freeze', 'poison',
            'petrify', 'hologram', 'chromatic', 'stretch', 'ripple', 'levitate',
            # Advanced effects
            'particles', 'motion_blur', 'speed_lines', 'motion_trail', 'ribbon_trail', 'keyframe',
            # Elemental effects (ADD visuals to sprite)
            'fire_element', 'water_element', 'ice_element',
            # Creature/wing effects
            'flap', 'hover_flap', 'glide_flap',
            # Window/light effects
            'window_flicker',
            # Color-targeted effects
            'color_pulse', 'shine',
            # Sunlight
            'sunlight'
        ],
        help='Animation effect (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '-f', '--frames',
        type=int,
        default=8,
        help='Number of animation frames (default: 8)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        default='gif',
        choices=['gif', 'spritesheet', 'frames'],
        help='Output format (default: gif)'
    )
    
    parser.add_argument(
        '-i', '--intensity',
        type=float,
        default=1.0,
        help='Effect intensity 0.0-2.0 (default: 1.0)'
    )
    
    parser.add_argument(
        '-s', '--speed',
        type=float,
        default=1.0,
        help='Animation speed multiplier (default: 1.0)'
    )
    
    parser.add_argument(
        '-q', '--quality',
        type=str,
        default='high',
        choices=['fast', 'high', 'best'],
        help='Rendering quality: fast (bilinear), high (gamma-correct), best (Lanczos)'
    )
    
    parser.add_argument(
        '--motion-blur',
        action='store_true',
        help='Enable motion blur for fast-moving effects (spin, shake)'
    )
    
    parser.add_argument(
        '--anticipation',
        type=float,
        default=0.0,
        help='Anticipation/windup amount 0.0-0.4 (e.g., 0.2 = 20%% windup before motion)'
    )
    
    parser.add_argument(
        '--overshoot',
        type=float,
        default=0.0,
        help='Overshoot/settle amount 0.0-0.3 (e.g., 0.15 = 15%% overshoot on settle)'
    )
    
    parser.add_argument(
        '--squash-stretch',
        type=float,
        default=0.0,
        help='Squash and stretch intensity 0.0-0.5 (deformation during motion)'
    )
    
    parser.add_argument(
        '--onion-skin',
        type=int,
        default=0,
        metavar='N',
        help='Export with onion skinning (N previous frames ghosted at 30%% opacity)'
    )
    
    parser.add_argument(
        '--onion-opacity',
        type=float,
        default=0.3,
        help='Onion skin base opacity 0.0-1.0 (default: 0.3)'
    )
    
    parser.add_argument(
        '--onion-tint',
        action='store_true',
        help='Tint onion skin frames (red=previous, blue=next)'
    )
    
    parser.add_argument(
        '--normal',
        type=str,
        default=None,
        metavar='PATH',
        help='Normal map file for 3D lighting effects (e.g., gem_n.png)'
    )
    
    parser.add_argument(
        '--light-dir',
        type=str,
        default='top-left',
        choices=['top-left', 'top', 'top-right', 'left', 'right', 
                 'bottom-left', 'bottom', 'bottom-right', 'front'],
        help='Light direction for normal-mapped effects (default: top-left)'
    )
    
    parser.add_argument(
        '--light-anim',
        type=str,
        default=None,
        choices=['orbit', 'pulse', 'color'],
        help='Animate light source (orbit=circle around, pulse=brightness, color=cycle colors)'
    )
    
    parser.add_argument(
        '--analyze',
        action='store_true',
        help="Only analyze sprite, don't animate"
    )
    
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Open real-time preview window (requires pygame)'
    )
    
    parser.add_argument(
        '--preview-zoom',
        type=float,
        default=4.0,
        help='Initial zoom level for preview (default: 4.0)'
    )
    
    parser.add_argument(
        '--preset',
        type=str,
        default=None,
        metavar='NAME',
        help='Use a preset configuration (e.g., torch_realistic, magic_crystal)'
    )
    
    parser.add_argument(
        '--list-presets',
        action='store_true',
        help='List all available presets and exit'
    )
    
    parser.add_argument(
        '--preset-info',
        type=str,
        default=None,
        metavar='NAME',
        help='Show detailed info about a preset and exit'
    )
    
    # Advanced effect parameters
    parser.add_argument(
        '--particle-type',
        type=str,
        default='spark',
        choices=['spark', 'dust', 'magic', 'fire', 'smoke', 'bubble', 'star', 'snow', 'rain', 'leaf', 'debris', 'energy'],
        help='Particle type for particles effect (default: spark)'
    )
    
    parser.add_argument(
        '--trail-style',
        type=str,
        default='afterimage',
        choices=['afterimage', 'ghost', 'echo', 'smear', 'stroboscopic'],
        help='Trail style for motion_trail effect (default: afterimage)'
    )
    
    parser.add_argument(
        '--trail-count',
        type=int,
        default=5,
        help='Number of trail copies (default: 5)'
    )
    
    parser.add_argument(
        '--blur-type',
        type=str,
        default='motion',
        choices=['linear', 'radial', 'zoom', 'directional', 'motion'],
        help='Motion blur type (default: motion)'
    )
    
    parser.add_argument(
        '--curve-preset',
        type=str,
        default='ease_in_out',
        choices=['linear', 'ease_in_out', 'bounce', 'overshoot', 'anticipation', 'elastic', 'heartbeat', 'breathing', 'squash_stretch', 'wobble'],
        help='Animation curve preset for keyframe effect (default: ease_in_out)'
    )
    
    # === PIXELATE COMMAND ===
    parser.add_argument(
        '--pixelate',
        action='store_true',
        help='Convert image to pixel art sprite (no animation)'
    )
    
    parser.add_argument(
        '--pixel-width',
        type=int,
        default=32,
        help='Output pixel art width (default: 32)'
    )
    
    parser.add_argument(
        '--pixel-height',
        type=int,
        default=0,
        help='Output pixel art height (0 = auto aspect ratio)'
    )
    
    parser.add_argument(
        '--colors',
        type=int,
        default=16,
        help='Number of colors in palette (default: 16)'
    )
    
    parser.add_argument(
        '--palette',
        type=str,
        default=None,
        choices=['gameboy', 'nes', 'pico8', 'endesga32', 'grayscale'],
        help='Use preset color palette'
    )
    
    parser.add_argument(
        '--dither',
        type=str,
        default='none',
        choices=['none', 'ordered', 'floyd'],
        help='Dithering style (default: none)'
    )
    
    parser.add_argument(
        '--pixel-outline',
        action='store_true',
        help='Add black outline to pixel art'
    )
    
    parser.add_argument(
        '--bg-tolerance',
        type=int,
        default=15,
        help='Background removal tolerance (0-255, higher = more aggressive, default: 15)'
    )
    
    parser.add_argument(
        '--no-remove-bg',
        action='store_true',
        help='Disable automatic background removal'
    )
    
    parser.add_argument(
        '--contrast',
        type=float,
        default=1.2,
        help='Contrast enhancement (1.0 = no change, default: 1.2)'
    )
    
    parser.add_argument(
        '--saturation',
        type=float,
        default=1.1,
        help='Saturation enhancement (1.0 = no change, default: 1.1)'
    )
    
    parser.add_argument(
        '--outline',
        type=float,
        default=0,
        metavar='THICKNESS',
        help='Add black outline to sprite (thickness in pixels, e.g., 1 or 2)'
    )
    
    parser.add_argument(
        '--outline-color',
        type=str,
        default='black',
        help='Outline color: black, white, or R,G,B (default: black)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed analysis'
    )
    
    args = parser.parse_args()
    
    # Handle preset listing/info (doesn't require input file)
    if args.list_presets:
        from src.core.presets import get_preset_manager
        manager = get_preset_manager()
        
        print("Available Effect Presets:\n")
        
        # Group by tags
        tags = manager.list_tags()
        for tag in ['fire', 'water', 'magic', 'nature', 'tech', 'ui', 'character', 'environment']:
            if tag not in tags:
                continue
            presets = manager.list_by_tag(tag)
            if presets:
                print(f"  [{tag.upper()}]")
                for name in presets:
                    preset = manager.get(name)
                    desc = preset.description[:50] + "..." if len(preset.description) > 50 else preset.description
                    print(f"    {name:<20} - {desc}")
                print()
        
        print(f"Total: {len(manager.list_all())} presets")
        print("\nUsage: --preset <name>")
        print("Details: --preset-info <name>")
        sys.exit(0)
    
    if args.preset_info:
        from src.core.presets import get_preset_manager
        manager = get_preset_manager()
        
        preset = manager.get(args.preset_info)
        if not preset:
            print(f"Error: Preset '{args.preset_info}' not found")
            print(f"Use --list-presets to see available presets")
            sys.exit(1)
        
        print(f"Preset: {preset.name}")
        print(f"Description: {preset.description}")
        print(f"\nSettings:")
        print(f"  Effect: {preset.effect}")
        print(f"  Frames: {preset.frames}")
        print(f"  Intensity: {preset.intensity}")
        print(f"  Speed: {preset.speed}")
        print(f"  Easing: {preset.easing}")
        
        if preset.secondary_effects:
            print(f"\nSecondary Effects:")
            for sec in preset.secondary_effects:
                print(f"  - {sec.get('effect')} (intensity: {sec.get('intensity', 1.0)})")
        
        if preset.color_ramp:
            print(f"\nColor Ramp: {', '.join(preset.color_ramp)}")
        
        if preset.motion_blur:
            print(f"\nMotion Blur: enabled ({preset.blur_samples} samples)")
        
        if any([preset.anticipation, preset.overshoot, preset.squash_stretch]):
            print(f"\nAnimation Principles:")
            if preset.anticipation:
                print(f"  Anticipation: {preset.anticipation}")
            if preset.overshoot:
                print(f"  Overshoot: {preset.overshoot}")
            if preset.squash_stretch:
                print(f"  Squash/Stretch: {preset.squash_stretch}")
        
        print(f"\nTags: {', '.join(preset.tags)}")
        sys.exit(0)
    
    # Check input is provided (required unless listing presets)
    if not args.input:
        print("Error: Input file is required")
        print("Usage: python main.py <input_image> [options]")
        print("       python main.py --list-presets")
        sys.exit(1)
    
    # Check input exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Apply preset if specified
    if args.preset:
        from src.core.presets import get_preset, apply_preset_to_args
        
        preset = get_preset(args.preset)
        if not preset:
            print(f"Error: Preset '{args.preset}' not found")
            print(f"Use --list-presets to see available presets")
            sys.exit(1)
        
        print(f"Using preset: {args.preset} ({preset.description})")
        args = apply_preset_to_args(preset, args)
    
    # Import here to avoid slow startup for --help
    from src import SpriteParser, SpriteAnalyzer, animate, analyze
    from src.core import preview_animation, check_pygame_available, SpriteExporter
    
    # Analyze mode
    if args.analyze:
        print(f"Analyzing: {args.input}\n")
        result = analyze(args.input)
        
        if args.verbose:
            print(result['explanation'])
        else:
            print(f"Best Effect: {result['best_effect'].upper()}")
            print(f"Confidence: {result['confidence']:.0%}")
            print("\nTop Suggestions:")
            for i, s in enumerate(result['suggestions'][:5], 1):
                print(f"  {i}. {s['effect']} ({s['confidence']:.0%})")
        
        return
    
    # === PIXELATE MODE ===
    if args.pixelate:
        from src.core.pixelate import pixelate_image, PALETTES
        
        print(f"Converting to pixel art: {args.input}")
        print(f"Output size: {args.pixel_width}x{args.pixel_height if args.pixel_height > 0 else 'auto'}")
        print(f"Colors: {args.colors}" + (f" ({args.palette} palette)" if args.palette else ""))
        print(f"Background removal: {'disabled' if args.no_remove_bg else f'enabled (tolerance: {args.bg_tolerance})'}")
        
        # Generate output path
        if args.output:
            output_path = args.output
        else:
            stem = input_path.stem
            output_path = str(input_path.parent / f"{stem}_pixel.png")
        
        result = pixelate_image(
            args.input,
            output_path,
            width=args.pixel_width,
            height=args.pixel_height,
            colors=args.colors,
            palette=args.palette,
            dither=args.dither,
            outline=args.pixel_outline,
            remove_background=not args.no_remove_bg,
            bg_tolerance=args.bg_tolerance,
            contrast=args.contrast,
            saturation=args.saturation
        )
        
        print(f"\nCreated: {output_path}")
        print(f"Size: {result.width}x{result.height} pixels")
        
        # Show preview if requested
        if args.preview:
            if not check_pygame_available():
                print("Error: Preview requires pygame. Install with: pip install pygame")
                sys.exit(1)
            
            import numpy as np
            frames = [np.array(result)]
            preview_animation(
                frames,
                original=None,
                fps=10,
                zoom=args.preview_zoom,
                title="Pixel Art Result"
            )
        
        return
    
    # Animation mode
    print(f"Animating: {args.input}")
    print(f"Quality: {args.quality}" + (" (with motion blur)" if args.motion_blur else ""))
    if args.onion_skin > 0:
        print(f"Onion skin: {args.onion_skin} frames at {args.onion_opacity:.0%} opacity")
    
    try:
        # Build extra config for quality settings
        extra_config = {
            'quality': args.quality,
            'motion_blur': args.motion_blur,
            'onion_skin': args.onion_skin,
            'onion_opacity': args.onion_opacity,
            'onion_tint': args.onion_tint,
        }
        
        # Preview mode - generate frames and open preview window
        if args.preview:
            if not check_pygame_available():
                print("Error: Preview requires pygame. Install with: pip install pygame")
                sys.exit(1)
            
            # Load sprite and generate frames without exporting
            from src.core import SpriteParser
            from src.detection import SpriteAnalyzer
            
            sprite = SpriteParser.parse(
                args.input, 
                remove_background=not args.no_remove_bg,
                bg_tolerance=args.bg_tolerance
            )
            
            # Get effect
            effect = args.effect
            if not effect:
                analyzer = SpriteAnalyzer()
                result = analyzer.analyze(sprite)
                effect = result['best_effect']
                print(f"Auto-detected effect: {effect}")
            
            # Import effect module and get effect class
            from src.procedural import get_effect, EffectConfig
            
            try:
                EffectClass = get_effect(effect)
            except ValueError as e:
                print(f"Error: {e}")
                sys.exit(1)
            
            # Create effect config based on effect type
            if effect == 'particles':
                from src.procedural.particles import ParticleEffectConfig
                config = ParticleEffectConfig(
                    frame_count=args.frames,
                    intensity=args.intensity,
                    speed=args.speed,
                    particle_type=args.particle_type
                )
            elif effect in ['motion_trail', 'afterimages', 'ghost_trail']:
                from src.procedural.trail import TrailConfig
                config = TrailConfig(
                    frame_count=args.frames,
                    intensity=args.intensity,
                    speed=args.speed,
                    style=args.trail_style,
                    trail_count=args.trail_count
                )
            elif effect in ['motion_blur', 'blur']:
                from src.procedural.motion_blur import MotionBlurConfig
                config = MotionBlurConfig(
                    frame_count=args.frames,
                    intensity=args.intensity,
                    blur_type=args.blur_type
                )
            elif effect == 'keyframe':
                from src.procedural.keyframes import KeyframeEffectConfig
                config = KeyframeEffectConfig(
                    frame_count=args.frames,
                    curve_preset=args.curve_preset,
                    value_min=0.0,
                    value_max=args.intensity
                )
            elif effect == 'fire_element':
                from src.procedural.elements import FireElementConfig
                config = FireElementConfig(
                    frame_count=args.frames,
                    intensity=args.intensity,
                    flame_height=0.5,
                    ember_count=8,
                    glow_intensity=0.6
                )
            elif effect == 'water_element':
                from src.procedural.elements import WaterElementConfig
                config = WaterElementConfig(
                    frame_count=args.frames,
                    intensity=args.intensity,
                    splash_intensity=0.6,
                    droplet_count=10,
                    ripple_count=3
                )
            elif effect == 'ice_element':
                from src.procedural.elements import IceElementConfig
                config = IceElementConfig(
                    frame_count=args.frames,
                    intensity=args.intensity,
                    crystal_count=6,
                    frost_intensity=0.5,
                    sparkle_count=12
                )
            elif effect in ['flap', 'hover_flap', 'glide_flap']:
                from src.procedural.flap import FlapConfig
                config = FlapConfig(
                    frame_count=args.frames,
                    intensity=args.intensity,
                    flap_angle=25.0,
                    flap_speed=1.0,
                    body_width=0.3,
                    vertical_bob=0.15
                )
            else:
                config = EffectConfig(
                    frame_count=args.frames,
                    intensity=args.intensity,
                    speed=args.speed
                )
            
            # Generate frames
            print(f"Generating {args.frames} frames...")
            effect_instance = EffectClass(config)
            frame_sprites = effect_instance.apply(sprite)
            frames = [s.pixels for s in frame_sprites]
            
            # Open preview window
            print(f"Opening preview window (zoom: {args.preview_zoom}x)...")
            print("Controls: SPACE=play/pause, ←/→=step, B=compare, H=help, ESC=quit")
            
            # Export callback
            def on_export(export_frames):
                output_path = args.output or str(input_path.stem) + "_animated.gif"
                exporter = SpriteExporter()
                if args.format == 'gif':
                    exporter.save_gif(export_frames, output_path, fps=int(12 * args.speed))
                elif args.format == 'spritesheet':
                    exporter.save_spritesheet(export_frames, output_path)
                else:
                    exporter.save_frames(export_frames, output_path)
                print(f"Exported: {output_path}")
            
            preview_animation(
                frames,
                original=[sprite.pixels] * len(frames),
                title=f"Preview: {input_path.name} ({effect})",
                fps=int(12 * args.speed),
                zoom=args.preview_zoom
            )
            print("Preview closed.")
            return
        
        # Parse outline color
        outline_color = (0, 0, 0)  # default black
        if args.outline > 0:
            if args.outline_color == 'black':
                outline_color = (0, 0, 0)
            elif args.outline_color == 'white':
                outline_color = (255, 255, 255)
            elif ',' in args.outline_color:
                outline_color = tuple(int(c) for c in args.outline_color.split(','))
        
        output = animate(
            args.input,
            effect=args.effect,
            output_path=args.output,
            frames=args.frames,
            format=args.format,
            intensity=args.intensity,
            speed=args.speed,
            extra=extra_config,
            remove_background=not args.no_remove_bg,
            bg_tolerance=args.bg_tolerance,
            outline=args.outline,
            outline_color=outline_color
        )
        
        print(f"Output: {output}")
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
