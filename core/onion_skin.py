"""
Onion Skinning Export

Creates frames with ghosted previous/next frames for hand-tweaking in
external editors like Aseprite, GraphicsGale, or Photoshop.

Features:
- Configurable number of onion skin frames (before and/or after)
- Adjustable opacity falloff
- Color tinting for previous vs next frames
- Multiple blend modes
- Export as layered files or flattened frames

Example:
    python main.py sprite.png --effect flame --onion-skin 3
    # Exports frames with previous 3 frames ghosted at decreasing opacity
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# Constants
# =============================================================================

GAMMA = 2.2
INV_GAMMA = 1.0 / GAMMA


# =============================================================================
# Onion Skin Configuration
# =============================================================================

class OnionBlendMode(Enum):
    """Blend modes for onion skin layers"""
    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"


class OnionTintMode(Enum):
    """Color tinting modes for distinguishing frames"""
    NONE = "none"
    RED_BLUE = "red_blue"       # Previous=red, Next=blue
    WARM_COOL = "warm_cool"     # Previous=orange, Next=cyan
    CUSTOM = "custom"


@dataclass
class OnionSkinConfig:
    """Configuration for onion skinning"""
    # Number of frames to show
    frames_before: int = 3       # Previous frames
    frames_after: int = 0        # Next frames (0 = previous only)
    
    # Opacity settings
    base_opacity: float = 0.3    # Opacity of nearest onion frame
    opacity_falloff: float = 0.5 # Multiplier per frame distance (0.5 = halves each frame)
    
    # Visual settings
    blend_mode: OnionBlendMode = OnionBlendMode.NORMAL
    tint_mode: OnionTintMode = OnionTintMode.RED_BLUE
    
    # Custom tint colors (RGB 0-255)
    tint_previous: Tuple[int, int, int] = (255, 100, 100)  # Reddish
    tint_next: Tuple[int, int, int] = (100, 100, 255)      # Bluish
    tint_strength: float = 0.3   # How strong the tint is (0-1)
    
    # Output options
    include_clean_layer: bool = True   # Include untinted current frame layer
    flatten_output: bool = True        # Flatten to single image (vs keep layers)
    
    @classmethod
    def default(cls) -> 'OnionSkinConfig':
        """Default config for typical hand-tweaking workflow"""
        return cls()
    
    @classmethod
    def for_review(cls) -> 'OnionSkinConfig':
        """Config for reviewing animation flow (more frames, subtle)"""
        return cls(
            frames_before=5,
            frames_after=2,
            base_opacity=0.2,
            opacity_falloff=0.6,
            tint_strength=0.2
        )
    
    @classmethod
    def for_editing(cls) -> 'OnionSkinConfig':
        """Config for detailed editing (fewer frames, stronger)"""
        return cls(
            frames_before=2,
            frames_after=1,
            base_opacity=0.4,
            opacity_falloff=0.5,
            tint_strength=0.4
        )


# =============================================================================
# Core Functions
# =============================================================================

def to_linear(srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB to linear color space"""
    return (srgb / 255.0) ** GAMMA


def to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear to sRGB"""
    return (np.clip(linear, 0, 1) ** INV_GAMMA * 255).astype(np.uint8)


def apply_tint(
    pixels: np.ndarray,
    tint_color: Tuple[int, int, int],
    strength: float
) -> np.ndarray:
    """Apply color tint to an image"""
    result = pixels.copy().astype(np.float32)
    
    # Only tint RGB, preserve alpha
    tint = np.array(tint_color, dtype=np.float32)
    
    # Blend toward tint color
    result[:, :, 0] = result[:, :, 0] * (1 - strength) + tint[0] * strength
    result[:, :, 1] = result[:, :, 1] * (1 - strength) + tint[1] * strength
    result[:, :, 2] = result[:, :, 2] * (1 - strength) + tint[2] * strength
    
    return np.clip(result, 0, 255).astype(np.uint8)


def blend_alpha(
    base: np.ndarray,
    overlay: np.ndarray,
    opacity: float = 1.0,
    mode: OnionBlendMode = OnionBlendMode.NORMAL
) -> np.ndarray:
    """
    Blend overlay onto base with alpha compositing.
    
    Args:
        base: Base RGBA image
        overlay: Overlay RGBA image
        opacity: Additional opacity multiplier (0-1)
        mode: Blend mode
    
    Returns:
        Blended RGBA image
    """
    result = base.copy().astype(np.float32)
    over = overlay.astype(np.float32)
    
    # Get alpha channels
    base_alpha = result[:, :, 3] / 255.0
    over_alpha = (over[:, :, 3] / 255.0) * opacity
    
    # Compute output alpha
    out_alpha = over_alpha + base_alpha * (1 - over_alpha)
    
    # Avoid division by zero
    safe_alpha = np.where(out_alpha > 0, out_alpha, 1)
    
    # Blend RGB based on mode
    for c in range(3):
        base_c = result[:, :, c]
        over_c = over[:, :, c]
        
        if mode == OnionBlendMode.NORMAL:
            blended = over_c
        elif mode == OnionBlendMode.MULTIPLY:
            blended = (base_c * over_c) / 255.0
        elif mode == OnionBlendMode.SCREEN:
            blended = 255 - ((255 - base_c) * (255 - over_c)) / 255.0
        elif mode == OnionBlendMode.OVERLAY:
            low = (2 * base_c * over_c) / 255.0
            high = 255 - 2 * ((255 - base_c) * (255 - over_c)) / 255.0
            blended = np.where(base_c < 128, low, high)
        else:
            blended = over_c
        
        # Alpha compositing
        result[:, :, c] = (blended * over_alpha + base_c * base_alpha * (1 - over_alpha)) / safe_alpha
    
    result[:, :, 3] = out_alpha * 255
    
    return np.clip(result, 0, 255).astype(np.uint8)


def create_onion_frame(
    frames: List[np.ndarray],
    current_index: int,
    config: OnionSkinConfig
) -> np.ndarray:
    """
    Create a single frame with onion skinning applied.
    
    Args:
        frames: List of all animation frames (RGBA arrays)
        current_index: Index of the current frame
        config: Onion skin configuration
    
    Returns:
        RGBA image with onion skins composited
    """
    if not frames:
        raise ValueError("No frames provided")
    
    current_frame = frames[current_index]
    h, w = current_frame.shape[:2]
    
    # Start with transparent canvas
    result = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Collect onion layers with their opacities
    layers = []
    
    # Previous frames (furthest first, so they're under closer frames)
    for i in range(config.frames_before, 0, -1):
        frame_idx = current_index - i
        if 0 <= frame_idx < len(frames):
            opacity = config.base_opacity * (config.opacity_falloff ** (i - 1))
            
            # Apply tint if enabled
            frame = frames[frame_idx].copy()
            if config.tint_mode != OnionTintMode.NONE:
                if config.tint_mode == OnionTintMode.RED_BLUE:
                    tint = (255, 100, 100)  # Red for previous
                elif config.tint_mode == OnionTintMode.WARM_COOL:
                    tint = (255, 180, 100)  # Orange for previous
                else:
                    tint = config.tint_previous
                
                frame = apply_tint(frame, tint, config.tint_strength)
            
            layers.append((frame, opacity, 'previous'))
    
    # Next frames (furthest first)
    for i in range(config.frames_after, 0, -1):
        frame_idx = current_index + i
        if 0 <= frame_idx < len(frames):
            opacity = config.base_opacity * (config.opacity_falloff ** (i - 1))
            
            # Apply tint if enabled
            frame = frames[frame_idx].copy()
            if config.tint_mode != OnionTintMode.NONE:
                if config.tint_mode == OnionTintMode.RED_BLUE:
                    tint = (100, 100, 255)  # Blue for next
                elif config.tint_mode == OnionTintMode.WARM_COOL:
                    tint = (100, 200, 255)  # Cyan for next
                else:
                    tint = config.tint_next
                
                frame = apply_tint(frame, tint, config.tint_strength)
            
            layers.append((frame, opacity, 'next'))
    
    # Composite onion layers onto result
    for frame, opacity, direction in layers:
        result = blend_alpha(result, frame, opacity, config.blend_mode)
    
    # Add current frame on top (full opacity)
    result = blend_alpha(result, current_frame, 1.0, OnionBlendMode.NORMAL)
    
    return result


def create_onion_animation(
    frames: List[np.ndarray],
    config: Optional[OnionSkinConfig] = None
) -> List[np.ndarray]:
    """
    Create an entire animation with onion skinning applied to each frame.
    
    Args:
        frames: List of animation frames (RGBA arrays)
        config: Onion skin configuration (uses default if None)
    
    Returns:
        List of frames with onion skinning applied
    """
    if config is None:
        config = OnionSkinConfig.default()
    
    result_frames = []
    
    for i in range(len(frames)):
        onion_frame = create_onion_frame(frames, i, config)
        result_frames.append(onion_frame)
    
    return result_frames


def create_onion_layers(
    frames: List[np.ndarray],
    current_index: int,
    config: Optional[OnionSkinConfig] = None
) -> List[Tuple[np.ndarray, str, float]]:
    """
    Create separate onion skin layers for export to layered formats.
    
    Returns list of (image, layer_name, opacity) tuples.
    Layers are ordered bottom to top.
    
    Args:
        frames: List of all animation frames
        current_index: Index of the current frame
        config: Onion skin configuration
    
    Returns:
        List of (pixels, name, opacity) tuples
    """
    if config is None:
        config = OnionSkinConfig.default()
    
    layers = []
    
    # Previous frames (furthest = lowest layer)
    for i in range(config.frames_before, 0, -1):
        frame_idx = current_index - i
        if 0 <= frame_idx < len(frames):
            opacity = config.base_opacity * (config.opacity_falloff ** (i - 1))
            frame = frames[frame_idx].copy()
            
            if config.tint_mode != OnionTintMode.NONE:
                if config.tint_mode == OnionTintMode.RED_BLUE:
                    tint = (255, 100, 100)
                elif config.tint_mode == OnionTintMode.WARM_COOL:
                    tint = (255, 180, 100)
                else:
                    tint = config.tint_previous
                frame = apply_tint(frame, tint, config.tint_strength)
            
            layers.append((frame, f"Previous {i}", opacity))
    
    # Next frames (furthest = lowest among next frames)
    for i in range(config.frames_after, 0, -1):
        frame_idx = current_index + i
        if 0 <= frame_idx < len(frames):
            opacity = config.base_opacity * (config.opacity_falloff ** (i - 1))
            frame = frames[frame_idx].copy()
            
            if config.tint_mode != OnionTintMode.NONE:
                if config.tint_mode == OnionTintMode.RED_BLUE:
                    tint = (100, 100, 255)
                elif config.tint_mode == OnionTintMode.WARM_COOL:
                    tint = (100, 200, 255)
                else:
                    tint = config.tint_next
                frame = apply_tint(frame, tint, config.tint_strength)
            
            layers.append((frame, f"Next {i}", opacity))
    
    # Current frame on top
    layers.append((frames[current_index].copy(), f"Frame {current_index + 1}", 1.0))
    
    return layers


# =============================================================================
# Export Functions
# =============================================================================

def export_onion_frames(
    frames: List[np.ndarray],
    output_dir: str,
    config: Optional[OnionSkinConfig] = None,
    prefix: str = "onion"
) -> List[str]:
    """
    Export animation frames with onion skinning as individual PNGs.
    
    Args:
        frames: List of animation frames
        output_dir: Output directory path
        config: Onion skin configuration
        prefix: Filename prefix
    
    Returns:
        List of output file paths
    """
    from pathlib import Path
    from PIL import Image
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    onion_frames = create_onion_animation(frames, config)
    
    output_files = []
    for i, frame in enumerate(onion_frames):
        filename = f"{prefix}_{i:04d}.png"
        filepath = output_path / filename
        
        img = Image.fromarray(frame, 'RGBA')
        img.save(filepath, 'PNG')
        output_files.append(str(filepath))
    
    return output_files


def export_onion_spritesheet(
    frames: List[np.ndarray],
    output_path: str,
    config: Optional[OnionSkinConfig] = None,
    columns: Optional[int] = None,
    padding: int = 0
) -> str:
    """
    Export animation with onion skinning as a spritesheet.
    
    Args:
        frames: List of animation frames
        output_path: Output file path
        config: Onion skin configuration
        columns: Number of columns (auto if None)
        padding: Padding between frames
    
    Returns:
        Output file path
    """
    from pathlib import Path
    from PIL import Image
    
    onion_frames = create_onion_animation(frames, config)
    
    if not onion_frames:
        raise ValueError("No frames to export")
    
    frame_count = len(onion_frames)
    h, w = onion_frames[0].shape[:2]
    
    # Calculate grid
    if columns is None:
        columns = min(frame_count, 8)
    rows = (frame_count + columns - 1) // columns
    
    # Create spritesheet
    sheet_w = columns * (w + padding) - padding
    sheet_h = rows * (h + padding) - padding
    
    sheet = np.zeros((sheet_h, sheet_w, 4), dtype=np.uint8)
    
    for i, frame in enumerate(onion_frames):
        row = i // columns
        col = i % columns
        x = col * (w + padding)
        y = row * (h + padding)
        sheet[y:y+h, x:x+w] = frame
    
    # Save
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    img = Image.fromarray(sheet, 'RGBA')
    img.save(path, 'PNG')
    
    return str(path)


def export_onion_gif(
    frames: List[np.ndarray],
    output_path: str,
    config: Optional[OnionSkinConfig] = None,
    duration: int = 100,
    loop: int = 0
) -> str:
    """
    Export animation with onion skinning as a GIF.
    
    Args:
        frames: List of animation frames
        output_path: Output file path
        config: Onion skin configuration
        duration: Frame duration in milliseconds
        loop: Loop count (0 = infinite)
    
    Returns:
        Output file path
    """
    from pathlib import Path
    from PIL import Image
    
    onion_frames = create_onion_animation(frames, config)
    
    images = []
    for frame in onion_frames:
        img = Image.fromarray(frame, 'RGBA')
        img = img.convert('P', palette=Image.ADAPTIVE, colors=256)
        images.append(img)
    
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        transparency=0,
        disposal=2
    )
    
    return str(path)


# =============================================================================
# Convenience Functions
# =============================================================================

def add_onion_skin(
    frames: List[np.ndarray],
    num_frames: int = 3,
    opacity: float = 0.3,
    tinted: bool = True
) -> List[np.ndarray]:
    """
    Simple function to add onion skinning to animation frames.
    
    Args:
        frames: List of animation frames
        num_frames: Number of previous frames to show
        opacity: Base opacity for onion frames
        tinted: Whether to tint onion frames
    
    Returns:
        Frames with onion skinning applied
    """
    config = OnionSkinConfig(
        frames_before=num_frames,
        frames_after=0,
        base_opacity=opacity,
        tint_mode=OnionTintMode.RED_BLUE if tinted else OnionTintMode.NONE
    )
    
    return create_onion_animation(frames, config)


def preview_onion_skin(
    frames: List[np.ndarray],
    frame_index: int,
    num_before: int = 3,
    num_after: int = 1
) -> np.ndarray:
    """
    Preview onion skinning for a specific frame.
    
    Args:
        frames: List of animation frames
        frame_index: Which frame to preview
        num_before: Number of previous frames to show
        num_after: Number of next frames to show
    
    Returns:
        Single frame with onion skinning
    """
    config = OnionSkinConfig(
        frames_before=num_before,
        frames_after=num_after
    )
    
    return create_onion_frame(frames, frame_index, config)
