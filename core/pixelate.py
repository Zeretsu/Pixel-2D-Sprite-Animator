"""
Pixel Art Converter - Convert images to pixel-perfect sprites

Takes any image and converts it to clean pixel art with:
- Automatic background detection and removal
- Configurable output size
- Color palette reduction
- Dithering options
- Edge enhancement for crisp pixels
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path
from collections import Counter


@dataclass
class PixelateConfig:
    """Configuration for pixel art conversion."""
    # Output size
    width: int = 32           # Target width in pixels
    height: int = 0           # Target height (0 = auto aspect ratio)
    
    # Color settings
    max_colors: int = 16      # Max colors in palette (2-256)
    palette: Optional[List[Tuple[int,int,int]]] = None  # Custom palette
    
    # Dithering
    dither: str = 'none'      # 'none', 'ordered', 'floyd', 'bayer'
    dither_strength: float = 0.5
    
    # Quality enhancement
    sharpen: bool = False     # Sharpen edges (disabled by default, can cause artifacts)
    enhance_contrast: float = 1.2  # Boost contrast (1.0 = no change)
    enhance_saturation: float = 1.1  # Boost saturation
    
    # Outline
    outline: bool = False     # Add outline
    outline_color: Tuple[int,int,int] = (0, 0, 0)
    
    # Background removal
    remove_background: bool = True   # Auto-detect and remove background
    bg_tolerance: int = 30           # Color tolerance for background detection
    edge_feather: int = 0            # Feather edges (0 = hard edges)


class PixelArtConverter:
    """Convert any image to pixel art sprite."""
    
    def __init__(self, config: Optional[PixelateConfig] = None):
        self.config = config or PixelateConfig()
    
    def convert(self, image_path: str, output_path: Optional[str] = None) -> Image.Image:
        """Convert image to pixel art."""
        # Load image
        img = Image.open(image_path).convert('RGBA')
        
        # Calculate output dimensions
        out_w = self.config.width
        if self.config.height == 0:
            # Auto aspect ratio
            aspect = img.height / img.width
            out_h = max(1, int(out_w * aspect))
        else:
            out_h = self.config.height
        
        # Step 1: Remove background BEFORE resizing for better detection
        if self.config.remove_background:
            img = self._remove_background(img)
        
        # Step 2: Enhance image before resizing
        img = self._enhance_image(img)
        
        # Step 3: High-quality resize
        resized = self._smart_resize(img, out_w, out_h)
        
        # Step 4: Reduce colors (preserving alpha)
        if self.config.palette:
            quantized = self._apply_custom_palette(resized)
        else:
            quantized = self._quantize_colors(resized)
        
        # Step 5: Apply dithering if requested
        if self.config.dither != 'none':
            quantized = self._apply_dithering(resized, quantized)
        
        # Step 6: Clean up semi-transparent pixels
        quantized = self._clean_alpha(quantized)
        
        # Step 7: Sharpen if requested
        if self.config.sharpen:
            quantized = self._sharpen_pixels(quantized)
        
        # Step 8: Add outline if requested
        if self.config.outline:
            quantized = self._add_outline(quantized)
        
        # Save if output path provided
        if output_path:
            quantized.save(output_path)
            print(f"Saved pixel art to: {output_path}")
        
        return quantized
    
    def _remove_background(self, img: Image.Image) -> Image.Image:
        """Automatically detect and remove background."""
        arr = np.array(img)
        h, w = arr.shape[:2]
        
        # Sample corners and edges to detect background color
        samples = []
        margin = max(1, min(5, w // 20, h // 20))
        
        # Top edge
        samples.extend([tuple(arr[0, x, :3]) for x in range(0, w, max(1, w // 10))])
        # Bottom edge
        samples.extend([tuple(arr[h-1, x, :3]) for x in range(0, w, max(1, w // 10))])
        # Left edge
        samples.extend([tuple(arr[y, 0, :3]) for y in range(0, h, max(1, h // 10))])
        # Right edge
        samples.extend([tuple(arr[y, w-1, :3]) for y in range(0, h, max(1, h // 10))])
        # Corners (multiple samples)
        for cy in [0, 1, 2, h-3, h-2, h-1]:
            for cx in [0, 1, 2, w-3, w-2, w-1]:
                if 0 <= cy < h and 0 <= cx < w:
                    samples.append(tuple(arr[cy, cx, :3]))
        
        # Find most common color in samples (the background)
        color_counts = Counter(samples)
        bg_color = np.array(color_counts.most_common(1)[0][0], dtype=np.float32)
        
        # Calculate color distance from background for each pixel
        rgb = arr[:, :, :3].astype(np.float32)
        distance = np.sqrt(np.sum((rgb - bg_color) ** 2, axis=2))
        
        # Create mask: pixels similar to background become transparent
        tolerance = self.config.bg_tolerance
        bg_mask = distance < tolerance
        
        # Expand background detection from edges using flood fill concept
        # Start from edges and grow inward
        visited = np.zeros((h, w), dtype=bool)
        to_remove = np.zeros((h, w), dtype=bool)
        
        # Initialize with edge pixels that match background
        stack = []
        for x in range(w):
            if bg_mask[0, x]:
                stack.append((0, x))
            if bg_mask[h-1, x]:
                stack.append((h-1, x))
        for y in range(h):
            if bg_mask[y, 0]:
                stack.append((y, 0))
            if bg_mask[y, w-1]:
                stack.append((y, w-1))
        
        # Flood fill from edges
        while stack:
            cy, cx = stack.pop()
            if visited[cy, cx]:
                continue
            visited[cy, cx] = True
            
            if bg_mask[cy, cx]:
                to_remove[cy, cx] = True
                # Add neighbors
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                        stack.append((ny, nx))
        
        # Apply mask
        result = arr.copy()
        result[to_remove, 3] = 0
        
        # Also handle any existing alpha
        result[arr[:, :, 3] < 128, 3] = 0
        
        return Image.fromarray(result, 'RGBA')
    
    def _enhance_image(self, img: Image.Image) -> Image.Image:
        """Enhance contrast and saturation for better pixel art."""
        # Preserve alpha
        if img.mode == 'RGBA':
            alpha = np.array(img)[:, :, 3]
            rgb = img.convert('RGB')
        else:
            alpha = None
            rgb = img
        
        # Enhance contrast
        if self.config.enhance_contrast != 1.0:
            enhancer = ImageEnhance.Contrast(rgb)
            rgb = enhancer.enhance(self.config.enhance_contrast)
        
        # Enhance saturation
        if self.config.enhance_saturation != 1.0:
            enhancer = ImageEnhance.Color(rgb)
            rgb = enhancer.enhance(self.config.enhance_saturation)
        
        # Restore alpha
        if alpha is not None:
            result = rgb.convert('RGBA')
            arr = np.array(result)
            arr[:, :, 3] = alpha
            return Image.fromarray(arr, 'RGBA')
        
        return rgb
    
    def _smart_resize(self, img: Image.Image, width: int, height: int) -> Image.Image:
        """Resize with best quality for pixel art."""
        # For downscaling, use box filter first then LANCZOS
        # This preserves more detail
        orig_w, orig_h = img.size
        
        if orig_w > width * 4 or orig_h > height * 4:
            # Large downscale: use intermediate step with BOX filter
            # BOX is better for large reductions (averages pixels)
            intermediate_w = max(width * 2, orig_w // 2)
            intermediate_h = max(height * 2, orig_h // 2)
            img = img.resize((intermediate_w, intermediate_h), Image.Resampling.BOX)
        
        # Final resize with LANCZOS for sharp edges
        resized = img.resize((width, height), Image.Resampling.LANCZOS)
        return resized
    
    def _quantize_colors(self, img: Image.Image) -> Image.Image:
        """Reduce image to limited color palette with better quality."""
        arr = np.array(img)
        h, w = arr.shape[:2]
        
        # Separate alpha
        alpha = arr[:, :, 3].copy()
        visible_mask = alpha > 128
        
        # Only quantize visible pixels
        if not np.any(visible_mask):
            return img
        
        # Get RGB of visible pixels only
        rgb = arr[:, :, :3]
        
        # Use k-means style quantization for better color selection
        # Sample colors from visible pixels only
        visible_colors = rgb[visible_mask].reshape(-1, 3)
        
        if len(visible_colors) == 0:
            return img
        
        # Get unique colors and their counts
        unique_colors, counts = np.unique(visible_colors, axis=0, return_counts=True)
        
        # If already few colors, no need to quantize
        if len(unique_colors) <= self.config.max_colors:
            return img
        
        # Use median cut quantization on just the visible pixels
        # Create a temporary image with just visible colors
        temp_rgb = Image.fromarray(visible_colors.reshape(1, -1, 3).astype(np.uint8), 'RGB')
        
        quantized_temp = temp_rgb.quantize(
            colors=self.config.max_colors,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE
        )
        
        # Get the palette
        palette = quantized_temp.getpalette()[:self.config.max_colors * 3]
        palette_colors = np.array(palette).reshape(-1, 3).astype(np.float32)
        
        # Map all visible pixels to nearest palette color (vectorized)
        visible_float = visible_colors.astype(np.float32)
        
        # Calculate distance to each palette color
        # Shape: (n_pixels, n_colors)
        distances = np.zeros((len(visible_colors), len(palette_colors)))
        for i, pc in enumerate(palette_colors):
            distances[:, i] = np.sum((visible_float - pc) ** 2, axis=1)
        
        nearest_idx = np.argmin(distances, axis=1)
        new_colors = palette_colors[nearest_idx].astype(np.uint8)
        
        # Build result
        result = arr.copy()
        result[visible_mask, :3] = new_colors.reshape(-1, 3)
        
        return Image.fromarray(result, 'RGBA')
    
    def _apply_custom_palette(self, img: Image.Image) -> Image.Image:
        """Apply a custom color palette (vectorized)."""
        arr = np.array(img)
        h, w = arr.shape[:2]
        alpha = arr[:, :, 3]
        visible_mask = alpha > 128
        
        if not np.any(visible_mask):
            return img
        
        # Get RGB of visible pixels
        rgb = arr[:, :, :3]
        visible_colors = rgb[visible_mask].astype(np.float32)
        
        # Palette as array
        palette = np.array(self.config.palette, dtype=np.float32)
        
        # Vectorized nearest neighbor
        distances = np.zeros((len(visible_colors), len(palette)))
        for i, pc in enumerate(palette):
            distances[:, i] = np.sum((visible_colors - pc) ** 2, axis=1)
        
        nearest_idx = np.argmin(distances, axis=1)
        new_colors = palette[nearest_idx].astype(np.uint8)
        
        # Build result
        result = arr.copy()
        result[visible_mask, :3] = new_colors
        
        return Image.fromarray(result, 'RGBA')
    
    def _apply_dithering(self, original: Image.Image, quantized: Image.Image) -> Image.Image:
        """Apply dithering based on config."""
        if self.config.dither == 'ordered' or self.config.dither == 'bayer':
            return self._ordered_dither(original, quantized)
        elif self.config.dither == 'floyd':
            return self._floyd_steinberg_dither(original)
        return quantized
    
    def _ordered_dither(self, original: Image.Image, quantized: Image.Image) -> Image.Image:
        """Apply ordered (Bayer matrix) dithering."""
        # 4x4 Bayer matrix
        bayer = np.array([
            [0,  8,  2,  10],
            [12, 4,  14, 6],
            [3,  11, 1,  9],
            [15, 7,  13, 5]
        ]) / 16.0 - 0.5
        
        orig_arr = np.array(original).astype(np.float32)
        quant_arr = np.array(quantized).astype(np.float32)
        h, w = orig_arr.shape[:2]
        
        # Preserve alpha
        alpha = orig_arr[:, :, 3].copy() if orig_arr.shape[2] == 4 else None
        
        # Tile the bayer matrix
        bayer_tiled = np.tile(bayer, (h // 4 + 1, w // 4 + 1))[:h, :w]
        
        # Apply threshold modulation
        strength = self.config.dither_strength * 32
        for c in range(3):
            threshold = bayer_tiled * strength
            diff = orig_arr[:, :, c] - quant_arr[:, :, c]
            mask = diff > threshold
            quant_arr[:, :, c] = np.where(
                mask,
                np.clip(quant_arr[:, :, c] + 16, 0, 255),
                quant_arr[:, :, c]
            )
        
        result = quant_arr.astype(np.uint8)
        
        if alpha is not None:
            result[:, :, 3] = alpha.astype(np.uint8)
            return Image.fromarray(result, 'RGBA')
        return Image.fromarray(result[:, :, :3], 'RGB')
    
    def _floyd_steinberg_dither(self, img: Image.Image) -> Image.Image:
        """Apply Floyd-Steinberg error diffusion dithering."""
        arr = np.array(img)
        alpha = arr[:, :, 3].copy() if img.mode == 'RGBA' else None
        
        rgb = img.convert('RGB')
        
        quantized = rgb.quantize(
            colors=self.config.max_colors,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.FLOYDSTEINBERG
        ).convert('RGB')
        
        if alpha is not None:
            quantized = quantized.convert('RGBA')
            arr = np.array(quantized)
            arr[:, :, 3] = alpha
            quantized = Image.fromarray(arr)
        
        return quantized
    
    def _clean_alpha(self, img: Image.Image) -> Image.Image:
        """Clean up alpha channel - make pixels fully opaque or transparent."""
        arr = np.array(img)
        
        # Threshold alpha to binary
        arr[:, :, 3] = np.where(arr[:, :, 3] > 128, 255, 0).astype(np.uint8)
        
        return Image.fromarray(arr, 'RGBA')
    
    def _sharpen_pixels(self, img: Image.Image) -> Image.Image:
        """Sharpen to make pixels more distinct."""
        arr = np.array(img)
        alpha = arr[:, :, 3].copy()
        
        # Simple contrast enhancement on colors
        for c in range(3):
            channel = arr[:, :, c].astype(np.float32)
            # Snap to nearest values for cleaner look
            snap = 16
            channel = np.round(channel / snap) * snap
            arr[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
        
        arr[:, :, 3] = alpha
        return Image.fromarray(arr, 'RGBA')
    
    def _add_outline(self, img: Image.Image) -> Image.Image:
        """Add outline around non-transparent pixels."""
        arr = np.array(img.convert('RGBA'))
        h, w = arr.shape[:2]
        
        # Find visible pixels
        visible = arr[:, :, 3] > 128
        
        # Create outline mask (pixels adjacent to visible but not visible)
        outline_mask = np.zeros((h, w), dtype=bool)
        
        # Check all 4 directions
        outline_mask[1:, :] |= visible[:-1, :] & ~visible[1:, :]
        outline_mask[:-1, :] |= visible[1:, :] & ~visible[:-1, :]
        outline_mask[:, 1:] |= visible[:, :-1] & ~visible[:, 1:]
        outline_mask[:, :-1] |= visible[:, 1:] & ~visible[:, :-1]
        
        # Apply outline color
        arr[outline_mask, 0] = self.config.outline_color[0]
        arr[outline_mask, 1] = self.config.outline_color[1]
        arr[outline_mask, 2] = self.config.outline_color[2]
        arr[outline_mask, 3] = 255
        
        return Image.fromarray(arr, 'RGBA')


# === PRESET PALETTES ===

PALETTES = {
    'gameboy': [
        (15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15)
    ],
    'nes': [
        (0, 0, 0), (255, 255, 255), (124, 124, 124), (188, 188, 188),
        (0, 120, 248), (0, 88, 248), (104, 68, 252), (216, 0, 204),
        (228, 0, 88), (248, 56, 0), (228, 92, 16), (172, 124, 0),
        (0, 184, 0), (0, 168, 0), (0, 168, 68), (0, 136, 136)
    ],
    'pico8': [
        (0, 0, 0), (29, 43, 83), (126, 37, 83), (0, 135, 81),
        (171, 82, 54), (95, 87, 79), (194, 195, 199), (255, 241, 232),
        (255, 0, 77), (255, 163, 0), (255, 236, 39), (0, 228, 54),
        (41, 173, 255), (131, 118, 156), (255, 119, 168), (255, 204, 170)
    ],
    'endesga32': [
        (190, 74, 47), (215, 118, 67), (234, 212, 170), (228, 166, 114),
        (184, 111, 80), (115, 62, 57), (62, 39, 49), (162, 38, 51),
        (228, 59, 68), (247, 118, 34), (254, 174, 52), (254, 231, 97),
        (99, 199, 77), (62, 137, 72), (38, 92, 66), (25, 60, 62),
        (18, 78, 137), (0, 149, 233), (44, 232, 245), (192, 203, 220),
        (139, 155, 180), (90, 105, 136), (58, 68, 102), (38, 43, 68),
        (24, 20, 37), (255, 0, 68), (104, 56, 108), (181, 80, 136),
        (246, 117, 122), (232, 183, 150), (194, 133, 105), (104, 71, 86)
    ],
    'grayscale': [
        (0, 0, 0), (34, 34, 34), (68, 68, 68), (102, 102, 102),
        (136, 136, 136), (170, 170, 170), (204, 204, 204), (255, 255, 255)
    ]
}


def pixelate_image(
    input_path: str,
    output_path: Optional[str] = None,
    width: int = 32,
    height: int = 0,
    colors: int = 16,
    palette: Optional[str] = None,
    dither: str = 'none',
    outline: bool = False,
    remove_background: bool = True,
    bg_tolerance: int = 30,
    contrast: float = 1.2,
    saturation: float = 1.1
) -> Image.Image:
    """
    Quick function to convert an image to pixel art.
    
    Args:
        input_path: Path to input image
        output_path: Path to save output (optional)
        width: Output width in pixels
        height: Output height (0 = auto aspect ratio)
        colors: Number of colors (ignored if palette specified)
        palette: Palette name ('gameboy', 'nes', 'pico8', 'endesga32', 'grayscale')
        dither: Dithering type ('none', 'ordered', 'floyd')
        outline: Add black outline
        remove_background: Auto-detect and remove background
        bg_tolerance: Color tolerance for background detection (0-255)
        contrast: Contrast enhancement (1.0 = no change)
        saturation: Saturation enhancement (1.0 = no change)
    
    Returns:
        PIL Image of the pixel art
    """
    config = PixelateConfig(
        width=width,
        height=height,
        max_colors=colors,
        palette=PALETTES.get(palette) if palette else None,
        dither=dither,
        outline=outline,
        remove_background=remove_background,
        bg_tolerance=bg_tolerance,
        enhance_contrast=contrast,
        enhance_saturation=saturation
    )
    
    converter = PixelArtConverter(config)
    return converter.convert(input_path, output_path)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pixelate.py <image> [--width 32] [--colors 16] [--palette pico8] [--dither ordered]")
        sys.exit(1)
    
    # Simple CLI
    import argparse
    parser = argparse.ArgumentParser(description='Convert image to pixel art')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('-o', '--output', help='Output path')
    parser.add_argument('-w', '--width', type=int, default=32, help='Output width')
    parser.add_argument('--height', type=int, default=0, help='Output height (0=auto)')
    parser.add_argument('-c', '--colors', type=int, default=16, help='Number of colors')
    parser.add_argument('-p', '--palette', choices=list(PALETTES.keys()), help='Use preset palette')
    parser.add_argument('-d', '--dither', choices=['none', 'ordered', 'floyd'], default='none')
    parser.add_argument('--outline', action='store_true', help='Add outline')
    parser.add_argument('--no-transparent', action='store_true', help='Keep background')
    
    args = parser.parse_args()
    
    output = args.output or args.input.rsplit('.', 1)[0] + '_pixel.png'
    
    result = pixelate_image(
        args.input,
        output,
        width=args.width,
        height=args.height,
        colors=args.colors,
        palette=args.palette,
        dither=args.dither,
        outline=args.outline,
        transparent=not args.no_transparent
    )
    
    print(f"Created {result.width}x{result.height} pixel art with {args.colors} colors")
