"""
Sprite Decomposition - Auto-detect and separate sprite layers
Extracts base, highlight, shadow, and outline layers for independent animation
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum, auto


class LayerType(Enum):
    """Types of auto-detected layers"""
    BASE = auto()       # Main body/color
    OUTLINE = auto()    # Edge pixels (usually darkest)
    HIGHLIGHT = auto()  # Bright spots/shine
    SHADOW = auto()     # Dark areas/shading
    MIDTONE = auto()    # Mid-brightness areas
    CUSTOM = auto()     # User-defined


@dataclass
class SpriteLayer:
    """A single extracted layer from sprite decomposition"""
    
    name: str
    layer_type: LayerType
    pixels: np.ndarray  # RGBA with transparency where layer doesn't exist
    mask: np.ndarray    # Boolean mask of layer coverage
    
    # Layer metadata
    coverage: float = 0.0       # Percentage of sprite covered
    avg_luminance: float = 0.0  # Average brightness
    avg_color: Tuple[int, int, int] = (128, 128, 128)
    bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    
    @property
    def width(self) -> int:
        return self.pixels.shape[1]
    
    @property
    def height(self) -> int:
        return self.pixels.shape[0]
    
    def to_premultiplied(self) -> np.ndarray:
        """Convert to premultiplied alpha for compositing"""
        result = self.pixels.astype(np.float32) / 255.0
        alpha = result[:, :, 3:4]
        result[:, :, :3] *= alpha
        return result


@dataclass
class DecomposedSprite:
    """Collection of layers extracted from a sprite"""
    
    original: np.ndarray
    layers: Dict[str, SpriteLayer] = field(default_factory=dict)
    
    # Standard layer references
    base: Optional[SpriteLayer] = None
    outline: Optional[SpriteLayer] = None
    highlight: Optional[SpriteLayer] = None
    shadow: Optional[SpriteLayer] = None
    
    @property
    def width(self) -> int:
        return self.original.shape[1]
    
    @property
    def height(self) -> int:
        return self.original.shape[0]
    
    def get_layer(self, name: str) -> Optional[SpriteLayer]:
        """Get layer by name"""
        return self.layers.get(name)
    
    def add_layer(self, layer: SpriteLayer) -> None:
        """Add a layer to the collection"""
        self.layers[layer.name] = layer
        
        # Set standard references
        if layer.layer_type == LayerType.BASE:
            self.base = layer
        elif layer.layer_type == LayerType.OUTLINE:
            self.outline = layer
        elif layer.layer_type == LayerType.HIGHLIGHT:
            self.highlight = layer
        elif layer.layer_type == LayerType.SHADOW:
            self.shadow = layer
    
    def composite(
        self,
        layer_order: Optional[List[str]] = None,
        layer_transforms: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Recomposite layers back into a single image.
        
        Args:
            layer_order: Order to composite (bottom to top). Default: shadow, base, highlight, outline
            layer_transforms: Optional per-layer pixel transforms (e.g., color shifts)
            
        Returns:
            Composited RGBA image
        """
        if layer_order is None:
            layer_order = ['shadow', 'base', 'midtone', 'highlight', 'outline']
        
        layer_transforms = layer_transforms or {}
        
        h, w = self.height, self.width
        result = np.zeros((h, w, 4), dtype=np.float32)
        
        for layer_name in layer_order:
            layer = self.layers.get(layer_name)
            if layer is None:
                continue
            
            # Get layer pixels (possibly transformed)
            layer_pixels = layer_transforms.get(layer_name, layer.pixels)
            layer_float = layer_pixels.astype(np.float32) / 255.0
            
            # Alpha compositing (premultiplied)
            src_alpha = layer_float[:, :, 3:4]
            src_rgb = layer_float[:, :, :3] * src_alpha
            
            dst_alpha = result[:, :, 3:4]
            dst_rgb = result[:, :, :3]
            
            # Porter-Duff over operation
            out_alpha = src_alpha + dst_alpha * (1 - src_alpha)
            safe_alpha = np.where(out_alpha > 1e-10, out_alpha, 1.0)
            
            result[:, :, :3] = (src_rgb + dst_rgb * (1 - src_alpha)) / safe_alpha * out_alpha
            result[:, :, 3:4] = out_alpha
        
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)


class SpriteDecomposer:
    """
    Analyzes sprites and decomposes them into logical layers.
    
    Methods:
        decompose(): Full automatic decomposition
        extract_outline(): Get outline pixels only
        extract_highlights(): Get bright/shiny pixels
        extract_shadows(): Get dark/shadow pixels
        extract_by_luminance(): Split by brightness ranges
        extract_by_color(): Split by color similarity
    """
    
    # Luminance coefficients (Rec. 709)
    LUMA_R = 0.2126
    LUMA_G = 0.7152
    LUMA_B = 0.0722
    
    @classmethod
    def decompose(
        cls,
        sprite: np.ndarray,
        outline_threshold: float = 0.25,
        highlight_threshold: float = 0.75,
        shadow_threshold: float = 0.35,
        detect_outline: bool = True
    ) -> DecomposedSprite:
        """
        Fully decompose a sprite into layers.
        
        Args:
            sprite: RGBA sprite image
            outline_threshold: Luminance below this at edges = outline
            highlight_threshold: Luminance above this = highlight
            shadow_threshold: Luminance below this = shadow
            detect_outline: Whether to detect outline layer
            
        Returns:
            DecomposedSprite with all detected layers
        """
        result = DecomposedSprite(original=sprite.copy())
        h, w = sprite.shape[:2]
        
        # Get alpha mask for visible pixels
        alpha = sprite[:, :, 3] if sprite.shape[2] == 4 else np.ones((h, w), dtype=np.uint8) * 255
        visible_mask = alpha > 10
        
        # Calculate luminance
        luminance = cls._calculate_luminance(sprite)
        
        # Normalize luminance within visible area
        if np.any(visible_mask):
            vis_lum = luminance[visible_mask]
            lum_min, lum_max = vis_lum.min(), vis_lum.max()
            if lum_max > lum_min:
                luminance_norm = (luminance - lum_min) / (lum_max - lum_min)
            else:
                luminance_norm = np.zeros_like(luminance)
        else:
            luminance_norm = luminance
        
        # Extract outline (edge detection + dark pixels)
        if detect_outline:
            outline_layer = cls.extract_outline(
                sprite, 
                luminance_norm, 
                visible_mask,
                threshold=outline_threshold
            )
            result.add_layer(outline_layer)
            outline_mask = outline_layer.mask
        else:
            outline_mask = np.zeros((h, w), dtype=bool)
        
        # Extract highlights (bright areas, not outline)
        highlight_layer = cls.extract_highlights(
            sprite,
            luminance_norm,
            visible_mask & ~outline_mask,
            threshold=highlight_threshold
        )
        result.add_layer(highlight_layer)
        
        # Extract shadows (dark areas, not outline)
        shadow_layer = cls.extract_shadows(
            sprite,
            luminance_norm,
            visible_mask & ~outline_mask,
            threshold=shadow_threshold
        )
        result.add_layer(shadow_layer)
        
        # Base layer = everything else
        used_mask = outline_mask | highlight_layer.mask | shadow_layer.mask
        base_mask = visible_mask & ~used_mask
        
        base_layer = cls._create_layer_from_mask(
            sprite, base_mask, 'base', LayerType.BASE
        )
        result.add_layer(base_layer)
        
        # Midtone layer (between shadow and highlight, excluding base)
        midtone_mask = (
            visible_mask & 
            ~outline_mask & 
            (luminance_norm >= shadow_threshold) & 
            (luminance_norm <= highlight_threshold)
        )
        midtone_layer = cls._create_layer_from_mask(
            sprite, midtone_mask, 'midtone', LayerType.MIDTONE
        )
        result.add_layer(midtone_layer)
        
        return result
    
    @classmethod
    def extract_outline(
        cls,
        sprite: np.ndarray,
        luminance: np.ndarray,
        visible_mask: np.ndarray,
        threshold: float = 0.25
    ) -> SpriteLayer:
        """
        Extract outline pixels using edge detection.
        
        Outline pixels are:
        1. At the edge of the sprite (adjacent to transparent)
        2. Dark compared to neighbors
        """
        h, w = sprite.shape[:2]
        
        # Find edge pixels (adjacent to transparency)
        alpha = sprite[:, :, 3] if sprite.shape[2] == 4 else np.ones((h, w), dtype=np.uint8) * 255
        edge_mask = cls._find_edge_pixels(alpha > 10)
        
        # Find dark pixels that form internal outlines
        # Use gradient magnitude to detect internal edges
        grad_y, grad_x = np.gradient(luminance)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize gradient
        if gradient_mag.max() > 0:
            gradient_norm = gradient_mag / gradient_mag.max()
        else:
            gradient_norm = gradient_mag
        
        # Outline = edge pixels OR (dark + high gradient)
        dark_mask = luminance < threshold
        internal_outline = dark_mask & (gradient_norm > 0.3) & visible_mask
        
        outline_mask = (edge_mask | internal_outline) & visible_mask & dark_mask
        
        return cls._create_layer_from_mask(
            sprite, outline_mask, 'outline', LayerType.OUTLINE
        )
    
    @classmethod
    def extract_highlights(
        cls,
        sprite: np.ndarray,
        luminance: np.ndarray,
        visible_mask: np.ndarray,
        threshold: float = 0.75
    ) -> SpriteLayer:
        """Extract bright highlight pixels"""
        highlight_mask = (luminance >= threshold) & visible_mask
        
        return cls._create_layer_from_mask(
            sprite, highlight_mask, 'highlight', LayerType.HIGHLIGHT
        )
    
    @classmethod
    def extract_shadows(
        cls,
        sprite: np.ndarray,
        luminance: np.ndarray,
        visible_mask: np.ndarray,
        threshold: float = 0.35
    ) -> SpriteLayer:
        """Extract dark shadow pixels (not outline)"""
        shadow_mask = (luminance < threshold) & visible_mask
        
        return cls._create_layer_from_mask(
            sprite, shadow_mask, 'shadow', LayerType.SHADOW
        )
    
    @classmethod
    def extract_by_luminance(
        cls,
        sprite: np.ndarray,
        num_levels: int = 4
    ) -> List[SpriteLayer]:
        """
        Split sprite into N luminance bands.
        
        Args:
            sprite: RGBA sprite
            num_levels: Number of luminance bands (2-8)
            
        Returns:
            List of layers from darkest to brightest
        """
        num_levels = max(2, min(8, num_levels))
        
        h, w = sprite.shape[:2]
        alpha = sprite[:, :, 3] if sprite.shape[2] == 4 else np.ones((h, w), dtype=np.uint8) * 255
        visible_mask = alpha > 10
        
        luminance = cls._calculate_luminance(sprite)
        
        # Normalize
        if np.any(visible_mask):
            vis_lum = luminance[visible_mask]
            lum_min, lum_max = vis_lum.min(), vis_lum.max()
            if lum_max > lum_min:
                luminance_norm = (luminance - lum_min) / (lum_max - lum_min)
            else:
                luminance_norm = np.zeros_like(luminance)
        else:
            return []
        
        layers = []
        level_names = ['darkest', 'dark', 'mid-dark', 'mid', 'mid-light', 'light', 'bright', 'brightest']
        
        for i in range(num_levels):
            low = i / num_levels
            high = (i + 1) / num_levels
            
            mask = visible_mask & (luminance_norm >= low) & (luminance_norm < high)
            
            # Last level includes max value
            if i == num_levels - 1:
                mask = visible_mask & (luminance_norm >= low)
            
            name = level_names[min(i, len(level_names) - 1)]
            layer = cls._create_layer_from_mask(
                sprite, mask, f'lum_{name}', LayerType.CUSTOM
            )
            layers.append(layer)
        
        return layers
    
    @classmethod
    def extract_by_color(
        cls,
        sprite: np.ndarray,
        num_colors: int = 4,
        method: str = 'kmeans'
    ) -> List[SpriteLayer]:
        """
        Split sprite by color clusters.
        
        Args:
            sprite: RGBA sprite
            num_colors: Number of color groups
            method: Clustering method ('kmeans' or 'dominant')
            
        Returns:
            List of layers, one per color cluster
        """
        h, w = sprite.shape[:2]
        alpha = sprite[:, :, 3] if sprite.shape[2] == 4 else np.ones((h, w), dtype=np.uint8) * 255
        visible_mask = alpha > 10
        
        if not np.any(visible_mask):
            return []
        
        # Get visible pixels
        vis_coords = np.where(visible_mask)
        vis_colors = sprite[vis_coords[0], vis_coords[1], :3]
        
        if method == 'kmeans':
            # Simple k-means clustering
            labels = cls._kmeans_colors(vis_colors, num_colors)
        else:
            # Dominant color extraction
            labels = cls._dominant_colors(vis_colors, num_colors)
        
        # Create layers from clusters
        layers = []
        label_img = np.full((h, w), -1, dtype=np.int32)
        label_img[vis_coords[0], vis_coords[1]] = labels
        
        for i in range(num_colors):
            mask = label_img == i
            if not np.any(mask):
                continue
            
            # Get dominant color in cluster
            cluster_colors = sprite[mask, :3]
            avg_color = tuple(cluster_colors.mean(axis=0).astype(int))
            
            layer = cls._create_layer_from_mask(
                sprite, mask, f'color_{i}', LayerType.CUSTOM
            )
            layer.avg_color = avg_color
            layers.append(layer)
        
        return layers
    
    @classmethod
    def extract_specular(
        cls,
        sprite: np.ndarray,
        saturation_threshold: float = 0.3,
        brightness_threshold: float = 0.85
    ) -> SpriteLayer:
        """
        Extract specular highlights (very bright, low saturation).
        These are typically white shine spots on materials.
        """
        h, w = sprite.shape[:2]
        alpha = sprite[:, :, 3] if sprite.shape[2] == 4 else np.ones((h, w), dtype=np.uint8) * 255
        visible_mask = alpha > 10
        
        # Convert to HSV-like values
        rgb = sprite[:, :, :3].astype(np.float32) / 255.0
        
        max_rgb = rgb.max(axis=2)
        min_rgb = rgb.min(axis=2)
        delta = max_rgb - min_rgb
        
        # Saturation
        saturation = np.where(max_rgb > 0, delta / max_rgb, 0)
        
        # Value/brightness
        brightness = max_rgb
        
        # Specular = bright + low saturation
        specular_mask = (
            visible_mask &
            (brightness >= brightness_threshold) &
            (saturation <= saturation_threshold)
        )
        
        return cls._create_layer_from_mask(
            sprite, specular_mask, 'specular', LayerType.HIGHLIGHT
        )
    
    @classmethod
    def _calculate_luminance(cls, sprite: np.ndarray) -> np.ndarray:
        """Calculate perceptual luminance"""
        rgb = sprite[:, :, :3].astype(np.float32) / 255.0
        return (
            cls.LUMA_R * rgb[:, :, 0] +
            cls.LUMA_G * rgb[:, :, 1] +
            cls.LUMA_B * rgb[:, :, 2]
        )
    
    @classmethod
    def _find_edge_pixels(cls, mask: np.ndarray) -> np.ndarray:
        """Find pixels at the edge of a mask (adjacent to False)"""
        h, w = mask.shape
        edge = np.zeros_like(mask)
        
        # Check all 8 neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                
                # Shift mask
                shifted = np.zeros_like(mask)
                
                src_y = slice(max(0, -dy), min(h, h - dy))
                src_x = slice(max(0, -dx), min(w, w - dx))
                dst_y = slice(max(0, dy), min(h, h + dy))
                dst_x = slice(max(0, dx), min(w, w + dx))
                
                shifted[dst_y, dst_x] = mask[src_y, src_x]
                
                # Edge where mask is True but neighbor is False
                edge |= mask & ~shifted
        
        return edge
    
    @classmethod
    def _create_layer_from_mask(
        cls,
        sprite: np.ndarray,
        mask: np.ndarray,
        name: str,
        layer_type: LayerType
    ) -> SpriteLayer:
        """Create a SpriteLayer from a boolean mask"""
        h, w = sprite.shape[:2]
        
        # Create layer pixels (transparent where mask is False)
        pixels = np.zeros((h, w, 4), dtype=np.uint8)
        pixels[mask] = sprite[mask]
        
        # Calculate metadata
        coverage = mask.sum() / mask.size if mask.size > 0 else 0.0
        
        if np.any(mask):
            avg_lum = cls._calculate_luminance(sprite)[mask].mean()
            avg_color = tuple(sprite[mask, :3].mean(axis=0).astype(int))
            
            ys, xs = np.where(mask)
            bounds = (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
        else:
            avg_lum = 0.0
            avg_color = (0, 0, 0)
            bounds = (0, 0, 0, 0)
        
        return SpriteLayer(
            name=name,
            layer_type=layer_type,
            pixels=pixels,
            mask=mask,
            coverage=coverage,
            avg_luminance=avg_lum,
            avg_color=avg_color,
            bounds=bounds
        )
    
    @classmethod
    def _kmeans_colors(cls, colors: np.ndarray, k: int, max_iter: int = 20) -> np.ndarray:
        """Simple k-means clustering for colors"""
        n = len(colors)
        if n == 0:
            return np.array([], dtype=np.int32)
        
        k = min(k, n)
        colors_float = colors.astype(np.float32)
        
        # Initialize centroids randomly
        indices = np.random.choice(n, k, replace=False)
        centroids = colors_float[indices].copy()
        
        labels = np.zeros(n, dtype=np.int32)
        
        for _ in range(max_iter):
            # Assign to nearest centroid
            for i, c in enumerate(colors_float):
                dists = np.sum((centroids - c) ** 2, axis=1)
                labels[i] = np.argmin(dists)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                cluster_mask = labels == j
                if np.any(cluster_mask):
                    new_centroids[j] = colors_float[cluster_mask].mean(axis=0)
                else:
                    new_centroids[j] = centroids[j]
            
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        return labels
    
    @classmethod
    def _dominant_colors(cls, colors: np.ndarray, k: int) -> np.ndarray:
        """Extract dominant colors using histogram binning"""
        n = len(colors)
        if n == 0:
            return np.array([], dtype=np.int32)
        
        # Quantize colors to bins
        bins = 8
        quantized = (colors // (256 // bins)).astype(np.int32)
        
        # Create color keys
        keys = quantized[:, 0] * bins * bins + quantized[:, 1] * bins + quantized[:, 2]
        
        # Count frequencies
        unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        
        # Map to k clusters based on frequency rank
        rank = np.zeros(len(unique), dtype=np.int32)
        sorted_indices = np.argsort(-counts)
        for i, idx in enumerate(sorted_indices):
            rank[idx] = min(i, k - 1)
        
        return rank[inverse]


# ============================================================================
# Layer-Aware Animation Helpers
# ============================================================================

def animate_layer(
    decomposed: DecomposedSprite,
    layer_name: str,
    transform_fn,
    num_frames: int = 8
) -> List[np.ndarray]:
    """
    Animate a single layer while keeping others static.
    
    Args:
        decomposed: DecomposedSprite with layers
        layer_name: Name of layer to animate
        transform_fn: Function(pixels, t) -> transformed_pixels
        num_frames: Number of frames
        
    Returns:
        List of composited frames
    """
    frames = []
    
    for i in range(num_frames):
        t = i / num_frames
        
        # Transform the target layer
        transforms = {}
        layer = decomposed.get_layer(layer_name)
        if layer is not None:
            transforms[layer_name] = transform_fn(layer.pixels.copy(), t)
        
        # Composite with transformation
        frame = decomposed.composite(layer_transforms=transforms)
        frames.append(frame)
    
    return frames


def pulse_highlights(
    decomposed: DecomposedSprite,
    num_frames: int = 8,
    intensity: float = 1.5,
    min_intensity: float = 0.8
) -> List[np.ndarray]:
    """
    Create pulsing glow on highlight layer only.
    
    This creates a breathing/glowing effect that only affects bright pixels,
    giving the sprite apparent depth.
    """
    def transform(pixels, t):
        # Sine wave pulse
        pulse = min_intensity + (intensity - min_intensity) * (0.5 + 0.5 * np.sin(t * 2 * np.pi))
        
        result = pixels.astype(np.float32)
        mask = pixels[:, :, 3] > 0
        
        # Brighten RGB
        result[mask, :3] = np.clip(result[mask, :3] * pulse, 0, 255)
        
        return result.astype(np.uint8)
    
    return animate_layer(decomposed, 'highlight', transform, num_frames)


def shift_shadows(
    decomposed: DecomposedSprite,
    num_frames: int = 8,
    offset_range: Tuple[int, int] = (0, 2)
) -> List[np.ndarray]:
    """
    Animate shadow layer shifting (simulates light source movement).
    """
    def transform(pixels, t):
        # Oscillate shadow position
        offset_y = int(offset_range[0] + (offset_range[1] - offset_range[0]) * (0.5 + 0.5 * np.sin(t * 2 * np.pi)))
        
        result = np.zeros_like(pixels)
        h, w = pixels.shape[:2]
        
        if offset_y > 0:
            result[offset_y:, :] = pixels[:-offset_y, :]
        elif offset_y < 0:
            result[:offset_y, :] = pixels[-offset_y:, :]
        else:
            result = pixels.copy()
        
        return result
    
    return animate_layer(decomposed, 'shadow', transform, num_frames)


def color_cycle_layer(
    decomposed: DecomposedSprite,
    layer_name: str,
    num_frames: int = 8,
    hue_shift: float = 1.0
) -> List[np.ndarray]:
    """
    Apply hue cycling to a specific layer.
    
    Args:
        decomposed: DecomposedSprite
        layer_name: Layer to color cycle
        num_frames: Animation frames
        hue_shift: Amount of hue rotation (1.0 = full cycle)
    """
    def transform(pixels, t):
        result = pixels.copy().astype(np.float32)
        mask = pixels[:, :, 3] > 0
        
        if not np.any(mask):
            return pixels
        
        # Convert to HSV-ish, shift hue, convert back
        rgb = result[mask, :3] / 255.0
        
        # RGB to HSV
        max_rgb = rgb.max(axis=1)
        min_rgb = rgb.min(axis=1)
        delta = max_rgb - min_rgb
        
        # Hue calculation
        hue = np.zeros(len(rgb))
        
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        
        mask_r = (delta > 0) & (max_rgb == r)
        mask_g = (delta > 0) & (max_rgb == g)
        mask_b = (delta > 0) & (max_rgb == b)
        
        hue[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
        hue[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2
        hue[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4
        
        hue = hue / 6.0  # Normalize to 0-1
        
        # Shift hue
        hue = (hue + t * hue_shift) % 1.0
        
        # HSV to RGB
        saturation = np.where(max_rgb > 0, delta / max_rgb, 0)
        value = max_rgb
        
        c = value * saturation
        x = c * (1 - np.abs((hue * 6) % 2 - 1))
        m = value - c
        
        h_sector = (hue * 6).astype(int) % 6
        
        new_rgb = np.zeros_like(rgb)
        
        for sector in range(6):
            sector_mask = h_sector == sector
            if sector == 0:
                new_rgb[sector_mask] = np.column_stack([c[sector_mask], x[sector_mask], np.zeros(sector_mask.sum())])
            elif sector == 1:
                new_rgb[sector_mask] = np.column_stack([x[sector_mask], c[sector_mask], np.zeros(sector_mask.sum())])
            elif sector == 2:
                new_rgb[sector_mask] = np.column_stack([np.zeros(sector_mask.sum()), c[sector_mask], x[sector_mask]])
            elif sector == 3:
                new_rgb[sector_mask] = np.column_stack([np.zeros(sector_mask.sum()), x[sector_mask], c[sector_mask]])
            elif sector == 4:
                new_rgb[sector_mask] = np.column_stack([x[sector_mask], np.zeros(sector_mask.sum()), c[sector_mask]])
            else:
                new_rgb[sector_mask] = np.column_stack([c[sector_mask], np.zeros(sector_mask.sum()), x[sector_mask]])
        
        new_rgb += m[:, np.newaxis]
        
        result[mask, :3] = np.clip(new_rgb * 255, 0, 255)
        return result.astype(np.uint8)
    
    return animate_layer(decomposed, layer_name, transform, num_frames)


def independent_layer_animation(
    decomposed: DecomposedSprite,
    layer_animations: Dict[str, callable],
    num_frames: int = 8
) -> List[np.ndarray]:
    """
    Apply different animations to different layers simultaneously.
    
    Args:
        decomposed: DecomposedSprite
        layer_animations: Dict of {layer_name: transform_fn(pixels, t)}
        num_frames: Number of frames
        
    Returns:
        List of composited frames
    """
    frames = []
    
    for i in range(num_frames):
        t = i / num_frames
        
        transforms = {}
        for layer_name, transform_fn in layer_animations.items():
            layer = decomposed.get_layer(layer_name)
            if layer is not None:
                transforms[layer_name] = transform_fn(layer.pixels.copy(), t)
        
        frame = decomposed.composite(layer_transforms=transforms)
        frames.append(frame)
    
    return frames


# ============================================================================
# Quick decomposition helpers
# ============================================================================

def quick_decompose(sprite: np.ndarray) -> DecomposedSprite:
    """Quick sprite decomposition with default settings"""
    return SpriteDecomposer.decompose(sprite)


def quick_glow_animation(
    sprite: np.ndarray,
    frames: int = 8,
    intensity: float = 1.5
) -> List[np.ndarray]:
    """Quick highlight-only glow animation"""
    decomposed = quick_decompose(sprite)
    return pulse_highlights(decomposed, frames, intensity)


def quick_shadow_dance(
    sprite: np.ndarray,
    frames: int = 8,
    offset: int = 2
) -> List[np.ndarray]:
    """Quick shadow shifting animation"""
    decomposed = quick_decompose(sprite)
    return shift_shadows(decomposed, frames, (0, offset))
