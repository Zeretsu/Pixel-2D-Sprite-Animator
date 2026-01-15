"""
Sprite Parser - Reads image files and extracts sprite data
Supports: PNG, GIF, and basic Aseprite JSON exports
"""

from PIL import Image
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from collections import Counter
import json


@dataclass
class Layer:
    """Represents a sprite layer"""
    name: str
    image: np.ndarray
    visible: bool = True
    opacity: float = 1.0


@dataclass  
class Sprite:
    """Represents a parsed sprite with all its data"""
    width: int
    height: int
    pixels: np.ndarray  # RGBA numpy array
    layers: List[Layer] = field(default_factory=list)
    palette: List[Tuple[int, int, int, int]] = field(default_factory=list)
    name: str = "sprite"
    source_path: Optional[Path] = None
    
    @property
    def has_transparency(self) -> bool:
        """Check if sprite has transparent pixels"""
        if self.pixels.shape[2] == 4:
            return np.any(self.pixels[:, :, 3] < 255)
        return False
    
    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Get bounding box of non-transparent pixels (x, y, w, h)"""
        if self.pixels.shape[2] == 4:
            alpha = self.pixels[:, :, 3]
            rows = np.any(alpha > 0, axis=1)
            cols = np.any(alpha > 0, axis=0)
            if not np.any(rows) or not np.any(cols):
                return (0, 0, self.width, self.height)
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            return (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))
        return (0, 0, self.width, self.height)
    
    def get_colors(self) -> List[Tuple[int, int, int, int]]:
        """Extract unique colors from sprite"""
        if len(self.palette) > 0:
            return self.palette
        
        # Flatten and get unique colors
        flat = self.pixels.reshape(-1, self.pixels.shape[2])
        unique = np.unique(flat, axis=0)
        return [tuple(c) for c in unique if c[3] > 0]  # Exclude transparent
    
    def copy(self) -> 'Sprite':
        """Create a deep copy of the sprite"""
        return Sprite(
            width=self.width,
            height=self.height,
            pixels=self.pixels.copy(),
            layers=[Layer(l.name, l.image.copy(), l.visible, l.opacity) for l in self.layers],
            palette=self.palette.copy(),
            name=self.name,
            source_path=self.source_path
        )


class SpriteParser:
    """Parses various image formats into Sprite objects"""
    
    SUPPORTED_FORMATS = {'.png', '.gif', '.jpg', '.jpeg', '.bmp', '.webp'}
    
    # Default background removal settings (conservative to avoid eating into sprites)
    DEFAULT_BG_TOLERANCE = 15
    
    @classmethod
    def parse(cls, path: str | Path, remove_background: bool = True, 
              bg_tolerance: int = None) -> Sprite:
        """Parse an image file into a Sprite object
        
        Args:
            path: Path to the image file
            remove_background: Detect and remove background (default: True)
            bg_tolerance: Color tolerance for background detection (default: 15)
        """
        path = Path(path)
        
        if bg_tolerance is None:
            bg_tolerance = cls.DEFAULT_BG_TOLERANCE
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        suffix = path.suffix.lower()
        
        if suffix in cls.SUPPORTED_FORMATS:
            sprite = cls._parse_image(path)
        elif suffix == '.json':
            sprite = cls._parse_aseprite_json(path)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
        
        # Auto-remove background if requested
        if remove_background:
            sprite = cls._remove_background(sprite, bg_tolerance)
        
        return sprite
    
    @classmethod
    def _parse_image(cls, path: Path) -> Sprite:
        """Parse a standard image file"""
        img = Image.open(path)
        
        # Convert to RGBA
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        pixels = np.array(img)
        
        # Extract palette if indexed
        palette = []
        if hasattr(img, 'palette') and img.palette:
            pal_data = img.palette.getdata()
            if pal_data:
                pal_bytes = pal_data[1]
                for i in range(0, len(pal_bytes), 3):
                    if i + 2 < len(pal_bytes):
                        palette.append((pal_bytes[i], pal_bytes[i+1], pal_bytes[i+2], 255))
        
        return Sprite(
            width=img.width,
            height=img.height,
            pixels=pixels,
            palette=palette,
            name=path.stem,
            source_path=path
        )
    
    @classmethod
    def _parse_aseprite_json(cls, path: Path) -> Sprite:
        """Parse Aseprite JSON export (requires accompanying PNG)"""
        with open(path) as f:
            data = json.load(f)
        
        # Find the source image
        image_path = path.parent / data.get('meta', {}).get('image', path.stem + '.png')
        
        if not image_path.exists():
            raise FileNotFoundError(f"Source image not found: {image_path}")
        
        sprite = cls._parse_image(image_path)
        sprite.name = path.stem
        
        # Extract layers if available
        if 'layers' in data.get('meta', {}):
            for layer_data in data['meta']['layers']:
                sprite.layers.append(Layer(
                    name=layer_data.get('name', 'Layer'),
                    image=sprite.pixels.copy(),  # Simplified - would need proper slicing
                    visible=True,
                    opacity=layer_data.get('opacity', 255) / 255.0
                ))
        
        return sprite
    
    @classmethod
    def from_array(cls, pixels: np.ndarray, name: str = "sprite") -> Sprite:
        """Create a Sprite from a numpy array"""
        if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
            raise ValueError("Pixels must be HxWx3 or HxWx4 array")
        
        # Ensure RGBA
        if pixels.shape[2] == 3:
            alpha = np.full((*pixels.shape[:2], 1), 255, dtype=pixels.dtype)
            pixels = np.concatenate([pixels, alpha], axis=2)
        
        return Sprite(
            width=pixels.shape[1],
            height=pixels.shape[0],
            pixels=pixels,
            name=name
        )
    
    @classmethod
    def _remove_background(cls, sprite: Sprite, tolerance: int = 30) -> Sprite:
        """Fast background removal - detects edge color and removes matching pixels."""
        arr = sprite.pixels.copy()
        h, w = arr.shape[:2]
        
        if h < 3 or w < 3:
            return sprite
        
        # Get background color from corners
        corners = [arr[0, 0, :3], arr[0, -1, :3], arr[-1, 0, :3], arr[-1, -1, :3]]
        bg_color = np.mean(corners, axis=0).astype(np.float32)
        
        # Simple distance-based removal (no loops)
        rgb = arr[:, :, :3].astype(np.float32)
        distance = np.sqrt(np.sum((rgb - bg_color) ** 2, axis=2))
        
        # Remove pixels close to background color
        to_remove = distance < tolerance
        arr[to_remove, 3] = 0
        
        return Sprite(
            width=sprite.width,
            height=sprite.height,
            pixels=arr,
            layers=sprite.layers,
            palette=sprite.palette,
            name=sprite.name,
            source_path=sprite.source_path
        )