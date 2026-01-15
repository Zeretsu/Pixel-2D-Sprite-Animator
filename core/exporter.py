"""
Sprite Exporter - Exports animated sprites to various formats
"""

from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from .parser import Sprite


class SpriteExporter:
    """Exports sprites and animations to various formats"""
    
    @classmethod
    def to_png(cls, sprite: Sprite, path: str | Path) -> Path:
        """Export a single sprite to PNG"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        img = Image.fromarray(sprite.pixels.astype(np.uint8), 'RGBA')
        img.save(path, 'PNG')
        
        return path
    
    @classmethod
    def to_gif(
        cls, 
        frames: List[Sprite], 
        path: str | Path,
        duration: int = 100,
        loop: int = 0
    ) -> Path:
        """Export animation frames to GIF"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if not frames:
            raise ValueError("No frames to export")
        
        images = []
        for frame in frames:
            img = Image.fromarray(frame.pixels.astype(np.uint8), 'RGBA')
            # Extract alpha channel to create proper transparency mask
            alpha = img.split()[3]
            # Create a mask where transparent pixels are marked
            mask = Image.eval(alpha, lambda a: 255 if a < 128 else 0)
            # Convert to palette mode with transparency support
            img_p = img.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=255)
            # Set the transparent color index
            img_p.paste(255, mask)
            images.append(img_p)
        
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop,
            transparency=255,
            disposal=2
        )
        
        return path
    
    @classmethod
    def to_spritesheet(
        cls,
        frames: List[Sprite],
        path: str | Path,
        columns: Optional[int] = None,
        padding: int = 0
    ) -> Tuple[Path, dict]:
        """Export animation frames to a spritesheet PNG with metadata"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if not frames:
            raise ValueError("No frames to export")
        
        frame_count = len(frames)
        frame_width = frames[0].width
        frame_height = frames[0].height
        
        # Calculate grid
        if columns is None:
            columns = min(frame_count, 8)
        rows = (frame_count + columns - 1) // columns
        
        # Create spritesheet
        sheet_width = columns * (frame_width + padding) - padding
        sheet_height = rows * (frame_height + padding) - padding
        
        sheet = np.zeros((sheet_height, sheet_width, 4), dtype=np.uint8)
        
        for i, frame in enumerate(frames):
            row = i // columns
            col = i % columns
            x = col * (frame_width + padding)
            y = row * (frame_height + padding)
            sheet[y:y+frame_height, x:x+frame_width] = frame.pixels
        
        # Save image
        img = Image.fromarray(sheet, 'RGBA')
        img.save(path, 'PNG')
        
        # Generate metadata
        metadata = {
            'frames': frame_count,
            'frame_width': frame_width,
            'frame_height': frame_height,
            'columns': columns,
            'rows': rows,
            'padding': padding,
            'sheet_width': sheet_width,
            'sheet_height': sheet_height
        }
        
        # Save JSON metadata alongside
        import json
        meta_path = path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return path, metadata
    
    @classmethod
    def to_frames(
        cls,
        frames: List[Sprite],
        directory: str | Path,
        prefix: str = "frame"
    ) -> List[Path]:
        """Export animation frames as individual PNGs"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        paths = []
        for i, frame in enumerate(frames):
            frame_path = directory / f"{prefix}_{i:04d}.png"
            cls.to_png(frame, frame_path)
            paths.append(frame_path)
        
        return paths
