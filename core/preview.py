"""
Real-Time Preview Window

Interactive preview for animations without waiting for export.

Features:
- Real-time animation playback
- Timeline scrubbing
- Parameter adjustment with keyboard/mouse
- Before/after comparison (split view)
- Zoom and pan
- Frame-by-frame stepping
- Export when satisfied

Controls:
    SPACE       - Play/pause
    LEFT/RIGHT  - Previous/next frame
    HOME/END    - First/last frame
    UP/DOWN     - Adjust speed (0.25x to 4x)
    +/-         - Zoom in/out
    1-9         - Jump to percentage (1=10%, 5=50%, 9=90%)
    B           - Toggle before/after split view
    G           - Toggle grid overlay
    O           - Toggle onion skinning preview
    S           - Save current frame as PNG
    E           - Export animation
    R           - Reset to defaults
    H           - Show/hide help
    ESC/Q       - Quit

Requires: pygame (pip install pygame)
"""

import sys
import numpy as np
from typing import List, Optional, Tuple, Callable, Any, Dict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

# Try to import pygame
try:
    import pygame
    from pygame import Surface, Rect
    from pygame.locals import *
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    pygame = None
    # Dummy types for annotations when pygame not installed
    Surface = Any
    Rect = Any


# =============================================================================
# Preview Configuration
# =============================================================================

@dataclass
class PreviewConfig:
    """Configuration for the preview window"""
    # Window settings
    window_width: int = 800
    window_height: int = 600
    window_title: str = "Sprite Animator Preview"
    background_color: Tuple[int, int, int] = (40, 40, 40)
    
    # Playback settings
    fps: int = 60
    animation_fps: int = 12
    default_speed: float = 1.0
    loop: bool = True
    
    # View settings
    initial_zoom: float = 4.0
    min_zoom: float = 1.0
    max_zoom: float = 16.0
    zoom_step: float = 1.5
    
    # Display settings
    show_grid: bool = False
    grid_color: Tuple[int, int, int] = (60, 60, 60)
    show_info: bool = True
    show_help: bool = False
    
    # Comparison settings
    split_view: bool = False
    split_position: float = 0.5  # 0-1, position of split line
    
    # Onion skin preview
    onion_skin: bool = False
    onion_frames: int = 2
    onion_opacity: float = 0.3


class ViewMode(Enum):
    """Preview view modes"""
    NORMAL = auto()
    SPLIT_HORIZONTAL = auto()
    SPLIT_VERTICAL = auto()
    SIDE_BY_SIDE = auto()


# =============================================================================
# Preview Window
# =============================================================================

class PreviewWindow:
    """
    Real-time animation preview window.
    
    Example:
        frames = [...]  # List of RGBA numpy arrays
        preview = PreviewWindow(frames)
        preview.run()
    """
    
    def __init__(
        self,
        frames: List[np.ndarray],
        original_frames: Optional[List[np.ndarray]] = None,
        config: Optional[PreviewConfig] = None,
        on_export: Optional[Callable[[List[np.ndarray]], None]] = None,
        on_parameter_change: Optional[Callable[[str, Any], List[np.ndarray]]] = None
    ):
        """
        Initialize preview window.
        
        Args:
            frames: Animated frames to preview
            original_frames: Original (before) frames for comparison
            config: Preview configuration
            on_export: Callback when user requests export
            on_parameter_change: Callback when parameter changes, returns new frames
        """
        if not PYGAME_AVAILABLE:
            raise ImportError(
                "pygame is required for preview. Install with: pip install pygame"
            )
        
        self.frames = frames
        self.original_frames = original_frames or frames
        self.config = config or PreviewConfig()
        self.on_export = on_export
        self.on_parameter_change = on_parameter_change
        
        # State
        self.current_frame = 0
        self.playing = True
        self.speed = self.config.default_speed
        self.zoom = self.config.initial_zoom
        self.pan_x = 0
        self.pan_y = 0
        self.view_mode = ViewMode.NORMAL
        self.show_original = False
        
        # Timing
        self.frame_time = 0.0
        self.frame_duration = 1.0 / self.config.animation_fps
        
        # UI state
        self.dragging = False
        self.drag_start = (0, 0)
        self.pan_start = (0, 0)
        
        # Parameter adjustments (for real-time tweaking)
        self.parameters: Dict[str, Any] = {
            'intensity': 1.0,
            'speed': 1.0,
        }
        
        # Initialize pygame
        self._init_pygame()
    
    def _init_pygame(self):
        """Initialize pygame and create window"""
        pygame.init()
        pygame.display.set_caption(self.config.window_title)
        
        self.screen = pygame.display.set_mode(
            (self.config.window_width, self.config.window_height),
            pygame.RESIZABLE
        )
        
        self.clock = pygame.time.Clock()
        
        # Load font
        try:
            self.font = pygame.font.SysFont('consolas', 14)
            self.font_large = pygame.font.SysFont('consolas', 18)
        except:
            self.font = pygame.font.Font(None, 16)
            self.font_large = pygame.font.Font(None, 20)
        
        # Pre-convert frames to pygame surfaces
        self._convert_frames()
    
    def _convert_frames(self):
        """Convert numpy arrays to pygame surfaces"""
        self.surfaces = []
        self.original_surfaces = []
        
        for frame in self.frames:
            surf = self._array_to_surface(frame)
            self.surfaces.append(surf)
        
        for frame in self.original_frames:
            surf = self._array_to_surface(frame)
            self.original_surfaces.append(surf)
    
    def _array_to_surface(self, array: np.ndarray) -> Surface:
        """Convert RGBA numpy array to pygame surface"""
        # Ensure RGBA
        if array.shape[2] == 3:
            rgba = np.zeros((*array.shape[:2], 4), dtype=np.uint8)
            rgba[:, :, :3] = array
            rgba[:, :, 3] = 255
            array = rgba
        
        # Pygame expects (width, height) but numpy is (height, width)
        # Also pygame uses RGB not RGBA for the base, need to handle alpha
        h, w = array.shape[:2]
        
        # Create surface with alpha
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        
        # Blit pixel data
        pygame.surfarray.pixels3d(surf)[:] = array[:, :, :3].swapaxes(0, 1)
        pygame.surfarray.pixels_alpha(surf)[:] = array[:, :, 3].swapaxes(0, 1)
        
        return surf
    
    def run(self):
        """Run the preview window main loop"""
        running = True
        
        while running:
            dt = self.clock.tick(self.config.fps) / 1000.0
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_mouse_down(event)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self._handle_mouse_up(event)
                elif event.type == pygame.MOUSEMOTION:
                    self._handle_mouse_motion(event)
                elif event.type == pygame.MOUSEWHEEL:
                    self._handle_mouse_wheel(event)
                elif event.type == pygame.VIDEORESIZE:
                    self.config.window_width = event.w
                    self.config.window_height = event.h
                    self.screen = pygame.display.set_mode(
                        (event.w, event.h), pygame.RESIZABLE
                    )
            
            # Update animation
            if self.playing:
                self.frame_time += dt * self.speed
                if self.frame_time >= self.frame_duration:
                    self.frame_time = 0
                    self.current_frame += 1
                    if self.current_frame >= len(self.frames):
                        if self.config.loop:
                            self.current_frame = 0
                        else:
                            self.current_frame = len(self.frames) - 1
                            self.playing = False
            
            # Render
            self._render()
            
            pygame.display.flip()
        
        pygame.quit()
    
    def _handle_key(self, key: int) -> bool:
        """Handle keyboard input. Returns False to quit."""
        # Quit
        if key in (K_ESCAPE, K_q):
            return False
        
        # Play/pause
        elif key == K_SPACE:
            self.playing = not self.playing
        
        # Frame navigation
        elif key == K_LEFT:
            self.current_frame = max(0, self.current_frame - 1)
            self.playing = False
        elif key == K_RIGHT:
            self.current_frame = min(len(self.frames) - 1, self.current_frame + 1)
            self.playing = False
        elif key == K_HOME:
            self.current_frame = 0
        elif key == K_END:
            self.current_frame = len(self.frames) - 1
        
        # Speed control
        elif key == K_UP:
            self.speed = min(4.0, self.speed * 1.5)
        elif key == K_DOWN:
            self.speed = max(0.25, self.speed / 1.5)
        
        # Zoom
        elif key in (K_PLUS, K_EQUALS, K_KP_PLUS):
            self.zoom = min(self.config.max_zoom, self.zoom * self.config.zoom_step)
        elif key in (K_MINUS, K_KP_MINUS):
            self.zoom = max(self.config.min_zoom, self.zoom / self.config.zoom_step)
        
        # Jump to percentage
        elif K_1 <= key <= K_9:
            pct = (key - K_0) / 10.0
            self.current_frame = int(pct * (len(self.frames) - 1))
        
        # Toggle views
        elif key == K_b:
            if self.view_mode == ViewMode.NORMAL:
                self.view_mode = ViewMode.SPLIT_VERTICAL
            elif self.view_mode == ViewMode.SPLIT_VERTICAL:
                self.view_mode = ViewMode.SPLIT_HORIZONTAL
            elif self.view_mode == ViewMode.SPLIT_HORIZONTAL:
                self.view_mode = ViewMode.SIDE_BY_SIDE
            else:
                self.view_mode = ViewMode.NORMAL
        
        elif key == K_g:
            self.config.show_grid = not self.config.show_grid
        
        elif key == K_o:
            self.config.onion_skin = not self.config.onion_skin
        
        elif key == K_h:
            self.config.show_help = not self.config.show_help
        
        elif key == K_i:
            self.config.show_info = not self.config.show_info
        
        # Save/export
        elif key == K_s:
            self._save_frame()
        elif key == K_e:
            self._export()
        
        # Reset
        elif key == K_r:
            self.zoom = self.config.initial_zoom
            self.pan_x = 0
            self.pan_y = 0
            self.speed = self.config.default_speed
        
        return True
    
    def _handle_mouse_down(self, event):
        """Handle mouse button press"""
        if event.button == 1:  # Left click
            self.dragging = True
            self.drag_start = event.pos
            self.pan_start = (self.pan_x, self.pan_y)
        elif event.button == 2:  # Middle click - reset view
            self.zoom = self.config.initial_zoom
            self.pan_x = 0
            self.pan_y = 0
    
    def _handle_mouse_up(self, event):
        """Handle mouse button release"""
        if event.button == 1:
            self.dragging = False
    
    def _handle_mouse_motion(self, event):
        """Handle mouse movement"""
        if self.dragging:
            dx = event.pos[0] - self.drag_start[0]
            dy = event.pos[1] - self.drag_start[1]
            self.pan_x = self.pan_start[0] + dx
            self.pan_y = self.pan_start[1] + dy
    
    def _handle_mouse_wheel(self, event):
        """Handle mouse wheel for zooming"""
        if event.y > 0:
            self.zoom = min(self.config.max_zoom, self.zoom * self.config.zoom_step)
        elif event.y < 0:
            self.zoom = max(self.config.min_zoom, self.zoom / self.config.zoom_step)
    
    def _render(self):
        """Render the preview"""
        # Clear screen
        self.screen.fill(self.config.background_color)
        
        # Get current frame surface
        if self.view_mode == ViewMode.NORMAL:
            self._render_single_view()
        elif self.view_mode == ViewMode.SPLIT_VERTICAL:
            self._render_split_vertical()
        elif self.view_mode == ViewMode.SPLIT_HORIZONTAL:
            self._render_split_horizontal()
        elif self.view_mode == ViewMode.SIDE_BY_SIDE:
            self._render_side_by_side()
        
        # Draw UI
        if self.config.show_info:
            self._render_info()
        
        if self.config.show_help:
            self._render_help()
        
        # Draw timeline
        self._render_timeline()
    
    def _render_single_view(self):
        """Render single animated view"""
        surf = self.surfaces[self.current_frame]
        
        # Apply onion skinning if enabled
        if self.config.onion_skin:
            surf = self._create_onion_surface(self.current_frame)
        
        self._blit_centered(surf)
    
    def _render_split_vertical(self):
        """Render vertical split (left=before, right=after)"""
        w = self.config.window_width
        h = self.config.window_height
        split_x = int(w * self.config.split_position)
        
        # Left side (original)
        orig_surf = self.original_surfaces[self.current_frame]
        self._blit_centered(orig_surf, clip_rect=Rect(0, 0, split_x, h))
        
        # Right side (animated)
        anim_surf = self.surfaces[self.current_frame]
        self._blit_centered(anim_surf, clip_rect=Rect(split_x, 0, w - split_x, h))
        
        # Draw split line
        pygame.draw.line(self.screen, (255, 255, 255), (split_x, 0), (split_x, h), 2)
        
        # Labels
        self._render_text("BEFORE", (10, 10))
        self._render_text("AFTER", (split_x + 10, 10))
    
    def _render_split_horizontal(self):
        """Render horizontal split (top=before, bottom=after)"""
        w = self.config.window_width
        h = self.config.window_height
        split_y = int(h * self.config.split_position)
        
        # Top (original)
        orig_surf = self.original_surfaces[self.current_frame]
        self._blit_centered(orig_surf, clip_rect=Rect(0, 0, w, split_y))
        
        # Bottom (animated)
        anim_surf = self.surfaces[self.current_frame]
        self._blit_centered(anim_surf, clip_rect=Rect(0, split_y, w, h - split_y))
        
        # Draw split line
        pygame.draw.line(self.screen, (255, 255, 255), (0, split_y), (w, split_y), 2)
        
        # Labels
        self._render_text("BEFORE", (10, 10))
        self._render_text("AFTER", (10, split_y + 10))
    
    def _render_side_by_side(self):
        """Render side-by-side comparison"""
        w = self.config.window_width
        h = self.config.window_height
        half_w = w // 2
        
        # Adjust zoom for side-by-side
        side_zoom = self.zoom * 0.5
        
        # Left (original)
        orig_surf = self.original_surfaces[self.current_frame]
        scaled_w = int(orig_surf.get_width() * side_zoom)
        scaled_h = int(orig_surf.get_height() * side_zoom)
        scaled = pygame.transform.scale(orig_surf, (scaled_w, scaled_h))
        x = (half_w - scaled_w) // 2
        y = (h - scaled_h) // 2
        self.screen.blit(scaled, (x, y))
        
        # Right (animated)
        anim_surf = self.surfaces[self.current_frame]
        scaled = pygame.transform.scale(anim_surf, (scaled_w, scaled_h))
        x = half_w + (half_w - scaled_w) // 2
        self.screen.blit(scaled, (x, y))
        
        # Divider
        pygame.draw.line(self.screen, (100, 100, 100), (half_w, 0), (half_w, h), 1)
        
        # Labels
        self._render_text("BEFORE", (10, 10))
        self._render_text("AFTER", (half_w + 10, 10))
    
    def _blit_centered(self, surf: Surface, clip_rect: Optional[Rect] = None):
        """Blit surface centered with zoom and pan"""
        # Scale
        scaled_w = int(surf.get_width() * self.zoom)
        scaled_h = int(surf.get_height() * self.zoom)
        scaled = pygame.transform.scale(surf, (scaled_w, scaled_h))
        
        # Center position with pan
        x = (self.config.window_width - scaled_w) // 2 + self.pan_x
        y = (self.config.window_height - scaled_h) // 2 + self.pan_y
        
        # Draw grid if enabled
        if self.config.show_grid:
            self._draw_grid(x, y, scaled_w, scaled_h)
        
        # Clip if needed
        if clip_rect:
            self.screen.set_clip(clip_rect)
        
        self.screen.blit(scaled, (x, y))
        
        if clip_rect:
            self.screen.set_clip(None)
    
    def _create_onion_surface(self, frame_idx: int) -> Surface:
        """Create surface with onion skinning"""
        base = self.surfaces[frame_idx].copy()
        
        for i in range(1, self.config.onion_frames + 1):
            prev_idx = frame_idx - i
            if 0 <= prev_idx < len(self.surfaces):
                opacity = int(255 * self.config.onion_opacity * (0.5 ** (i - 1)))
                ghost = self.surfaces[prev_idx].copy()
                ghost.set_alpha(opacity)
                
                # Tint red
                tint = pygame.Surface(ghost.get_size(), pygame.SRCALPHA)
                tint.fill((255, 100, 100, 50))
                ghost.blit(tint, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
                
                # Create temp surface for compositing
                temp = pygame.Surface(base.get_size(), pygame.SRCALPHA)
                temp.blit(ghost, (0, 0))
                temp.blit(base, (0, 0))
                base = temp
        
        return base
    
    def _draw_grid(self, x: int, y: int, w: int, h: int):
        """Draw pixel grid overlay"""
        if self.zoom < 4:
            return  # Don't draw grid at low zoom
        
        pixel_size = self.zoom
        color = self.config.grid_color
        
        # Vertical lines
        for px in range(0, w + 1, int(pixel_size)):
            pygame.draw.line(self.screen, color, (x + px, y), (x + px, y + h), 1)
        
        # Horizontal lines
        for py in range(0, h + 1, int(pixel_size)):
            pygame.draw.line(self.screen, color, (x, y + py), (x + w, y + py), 1)
    
    def _render_info(self):
        """Render info panel"""
        lines = [
            f"Frame: {self.current_frame + 1}/{len(self.frames)}",
            f"Speed: {self.speed:.2f}x",
            f"Zoom: {self.zoom:.1f}x",
            f"{'▶ PLAYING' if self.playing else '⏸ PAUSED'}",
        ]
        
        if self.config.onion_skin:
            lines.append("🧅 Onion skin ON")
        
        if self.view_mode != ViewMode.NORMAL:
            lines.append(f"View: {self.view_mode.name}")
        
        y = 10
        for line in lines:
            self._render_text(line, (self.config.window_width - 150, y))
            y += 20
    
    def _render_help(self):
        """Render help overlay"""
        help_text = [
            "CONTROLS:",
            "",
            "SPACE      Play/Pause",
            "←/→        Prev/Next frame",
            "↑/↓        Speed up/down",
            "+/-        Zoom in/out",
            "1-9        Jump to %",
            "",
            "B          Toggle split view",
            "G          Toggle grid",
            "O          Toggle onion skin",
            "I          Toggle info",
            "",
            "S          Save frame",
            "E          Export animation",
            "R          Reset view",
            "",
            "H          Hide this help",
            "ESC/Q      Quit",
        ]
        
        # Semi-transparent background
        overlay = pygame.Surface((280, len(help_text) * 20 + 20), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        x = (self.config.window_width - 280) // 2
        y = (self.config.window_height - len(help_text) * 20) // 2
        self.screen.blit(overlay, (x, y))
        
        for i, line in enumerate(help_text):
            self._render_text(line, (x + 20, y + 10 + i * 20), color=(255, 255, 255))
    
    def _render_timeline(self):
        """Render timeline at bottom"""
        h = 30
        y = self.config.window_height - h
        w = self.config.window_width
        
        # Background
        pygame.draw.rect(self.screen, (30, 30, 30), (0, y, w, h))
        
        # Progress bar
        progress = (self.current_frame + 1) / len(self.frames)
        bar_w = int((w - 20) * progress)
        pygame.draw.rect(self.screen, (80, 80, 80), (10, y + 10, w - 20, 10))
        pygame.draw.rect(self.screen, (100, 180, 255), (10, y + 10, bar_w, 10))
        
        # Frame markers
        if len(self.frames) <= 32:
            marker_w = (w - 20) / len(self.frames)
            for i in range(len(self.frames)):
                mx = 10 + i * marker_w
                color = (255, 200, 100) if i == self.current_frame else (60, 60, 60)
                pygame.draw.rect(self.screen, color, (mx, y + 8, max(2, marker_w - 1), 14))
    
    def _render_text(
        self,
        text: str,
        pos: Tuple[int, int],
        color: Tuple[int, int, int] = (200, 200, 200)
    ):
        """Render text with shadow"""
        # Shadow
        shadow = self.font.render(text, True, (0, 0, 0))
        self.screen.blit(shadow, (pos[0] + 1, pos[1] + 1))
        
        # Text
        surface = self.font.render(text, True, color)
        self.screen.blit(surface, pos)
    
    def _save_frame(self):
        """Save current frame as PNG"""
        filename = f"frame_{self.current_frame:04d}.png"
        pygame.image.save(self.surfaces[self.current_frame], filename)
        print(f"Saved: {filename}")
    
    def _export(self):
        """Trigger export callback"""
        if self.on_export:
            self.on_export(self.frames)
            print("Export triggered")
        else:
            print("No export handler configured")


# =============================================================================
# Convenience Functions
# =============================================================================

def preview_animation(
    frames: List[np.ndarray],
    original: Optional[List[np.ndarray]] = None,
    title: str = "Animation Preview",
    fps: int = 12,
    zoom: float = 4.0
) -> None:
    """
    Quick preview of animation frames.
    
    Args:
        frames: List of RGBA numpy arrays
        original: Optional original frames for comparison
        title: Window title
        fps: Animation frames per second
        zoom: Initial zoom level
    """
    if not PYGAME_AVAILABLE:
        print("Preview requires pygame. Install with: pip install pygame")
        print("Alternatively, export to GIF and view in external program.")
        return
    
    config = PreviewConfig(
        window_title=title,
        animation_fps=fps,
        initial_zoom=zoom
    )
    
    preview = PreviewWindow(frames, original, config)
    preview.run()


def preview_sprite(
    pixels: np.ndarray,
    title: str = "Sprite Preview",
    zoom: float = 8.0
) -> None:
    """
    Preview a single sprite (static).
    
    Args:
        pixels: RGBA numpy array
        title: Window title
        zoom: Initial zoom level
    """
    preview_animation([pixels], title=title, fps=1, zoom=zoom)


def check_pygame_available() -> bool:
    """Check if pygame is available for preview"""
    return PYGAME_AVAILABLE


# =============================================================================
# Integration Helper
# =============================================================================

class PreviewSession:
    """
    Interactive preview session with parameter adjustment.
    
    Example:
        def regenerate(params):
            return generate_animation(sprite, **params)
        
        session = PreviewSession(initial_frames, regenerate)
        session.add_parameter('intensity', 0.0, 2.0, 1.0)
        session.add_parameter('speed', 0.1, 3.0, 1.0)
        session.run()
    """
    
    def __init__(
        self,
        initial_frames: List[np.ndarray],
        regenerate_callback: Callable[[Dict[str, Any]], List[np.ndarray]],
        original_frames: Optional[List[np.ndarray]] = None
    ):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame required for preview sessions")
        
        self.frames = initial_frames
        self.original = original_frames or initial_frames
        self.regenerate = regenerate_callback
        self.parameters: Dict[str, Dict[str, Any]] = {}
        self.current_values: Dict[str, Any] = {}
    
    def add_parameter(
        self,
        name: str,
        min_val: float,
        max_val: float,
        default: float,
        step: Optional[float] = None
    ):
        """Add an adjustable parameter"""
        if step is None:
            step = (max_val - min_val) / 20
        
        self.parameters[name] = {
            'min': min_val,
            'max': max_val,
            'step': step,
            'default': default
        }
        self.current_values[name] = default
    
    def run(self):
        """Run the preview session"""
        def on_param_change(name: str, value: Any) -> List[np.ndarray]:
            self.current_values[name] = value
            return self.regenerate(self.current_values)
        
        config = PreviewConfig(
            window_title="Sprite Animator - Interactive Preview",
            initial_zoom=4.0
        )
        
        preview = PreviewWindow(
            self.frames,
            self.original,
            config,
            on_parameter_change=on_param_change
        )
        preview.parameters = self.current_values
        preview.run()
