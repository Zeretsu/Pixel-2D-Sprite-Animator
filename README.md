<p align="center">
  <h1 align="center">🎮 Sprite Animator ✨</h1>
  <p align="center">
    <b>Smart procedural animation for pixel art sprites</b>
  </p>
  <p align="center">
    <a href="#installation">Installation</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#effects-gallery">Effects</a> •
    <a href="#documentation">Docs</a>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License MIT">
  <img src="https://img.shields.io/badge/effects-35+-orange.svg" alt="35+ Effects">
</p>

---

## ✨ What is Sprite Animator?

Sprite Animator is a **command-line tool** that brings your pixel art to life with procedural animations. Simply provide a sprite, and the tool will **automatically analyze** its colors, shapes, and edges to suggest the perfect animation effect — or choose from **35+ handcrafted effects** yourself.

> **No keyframing required.** Just point, click, and animate.

---

## 🎬 Demo

<!-- Replace with actual GIF demos of your effects -->
```
┌─────────────────────────────────────────────────────────────┐
│  🔥 Flame    💧 Water    ⚡ Electric    ✨ Sparkle    🌀 Spin │
│                                                             │
│     [torch]     [pool]     [sword]      [gem]      [coin]   │
│       ↓           ↓          ↓           ↓          ↓       │
│    animated   animated   animated    animated   animated    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Features

| Feature | Description |
|:--------|:------------|
| 🔍 **Auto-Detection** | Analyzes sprite color, shape & edges to recommend the best effect |
| 🎨 **35+ Effects** | Flame, water, glitch, bounce, dissolve, electric, and many more |
| 🎭 **Animation Principles** | Built-in squash & stretch, anticipation, overshoot, motion blur |
| 🖼️ **Pixel Art Converter** | Convert any image to pixel art with palette support (Gameboy, NES, PICO-8) |
| ⚡ **Particle Systems** | Sparks, smoke, snow, magic dust, and customizable emitters |
| 👁️ **Real-time Preview** | Instant preview window to tweak settings live |
| 📦 **Multiple Formats** | Export as GIF, spritesheet, or individual frames |
| 🎛️ **Presets System** | Save and share complex effect configurations |

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Steps

```bash
# Clone the repository
git clone https://github.com/Zeretsu/Pixel-2D-Sprite-Animator.git
cd sprite-animator

# Install required dependencies
pip install -r requirements.txt

# (Optional) Install pygame for real-time preview
pip install pygame
```

---

## ⚡ Quick Start

### Auto-detect the best effect
```bash
python main.py my_sprite.png
```

### Use a specific effect
```bash
python main.py torch.png --effect flame
```

### Customize your animation
```bash
python main.py character.png --effect bounce --frames 12 --intensity 1.5 --format spritesheet
```

### Preview in real-time
```bash
python main.py gem.png --effect sparkle --preview
```

---

## 🎨 Effects Gallery

<details>
<summary><b>🔥 Basic Effects</b> (click to expand)</summary>

| Effect | Description |
|:-------|:------------|
| `flame` | Fire/flickering with upward drift |
| `water` | Wave distortion and ripples |
| `float` | Gentle up-and-down bobbing |
| `sparkle` | Magic glitter particles |
| `sway` | Side-to-side swaying (plants, candles) |
| `pulse` | Breathing/scaling animation |
| `smoke` | Soft drifting for clouds/smoke |
| `wobble` | Jelly-like elastic deformation |
| `glitch` | Digital corruption/RGB split |
| `shake` | Screen shake/vibration |
| `bounce` | Bouncing with squash/stretch |
| `spin` | Rotation animation |
| `melt` | Melting/dripping |
| `electric` | Lightning/electricity |

</details>

<details>
<summary><b>⚔️ Status & Game Effects</b> (click to expand)</summary>

| Effect | Description |
|:-------|:------------|
| `shadow` | Shadow/afterimage trail |
| `teleport` | Warp/materialize effect |
| `charge` | Power/energy charging |
| `damage` | Hit/hurt flash effect |
| `freeze` | Ice/frozen effect |
| `poison` | Toxic/venom dripping |
| `petrify` | Stone/statue transformation |
| `hologram` | Holographic projection |
| `chromatic` | RGB aberration/split |
| `levitate` | Magical floating hover |

</details>

<details>
<summary><b>🌟 Advanced Effects</b> (click to expand)</summary>

| Effect | Description |
|:-------|:------------|
| `particles` | Customizable particle systems |
| `motion_blur` | Linear, radial, or zoom blur |
| `speed_lines` | Anime-style action lines |
| `motion_trail` | Afterimage/ghost trails |
| `ribbon_trail` | Flowing ribbon trails |
| `keyframe` | Custom keyframe animations |
| `fire_element` | Flames emanating from sprite |
| `water_element` | Ripples and water effects |
| `ice_element` | Frost and crystal sparkles |

</details>

<details>
<summary><b>🦋 Creature Effects</b> (click to expand)</summary>

| Effect | Description |
|:-------|:------------|
| `flap` | Wing flapping for bats, birds, butterflies |
| `hover_flap` | Hovering with figure-8 wing motion |
| `glide_flap` | Gliding with occasional flaps |

</details>

---

## 📖 Documentation

### Command Reference

```
python main.py <input> [options]
```

| Option | Default | Description |
|:-------|:--------|:------------|
| `-e, --effect` | auto | Animation effect to apply |
| `-f, --frames` | 8 | Number of animation frames |
| `-s, --speed` | 1.0 | Animation speed multiplier |
| `-i, --intensity` | 1.0 | Effect intensity (0.0 - 2.0) |
| `-o, --output` | auto | Output file path |
| `--format` | gif | Output format: `gif`, `spritesheet`, `frames` |
| `--preview` | - | Open real-time preview window |
| `--analyze` | - | Show sprite analysis without animating |

### Animation Principles

Add Disney-style life to your animations:

```bash
python main.py ball.png --effect bounce \
    --squash-stretch 0.2 \
    --anticipation 0.2 \
    --overshoot 0.1 \
    --motion-blur
```

| Option | Description |
|:-------|:------------|
| `--squash-stretch` | Deformation during motion (0.0 - 0.5) |
| `--anticipation` | Wind-up before motion (0.0 - 0.4) |
| `--overshoot` | Settle past target (0.0 - 0.3) |
| `--motion-blur` | Enable motion blur |

### Pixel Art Converter

Convert any image to pixel art:

```bash
python main.py photo.jpg --pixelate \
    --pixel-width 64 \
    --palette gameboy \
    --dither ordered \
    --pixel-outline
```

**Available palettes:** `gameboy`, `nes`, `pico8`, `endesga32`, `grayscale`

### Using Presets

```bash
# List all presets
python main.py --list-presets

# Get preset info
python main.py --preset-info torch_realistic

# Use a preset
python main.py torch.png --preset torch_realistic
```

---

## ⚙️ Configuration

Customize default settings in `config.yaml`:

```yaml
animation:
  frame_count: 8
  frame_duration: 100  # milliseconds

effects:
  flame:
    speed: 1.0
    intensity: 0.8
    ember_count: 5
```

---

## 📁 Project Structure

```
sprite-animator/
├── 📄 config.yaml        # Global configuration
├── 🐍 main.py            # CLI entry point
├── 📋 requirements.txt   # Python dependencies
└── 📂 src/
    ├── 🤖 ai/            # Prompt interpretation (experimental)
    ├── ⚙️ core/           # Animation engine, parser, exporter
    ├── 🔍 detection/      # Sprite analysis (color, shape, edges)
    └── 🎨 procedural/     # All effect implementations
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/amazing-effect`)
3. 💾 Commit your changes (`git commit -m 'Add amazing effect'`)
4. 📤 Push to the branch (`git push origin feature/amazing-effect`)
5. 🎉 Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ for pixel art enthusiasts
</p>
