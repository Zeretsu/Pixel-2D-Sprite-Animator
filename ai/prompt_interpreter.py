"""
AI Prompt Interpreter - Smart animation intent detection from natural language
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class AnimationIntent:
    """Parsed animation intent from user prompt"""
    effect: str  # flame, water, sparkle, sway, float, pulse, smoke, wobble
    confidence: float
    intensity: float = 1.0
    speed: float = 1.0
    frame_count: int = 8
    reasoning: str = ""
    extra_params: Dict = field(default_factory=dict)


class PromptInterpreter:
    """
    Interprets natural language prompts to determine animation parameters.
    Uses keyword matching, context analysis, and fuzzy matching for smart detection.
    """
    
    # Effect keyword mappings with weights
    EFFECT_KEYWORDS = {
        'flame': {
            'primary': ['flame', 'fire', 'burn', 'burning', 'blaze', 'inferno', 'torch', 'candle', 'ember', 'fiery'],
            'secondary': ['hot', 'heat', 'warm', 'lava', 'magma', 'hell', 'demon', 'phoenix', 'dragon'],
            'context': ['flicker', 'glow', 'orange', 'red', 'yellow'],
        },
        'water': {
            'primary': ['water', 'wave', 'ocean', 'sea', 'river', 'lake', 'pond', 'aqua', 'liquid', 'fluid'],
            'secondary': ['swim', 'splash', 'ripple', 'tide', 'current', 'flow', 'stream', 'pool'],
            'context': ['blue', 'wet', 'cool', 'fish', 'underwater', 'reflection'],
        },
        'sparkle': {
            'primary': ['sparkle', 'glitter', 'twinkle', 'shimmer', 'shine', 'star', 'magic', 'fairy'],
            'secondary': ['crystal', 'gem', 'diamond', 'jewel', 'treasure', 'gold', 'silver', 'metallic'],
            'context': ['bright', 'light', 'glow', 'magical', 'enchanted', 'spell', 'power'],
        },
        'sway': {
            'primary': ['sway', 'swing', 'wave', 'bend', 'grass', 'plant', 'tree', 'leaf', 'branch'],
            'secondary': ['wind', 'breeze', 'blow', 'flutter', 'rustle', 'garden', 'forest', 'nature'],
            'context': ['gentle', 'soft', 'natural', 'organic', 'green', 'alive'],
        },
        'float': {
            'primary': ['float', 'hover', 'levitate', 'fly', 'drift', 'bob', 'suspended', 'ghost'],
            'secondary': ['air', 'cloud', 'balloon', 'feather', 'spirit', 'ethereal', 'weightless'],
            'context': ['up', 'down', 'gentle', 'slow', 'peaceful', 'dreamy'],
        },
        'pulse': {
            'primary': ['pulse', 'beat', 'throb', 'heart', 'heartbeat', 'pump', 'rhythm', 'breathing'],
            'secondary': ['glow', 'expand', 'contract', 'alive', 'living', 'organic', 'life'],
            'context': ['steady', 'regular', 'grow', 'shrink', 'scale'],
        },
        'smoke': {
            'primary': ['smoke', 'fog', 'mist', 'cloud', 'steam', 'vapor', 'haze', 'smog'],
            'secondary': ['chimney', 'factory', 'cigarette', 'incense', 'potion', 'cauldron', 'mysterious'],
            'context': ['gray', 'white', 'wispy', 'drift', 'fade', 'dissipate'],
        },
        'wobble': {
            'primary': ['wobble', 'jiggle', 'shake', 'wiggle', 'bounce', 'jelly', 'slime', 'blob'],
            'secondary': ['rubber', 'elastic', 'squishy', 'soft', 'gelatinous', 'pudding'],
            'context': ['funny', 'cute', 'silly', 'playful', 'cartoon'],
        },
    }
    
    # Intensity modifiers
    INTENSITY_MODIFIERS = {
        'high': ['intense', 'strong', 'powerful', 'extreme', 'heavy', 'big', 'large', 'dramatic', 'violent', 'wild', 'crazy', 'maximum', 'max', 'aggressive'],
        'medium': ['normal', 'medium', 'moderate', 'regular', 'standard', 'balanced'],
        'low': ['subtle', 'gentle', 'soft', 'light', 'small', 'tiny', 'mild', 'calm', 'minimal', 'slight', 'delicate'],
    }
    
    # Speed modifiers
    SPEED_MODIFIERS = {
        'fast': ['fast', 'quick', 'rapid', 'speedy', 'swift', 'hasty', 'energetic', 'frantic', 'hyper'],
        'medium': ['normal', 'medium', 'moderate', 'regular', 'standard'],
        'slow': ['slow', 'lazy', 'gentle', 'calm', 'peaceful', 'relaxed', 'steady', 'gradual', 'smooth'],
    }
    
    # Frame count hints
    FRAME_HINTS = {
        'more': ['smooth', 'detailed', 'fluid', 'many', 'lots', 'complex', 'long'],
        'less': ['simple', 'basic', 'short', 'few', 'quick', 'snappy'],
    }
    
    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency"""
        self.word_pattern = re.compile(r'\b\w+\b')
    
    def interpret(self, prompt: str, sprite_analysis: Optional[Dict] = None) -> AnimationIntent:
        """
        Interpret a natural language prompt to determine animation intent.
        
        Args:
            prompt: User's description of desired animation
            sprite_analysis: Optional analysis results from SpriteAnalyzer
            
        Returns:
            AnimationIntent with detected effect and parameters
        """
        prompt_lower = prompt.lower().strip()
        words = set(self.word_pattern.findall(prompt_lower))
        
        # Detect effect
        effect, effect_confidence, reasoning = self._detect_effect(words, prompt_lower)
        
        # If no clear effect from prompt, use sprite analysis
        if effect_confidence < 0.3 and sprite_analysis:
            suggestions = sprite_analysis.get('suggestions', [])
            if suggestions:
                effect = suggestions[0].get('effect', effect)
                effect_confidence = max(effect_confidence, suggestions[0].get('confidence', 0.5))
                reasoning += f" Combined with visual analysis suggesting {effect}."
        
        # Detect modifiers
        intensity = self._detect_intensity(words)
        speed = self._detect_speed(words)
        frame_count = self._detect_frame_count(words)
        
        # Build extra params based on prompt analysis
        extra_params = self._extract_extra_params(words, effect)
        
        return AnimationIntent(
            effect=effect,
            confidence=effect_confidence,
            intensity=intensity,
            speed=speed,
            frame_count=frame_count,
            reasoning=reasoning,
            extra_params=extra_params
        )
    
    def _detect_effect(self, words: set, full_prompt: str) -> Tuple[str, float, str]:
        """Detect the intended effect from keywords"""
        scores = {}
        reasons = {}
        
        for effect, keywords in self.EFFECT_KEYWORDS.items():
            score = 0.0
            matched = []
            
            # Primary keywords (high weight)
            primary_matches = words & set(keywords['primary'])
            if primary_matches:
                score += len(primary_matches) * 0.4
                matched.extend(primary_matches)
            
            # Secondary keywords (medium weight)
            secondary_matches = words & set(keywords['secondary'])
            if secondary_matches:
                score += len(secondary_matches) * 0.25
                matched.extend(secondary_matches)
            
            # Context keywords (low weight)
            context_matches = words & set(keywords['context'])
            if context_matches:
                score += len(context_matches) * 0.1
                matched.extend(context_matches)
            
            # Phrase matching bonus
            for phrase in keywords['primary']:
                if phrase in full_prompt:
                    score += 0.15
            
            scores[effect] = min(score, 1.0)  # Cap at 1.0
            if matched:
                reasons[effect] = f"Matched: {', '.join(matched[:3])}"
        
        # Find best match
        if scores:
            best_effect = max(scores, key=scores.get)
            best_score = scores[best_effect]
            
            if best_score > 0:
                return best_effect, best_score, reasons.get(best_effect, "")
        
        # Default to float if nothing detected
        return 'float', 0.2, "No specific effect detected, defaulting to float"
    
    def _detect_intensity(self, words: set) -> float:
        """Detect intensity modifier from keywords"""
        high_matches = words & set(self.INTENSITY_MODIFIERS['high'])
        low_matches = words & set(self.INTENSITY_MODIFIERS['low'])
        
        if high_matches:
            return 1.5 + (len(high_matches) - 1) * 0.2  # 1.5 - 2.0
        elif low_matches:
            return 0.5 - (len(low_matches) - 1) * 0.1  # 0.3 - 0.5
        return 1.0
    
    def _detect_speed(self, words: set) -> float:
        """Detect speed modifier from keywords"""
        fast_matches = words & set(self.SPEED_MODIFIERS['fast'])
        slow_matches = words & set(self.SPEED_MODIFIERS['slow'])
        
        if fast_matches:
            return 1.5 + (len(fast_matches) - 1) * 0.3  # 1.5 - 2.5
        elif slow_matches:
            return 0.5 - (len(slow_matches) - 1) * 0.1  # 0.3 - 0.5
        return 1.0
    
    def _detect_frame_count(self, words: set) -> int:
        """Detect frame count hints from keywords"""
        more_frames = words & set(self.FRAME_HINTS['more'])
        less_frames = words & set(self.FRAME_HINTS['less'])
        
        if more_frames:
            return 12 + len(more_frames) * 2  # 14-20
        elif less_frames:
            return max(4, 8 - len(less_frames) * 2)  # 4-6
        return 8
    
    def _extract_extra_params(self, words: set, effect: str) -> Dict:
        """Extract effect-specific extra parameters"""
        params = {}
        
        if effect == 'flame':
            if 'rising' in words or 'upward' in words:
                params['rise_speed'] = 1.5
            if 'candle' in words or 'torch' in words:
                params['flicker_intensity'] = 0.8
        
        elif effect == 'water':
            if 'calm' in words or 'still' in words:
                params['wave_amplitude'] = 0.5
            if 'stormy' in words or 'rough' in words:
                params['wave_amplitude'] = 2.0
                params['wave_frequency'] = 2.0
        
        elif effect == 'sparkle':
            if 'rainbow' in words or 'colorful' in words:
                params['color_shift'] = True
            if 'random' in words or 'scattered' in words:
                params['density'] = 1.5
        
        elif effect == 'sway':
            if 'strong' in words or 'windy' in words:
                params['sway_amount'] = 1.5
        
        elif effect == 'smoke':
            if 'thick' in words or 'heavy' in words:
                params['density'] = 1.5
            if 'wispy' in words or 'thin' in words:
                params['fade'] = True
                params['turbulence'] = 0.2
        
        return params
    
    def suggest_prompt(self, effect: str) -> str:
        """Generate a helpful prompt suggestion for an effect"""
        suggestions = {
            'flame': "Try: 'flickering torch flame' or 'intense blazing fire'",
            'water': "Try: 'calm rippling water' or 'stormy ocean waves'",
            'sparkle': "Try: 'magical sparkles' or 'twinkling stars'",
            'sway': "Try: 'grass swaying in breeze' or 'tree branches in wind'",
            'float': "Try: 'gently floating ghost' or 'hovering magical orb'",
            'pulse': "Try: 'glowing heartbeat' or 'pulsing energy'",
            'smoke': "Try: 'wispy fog drifting' or 'thick chimney smoke'",
            'wobble': "Try: 'jelly wobbling' or 'bouncy slime blob'",
        }
        return suggestions.get(effect, "Describe your desired animation")


class SmartAnimator:
    """
    High-level AI animator that combines prompt interpretation with sprite analysis
    """
    
    def __init__(self):
        self.interpreter = PromptInterpreter()
    
    def analyze_and_suggest(
        self,
        prompt: str,
        sprite_analysis: Optional[Dict] = None
    ) -> AnimationIntent:
        """
        Analyze prompt and sprite to suggest best animation approach
        """
        intent = self.interpreter.interpret(prompt, sprite_analysis)
        
        # Boost confidence if prompt and analysis agree
        if sprite_analysis:
            suggestions = sprite_analysis.get('suggestions', [])
            if suggestions and suggestions[0].get('effect') == intent.effect:
                intent.confidence = min(1.0, intent.confidence + 0.2)
                intent.reasoning += " (Confirmed by visual analysis)"
        
        return intent
