import hashlib
import os

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

class PaletteManager:
    """
    Manages class-to-color mapping for error visualizations.
    Supports loading from files and dynamic color generation.
    """
    def __init__(self, config_path=None):
        self.palette = {}
        # Base colors for error types
        self.error_type_defaults = {
            'correct': '#00ff00',           # Green
            'false positive': '#ff0000',    # Red
            'false negative': '#0080ff',    # Blue
            'misclassification': '#ffea00'  # Yellow
        }
        
        if config_path:
            self.load_config(config_path)

    def load_config(self, path):
        """Loads palette from a text file (class - type: #hex)"""
        if not os.path.exists(path):
            return
        
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                key, val = line.split(':', 1)
                self.palette[key.strip().lower()] = val.strip()

    def _generate_color(self, name):
        """Generates a stable color from a string."""
        hash_obj = hashlib.md5(name.encode())
        hex_digest = hash_obj.hexdigest()
        return (
            int(hex_digest[0:2], 16),
            int(hex_digest[2:4], 16),
            int(hex_digest[4:6], 16)
        )

    def get_color(self, class_name, error_type):
        """
        error_type: 'Correct', 'False Positive', 'False Negative', 'Misclassification'
        """
        class_name_norm = class_name.lower()
        error_type_norm = error_type.lower()
        
        key = f"{class_name_norm} - {error_type_norm}"
        if key in self.palette:
            return hex_to_rgb(self.palette[key])
        
        # If no specific mapping, generate base class color and adjust?
        # For consistency with previous requests, we prioritize error type "vibe"
        # but try to keep class distinct if possible.
        
        # Simple strategy: 
        # Correct -> Green-ish
        # FP -> Red-ish
        # FN -> Blue-ish
        # Miscalc -> Yellow-ish
        
        # To make classes distinct while keeping error "meaning":
        # We blend a stable class color with the error type color.
        base_error_hex = self.error_type_defaults.get(error_type_norm, '#ffffff')
        base_error_rgb = hex_to_rgb(base_error_hex)
        
        class_rgb = self._generate_color(class_name_norm)
        
        # Blend (70% error type color, 30% class unique color)
        # This keeps the error type immediately recognizable while making different classes distinct.
        r = int(base_error_rgb[0] * 0.7 + class_rgb[0] * 0.3)
        g = int(base_error_rgb[1] * 0.7 + class_rgb[1] * 0.3)
        b = int(base_error_rgb[2] * 0.7 + class_rgb[2] * 0.3)
        
        return (r, g, b)

# Legacy global instance and function for backward compatibility
_GLOBAL_PALETTE = PaletteManager()

def get_error_color(class_name, error_type):
    return _GLOBAL_PALETTE.get_color(class_name, error_type)

# Expose internal defaults if needed
ERROR_PALETTE = _GLOBAL_PALETTE.palette 
