"""
palette Finder - Image Analysis Tool

This module provides functionality to:
1. Parse and load images
2. Reduce the number of colors using k-means clustering in OKLAB color space
"""

import numpy as np
import sys
from typing import Tuple, Optional, List
import signal

# Import utility functions
from utils.load_image import load_image as _load_image
from utils.reduce_colors import reduce_colors as _reduce_colors
from utils.save_results import save_results as _save_results


class PaletteFinder:
    """
    A class for analyzing images to find color palettes.
    """
    
    def __init__(self, image_path: str):
        """
        Initialize the PaletteFinder with an image.
        
        Args:
            image_path: Path to the image file
        """
        self.image_path = image_path
        self.image = None
        self.image_array = None
        self.reduced_image = None
        self.color_palette = None
    
    def load_image(self) -> np.ndarray:
        """
        Load and parse the image file.
        
        Returns:
            numpy array representation of the image
        """
        self.image, self.image_array = _load_image(self.image_path)
        return self.image_array
    
    def reduce_colors(self, n_colors: int = 8, random_state: int = 42,
                     weight_L: float = 1.0, weight_a: float = 1.0, weight_b: float = 1.0,
                     seed_colors: Optional[List[Tuple[int, int, int]]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reduce the number of colors in the image using k-means clustering in OKLAB space.
        
        Args:
            n_colors: Number of colors to reduce to
            random_state: Random state for reproducibility
            weight_L: Weight for L (lightness) channel
            weight_a: Weight for a (green-red) channel
            weight_b: Weight for b (blue-yellow) channel
            seed_colors: Optional list of RGB tuples to use as initial centers
            
        Returns:
            Tuple of (reduced image array, color palette)
        """
        if self.image_array is None:
            self.load_image()
        
        self.reduced_image, self.color_palette = _reduce_colors(
            self.image_array, 
            n_colors=n_colors, 
            random_state=random_state,
            weight_L=weight_L,
            weight_a=weight_a,
            weight_b=weight_b,
            seed_colors=seed_colors
        )
        return self.reduced_image, self.color_palette
    
    def save_results(self, output_prefix: str = "output", seed_colors=None):
        """
        Save the processed images to files.
        
        Args:
            output_prefix: Prefix for output filenames
            seed_colors: Optional list of seed colors to show matches
        """
        _save_results(self.reduced_image, output_prefix=output_prefix,
                     seed_colors=seed_colors, color_palette_rgb=self.color_palette)


def parse_seed_colors(seed_file: str) -> List[Tuple[int, int, int]]:
    """
    Parse seed colors from a file.
    
    File format (one color per line):
        255,0,0       # Red (comment)
        #00ff00       # Green (hex)
        0 0 255       # Blue (space separated)
        # This is a comment line
    
    Args:
        seed_file: Path to file containing seed colors
        
    Returns:
        List of RGB tuples
    """
    seed_colors = []
    
    with open(seed_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            try:
                # Check if line starts with #
                if line.startswith('#'):
                    # Try to parse as hex color (need at least #RRGGBB = 7 chars)
                    if len(line) >= 7:
                        hex_part = line[1:7]
                        # Validate that it's actually hex digits
                        try:
                            r = int(hex_part[0:2], 16)
                            g = int(hex_part[2:4], 16)
                            b = int(hex_part[4:6], 16)
                            
                            # Successfully parsed as hex color
                            seed_colors.append((r, g, b))
                            print(f"Parsed seed color {len(seed_colors)}: RGB({r:3d}, {g:3d}, {b:3d}) = #{r:02x}{g:02x}{b:02x}")
                            continue
                        except ValueError:
                            # Not valid hex, treat as comment
                            pass
                    
                    # If we get here, it's a comment line - skip it
                    continue
                
                # Not a # line, so try to parse as RGB values
                # Remove inline comments
                if '#' in line:
                    line = line.split('#')[0].strip()
                
                # Try comma or space separated
                parts = line.replace(',', ' ').split()
                if len(parts) != 3:
                    print(f"Warning: Invalid color format on line {line_num}: {line}")
                    continue
                
                r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                
                # Validate range
                if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                    print(f"Warning: Color values out of range on line {line_num}: RGB({r}, {g}, {b})")
                    continue
                
                seed_colors.append((r, g, b))
                print(f"Parsed seed color {len(seed_colors)}: RGB({r:3d}, {g:3d}, {b:3d}) = #{r:02x}{g:02x}{b:02x}")
                
            except (ValueError, IndexError) as e:
                print(f"Warning: Failed to parse color on line {line_num}: '{line}' - {e}")
                continue
    
    return seed_colors


if __name__ == "__main__":
    
    def signal_handler(sig, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\nProcess interrupted by user (Ctrl+C)")
        print("Cleaning up and exiting...")
        sys.exit(0)
    
    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    if len(sys.argv) < 2:
        print("Usage: python palette_finder.py <image_path> [n_colors] [weight_L] [weight_a] [weight_b] [--seeds <file>]")
        print("\nExamples:")
        print("  python palette_finder.py image.jpg 8")
        print("  python palette_finder.py image.jpg 32 1 2 2  # Emphasize color over lightness")
        print("  python palette_finder.py image.jpg 16 --seeds colors.txt  # Force specific colors")
        print("\nWeights:")
        print("  weight_L: Weight for lightness (default 1.0)")
        print("  weight_a: Weight for green-red axis (default 1.0)")
        print("  weight_b: Weight for blue-yellow axis (default 1.0)")
        print("\nSeed Colors File Format:")
        print("  255,0,0       # Red")
        print("  #00ff00       # Green (hex)")
        print("  0 0 255       # Blue")
        print("\nOutput files will be saved with the input filename as prefix")
        sys.exit(1)
    
    # Parse arguments
    image_path = sys.argv[1]
    args = sys.argv[2:]
    
    # Defaults
    n_colors = 8
    weight_L = 1.0
    weight_a = 1.0
    weight_b = 1.0
    seed_file = None
    
    # Parse positional and flag arguments
    pos_args = []
    i = 0
    while i < len(args):
        if args[i] == '--seeds':
            if i + 1 < len(args):
                seed_file = args[i + 1]
                i += 2
            else:
                print("Error: --seeds requires a filename")
                sys.exit(1)
        else:
            pos_args.append(args[i])
            i += 1
    
    # Parse positional arguments
    if len(pos_args) > 0:
        n_colors = int(pos_args[0])
    if len(pos_args) > 1:
        weight_L = float(pos_args[1])
    if len(pos_args) > 2:
        weight_a = float(pos_args[2])
    if len(pos_args) > 3:
        weight_b = float(pos_args[3])
    
    seed_weight = 1000.0  # Default seed weight
    
    # Parse seed colors if provided
    seed_colors = None
    if seed_file:
        try:
            seed_colors = parse_seed_colors(seed_file)
            if len(seed_colors) == 0:
                print(f"Warning: No valid seed colors found in {seed_file}")
        except FileNotFoundError:
            print(f"Error: Seed file not found: {seed_file}")
            sys.exit(1)
    
    # Extract filename without extension for output prefix
    output_prefix = image_path.rsplit('.', 1)[0]
    
    print(f"Analyzing image: {image_path}")
    print(f"Reducing to {n_colors} colors using OKLAB color space...")
    print(f"Channel weights: L={weight_L:.2f}, a={weight_a:.2f}, b={weight_b:.2f}")
    if seed_colors:
        print(f"Using {len(seed_colors)} seed colors from: {seed_file}")
        print(f"Seed weight multiplier: {seed_weight:.0f}x")
    print(f"Output files will be saved as: {output_prefix}_*.png")
    print("(Press Ctrl+C to abort at any time)\n")
    
    try:
        # Create finder and process
        finder = PaletteFinder(image_path)
        finder.load_image()
        reduced_image, color_palette = finder.reduce_colors(
            n_colors=n_colors,
            weight_L=weight_L,
            weight_a=weight_a,
            weight_b=weight_b,
            seed_colors=seed_colors
        )
        finder.save_results(output_prefix=output_prefix, seed_colors=seed_colors)
        
        print(f"\n{'=' * 60}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'=' * 60}")
        print(f"Found {len(color_palette)} colors")
        print(f"\nOutput files:")
        print(f"  - {output_prefix}_reduced.png")
        print(f"  - {output_prefix}_palette.txt")
        print(f"{'=' * 60}")
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
        print("Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
