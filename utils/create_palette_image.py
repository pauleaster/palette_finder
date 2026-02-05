import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple
import colorsys

def sort_colors_by_hue_lightness(colors_rgb: np.ndarray) -> np.ndarray:
    """
    Sort colors by hue, then lightness for a natural arrangement.
    
    Args:
        colors_rgb: Array of RGB colors (n, 3)
        
    Returns:
        Indices to sort colors
    """
    # Convert RGB to HSL for sorting
    hsl_colors = []
    for color in colors_rgb:
        r, g, b = color / 255.0
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        hsl_colors.append((h, l, s))
    
    hsl_array = np.array(hsl_colors)
    
    # Sort by hue first, then lightness
    # For grayscale colors (low saturation), sort by lightness
    indices = np.arange(len(hsl_array))
    
    # Separate grays (low saturation) from chromatic colors
    gray_threshold = 0.1
    is_gray = hsl_array[:, 2] < gray_threshold  # saturation < 0.1
    
    gray_indices = indices[is_gray]
    color_indices = indices[~is_gray]
    
    # Sort grays by lightness (dark to light)
    if len(gray_indices) > 0:
        gray_lightness = hsl_array[gray_indices, 1]
        gray_indices = gray_indices[np.argsort(gray_lightness)]
    
    # Sort chromatic colors by hue, then lightness
    if len(color_indices) > 0:
        color_hues = hsl_array[color_indices, 0]
        color_lightness = hsl_array[color_indices, 1]
        # Sort by hue primarily, lightness secondarily
        color_indices = color_indices[np.lexsort((color_lightness, color_hues))]
    
    # Combine: grays first (dark to light), then chromatic (by hue)
    return np.concatenate([gray_indices, color_indices])


def create_palette_image(color_palette_rgb: np.ndarray, 
                         output_path: str,
                         width: int = 1400,
                         show_hex: bool = True) -> None:
    """
    Create a palette swatch image arranged in color space.
    
    Image ratio is 1.4:1 (suitable for A-series paper orientation).
    Swatches are square with mid-grey (#808080) separator.
    
    Args:
        color_palette_rgb: Array of RGB colors (n, 3)
        output_path: Path to save the image
        width: Width of output image (height will be width/1.4)
        show_hex: Whether to show hex codes on swatches
    """
    n_colors = len(color_palette_rgb)
    
    # Calculate dimensions
    height = int(width / 1.4)
    
    # Sort colors for better visual arrangement
    sort_indices = sort_colors_by_hue_lightness(color_palette_rgb)
    sorted_colors = color_palette_rgb[sort_indices]
    
    # Calculate grid layout
    # Try to make roughly square grid
    cols = int(np.ceil(np.sqrt(n_colors * 1.4)))  # Adjusted for aspect ratio
    rows = int(np.ceil(n_colors / cols))
    
    # Calculate swatch size with separator
    separator_width = max(2, width // 200)  # ~0.5% of width
    available_width = width - (cols + 1) * separator_width
    available_height = height - (rows + 1) * separator_width
    
    swatch_size = min(available_width // cols, available_height // rows)
    
    # Create image with mid-grey background
    mid_grey = (128, 128, 128)
    img = Image.new('RGB', (width, height), mid_grey)
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fall back to default if not available
    try:
        # Try to find a truetype font
        font_size = max(12, swatch_size // 8)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Calculate starting position to center the grid
    total_grid_width = cols * swatch_size + (cols + 1) * separator_width
    total_grid_height = rows * swatch_size + (rows + 1) * separator_width
    start_x = (width - total_grid_width) // 2
    start_y = (height - total_grid_height) // 2
    
    # Draw swatches
    for i, color in enumerate(sorted_colors):
        row = i // cols
        col = i % cols
        
        x = start_x + separator_width + col * (swatch_size + separator_width)
        y = start_y + separator_width + row * (swatch_size + separator_width)
        
        # Convert color to integer RGB
        r = int(np.clip(color[0], 0, 255))
        g = int(np.clip(color[1], 0, 255))
        b = int(np.clip(color[2], 0, 255))
        
        # Draw swatch
        draw.rectangle(
            [x, y, x + swatch_size, y + swatch_size],
            fill=(r, g, b)
        )
        
        # Add hex code if requested and swatch is large enough
        if show_hex and swatch_size > 60:
            hex_code = f"#{r:02x}{g:02x}{b:02x}"
            
            # Calculate text color (black or white) based on luminance
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = (255, 255, 255) if luminance < 128 else (0, 0, 0)
            
            # Get text bounding box
            bbox = draw.textbbox((0, 0), hex_code, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Center text in swatch
            text_x = x + (swatch_size - text_width) // 2
            text_y = y + (swatch_size - text_height) // 2
            
            draw.text((text_x, text_y), hex_code, fill=text_color, font=font)
    
    # Save image
    img.save(output_path)
    print(f"Saved palette swatch image to: {output_path}")
    print(f"  Image size: {width} × {height} pixels (1.4:1 ratio)")
    print(f"  Grid: {rows} rows × {cols} columns")
    print(f"  Swatch size: {swatch_size} × {swatch_size} pixels")
    print(f"  Colors arranged: grays (dark→light), then chromatic (by hue)")