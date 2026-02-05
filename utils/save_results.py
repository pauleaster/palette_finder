import numpy as np
from PIL import Image
from .create_palette_image import create_palette_image

def save_results(reduced_image: np.ndarray, output_prefix: str = "output", 
                seed_colors=None, color_palette_rgb=None):
    """
    Save the reduced image and color palette to files.
    
    Args:
        reduced_image: Reduced image array in RGB
        output_prefix: Prefix for output filenames
        seed_colors: Optional list of seed RGB tuples
        color_palette_rgb: Optional color palette array
    """
    # Convert to uint8 and save
    reduced_image_uint8 = reduced_image.astype('uint8')
    reduced_pil = Image.fromarray(reduced_image_uint8)
    
    output_image = f"{output_prefix}_reduced.png"
    reduced_pil.save(output_image)
    print(f"\nSaved reduced image to: {output_image}")
    
    # Save palette info if provided
    if color_palette_rgb is not None:
        output_palette = f"{output_prefix}_palette.txt"
        
        with open(output_palette, 'w') as f:
            f.write("Color Palette from Reduced Image (OKLAB K-means)\n")
            f.write("=" * 60 + "\n\n")
            
            # Write palette colors
            for i in range(len(color_palette_rgb)):
                color = color_palette_rgb[i]
                r = int(np.clip(color[0], 0, 255))
                g = int(np.clip(color[1], 0, 255))
                b = int(np.clip(color[2], 0, 255))
                f.write(f"Color {i:2d}: RGB({r:3d}, {g:3d}, {b:3d}) = #{r:02x}{g:02x}{b:02x}\n")
            
            # If seeds were provided, show closest matches
            if seed_colors and len(seed_colors) > 0:
                f.write("\n" + "=" * 60 + "\n")
                f.write("Seed Color Matches (first colors are frozen seeds)\n")
                f.write("=" * 60 + "\n\n")
                
                n_seeds = len(seed_colors)
                for i in range(n_seeds):
                    sr, sg, sb = seed_colors[i]
                    pr = int(np.clip(color_palette_rgb[i][0], 0, 255))
                    pg = int(np.clip(color_palette_rgb[i][1], 0, 255))
                    pb = int(np.clip(color_palette_rgb[i][2], 0, 255))
                    
                    f.write(f"Seed {i}: RGB({sr:3d}, {sg:3d}, {sb:3d}) = #{sr:02x}{sg:02x}{sb:02x}\n")
                    f.write(f"  → Color {i:2d} [FROZEN]: RGB({pr:3d}, {pg:3d}, {pb:3d}) = #{pr:02x}{pg:02x}{pb:02x}\n\n")
        
        print(f"Saved palette info to: {output_palette}")
        
        # Create palette swatch image
        output_swatches = f"{output_prefix}_swatches.png"
        create_palette_image(
            color_palette_rgb=color_palette_rgb,
            output_path=output_swatches,
            width=1400,  # A4 landscape at ~300 DPI ≈ 1400 pixels
            show_hex=True
        )
