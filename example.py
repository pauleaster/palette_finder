"""
Example script demonstrating the palette_finder functionality.
This creates a sample image and analyzes it.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from palette_finder import PaletteFinder, analyze_image


def create_sample_image(filename: str = "sample_image.png", size: int = 200):
    """
    Create a sample image with distinct colored regions for testing.
    
    Args:
        filename: Output filename for the sample image
        size: Size of the square image
    """
    # Create an image with distinct colored regions
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Red region (top-left)
    img[0:size//2, 0:size//2] = [255, 0, 0]
    
    # Green region (top-right)
    img[0:size//2, size//2:] = [0, 255, 0]
    
    # Blue region (bottom-left)
    img[size//2:, 0:size//2] = [0, 0, 255]
    
    # Yellow region (bottom-right)
    img[size//2:, size//2:] = [255, 255, 0]
    
    # Add some noise to make it more interesting
    noise = np.random.randint(-30, 30, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Save the sample image
    Image.fromarray(img).save(filename)
    print(f"Sample image created: {filename}")
    return filename


def main():
    """Run the example analysis."""
    print("palette Finder - Example Usage")
    print("=" * 50)
    
    # Create a sample image
    sample_image = create_sample_image()
    
    # Analyze the image
    print("\nAnalyzing sample image...")
    finder = PaletteFinder(sample_image)
    
    # Step 1: Load the image
    print("1. Loading image...")
    image_array = finder.load_image()
    print(f"   Image shape: {image_array.shape}")
    
    # Step 2: Reduce colors using k-means
    print("\n2. Reducing colors using k-means...")
    reduced_image, palette = finder.reduce_colors(n_colors=6)
    print(f"   Reduced to {len(palette)} colors")
    print(f"   Color palette:")
    for i, color in enumerate(palette):
        print(f"      Color {i+1}: RGB{tuple(color)}")
    
    # Step 3: Segment regions
    print("\n3. Segmenting regions...")
    segments = finder.segment_regions()
    n_regions = len(np.unique(segments))
    print(f"   Found {n_regions} distinct regions")
    
    # Step 4: Visualize results
    print("\n4. Visualizing results...")
    finder.visualize()
    plt.savefig('example_output.png', dpi=150, bbox_inches='tight')
    print("   Visualization saved to: example_output.png")
    
    # Step 5: Save results
    print("\n5. Saving processed images...")
    finder.save_results(output_prefix="example")
    print("   Saved: example_reduced.png")
    print("   Saved: example_segmented.png")
    
    print("\n" + "=" * 50)
    print("Example complete! Check the output files.")
    
    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
