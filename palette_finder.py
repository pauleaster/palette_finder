"""
palette Finder - Image Analysis Tool

This module provides functionality to:
1. Parse and load images
2. Separate images into similar color adjacent areas (segmentation)
3. Reduce the number of colors using k-means clustering
"""

import numpy as np
import sys
from PIL import Image
from sklearn.cluster import KMeans
from scipy.ndimage import label
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import time
import signal


class PaletteFinder:
    """
    A class for analyzing images to find color palettes and segment regions.
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
        self.segmented_image = None
        self.color_palette = None
        
    def load_image(self) -> np.ndarray:
        """
        Load and parse the image file.
        
        Returns:
            numpy array representation of the image
        """
        print(f"Loading image: {self.image_path}")
        start_time = time.time()
        
        self.image = Image.open(self.image_path)
        # Convert to RGB if necessary
        if self.image.mode != 'RGB':
            print(f"Converting from {self.image.mode} to RGB...")
            self.image = self.image.convert('RGB')
        self.image_array = np.array(self.image)
        
        elapsed = time.time() - start_time
        print(f"Image loaded in {elapsed:.2f} seconds")
        print(f"Image size: {self.image_array.shape[1]} × {self.image_array.shape[0]} pixels")
        
        return self.image_array
    
    def reduce_colors(self, n_colors: int = 8, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reduce the number of colors in the image using k-means clustering.
        
        Args:
            n_colors: Number of colors to reduce to
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (reduced image array, color palette)
            
        Raises:
            KeyboardInterrupt: If user presses Ctrl+C during processing
        """
        if self.image_array is None:
            self.load_image()
        
        print("\n" + "=" * 60)
        print("STARTING COLOR REDUCTION")
        print("=" * 60)
        
        # Reshape image to be a list of pixels
        h, w, c = self.image_array.shape
        
        print(f"Reshaping image array...")
        reshape_start = time.time()
        pixels = self.image_array.reshape(-1, 3)
        reshape_time = time.time() - reshape_start
        print(f"Reshaped to {len(pixels):,} pixels in {reshape_time:.2f} seconds")
        
        n_init = 10  # Number of times k-means will run with different initializations
        
        print(f"\nProcessing {len(pixels):,} pixels with {n_colors} colors...")
        print(f"Image dimensions: {w} × {h}")
        print(f"Running K-means clustering with {n_init} different initializations...")
        print(f"Expected output: {n_init} convergence messages (one per initialization)")
        print("(This may take a while for large images - watch for iteration progress below)")
        print("=" * 60)
        
        kmeans_start = time.time()
        
        try:
            # Apply k-means clustering
            kmeans = KMeans(n_clusters=n_colors, random_state=random_state, n_init=n_init, verbose=1)
            labels = kmeans.fit_predict(pixels)
        except KeyboardInterrupt:
            print("\n" + "=" * 60)
            print("K-means clustering interrupted by user")
            raise  # Re-raise to propagate up
        
        kmeans_time = time.time() - kmeans_start
        print("=" * 60)
        print(f"K-means completed in {kmeans_time:.2f} seconds ({kmeans_time/60:.1f} minutes)")
        
        # Get the color palette (cluster centers)
        print("\nExtracting color palette...")
        palette_start = time.time()
        self.color_palette = kmeans.cluster_centers_.astype(int)
        palette_time = time.time() - palette_start
        print(f"Color palette extracted in {palette_time:.4f} seconds")
        
        # Replace each pixel with its cluster center
        print("Applying palette to image pixels...")
        apply_start = time.time()
        reduced_pixels = self.color_palette[labels]
        self.reduced_image = reduced_pixels.reshape(h, w, c)
        apply_time = time.time() - apply_start
        print(f"Palette applied in {apply_time:.2f} seconds")
        
        total_time = time.time() - reshape_start
        print("\n" + "=" * 60)
        print(f"COLOR REDUCTION COMPLETE - Total: {total_time:.2f} seconds")
        print("=" * 60)
        
        return self.reduced_image, self.color_palette
    
    def segment_regions(self) -> np.ndarray:
        """
        Segment the image into regions of similar adjacent colors.
        
        This uses connected component labeling to identify regions where
        adjacent pixels have similar colors.
        
        Returns:
            Array of labeled regions
        """
        if self.reduced_image is None:
            raise ValueError("Must call reduce_colors() before segment_regions()")
        
        print("\n" + "=" * 60)
        print("STARTING REGION SEGMENTATION")
        print("=" * 60)
        
        total_start = time.time()
        
        h, w, c = self.reduced_image.shape
        
        print("Initializing label array...")
        init_start = time.time()
        labels = np.zeros((h, w), dtype=int)
        current_label = 0
        init_time = time.time() - init_start
        print(f"Initialized in {init_time:.4f} seconds")
        
        # Create a structure for 4-connectivity (up, down, left, right)
        structure = np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]])
        
        # For each unique color in the reduced image, find connected components
        print("\nFinding unique colors...")
        unique_start = time.time()
        unique_colors = np.unique(self.reduced_image.reshape(-1, 3), axis=0)
        unique_time = time.time() - unique_start
        print(f"Found {len(unique_colors)} unique colors in {unique_time:.2f} seconds")
        
        print("\nProcessing connected components for each color...")
        component_start = time.time()
        
        for i, color in enumerate(unique_colors):
            if i % 5 == 0:  # Progress update every 5 colors
                elapsed = time.time() - component_start
                avg_time = elapsed / (i + 1) if i > 0 else 0
                remaining = avg_time * (len(unique_colors) - i - 1)
                print(f"Processing color {i+1}/{len(unique_colors)} (elapsed: {elapsed:.1f}s, est. remaining: {remaining:.1f}s)...")
            
            # Create a mask for pixels of this color
            mask = np.all(self.reduced_image == color, axis=2)
            
            # Find connected components in this mask
            color_labels, n_features = label(mask, structure=structure)
            
            # Add these labels to our main label array (with offset)
            labels[mask] = color_labels[mask] + current_label
            current_label += n_features
        
        component_time = time.time() - component_start
        print(f"Connected components processed in {component_time:.2f} seconds")
        
        self.segmented_image = labels
        
        total_time = time.time() - total_start
        num_regions = len(np.unique(labels))
        print("\n" + "=" * 60)
        print(f"SEGMENTATION COMPLETE - Total: {total_time:.2f} seconds")
        print(f"Found {num_regions} total regions")
        print("=" * 60)
        
        return self.segmented_image
    
    def visualize(self, show_original: bool = True, show_reduced: bool = True, 
                  show_segmented: bool = True, show_palette: bool = True):
        """
        Visualize the original image, color-reduced image, segmented regions, and color palette.
        
        Args:
            show_original: Whether to show the original image
            show_reduced: Whether to show the color-reduced image
            show_segmented: Whether to show the segmented regions
            show_palette: Whether to show the color palette
        """
        print("\n" + "=" * 60)
        print("CREATING VISUALIZATION")
        print("=" * 60)
        
        viz_start = time.time()
        
        plots_to_show = sum([show_original, show_reduced, show_segmented, show_palette])
        
        if plots_to_show == 0:
            print("No plots to show")
            return
        
        print(f"Creating figure with {plots_to_show} subplots...")
        fig, axes = plt.subplots(1, plots_to_show, figsize=(5 * plots_to_show, 5))
        
        if plots_to_show == 1:
            axes = [axes]
        
        idx = 0
        
        if show_original and self.image_array is not None:
            print("Plotting original image...")
            axes[idx].imshow(self.image_array)
            axes[idx].set_title('Original Image')
            axes[idx].axis('off')
            idx += 1
        
        if show_reduced and self.reduced_image is not None:
            print("Plotting color-reduced image...")
            axes[idx].imshow(self.reduced_image.astype(np.uint8))
            axes[idx].set_title('Color Reduced Image')
            axes[idx].axis('off')
            idx += 1
        
        if show_segmented and self.segmented_image is not None:
            print("Plotting segmented regions...")
            axes[idx].imshow(self.segmented_image, cmap='nipy_spectral')
            axes[idx].set_title('Segmented Regions')
            axes[idx].axis('off')
            idx += 1
        
        if show_palette and self.color_palette is not None:
            print("Plotting color palette...")
            # Create a palette visualization
            palette_img = np.zeros((50, len(self.color_palette) * 50, 3), dtype=np.uint8)
            for i, color in enumerate(self.color_palette):
                palette_img[:, i*50:(i+1)*50] = color
            axes[idx].imshow(palette_img)
            axes[idx].set_title('Color Palette')
            axes[idx].axis('off')
            idx += 1
        
        print("Applying tight layout...")
        plt.tight_layout()
        
        viz_time = time.time() - viz_start
        print(f"Visualization created in {viz_time:.2f} seconds")
        print("=" * 60)
        
        return fig
    
    def save_results(self, output_prefix: str = "output", save_visualization: bool = True):
        """
        Save the processed images to files.
        
        Args:
            output_prefix: Prefix for output filenames
            save_visualization: Whether to save the color-segmented visualization
        """
        print("\n" + "=" * 60)
        print("SAVING RESULTS")
        print("=" * 60)
        
        save_start = time.time()
        
        if self.reduced_image is not None:
            print(f"Saving reduced color image...")
            file_start = time.time()
            reduced_img = Image.fromarray(self.reduced_image.astype(np.uint8))
            reduced_img.save(f"{output_prefix}_reduced.png")
            file_time = time.time() - file_start
            print(f"Saved: {output_prefix}_reduced.png ({file_time:.2f}s)")
        
        if self.segmented_image is not None:
            # Save normalized segmented image (grayscale labels)
            print(f"Saving segmented image (grayscale)...")
            file_start = time.time()
            seg_normalized = (self.segmented_image - self.segmented_image.min())
            max_val = seg_normalized.max()
            # prevent division by zero
            if max_val > 0:
                seg_normalized = (seg_normalized / max_val * 255).astype(np.uint8)
            seg_img = Image.fromarray(seg_normalized)
            seg_img.save(f"{output_prefix}_segmented.png")
            file_time = time.time() - file_start
            print(f"Saved: {output_prefix}_segmented.png ({file_time:.2f}s)")
            
            # Save color-segmented visualization
            if save_visualization:
                print(f"Saving colorized segmented image...")
                file_start = time.time()
                # Create colorized version using matplotlib colormap
                import matplotlib.cm as cm
                cmap = cm.get_cmap('nipy_spectral')
                # Normalize to [0, 1] range
                seg_norm = seg_normalized / 255.0
                # Apply colormap
                colored_seg = cmap(seg_norm)
                # Convert to RGB (remove alpha channel)
                colored_seg_rgb = (colored_seg[:, :, :3] * 255).astype(np.uint8)
                colored_img = Image.fromarray(colored_seg_rgb)
                colored_img.save(f"{output_prefix}_segmented_colored.png")
                file_time = time.time() - file_start
                print(f"Saved: {output_prefix}_segmented_colored.png ({file_time:.2f}s)")
        
        total_save_time = time.time() - save_start
        print("=" * 60)
        print(f"ALL FILES SAVED - Total: {total_save_time:.2f} seconds")
        print("=" * 60)


def analyze_image(image_path: str, n_colors: int = 8, visualize: bool = True) -> PaletteFinder:
    """
    Convenience function to analyze an image with default settings.
    
    Args:
        image_path: Path to the image file
        n_colors: Number of colors to reduce to
        visualize: Whether to display visualization
        
    Returns:
        PaletteFinder object with analysis results
    """
    overall_start = time.time()
    
    finder = PaletteFinder(image_path)
    finder.load_image()
    finder.reduce_colors(n_colors=n_colors)
    finder.segment_regions()
    
    if visualize:
        finder.visualize()
        print("\nDisplaying visualization (close window to continue)...")
        display_start = time.time()
        plt.show()
        display_time = time.time() - display_start
        print(f"Visualization displayed for {display_time:.2f} seconds")
    
    overall_time = time.time() - overall_start
    print("\n" + "=" * 60)
    print(f"TOTAL PROCESSING TIME: {overall_time:.2f} seconds ({overall_time/60:.1f} minutes)")
    print("=" * 60)
    
    return finder


if __name__ == "__main__":
    
    def signal_handler(sig, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\nProcess interrupted by user (Ctrl+C)")
        print("Cleaning up and exiting...")
        sys.exit(0)
    
    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    if len(sys.argv) < 2:
        print("Usage: python palette_finder.py <image_path> [n_colors]")
        print("Example: python palette_finder.py image.jpg 8")
        sys.exit(1)
    
    image_path = sys.argv[1]
    n_colors = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    
    print(f"Analyzing image: {image_path}")
    print(f"Reducing to {n_colors} colors...")
    print("(Press Ctrl+C to abort at any time)\n")
    
    try:
        finder = analyze_image(image_path, n_colors=n_colors, visualize=True)
        
        print(f"\nFound {len(finder.color_palette)} colors")
        print(f"Segmented into {len(np.unique(finder.segmented_image))} regions")
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
        print("Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during processing: {e}")
        sys.exit(1)
