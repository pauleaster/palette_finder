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
        self.image = Image.open(self.image_path)
        # Convert to RGB if necessary
        if self.image.mode != 'RGB':
            self.image = self.image.convert('RGB')
        self.image_array = np.array(self.image)
        return self.image_array
    
    def reduce_colors(self, n_colors: int = 8, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reduce the number of colors in the image using k-means clustering.
        
        Args:
            n_colors: Number of colors to reduce to
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (reduced image array, color palette)
        """
        if self.image_array is None:
            self.load_image()
        
        # Reshape image to be a list of pixels
        h, w, c = self.image_array.shape
        pixels = self.image_array.reshape(-1, 3)
        
        # Apply k-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # Get the color palette (cluster centers)
        self.color_palette = kmeans.cluster_centers_.astype(int)
        
        # Replace each pixel with its cluster center
        reduced_pixels = self.color_palette[labels]
        self.reduced_image = reduced_pixels.reshape(h, w, c)
        
        return self.reduced_image, self.color_palette
    
    def segment_regions(self) -> np.ndarray:
        """
        Segment the image into regions of similar adjacent colors.
        
        This uses connected component labeling to identify regions where
        adjacent pixels have similar colors.
        
        Returns:
            Array of labeled regions with consecutive integer labels
        """
        if self.reduced_image is None:
            raise ValueError("Must call reduce_colors() before segment_regions()")
        
        h, w, c = self.reduced_image.shape
        labels = np.zeros((h, w), dtype=int)
        current_label = 0
        
        # Create a structure for 4-connectivity (up, down, left, right)
        structure = np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]])
        
        # For each unique color in the reduced image, find connected components
        unique_colors = np.unique(self.reduced_image.reshape(-1, 3), axis=0)
        
        for color in unique_colors:
            # Create a mask for pixels of this color
            mask = np.all(self.reduced_image == color, axis=2)
            
            # Find connected components in this mask
            color_labels, n_features = label(mask, structure=structure)
            
            # Add these labels to our main label array (with offset)
            labels[mask] = color_labels[mask] + current_label
            current_label += n_features
        
        # Renumber labels to be consecutive starting from 0
        unique_labels = np.unique(labels)
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        
        # Apply the mapping to create consecutive labels
        consecutive_labels = np.zeros_like(labels)
        for old_label, new_label in label_map.items():
            consecutive_labels[labels == old_label] = new_label
        
        self.segmented_image = consecutive_labels
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
        plots_to_show = sum([show_original, show_reduced, show_segmented, show_palette])
        
        if plots_to_show == 0:
            return
        
        fig, axes = plt.subplots(1, plots_to_show, figsize=(5 * plots_to_show, 5))
        
        if plots_to_show == 1:
            axes = [axes]
        
        idx = 0
        
        if show_original and self.image_array is not None:
            axes[idx].imshow(self.image_array)
            axes[idx].set_title('Original Image')
            axes[idx].axis('off')
            idx += 1
        
        if show_reduced and self.reduced_image is not None:
            axes[idx].imshow(self.reduced_image.astype(np.uint8))
            axes[idx].set_title('Color Reduced Image')
            axes[idx].axis('off')
            idx += 1
        
        if show_segmented and self.segmented_image is not None:
            axes[idx].imshow(self.segmented_image, cmap='nipy_spectral')
            axes[idx].set_title('Segmented Regions')
            axes[idx].axis('off')
            idx += 1
        
        if show_palette and self.color_palette is not None:
            # Create a palette visualization
            palette_img = np.zeros((50, len(self.color_palette) * 50, 3), dtype=np.uint8)
            for i, color in enumerate(self.color_palette):
                palette_img[:, i*50:(i+1)*50] = color
            axes[idx].imshow(palette_img)
            axes[idx].set_title('Color Palette')
            axes[idx].axis('off')
            idx += 1
        
        plt.tight_layout()
        return fig
    
    def save_results(self, output_prefix: str = "output", save_visualization: bool = True):
        """
        Save the processed images to files.
        
        Args:
            output_prefix: Prefix for output filenames
            save_visualization: Whether to save the color-segmented visualization
        """
        if self.reduced_image is not None:
            reduced_img = Image.fromarray(self.reduced_image.astype(np.uint8))
            reduced_img.save(f"{output_prefix}_reduced.png")
            print(f"Saved: {output_prefix}_reduced.png")
        
        if self.segmented_image is not None:
            # Save normalized segmented image (grayscale labels)
            seg_normalized = (self.segmented_image - self.segmented_image.min())
            max_val = seg_normalized.max()
            # prevent division by zero
            if max_val > 0:
                seg_normalized = (seg_normalized / max_val * 255).astype(np.uint8)
            seg_img = Image.fromarray(seg_normalized)
            seg_img.save(f"{output_prefix}_segmented.png")
            print(f"Saved: {output_prefix}_segmented.png")
            
            # Save color-segmented visualization
            if save_visualization:
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
                print(f"Saved: {output_prefix}_segmented_colored.png")


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
    finder = PaletteFinder(image_path)
    finder.load_image()
    finder.reduce_colors(n_colors=n_colors)
    finder.segment_regions()
    
    if visualize:
        finder.visualize()
        plt.show()
    
    return finder


if __name__ == "__main__":
    
    
    if len(sys.argv) < 2:
        print("Usage: python palette_finder.py <image_path> [n_colors]")
        print("Example: python palette_finder.py image.jpg 8")
        sys.exit(1)
    
    image_path = sys.argv[1]
    n_colors = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    
    print(f"Analyzing image: {image_path}")
    print(f"Reducing to {n_colors} colors...")
    
    finder = analyze_image(image_path, n_colors=n_colors, visualize=True)
    
    print(f"Found {len(finder.color_palette)} colors")
    print(f"Segmented into {len(np.unique(finder.segmented_image))} regions")
