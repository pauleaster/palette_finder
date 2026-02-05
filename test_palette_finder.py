"""
Unit tests for the palette_finder module.
"""

import unittest
import numpy as np
from PIL import Image
import os
import tempfile
from palette_finder import PaletteFinder, analyze_image


class TestPaletteFinder(unittest.TestCase):
    """Test cases for PaletteFinder class."""
    
    def setUp(self):
        """Create a temporary test image."""
        # Create a simple test image with 4 distinct colors
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image[0:50, 0:50] = [255, 0, 0]      # Red
        self.test_image[0:50, 50:100] = [0, 255, 0]    # Green
        self.test_image[50:100, 0:50] = [0, 0, 255]    # Blue
        self.test_image[50:100, 50:100] = [255, 255, 0]  # Yellow
        
        # Save to temporary file
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        Image.fromarray(self.test_image).save(self.temp_file.name)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)
    
    def test_load_image(self):
        """Test image loading functionality."""
        finder = PaletteFinder(self.temp_file.name)
        image_array = finder.load_image()
        
        self.assertIsNotNone(image_array)
        self.assertEqual(image_array.shape, (100, 100, 3))
        self.assertTrue(np.array_equal(image_array, self.test_image))
    
    def test_reduce_colors(self):
        """Test color reduction using k-means."""
        finder = PaletteFinder(self.temp_file.name)
        finder.load_image()
        
        n_colors = 4
        reduced_image, palette = finder.reduce_colors(n_colors=n_colors)
        
        # Check that we got the right number of colors
        self.assertEqual(len(palette), n_colors)
        
        # Check dimensions are preserved
        self.assertEqual(reduced_image.shape, (100, 100, 3))
        
        # Check that palette contains valid RGB values
        self.assertTrue(np.all(palette >= 0))
        self.assertTrue(np.all(palette <= 255))
    
    def test_segment_regions(self):
        """Test region segmentation."""
        finder = PaletteFinder(self.temp_file.name)
        finder.load_image()
        finder.reduce_colors(n_colors=4)
        
        segments = finder.segment_regions()
        
        # Check that segmentation returns an array
        self.assertIsNotNone(segments)
        self.assertEqual(segments.shape, (100, 100))
        
        # Check that we have multiple regions
        unique_labels = np.unique(segments)
        self.assertGreater(len(unique_labels), 1)
    
    def test_segment_regions_requires_reduction(self):
        """Test that segment_regions requires reduce_colors to be called first."""
        finder = PaletteFinder(self.temp_file.name)
        finder.load_image()
        
        with self.assertRaises(ValueError):
            finder.segment_regions()
    
    def test_analyze_image_convenience_function(self):
        """Test the convenience function."""
        finder = analyze_image(self.temp_file.name, n_colors=4, visualize=False)
        
        self.assertIsNotNone(finder.image_array)
        self.assertIsNotNone(finder.reduced_image)
        self.assertIsNotNone(finder.segmented_image)
        self.assertIsNotNone(finder.color_palette)
    
    def test_save_results(self):
        """Test saving results to files."""
        finder = PaletteFinder(self.temp_file.name)
        finder.load_image()
        finder.reduce_colors(n_colors=4)
        finder.segment_regions()
        
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            output_prefix = os.path.join(tmpdir, "test")
            finder.save_results(output_prefix=output_prefix)
            
            # Check that files were created
            self.assertTrue(os.path.exists(f"{output_prefix}_reduced.png"))
            self.assertTrue(os.path.exists(f"{output_prefix}_segmented.png"))


class TestImageFormats(unittest.TestCase):
    """Test handling of different image formats."""
    
    def test_grayscale_conversion(self):
        """Test that grayscale images are converted to RGB."""
        # Create a grayscale image
        gray_image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            Image.fromarray(gray_image, mode='L').save(f.name)
            
            try:
                finder = PaletteFinder(f.name)
                image_array = finder.load_image()
                
                # Should be converted to RGB
                self.assertEqual(image_array.shape[2], 3)
            finally:
                os.remove(f.name)


if __name__ == '__main__':
    unittest.main()
