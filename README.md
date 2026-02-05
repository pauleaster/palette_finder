# palette Finder

An image analysis Python application that uses matplotlib and scikit-learn to parse images, separate them into similar color regions, and reduce the number of colors using k-means clustering.

## Features

- **Image Loading**: Parse and load images in various formats (PNG, JPEG, etc.)
- **Color Reduction**: Use k-means clustering to reduce the number of colors in an image
- **Region Segmentation**: Identify and segment areas of similar adjacent colors
- **Visualization**: Display original, reduced, segmented images and color palettes using matplotlib
- **Export Results**: Save processed images and visualizations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/pauleaster/palette_finder.git
cd palette_finder
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line

Analyze an image directly from the command line:

```bash
python palette_finder.py <image_path> [n_colors]
```

Example:
```bash
python palette_finder.py my_image.jpg 8
```

### Python API

Use the `PaletteFinder` class in your Python code:

```python
from palette_finder import PaletteFinder

# Create a PaletteFinder instance
finder = PaletteFinder('path/to/image.jpg')

# Load the image
finder.load_image()

# Reduce colors using k-means (default: 8 colors)
reduced_image, color_palette = finder.reduce_colors(n_colors=8)

# Segment regions of similar colors
segments = finder.segment_regions()

# Visualize results
finder.visualize()

# Save results
finder.save_results(output_prefix='output')
```

### Convenience Function

For quick analysis:

```python
from palette_finder import analyze_image

# Analyze and visualize in one call
finder = analyze_image('image.jpg', n_colors=8, visualize=True)
```

## Example

Run the included example script:

```bash
python example.py
```

This will:
1. Create a sample image with colored regions
2. Reduce colors using k-means
3. Segment similar color regions
4. Display and save visualizations

## How It Works

### 1. Image Loading
The application loads images using PIL (Pillow) and converts them to numpy arrays for processing.

### 2. Color Reduction (K-means Clustering)
- Reshapes the image into a list of RGB pixels
- Applies k-means clustering to group similar colors
- Replaces each pixel with its cluster center color
- This reduces the total number of colors to the specified `n_colors`

### 3. Region Segmentation
- Uses connected component labeling to identify adjacent pixels with similar colors
- Groups pixels of the same color that are spatially connected
- Creates a labeled map of distinct regions

### 4. Visualization
- Displays the original image
- Shows the color-reduced version
- Visualizes segmented regions with distinct colors
- Presents the extracted color palette

## Dependencies

- **matplotlib** (>=3.5.0): For visualization
- **scikit-learn** (>=1.0.0): For k-means clustering
- **numpy** (>=1.21.0): For array operations
- **Pillow** (>=9.0.0): For image I/O
- **scipy** (>=1.7.0): For image segmentation

## Testing

Run the test suite:

```bash
python -m unittest test_palette_finder.py -v
```

## API Reference

### `PaletteFinder` Class

#### `__init__(image_path: str)`
Initialize with path to an image file.

#### `load_image() -> np.ndarray`
Load and parse the image file. Returns numpy array representation.

#### `reduce_colors(n_colors: int = 8, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]`
Reduce colors using k-means clustering. Returns tuple of (reduced_image, color_palette).

#### `segment_regions(color_tolerance: int = 10) -> np.ndarray`
Segment image into regions of similar adjacent colors. Returns labeled region array.

#### `visualize(show_original: bool = True, show_reduced: bool = True, show_segmented: bool = True, show_palette: bool = True)`
Display visualization of results.

#### `save_results(output_prefix: str = "output")`
Save processed images to files.

### `analyze_image(image_path: str, n_colors: int = 8, visualize: bool = True) -> PaletteFinder`
Convenience function for quick image analysis.

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.