import numpy as np
import time
from PIL import Image

def load_image(image_path: str) -> tuple[Image.Image, np.ndarray]:
    """
    Load and parse the image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (PIL Image object, numpy array representation)
    """
    print(f"Loading image: {image_path}")
    start_time = time.time()
    
    image = Image.open(image_path)
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        print(f"Converting from {image.mode} to RGB...")
        image = image.convert('RGB')
    image_array = np.array(image)
    
    elapsed = time.time() - start_time
    print(f"Image loaded in {elapsed:.2f} seconds")
    print(f"Image size: {image_array.shape[1]} × {image_array.shape[0]} pixels")
    
    return image, image_array