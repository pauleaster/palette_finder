import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from typing import Tuple, Optional, List

def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB to OKLAB color space.
    
    Args:
        rgb: RGB values in range [0, 255], shape (..., 3)
        
    Returns:
        OKLAB values, shape (..., 3)
    """
    # Normalize RGB to [0, 1]
    rgb_norm = rgb / 255.0
    
    # Convert to linear RGB
    def srgb_to_linear(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    
    r_lin = srgb_to_linear(rgb_norm[..., 0])
    g_lin = srgb_to_linear(rgb_norm[..., 1])
    b_lin = srgb_to_linear(rgb_norm[..., 2])
    
    # Convert to LMS
    l = 0.4122214708 * r_lin + 0.5363325363 * g_lin + 0.0514459929 * b_lin
    m = 0.2119034982 * r_lin + 0.6806995451 * g_lin + 0.1073969566 * b_lin
    s = 0.0883024619 * r_lin + 0.2817188376 * g_lin + 0.6299787005 * b_lin
    
    # Apply cube root
    l_ = np.cbrt(l)
    m_ = np.cbrt(m)
    s_ = np.cbrt(s)
    
    # Convert to OKLAB
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    
    return np.stack([L, a, b], axis=-1)


def oklab_to_rgb(oklab: np.ndarray) -> np.ndarray:
    """
    Convert OKLAB to RGB color space.
    
    Args:
        oklab: OKLAB values, shape (..., 3)
        
    Returns:
        RGB values in range [0, 255], shape (..., 3)
    """
    L = oklab[..., 0]
    a = oklab[..., 1]
    b = oklab[..., 2]
    
    # Convert to LMS
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    
    # Cube
    l = l_ ** 3
    m = m_ ** 3
    s = s_ ** 3
    
    # Convert to linear RGB
    r_lin = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g_lin = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b_lin = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    
    # Clip linear RGB to valid range before gamma conversion
    # (Negative values would produce NaN when raised to fractional power)
    r_lin = np.clip(r_lin, 0.0, None)
    g_lin = np.clip(g_lin, 0.0, None)
    b_lin = np.clip(b_lin, 0.0, None)
    
    # Convert to sRGB
    def linear_to_srgb(c):
        # c is already clipped to >= 0 above, but keep this for safety
        c = np.clip(c, 0.0, None)
        return np.where(c <= 0.0031308, 12.92 * c, 1.055 * (c ** (1/2.4)) - 0.055)
    
    r = linear_to_srgb(r_lin)
    g = linear_to_srgb(g_lin)
    b = linear_to_srgb(b_lin)
    
    # Clip and convert to [0, 255]
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb * 255, 0, 255)
    
    return rgb


def reduce_colors(image_array: np.ndarray, n_colors: int = 8, random_state: int = 42,
                  weight_L: float = 1.0, weight_a: float = 1.0, weight_b: float = 1.0,
                  seed_colors: Optional[List[Tuple[int, int, int]]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce the number of colors in the image using k-means clustering in OKLAB space.
    
    Args:
        image_array: Input image as numpy array (height, width, 3) in RGB
        n_colors: Number of colors to reduce to
        random_state: Random state for reproducibility
        weight_L: Weight for L (lightness) channel
        weight_a: Weight for a (green-red) channel
        weight_b: Weight for b (blue-yellow) channel
        seed_colors: Optional list of RGB tuples to use as initial cluster centers
                     e.g., [(255, 0, 0), (0, 255, 0)] for red and green
        
    Returns:
        Tuple of (reduced image array in RGB, color palette in RGB)
        
    Raises:
        KeyboardInterrupt: If user presses Ctrl+C during processing
        ValueError: If more seed colors than n_colors
    """
    print("\n" + "=" * 60)
    print("STARTING COLOR REDUCTION (OKLAB COLOR SPACE)")
    print(f"Channel weights: L={weight_L:.2f}, a={weight_a:.2f}, b={weight_b:.2f}")
    
    n_seed_colors = len(seed_colors) if seed_colors else 0
    if n_seed_colors > 0:
        print(f"Using {n_seed_colors} seed colors as initial cluster centers")
        for i, (r, g, b) in enumerate(seed_colors):
            print(f"  Seed {i}: RGB({r:3d}, {g:3d}, {b:3d}) = #{r:02x}{g:02x}{b:02x}")
        
        if n_seed_colors >= n_colors:
            raise ValueError(f"More seed colors ({n_seed_colors}) than n_colors ({n_colors}). "
                           f"Increase n_colors or reduce seeds.")
    
    print("=" * 60)
    
    # Reshape image to be a list of pixels
    h, w, c = image_array.shape
    
    print(f"Converting RGB to OKLAB...")
    convert_start = time.time()
    pixels_rgb = image_array.reshape(-1, 3)
    pixels_oklab = rgb_to_oklab(pixels_rgb)
    
    # Apply weights to OKLAB channels
    pixels_oklab_weighted = pixels_oklab.copy()
    pixels_oklab_weighted[:, 0] *= weight_L  # L channel
    pixels_oklab_weighted[:, 1] *= weight_a  # a channel
    pixels_oklab_weighted[:, 2] *= weight_b  # b channel
    
    convert_time = time.time() - convert_start
    print(f"Converted {len(pixels_rgb):,} pixels in {convert_time:.2f} seconds")
    
    # Prepare initial centers for k-means
    init_centers = None
    n_init = 10  # Default: multiple random initializations
    
    if seed_colors and len(seed_colors) > 0:
        # Convert seed colors to weighted OKLAB
        seed_colors_rgb = np.array(seed_colors, dtype=float)
        seed_colors_oklab = rgb_to_oklab(seed_colors_rgb)
        seed_colors_oklab_weighted = seed_colors_oklab.copy()
        seed_colors_oklab_weighted[:, 0] *= weight_L
        seed_colors_oklab_weighted[:, 1] *= weight_a
        seed_colors_oklab_weighted[:, 2] *= weight_b
        
        # Pick remaining centers randomly from pixels
        n_remaining = n_colors - n_seed_colors
        rng = np.random.default_rng(random_state)
        random_indices = rng.choice(pixels_oklab_weighted.shape[0], size=n_remaining, replace=False)
        remaining_centers = pixels_oklab_weighted[random_indices]
        
        # Combine seed colors and random pixels as initial centers
        init_centers = np.vstack([seed_colors_oklab_weighted, remaining_centers]).astype(np.float32, copy=False)
        n_init = 1  # Only one initialization when using explicit init
        
        print(f"Initialized with {n_seed_colors} seed colors + {n_remaining} random pixels")
    
    # Run k-means
    print(f"\nProcessing {len(pixels_oklab_weighted):,} pixels with {n_colors} colors...")
    print(f"Image dimensions: {w} × {h}")
    if seed_colors and len(seed_colors) > 0:
        print(f"Running frozen K-means (seed colors remain fixed)...")
        print(f"Max iterations: 50, tolerance: 1e-6")
    else:
        print(f"Running K-means clustering in weighted OKLAB space with {n_init} different initializations...")
        print(f"Expected output: {n_init} convergence messages (one per initialization)")
    print("(This may take a while for large images - watch for iteration progress below)")
    print("=" * 60)
    
    kmeans_start = time.time()
    
    try:
        # Apply k-means clustering
        if seed_colors and len(seed_colors) > 0:
            # Use frozen k-means to keep seed colors fixed
            color_palette_oklab_weighted, labels = frozen_kmeans(
                X=pixels_oklab_weighted,
                fixed_centers=seed_colors_oklab_weighted,
                n_clusters=n_colors,
                max_iter=50,
                tol=1e-6,
                rng_seed=random_state
            )
        else:
            # Standard k-means without seeds
            kmeans = KMeans(
                n_clusters=n_colors,
                random_state=random_state,
                n_init=n_init,
                verbose=1
            )
            labels = kmeans.fit_predict(pixels_oklab_weighted)
            color_palette_oklab_weighted = kmeans.cluster_centers_
        
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("K-means clustering interrupted by user")
        raise
    
    kmeans_time = time.time() - kmeans_start
    print("=" * 60)
    print(f"K-means completed in {kmeans_time:.2f} seconds ({kmeans_time/60:.1f} minutes)")
    
    # Get the color palette (cluster centers)
    print("\nExtracting color palette...")
    palette_start = time.time()
    
    # color_palette_oklab_weighted is already set above in both branches
    
    # Un-weight the palette to get back to original OKLAB space
    color_palette_oklab = color_palette_oklab_weighted.copy()
    color_palette_oklab[:, 0] /= weight_L  # L channel
    color_palette_oklab[:, 1] /= weight_a  # a channel
    color_palette_oklab[:, 2] /= weight_b  # b channel
    
    # Convert palette back to RGB
    print("Converting palette from OKLAB to RGB...")
    color_palette_rgb = oklab_to_rgb(color_palette_oklab)
    palette_time = time.time() - palette_start
    print(f"Color palette extracted and converted in {palette_time:.4f} seconds")
    
    # Log the palette colors
    print("\nColor Palette (RGB values):")
    for i in range(len(color_palette_rgb)):
        color = color_palette_rgb[i]
        r = int(np.clip(color[0], 0, 255))
        g = int(np.clip(color[1], 0, 255))
        b = int(np.clip(color[2], 0, 255))
        print(f"  Color {i:2d}: RGB({r:3d}, {g:3d}, {b:3d}) = #{r:02x}{g:02x}{b:02x}")
    
    # Replace each pixel with its cluster center (convert back to RGB)
    print("\nApplying palette to image pixels...")
    apply_start = time.time()
    reduced_pixels_oklab = color_palette_oklab[labels]
    reduced_pixels_rgb = oklab_to_rgb(reduced_pixels_oklab)
    reduced_image = reduced_pixels_rgb.reshape(h, w, c)
    apply_time = time.time() - apply_start
    print(f"Palette applied and converted to RGB in {apply_time:.2f} seconds")
    
    total_time = time.time() - convert_start
    print("\n" + "=" * 60)
    print(f"COLOR REDUCTION COMPLETE - Total: {total_time:.2f} seconds")
    print("=" * 60)
    
    # Show seed matches in console if seeds were provided
    if seed_colors and len(seed_colors) > 0:
        print("\nSeed Color Matches (first {n_seed_colors} colors are frozen seeds):")
        for i in range(n_seed_colors):
            sr, sg, sb = seed_colors[i]
            pr = int(np.clip(color_palette_rgb[i][0], 0, 255))
            pg = int(np.clip(color_palette_rgb[i][1], 0, 255))
            pb = int(np.clip(color_palette_rgb[i][2], 0, 255))
            
            print(f"  Color {i:2d} [SEED]: #{pr:02x}{pg:02x}{pb:02x} (original: #{sr:02x}{sg:02x}{sb:02x})")
    
    return reduced_image, color_palette_rgb


def frozen_kmeans(X, fixed_centers, n_clusters, max_iter=50, tol=1e-6, rng_seed=0):
    """
    K-means with fixed (frozen) cluster centers.
    
    Args:
        X: Data points to cluster
        fixed_centers: Centers that remain fixed during optimization
        n_clusters: Total number of clusters
        max_iter: Maximum iterations
        tol: Convergence tolerance
        rng_seed: Random seed
        
    Returns:
        Tuple of (centers, labels)
    """
    X = np.asarray(X, dtype=np.float32)
    fixed = np.asarray(fixed_centers, dtype=np.float32)
    m = fixed.shape[0]
    if n_clusters < m:
        raise ValueError("n_clusters must be >= number of fixed_centers")

    rng = np.random.default_rng(rng_seed)
    k_free = n_clusters - m

    # init free centers from data
    if k_free > 0:
        free = X[rng.choice(X.shape[0], size=k_free, replace=False)]
        centers = np.vstack([fixed, free]).astype(np.float32, copy=False)
    else:
        centers = fixed.copy()

    prev = centers.copy()

    for iteration in range(max_iter):
        labels = pairwise_distances_argmin(X, centers, metric="euclidean")

        # update only free centers
        if k_free > 0:
            for j in range(m, n_clusters):
                mask = labels == j
                if mask.any():
                    centers[j] = X[mask].mean(axis=0)
                else:
                    centers[j] = X[rng.integers(0, X.shape[0])]

        shift = np.sqrt(((centers - prev) ** 2).sum(axis=1)).max()
        print(f"Iteration {iteration}, max center shift: {shift:.6f}")
        
        if shift < tol:
            print(f"Converged at iteration {iteration}")
            break
        prev[:] = centers

    labels = pairwise_distances_argmin(X, centers, metric="euclidean")
    return centers, labels

