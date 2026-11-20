"""Content-aware initial brush-stroke placement.

This module only selects initial stroke centres. It does not alter stroke
geometry, rendering, losses, or optimization.
"""
import os

import cv2
import numpy as np


def _to_rgb_float(image):
    """Accept HWC RGB or a [1, 3, H, W] / [3, H, W] torch or NumPy image."""
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)

    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("Expected a single image in the batch.")
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
        raise ValueError("Expected an RGB image with shape HxWx3 or 3xHxW.")
    if image.shape[2] == 4:
        image = image[..., :3]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    image = image.astype(np.float32)
    if not np.isfinite(image).all():
        raise ValueError("Feature source contains NaN or infinite values.")
    if image.max() > 1.5:
        image /= 255.0
    return np.clip(image, 0.0, 1.0)


def _normalize_map(feature):
    feature = np.asarray(feature, dtype=np.float32)
    lo, hi = float(feature.min()), float(feature.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        return np.zeros_like(feature, dtype=np.float32)
    return (feature - lo) / (hi - lo)


def build_density_maps(image, alpha=0.6, beta=0.4, window=9, sigma=1.0):
    """Return RGB source, normalized Sobel edges, local variance, and density."""
    if alpha < 0 or beta < 0 or alpha + beta <= 0:
        raise ValueError("alpha and beta must be non-negative and not both zero.")
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    rgb = _to_rgb_float(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = _normalize_map(cv2.magnitude(gx, gy))

    mean = cv2.boxFilter(gray, cv2.CV_32F, (window, window),
                         normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(gray * gray, cv2.CV_32F, (window, window),
                            normalize=True, borderType=cv2.BORDER_REFLECT)
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    texture = _normalize_map(variance)

    density = (alpha * edges + beta * texture) / (alpha + beta)
    if sigma > 0:
        density = cv2.GaussianBlur(density, (0, 0), sigmaX=sigma, sigmaY=sigma)
    density = _normalize_map(density)
    return rgb, edges, texture, density


def sample_adaptive_locations(image, n_strokes, alpha=0.6, beta=0.4,
                              window=9, sigma=1.0, epsilon=0.05,
                              uniform_mix=0.15, seed=None, diag_dir=None):
    """Sample (x,y) centres in renderer coordinates [-0.95, 0.95].

    The baseline initializer creates floor(sqrt(n_strokes))**2 strokes; this
    function deliberately uses the same effective count. epsilon is a
    per-pixel probability floor before normalization, not a stroke quota.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be greater than zero.")
    if not 0.0 <= uniform_mix <= 1.0:
        raise ValueError("uniform_mix must be between 0 and 1.")
    side = int(np.sqrt(n_strokes))
    count = side * side
    if count < 1:
        raise ValueError("n_strokes must be at least 1.")

    rgb, edges, texture, density = build_density_maps(
        image, alpha=alpha, beta=beta, window=window, sigma=sigma)
    height, width = density.shape
    flat = density.reshape(-1).astype(np.float64)

    adaptive_prob = flat + float(epsilon)
    adaptive_prob /= adaptive_prob.sum()
    uniform_prob = np.full(flat.shape, 1.0 / flat.size, dtype=np.float64)
    probabilities = ((1.0 - uniform_mix) * adaptive_prob
                     + uniform_mix * uniform_prob)
    probabilities /= probabilities.sum()

    rng = np.random.default_rng(seed)
    indices = rng.choice(flat.size, size=count, replace=False, p=probabilities)
    rows, cols = np.divmod(indices, width)

    # Jitter within the selected pixel, then use the exact [-1,1] mapping
    # consumed by RigidBodyTransformation (x -> width, y -> height).
    jitter_x = rng.uniform(-0.5, 0.5, size=count)
    jitter_y = rng.uniform(-0.5, 0.5, size=count)
    x = 2.0 * (cols + 0.5 + jitter_x) / width - 1.0
    y = 2.0 * (rows + 0.5 + jitter_y) / height - 1.0
    locations = np.column_stack((np.clip(x, -0.95, 0.95),
                                 np.clip(y, -0.95, 0.95))).astype(np.float32)

    if diag_dir:
        save_diagnostics(diag_dir, rgb, edges, texture, density,
                         locations, probabilities.reshape(height, width))
    return locations


def save_diagnostics(directory, rgb, edges, texture, density, locations, probabilities):
    """Write source/features/density and sampled centres for visual inspection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(directory, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    panels = [
        (rgb, "NST feature source", None),
        (edges, "Sobel edge magnitude", "gray"),
        (texture, "Local variance", "gray"),
        (density, "Combined density", "magma"),
        (probabilities, "Final sampling probability", "magma"),
    ]
    for ax, (data, title, cmap) in zip(axes.flat, panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    # Renderer coordinates map x to columns and y to rows.
    height, width = density.shape
    px = (locations[:, 0] + 1.0) * width / 2.0
    py = (locations[:, 1] + 1.0) * height / 2.0
    axes.flat[5].imshow(rgb)
    axes.flat[5].scatter(px, py, s=12, c="red", alpha=0.75)
    axes.flat[5].set_title("Sampled centres (renderer coordinate mapping)")
    axes.flat[5].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(directory, "adaptive_placement_diagnostics.png"), dpi=160)
    plt.close(fig)
