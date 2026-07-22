import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def load_image(path: str) -> np.ndarray:
    """Load image and return as float32 RGB numpy array [0, 1]."""
    try:
        img = Image.open(path).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {path}")
    except Exception as e:
        raise IOError(f"Failed to load image '{path}': {e}")
    arr = np.array(img, dtype=np.float32) / 255.0
    logger.debug("Loaded image '%s': %dx%d", path, img.width, img.height)
    return arr


def save_image(arr: np.ndarray, path: str) -> None:
    """Save float32 [0,1] or uint8 numpy array to image file."""
    try:
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        img.save(path)
    except OSError as e:
        raise IOError(f"Failed to save image to '{path}': {e}")
    logger.debug("Saved image to '%s'", path)


def invert_negative(arr: np.ndarray) -> np.ndarray:
    """Invert a negative scan to positive."""
    return 1.0 - arr


def _rgb_to_linear(arr: np.ndarray) -> np.ndarray:
    """Apply approximate sRGB gamma decode."""
    threshold = 0.04045
    linear = np.where(arr <= threshold, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    return linear


def _linear_to_rgb(arr: np.ndarray) -> np.ndarray:
    """Apply approximate sRGB gamma encode."""
    threshold = 0.0031308
    rgb = np.where(arr <= threshold, arr * 12.92, 1.055 * (arr ** (1.0 / 2.4)) - 0.055)
    return rgb


def _estimate_mask_from_border(arr: np.ndarray, border_size: float = 0.05) -> np.ndarray:
    """Estimate the color mask color from the image border region.

    The border of a film negative scan usually contains the unexposed
    film base, which reveals the pure color mask.
    """
    h, w = arr.shape[:2]
    bh = max(1, int(h * border_size))
    bw = max(1, int(w * border_size))

    border_pixels = np.concatenate([
        arr[:bh, :, :].reshape(-1, 3),
        arr[-bh:, :, :].reshape(-1, 3),
        arr[:, :bw, :].reshape(-1, 3),
        arr[:, -bw:, :].reshape(-1, 3),
    ], axis=0)

    mask_color = np.median(border_pixels, axis=0)
    logger.debug("Border mask estimate: R=%.4f G=%.4f B=%.4f", *mask_color)
    return mask_color


def _auto_estimate_mask(arr: np.ndarray) -> np.ndarray:
    """Auto-estimate the mask color using dark pixel analysis.

    In a negative scan, the darkest areas (film base) represent the
    pure color mask. We sample the darkest 2% of pixels.
    """
    h, w = arr.shape[:2]
    flat = arr.reshape(-1, 3)

    luminance = 0.299 * flat[:, 0] + 0.587 * flat[:, 1] + 0.114 * flat[:, 2]
    num_dark = max(1, len(flat) // 50)

    dark_idx = np.argpartition(luminance, num_dark)[:num_dark]
    mask_color = np.median(flat[dark_idx], axis=0)
    logger.debug("Auto mask estimate: R=%.4f G=%.4f B=%.4f", *mask_color)
    return mask_color


def remove_color_mask(
    arr: np.ndarray,
    mask_color: np.ndarray | None = None,
    mode: str = "auto",
    border_size: float = 0.05,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
) -> np.ndarray:
    """Remove the orange color mask from an inverted negative scan.

    Parameters
    ----------
    arr : np.ndarray
        Input RGB image as float32 in [0, 1]. This should be the
        **already inverted** positive image (use invert_negative() first).
    mask_color : np.ndarray or None
        Manual mask color as (R, G, B) in [0, 1]. Used when mode="manual".
    mode : str
        One of "auto", "border", "manual".
    border_size : float
        Fraction of image size for border analysis (mode="border" only).
    brightness : float
        Brightness multiplier after mask removal (default 1.0).
    contrast : float
        Contrast multiplier after mask removal (default 1.0).
    saturation : float
        Saturation multiplier after mask removal (default 1.0).

    Returns
    -------
    np.ndarray
        Color-corrected RGB image as float32 in [0, 1].
    """
    if mode == "auto":
        mask_color = _auto_estimate_mask(arr)
    elif mode == "border":
        mask_color = _estimate_mask_from_border(arr, border_size)
    elif mode == "manual":
        if mask_color is None:
            raise ValueError("mask_color must be provided when mode='manual'")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    mask_color = np.clip(mask_color, 1e-6, 1.0)

    arr_linear = _rgb_to_linear(arr)
    mask_linear = _rgb_to_linear(mask_color)

    corrected = arr_linear / mask_linear[np.newaxis, np.newaxis, :]

    result = _linear_to_rgb(corrected)

    if contrast != 1.0:
        result = (result - 0.5) * contrast + 0.5

    result = result * brightness

    if saturation != 1.0:
        gray = 0.299 * result[:, :, 0] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 2]
        result[:, :, 0] = gray + saturation * (result[:, :, 0] - gray)
        result[:, :, 1] = gray + saturation * (result[:, :, 1] - gray)
        result[:, :, 2] = gray + saturation * (result[:, :, 2] - gray)

    result = np.clip(result, 0, 1)
    return result


def process_negative(
    input_path: str,
    output_path: str,
    mode: str = "auto",
    border_size: float = 0.05,
    mask_r: float | None = None,
    mask_g: float | None = None,
    mask_b: float | None = None,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
) -> np.ndarray:
    """Full pipeline: load negative scan, invert, remove mask, save."""
    arr = load_image(input_path)
    positive = invert_negative(arr)

    if mode == "manual" and mask_r is not None and mask_g is not None and mask_b is not None:
        mask_color = np.array([mask_r, mask_g, mask_b], dtype=np.float32)
    else:
        mask_color = None

    result = remove_color_mask(
        positive,
        mask_color=mask_color,
        mode=mode,
        border_size=border_size,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
    )

    save_image(result, output_path)
    return result


def process_digital(
    input_path: str,
    output_path: str,
    mode: str = "auto",
    border_size: float = 0.05,
    mask_r: float | None = None,
    mask_g: float | None = None,
    mask_b: float | None = None,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
) -> np.ndarray:
    """Process a digital image that has a color cast applied (no inversion)."""
    arr = load_image(input_path)

    if mode == "manual" and mask_r is not None and mask_g is not None and mask_b is not None:
        mask_color = np.array([mask_r, mask_g, mask_b], dtype=np.float32)
    else:
        mask_color = None

    result = remove_color_mask(
        arr,
        mask_color=mask_color,
        mode=mode,
        border_size=border_size,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
    )

    save_image(result, output_path)
    return result


def detect_mask_color(arr: np.ndarray, mode: str = "auto", border_size: float = 0.05) -> np.ndarray:
    """Detect and return the estimated mask color without processing.

    Returns (R, G, B) values in [0, 1] range.
    """
    if mode == "auto":
        return _auto_estimate_mask(arr)
    elif mode == "border":
        return _estimate_mask_from_border(arr, border_size)
    else:
        raise ValueError(f"Unknown mode: {mode}")
