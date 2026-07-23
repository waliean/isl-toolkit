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
    w, h = img.width, img.height
    if w % 2 != 0 or h % 2 != 0:
        w_even = w - (w % 2)
        h_even = h - (h % 2)
        img = img.crop((0, 0, w_even, h_even))
        logger.debug("Cropped image to even dimensions: %dx%d -> %dx%d", w, h, w_even, h_even)
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


def _rgb_to_linear(arr: np.ndarray) -> np.ndarray:
    """sRGB gamma decode to linear light."""
    threshold = 0.04045
    return np.where(arr <= threshold, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)


def _linear_to_rgb(arr: np.ndarray) -> np.ndarray:
    """Linear light to sRGB gamma encode."""
    threshold = 0.0031308
    return np.where(arr <= threshold, arr * 12.92, 1.055 * (arr ** (1.0 / 2.4)) - 0.055)


def estimate_color_mask(
    arr: np.ndarray,
    method: str = "gray_world",
    percentile: float = 95.0,
    border_size: float = 0.05,
) -> np.ndarray:
    """Estimate the dominant color mask from an image.

    Returns the (R, G, B) color of the mask/cast, each in [0, 1].

    Parameters
    ----------
    method : str
        "gray_world"   - average pixel color IS the mask.
        "white_patch"  - brightest pixel color IS the mask.
        "percentile"   - top-N% brightest average IS the mask.
        "dark_pixel"   - darkest 2% median IS the mask (for film scans).
        "border"       - border region median IS the mask (for framed scans).
    """
    if method == "gray_world":
        mask = np.mean(arr, axis=(0, 1))
    elif method == "white_patch":
        mask = np.max(arr, axis=(0, 1))
    elif method == "percentile":
        flat = arr.reshape(-1, 3)
        lum = 0.299 * flat[:, 0] + 0.587 * flat[:, 1] + 0.114 * flat[:, 2]
        n = max(1, int(len(flat) * (100.0 - percentile) / 100.0))
        idx = np.argpartition(-lum, n)[:n]
        mask = np.mean(flat[idx], axis=0)
    elif method == "dark_pixel":
        flat = arr.reshape(-1, 3)
        lum = 0.299 * flat[:, 0] + 0.587 * flat[:, 1] + 0.114 * flat[:, 2]
        n = max(1, len(flat) // 50)
        idx = np.argpartition(lum, n)[:n]
        mask = np.median(flat[idx], axis=0)
    elif method == "border":
        h, w = arr.shape[:2]
        bh = max(1, int(h * border_size))
        bw = max(1, int(w * border_size))
        border = np.concatenate([
            arr[:bh, :, :].reshape(-1, 3),
            arr[-bh:, :, :].reshape(-1, 3),
            arr[:, :bw, :].reshape(-1, 3),
            arr[:, -bw:, :].reshape(-1, 3),
        ], axis=0)
        mask = np.median(border, axis=0)
    else:
        raise ValueError(f"Unknown mask detection method: {method}")
    mask = np.clip(mask, 1e-6, 1.0)
    logger.debug("Mask estimate (%s): R=%.4f G=%.4f B=%.4f", method, mask[0], mask[1], mask[2])
    return mask.astype(np.float32)


def remove_color_mask(
    arr: np.ndarray,
    mask_color: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    """Remove a color mask by applying per-channel gains in linear space.

    The "mask" is the unwanted color tint overlaid on the image.
    A pixel with exactly mask_color will become neutral gray at full strength.

    Parameters
    ----------
    arr : np.ndarray
        Input RGB image as float32 in [0, 1].
    mask_color : np.ndarray
        (R, G, B) color of the mask to remove, each in [0, 1].
    strength : float
        0.0 = no removal, 1.0 = full removal.

    Returns
    -------
    np.ndarray
        Corrected RGB image as float32 in [0, 1].
    """
    if strength <= 0.0:
        return arr
    mask_color = np.clip(mask_color, 1e-6, 1.0)
    arr_lin = _rgb_to_linear(arr)
    mask_lin = _rgb_to_linear(mask_color)

    # Gains to bring mask color to neutral gray (same luminance).
    neutral_lum = np.mean(mask_lin)
    gains = neutral_lum / mask_lin

    # Interpolate gains toward identity (1,1,1), NOT divisor.
    effective_gains = 1.0 + (gains - 1.0) * strength

    corrected = arr_lin * effective_gains[np.newaxis, np.newaxis, :]
    corrected = np.clip(corrected, 0, None)

    return _linear_to_rgb(corrected)


def correct_cross_process(
    arr: np.ndarray,
    method: str = "gray_world",
    percentile: float = 95.0,
    border_size: float = 0.05,
    mask_r: float | None = None,
    mask_g: float | None = None,
    mask_b: float | None = None,
    strength: float = 0.6,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
) -> np.ndarray:
    """Remove color mask from a cross-process styled image.

    Detects or uses a specified mask color, removes it by division in
    linear space, then applies brightness/contrast/saturation adjustments.

    Parameters
    ----------
    arr : np.ndarray
        Input RGB image as float32 in [0, 1].
    method : str
        Mask detection method: "gray_world", "white_patch", "percentile",
        "dark_pixel", "border". Ignored when manual mask is provided.
    percentile : float
        Percentile for "percentile" method (0-100).
    border_size : float
        Fraction for "border" method (0.0-1.0).
    mask_r, mask_g, mask_b : float or None
        Manual mask color as RGB in [0, 1]. Overrides auto detection.
    strength : float
        Correction strength 0.0-1.0. Default 0.6 preserves some character.
    brightness : float
        Brightness multiplier (default 1.0).
    contrast : float
        Contrast multiplier around mid-gray (default 1.0).
    saturation : float
        Saturation multiplier (default 1.0).

    Returns
    -------
    np.ndarray
        Corrected RGB image as float32 in [0, 1].
    """
    if mask_r is not None and mask_g is not None and mask_b is not None:
        mask_color = np.array([mask_r, mask_g, mask_b], dtype=np.float32)
        logger.debug("Manual mask: RGB=%.4f,%.4f,%.4f", mask_r, mask_g, mask_b)
    else:
        mask_color = estimate_color_mask(arr, method, percentile, border_size)

    result = remove_color_mask(arr, mask_color, strength)

    if contrast != 1.0:
        result = (result - 0.5) * contrast + 0.5

    result = result * brightness

    if saturation != 1.0:
        gray = 0.299 * result[:, :, 0] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 2]
        result[:, :, 0] = gray + saturation * (result[:, :, 0] - gray)
        result[:, :, 1] = gray + saturation * (result[:, :, 1] - gray)
        result[:, :, 2] = gray + saturation * (result[:, :, 2] - gray)

    return np.clip(result, 0, 1)


def process_image(
    input_path: str,
    output_path: str,
    method: str = "gray_world",
    percentile: float = 95.0,
    border_size: float = 0.05,
    mask_r: float | None = None,
    mask_g: float | None = None,
    mask_b: float | None = None,
    strength: float = 0.6,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
) -> np.ndarray:
    """Load, remove color mask, and save.  Parameters match correct_cross_process."""
    arr = load_image(input_path)
    result = correct_cross_process(
        arr,
        method=method,
        percentile=percentile,
        border_size=border_size,
        mask_r=mask_r,
        mask_g=mask_g,
        mask_b=mask_b,
        strength=strength,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
    )
    save_image(result, output_path)
    return result
