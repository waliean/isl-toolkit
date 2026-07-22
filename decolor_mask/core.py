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


def estimate_white_balance(
    arr: np.ndarray,
    method: str = "gray_world",
    percentile: float = 95.0,
) -> np.ndarray:
    """Estimate white balance gains from an image.

    Works in linear light space for physical accuracy.

    Parameters
    ----------
    arr : np.ndarray
        Input RGB image as float32 in [0, 1].
    method : str
        "gray_world" - average pixel should be neutral gray.
        "white_patch" - brightest pixel should be white.
        "percentile" - top percentile brightest pixels average to white.
    percentile : float
        Percentile for "percentile" method (0-100, default 95).

    Returns
    -------
    np.ndarray
        Gains (g_r, g_g, g_b) normalized so g_g = 1.0.
    """
    arr_lin = _rgb_to_linear(arr)

    if method == "gray_world":
        means = np.mean(arr_lin, axis=(0, 1))
    elif method == "white_patch":
        means = np.max(arr_lin, axis=(0, 1))
    elif method == "percentile":
        flat = arr_lin.reshape(-1, 3)
        luminance = 0.2126 * flat[:, 0] + 0.7152 * flat[:, 1] + 0.0722 * flat[:, 2]
        n_bright = max(1, int(len(flat) * (100.0 - percentile) / 100.0))
        bright_idx = np.argpartition(-luminance, n_bright)[:n_bright]
        means = np.mean(flat[bright_idx], axis=0)
    else:
        raise ValueError(f"Unknown white balance method: {method}")

    gains = np.where(means > 1e-8, np.mean(means) / means, 1.0)
    gains = gains / gains[1]
    logger.debug(
        "White balance (%s): gains R=%.4f G=%.4f B=%.4f",
        method, gains[0], gains[1], gains[2],
    )
    return gains.astype(np.float32)


def apply_white_balance(
    arr: np.ndarray,
    gains: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    """Apply white balance correction in linear space.

    Parameters
    ----------
    arr : np.ndarray
        Input RGB image as float32 in [0, 1].
    gains : np.ndarray
        White balance gains (g_r, g_g, g_b) from estimate_white_balance().
    strength : float
        Blend strength 0.0-1.0. 0 = no correction, 1 = full correction.

    Returns
    -------
    np.ndarray
        White-balanced RGB image in [0, 1].
    """
    if strength <= 0.0:
        return arr
    arr_lin = _rgb_to_linear(arr)
    g = gains[np.newaxis, np.newaxis, :]
    g = 1.0 + (g - 1.0) * strength
    corrected = arr_lin * g
    corrected = np.clip(corrected, 0, None)
    return _linear_to_rgb(corrected)


def correct_cross_process(
    arr: np.ndarray,
    method: str = "gray_world",
    percentile: float = 95.0,
    white_r: float | None = None,
    white_g: float | None = None,
    white_b: float | None = None,
    strength: float = 0.6,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
) -> np.ndarray:
    """Correct the color cast of a Pentax cross-process (正负逆冲) image.

    Balances colors toward natural while preserving film-like tonality.
    Auto-detects white balance then blends with the original by a
    controlled strength factor.

    Parameters
    ----------
    arr : np.ndarray
        Input RGB image as float32 in [0, 1].
    method : str
        Auto white balance method: "gray_world", "white_patch", "percentile".
        Ignored when manual white point is provided.
    percentile : float
        Percentile for "percentile" method (0-100).
    white_r, white_g, white_b : float or None
        Manual reference white point as RGB in [0, 1].
        When all three are provided, overrides auto detection.
        The reference is: "what color in the image should be white?"
    strength : float
        Correction strength 0.0-1.0. Higher = more correction toward natural.
        Default 0.6 preserves some cross-process character.
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
    if white_r is not None and white_g is not None and white_b is not None:
        ref = np.array([white_r, white_g, white_b], dtype=np.float32)
        ref = np.clip(ref, 1e-6, 1.0)
        gains = np.mean(ref) / ref
        gains = gains / gains[1]
        logger.debug(
            "Manual white reference: RGB=%.4f,%.4f,%.4f gains=%.4f,%.4f,%.4f",
            white_r, white_g, white_b, gains[0], gains[1], gains[2],
        )
    else:
        gains = estimate_white_balance(arr, method, percentile)

    result = apply_white_balance(arr, gains, strength)

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
    white_r: float | None = None,
    white_g: float | None = None,
    white_b: float | None = None,
    strength: float = 0.6,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
) -> np.ndarray:
    """Load, correct cross-process color cast, and save.

    Parameters match correct_cross_process().
    """
    arr = load_image(input_path)
    result = correct_cross_process(
        arr,
        method=method,
        percentile=percentile,
        white_r=white_r,
        white_g=white_g,
        white_b=white_b,
        strength=strength,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
    )
    save_image(result, output_path)
    return result
