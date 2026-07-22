"""Generate synthetic test images for the decolor mask tool.

Creates:
1. test_original.png - A color gradient with color patches (ground truth)
2. test_cross_process.png - Same image with warm/orange cross-process cast applied
"""
import numpy as np
from PIL import Image

from decolor_mask.core import save_image as _save


def create_test_image() -> np.ndarray:
    """Create a 512x512 test image with color gradients and patches."""
    h, w = 512, 512
    nx, ny = np.meshgrid(np.arange(w) / w, np.arange(h) / h)

    r = 0.5 + 0.5 * np.sin(nx * 6.0)
    g = 0.5 + 0.5 * np.sin(ny * 6.0 + 2.0)
    b = 0.5 + 0.5 * np.sin((nx + ny) * 4.0)

    mask_red = (nx - 0.25) ** 2 + (ny - 0.25) ** 2 < 0.0225
    mask_green = (nx - 0.75) ** 2 + (ny - 0.25) ** 2 < 0.0225
    mask_blue = (nx - 0.25) ** 2 + (ny - 0.75) ** 2 < 0.0225
    mask_yellow = (nx - 0.75) ** 2 + (ny - 0.75) ** 2 < 0.0225
    mask_white = (nx - 0.5) ** 2 + (ny - 0.5) ** 2 < 0.01

    r[mask_red] = 1.0; g[mask_red] = 0.0; b[mask_red] = 0.0
    r[mask_green] = 0.0; g[mask_green] = 1.0; b[mask_green] = 0.0
    r[mask_blue] = 0.0; g[mask_blue] = 0.0; b[mask_blue] = 1.0
    r[mask_yellow] = 1.0; g[mask_yellow] = 1.0; b[mask_yellow] = 0.0
    r[mask_white] = 1.0; g[mask_white] = 1.0; b[mask_white] = 1.0

    return np.stack([r, g, b], axis=-1).astype(np.float32)


def apply_cross_process_cast(arr: np.ndarray) -> np.ndarray:
    """Apply a warm orange cross-process color cast to simulate Pentax style."""
    cast = np.array([0.88, 0.72, 0.52], dtype=np.float32)
    result = arr * cast[np.newaxis, np.newaxis, :] * 1.2 + cast * 0.15
    result = np.clip(result, 0, 1)
    return result


def main():
    print("Creating test image...")
    positive = create_test_image()
    _save(positive, "test_original.png")

    print("Applying cross-process color cast...")
    cross_process = apply_cross_process_cast(positive)
    _save(cross_process, "test_negative_scan.png")

    print("\nTest images created:")
    print("  test_original.png      - Original reference image")
    print("  test_negative_scan.png - Image with cross-process color cast")
    print("\nNow run:")
    print("  python -m decolor_mask.cli test_negative_scan.png test_result.png --method gray_world --strength 0.6")
    print("  python -m decolor_mask.cli test_negative_scan.png test_result.png --method manual --white 0.88 0.72 0.52 --strength 0.8")


if __name__ == "__main__":
    main()
