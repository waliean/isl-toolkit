"""Generate a synthetic test image and validate the decolor mask tool.

Creates a simulated "scanned negative" by:
1. Taking a known color image (a simple gradient + color patches)
2. Inverting it to simulate negative film
3. Applying an orange color mask
4. Then processing it with our tool to verify the mask is removed.
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


def simulate_negative_scan(positive: np.ndarray, mask_color=(0.85, 0.55, 0.28)) -> np.ndarray:
    """Simulate a film negative scan with an orange mask."""
    negative = 1.0 - positive
    mask = np.array(mask_color, dtype=np.float32)
    scan = negative * (1.0 - mask) * 0.8 + mask * 0.6
    scan = np.clip(scan, 0, 1)
    return scan


def main():
    print("Creating test image...")
    positive = create_test_image()
    _save(positive, "test_original.png")

    print("Simulating negative scan with orange mask...")
    negative_scan = simulate_negative_scan(positive)
    _save(negative_scan, "test_negative_scan.png")

    print("\nTest images created:")
    print("  test_original.png      - Original positive image (ground truth)")
    print("  test_negative_scan.png - Simulated negative scan with orange mask")
    print("\nNow run:")
    print("  python -m decolor_mask.cli test_negative_scan.png test_result_auto.png --type negative --mode auto")
    print("  python -m decolor_mask.cli test_negative_scan.png test_result_border.png --type negative --mode border")


if __name__ == "__main__":
    main()
