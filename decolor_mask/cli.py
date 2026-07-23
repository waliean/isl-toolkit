"""CLI tool for removing arbitrary color mask from cross-process styled images.

Usage:
    python -m decolor_mask.cli input.jpg output.jpg
    decolor-mask input.jpg output.jpg --method white_patch --strength 0.7
    decolor-mask input.jpg output.jpg --method manual --mask 0.85 0.55 0.28
    decolor-mask input.jpg output.jpg --method dark_pixel  (for film scans)
    decolor-mask "photos/*.jpg" output/ --batch
"""

import argparse
import glob
import logging
import os
import sys

from decolor_mask.core import load_image, process_image, estimate_color_mask

logger = logging.getLogger("decolor_mask")


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging level and format."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    root_logger = logging.getLogger("decolor_mask")
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        root_logger.addHandler(handler)
    root_logger.setLevel(level)


def _build_kwargs(args: argparse.Namespace) -> dict:
    return dict(
        method=args.method,
        percentile=args.percentile,
        border_size=args.border_size,
        mask_r=args.mask_r,
        mask_g=args.mask_g,
        mask_b=args.mask_b,
        strength=args.strength,
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
    )


def process_single(args: argparse.Namespace) -> None:
    logger.info("Processing: %s", args.input)
    logger.info("  Method: %s  Strength: %.2f", args.method, args.strength)
    if args.method == "manual":
        logger.info("  Mask (RGB): %.3f, %.3f, %.3f", args.mask_r, args.mask_g, args.mask_b)

    process_image(args.input, args.output, **_build_kwargs(args))

    logger.info("Done. Output saved to: %s", args.output)


def process_batch(args: argparse.Namespace) -> None:
    inputs = sorted(glob.glob(args.input, recursive=True))
    if not inputs:
        logger.error("No files matched pattern: %s", args.input)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    logger.info("Batch processing %d files...", len(inputs))
    kwargs = _build_kwargs(args)

    for i, path in enumerate(inputs, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(args.output, f"{name}_corrected.png")
        try:
            process_image(path, out, **kwargs)
            logger.info("[%d/%d] %s -> %s", i, len(inputs), path, out)
        except Exception as e:
            logger.error("[%d/%d] Failed: %s - %s", i, len(inputs), path, e)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="正负逆冲 - Remove arbitrary color mask from images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  decolor-mask photo.jpg output.jpg                              Auto (gray_world)
  decolor-mask photo.jpg output.jpg --method white_patch         White patch
  decolor-mask photo.jpg output.jpg --method percentile --percentile 90
  decolor-mask photo.jpg output.jpg --method dark_pixel          Film scan mode
  decolor-mask photo.jpg output.jpg --method manual --mask 0.85 0.55 0.28
  decolor-mask photo.jpg output.jpg --strength 0.35              Gentle correction
  decolor-mask "photos/*.jpg" output/ --batch                    Batch processing
  decolor-mask photo.jpg --detect-only                           Detect mask color
""",
    )

    parser.add_argument("input", help="Path to input image, or glob pattern with --batch")
    parser.add_argument("output", nargs="?", default=None,
                        help="Path to output image or directory (with --batch)")

    parser.add_argument(
        "--method", "-m",
        choices=["gray_world", "white_patch", "percentile", "dark_pixel", "border", "manual"],
        default="gray_world",
        help="Mask detection method. Default: gray_world",
    )
    parser.add_argument(
        "--percentile", type=float, default=95.0,
        help="Percentile for 'percentile' method (0-100). Default: 95",
    )
    parser.add_argument(
        "--border-size", type=float, default=0.05,
        help="Border fraction for 'border' method. Default: 0.05",
    )
    parser.add_argument(
        "--mask", nargs=3, type=float, metavar=("R", "G", "B"),
        help="Manual mask color as RGB in [0, 1]. Use with --method manual.",
    )
    parser.add_argument(
        "--strength", type=float, default=0.6,
        help="Removal strength 0.0-1.0. 0=none, 1=full. Default: 0.6",
    )
    parser.add_argument(
        "--brightness", "-b", type=float, default=1.0,
        help="Brightness multiplier. Default: 1.0",
    )
    parser.add_argument(
        "--contrast", "-c", type=float, default=1.0,
        help="Contrast multiplier. Default: 1.0",
    )
    parser.add_argument(
        "--saturation", "-s", type=float, default=1.0,
        help="Saturation multiplier. Default: 1.0",
    )
    parser.add_argument(
        "--detect-only", action="store_true",
        help="Only detect and print mask color, don't process",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Batch mode: INPUT is a glob pattern, OUTPUT is a directory",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output (debug level)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress non-error output",
    )

    args = parser.parse_args()

    setup_logging(verbose=args.verbose, quiet=args.quiet)

    if args.method == "manual" and args.mask is None:
        parser.error("--mask R G B is required when --method manual")

    args.mask_r = args.mask_g = args.mask_b = None
    if args.mask:
        args.mask_r, args.mask_g, args.mask_b = args.mask

    if args.batch and args.output is None:
        parser.error("--batch requires OUTPUT to be a directory path")

    if args.detect_only:
        arr = load_image(args.input)
        mask = estimate_color_mask(
            arr, method=args.method, percentile=args.percentile, border_size=args.border_size,
        )
        r, g, b = mask
        print(f"Mask color (R, G, B): {r:.4f}, {g:.4f}, {b:.4f}")
        print(f"Mask color (0-255): {int(r * 255)}, {int(g * 255)}, {int(b * 255)}")
        return

    if args.batch:
        process_batch(args)
    else:
        if args.output is None:
            parser.error("OUTPUT is required (or use --batch for glob input)")
        process_single(args)


if __name__ == "__main__":
    main()
