"""CLI tool for removing color mask from film negative scans.

Usage:
    python -m decolor_mask.cli input.jpg output.jpg
    decolor-mask input.jpg output.jpg --mode border
    decolor-mask input.jpg output.jpg --mode manual --mask 0.8 0.5 0.3
    decolor-mask "scans/*.jpg" output/ --batch
"""

import argparse
import glob
import logging
import os
import sys

from decolor_mask.core import (
    load_image,
    process_negative,
    process_digital,
    detect_mask_color,
    invert_negative,
)

logger = logging.getLogger("decolor_mask")


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging level and format."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger("decolor_mask").addHandler(handler)
    logging.getLogger("decolor_mask").setLevel(level)


def process_single(args: argparse.Namespace) -> None:
    """Process a single input file."""
    kwargs = dict(
        mode=args.mode,
        border_size=args.border_size,
        mask_r=args.mask_r,
        mask_g=args.mask_g,
        mask_b=args.mask_b,
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
    )

    logger.info("Processing: %s", args.input)
    logger.info("  Type: %s  Mode: %s", args.type, args.mode)
    if args.mask_r is not None:
        logger.info("  Mask (RGB): %.3f, %.3f, %.3f", args.mask_r, args.mask_g, args.mask_b)

    if args.type == "negative":
        process_negative(args.input, args.output, **kwargs)
    else:
        process_digital(args.input, args.output, **kwargs)

    logger.info("Done. Output saved to: %s", args.output)


def process_batch(args: argparse.Namespace) -> None:
    """Process multiple input files via glob pattern."""
    inputs = sorted(glob.glob(args.input, recursive=True))
    if not inputs:
        logger.error("No files matched pattern: %s", args.input)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    logger.info("Batch processing %d files...", len(inputs))
    kwargs = dict(
        mode=args.mode,
        border_size=args.border_size,
        mask_r=args.mask_r,
        mask_g=args.mask_g,
        mask_b=args.mask_b,
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
    )

    for i, path in enumerate(inputs, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(args.output, f"{name}_corrected.png")
        try:
            if args.type == "negative":
                process_negative(path, out, **kwargs)
            else:
                process_digital(path, out, **kwargs)
            logger.info("[%d/%d] %s -> %s", i, len(inputs), path, out)
        except Exception as e:
            logger.error("[%d/%d] Failed: %s - %s", i, len(inputs), path, e)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="正负逆冲 - Remove color mask from film negative scans or digital images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  decolor-mask scan.jpg output.jpg                     Auto-detect mask
  decolor-mask scan.jpg output.jpg --mode border       Border analysis
  decolor-mask scan.jpg output.jpg --mode manual --mask 0.8 0.5 0.3
  decolor-mask "scans/*.jpg" output/ --batch           Batch processing
  decolor-mask scan.jpg --detect-only                  Only detect mask color
""",
    )

    parser.add_argument("input", help="Path to input image, or glob pattern with --batch")
    parser.add_argument("output", nargs="?", default=None, help="Path to output image or directory (with --batch)")

    parser.add_argument(
        "--type", "-t",
        choices=["negative", "digital"],
        default="negative",
        help="Processing type. Default: negative",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["auto", "border", "manual"],
        default="auto",
        help="Mask detection mode. Default: auto",
    )
    parser.add_argument(
        "--mask", nargs=3, type=float, metavar=("R", "G", "B"),
        help="Manual mask color as RGB values in [0, 1]",
    )
    parser.add_argument(
        "--border-size", type=float, default=0.05,
        help="Fraction of image for border analysis. Default: 0.05",
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
        help="Only detect and print the mask color, don't process",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Batch mode: INPUT is a glob pattern, OUTPUT is a directory",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output (debug level)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress non-error output",
    )

    args = parser.parse_args()

    setup_logging(verbose=args.verbose, quiet=args.quiet)

    if args.mode == "manual" and args.mask is None:
        parser.error("--mask R G B is required when --mode manual")

    args.mask_r = args.mask_g = args.mask_b = None
    if args.mask:
        args.mask_r, args.mask_g, args.mask_b = args.mask

    if args.batch and args.output is None:
        parser.error("--batch requires OUTPUT to be a directory path")

    if args.detect_only:
        arr = load_image(args.input)
        if args.type == "negative":
            arr = invert_negative(arr)
        mask = detect_mask_color(arr, mode=args.mode, border_size=args.border_size)
        r, g, b = mask
        print(f"Detected mask color (R, G, B): {r:.4f}, {g:.4f}, {b:.4f}")
        print(f"Detected mask color (0-255): {int(r * 255)}, {int(g * 255)}, {int(b * 255)}")
        return

    if args.batch:
        process_batch(args)
    else:
        if args.output is None:
            parser.error("OUTPUT is required (or use --batch for glob input)")
        process_single(args)


if __name__ == "__main__":
    main()
