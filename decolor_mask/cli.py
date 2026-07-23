"""RAW去色罩 CLI — 一键批量处理RAW文件。

用法:
    decolor-mask                         处理当前文件夹所有RAW
    decolor-mask D:\\photos               处理指定文件夹
    decolor-mask D:\\photos -o D:\\out     指定输出目录
    decolor-mask . --strength 0.5        自定义强度 (0=保留原色, 1=完全中性化)
    decolor-mask . --wb daylight         使用日光白平衡
    decolor-mask . -r                    递归搜索子文件夹
    decolor-mask . --pipeline            使用完整滤镜管线
    decolor-mask . --pipeline --chroma-nr 0.3 --dehaze 0.2
"""

import argparse
import logging
import os
import sys

from decolor_mask.core import find_raw_files, process_folder, process_raw

logger = logging.getLogger("decolor_mask")


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAW去色罩 — 自动识别文件夹内RAW文件，通过白平衡混合或滤镜管线去除色罩。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  decolor-mask                         处理当前文件夹所有RAW
  decolor-mask D:\\photos               处理指定文件夹
  decolor-mask D:\\photos -o D:\\out     指定输出目录
  decolor-mask . --strength 0.5        自定义强度(0=保留原色, 1=完全中性化)
  decolor-mask . --wb daylight         使用日光白平衡
  decolor-mask . --wb camera           仅用相机WB
  decolor-mask . -r                    递归搜索子文件夹
  decolor-mask . -b 1.2 -c 1.1         调整亮度与对比度
  decolor-mask . --pipeline            启用完整滤镜管线
  decolor-mask . --pipeline --chroma-nr 0.3 --dehaze 0.2  降噪+去雾
""",
    )

    parser.add_argument(
        "folder", nargs="?", default=".",
        help="包含RAW文件的文件夹 (默认: 当前目录)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="输出目录 (默认: 输入文件夹下的 corrected/)",
    )
    parser.add_argument(
        "--strength", "-s", type=float, default=0.8,
        help="去色罩强度 0.0-1.0 (0=保留原色, 1=完全中性化). 默认: 0.8",
    )
    parser.add_argument(
        "--wb", choices=["auto", "camera", "daylight"], default="auto",
        help="目标白平衡模式. 默认: auto",
    )
    parser.add_argument(
        "--brightness", "-b", type=float, default=1.0,
        help="后处理亮度倍率. 默认: 1.0",
    )
    parser.add_argument(
        "--contrast", "-c", type=float, default=1.0,
        help="后处理对比度倍率. 默认: 1.0",
    )
    parser.add_argument(
        "--saturation", type=float, default=1.0,
        help="后处理饱和度倍率. 默认: 1.0",
    )
    parser.add_argument(
        "--recursive", "-r", action="store_true",
        help="递归搜索子文件夹",
    )

    # 新增: 滤镜管线参数
    parser.add_argument(
        "--pipeline", action="store_true",
        help="启用完整滤镜管线 (LCC反转 + 降噪 + 去雾 + 平场校正)",
    )
    parser.add_argument(
        "--chroma-nr", type=float, default=0.0,
        help="色度降噪强度 0.0-1.0 (需 --pipeline). 推荐: 0.2-0.5",
    )
    parser.add_argument(
        "--band-nr", type=float, default=0.0,
        help="频带降噪强度 0.0-1.0 (需 --pipeline). 推荐: 0.1-0.3",
    )
    parser.add_argument(
        "--flat-field", type=str, default=None,
        help="白帧参考图路径 (需 --pipeline). 不指定则自动估计",
    )
    parser.add_argument(
        "--dehaze", type=float, default=0.0,
        help="去雾强度 0.0-1.0 (需 --pipeline). 推荐: 0.2-0.5",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="详细输出 (debug级别)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="静默模式",
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    input_dir = os.path.abspath(args.folder)
    output_dir = args.output or os.path.join(input_dir, "corrected")

    raw_files = find_raw_files(input_dir, args.recursive)
    if not raw_files:
        print(f"在 '{input_dir}' 中未找到RAW文件")
        sys.exit(1)

    print(f"文件夹   : {input_dir}")
    print(f"RAW文件  : {len(raw_files)} 个")
    print(f"强度     : {args.strength}")
    print(f"目标WB   : {args.wb}")
    if args.brightness != 1.0:
        print(f"亮度     : {args.brightness}")
    if args.contrast != 1.0:
        print(f"对比度   : {args.contrast}")
    if args.saturation != 1.0:
        print(f"饱和度   : {args.saturation}")
    if args.pipeline:
        print(f"滤镜管线 : 启用")
        if args.chroma_nr > 0:
            print(f"  色度降噪: {args.chroma_nr}")
        if args.band_nr > 0:
            print(f"  频带降噪: {args.band_nr}")
        if args.dehaze > 0:
            print(f"  去雾    : {args.dehaze}")
        if args.flat_field:
            print(f"  平场校正: {args.flat_field}")
    print(f"输出     : {output_dir}")
    print()

    results = process_folder(
        input_dir, output_dir,
        wb_mode=args.wb,
        strength=args.strength,
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
        recursive=args.recursive,
        use_pipeline=args.pipeline,
        chroma_nr=args.chroma_nr,
        band_nr=args.band_nr,
        flat_field=args.flat_field,
        dehaze=args.dehaze,
    )

    print(f"\n完成! 成功 {len(results)} 个, 失败 {len(raw_files) - len(results)} 个")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
