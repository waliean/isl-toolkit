"""RAW去色罩 CLI — 一键批量导出RAW文件。

用法:
    decolor-mask                         导出当前文件夹所有 RAW（正负逆冲模式）
    decolor-mask --mode invert           负片反转模式
    decolor-mask --mode filter           滤镜处理模式
    decolor-mask D:\\photos               导出指定文件夹
    decolor-mask D:\\photos -o D:\\out     指定导出目录
    decolor-mask . --strength 0.5        自定义强度
    decolor-mask . --wb daylight         使用日光白平衡
    decolor-mask . -r                    递归搜索子文件夹
    decolor-mask . -b 1.2 -c 1.1         调整亮度与对比度
    decolor-mask . --mode invert --chroma-nr 0.3 --dehaze 0.2
    decolor-mask . --sharpen 0.5 --vignette 0.4 --grain 0.2   修图参数
"""

import argparse
import logging
import os
import sys

from .core import find_raw_files, process_folder, process_raw
from .kernel.gpu import get_status_text
from .plugins import get, list_all

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
        description="RAW去色罩 — 自动识别文件夹内 RAW 文件，支持多种处理模式与导出方式。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""模式:
  cross    正负逆冲 (Cross Process) — Camera WB + Target WB 混合交叉冲洗 (默认)
  invert   负片反转 (Film Inversion) — LCC 空间反转负片为正片
  filter   滤镜处理 (Filter Pipeline) — 降噪/去雾/平场校正

示例:
  decolor-mask                         正负逆冲模式，导出当前文件夹
  decolor-mask D:\\photos               导出指定文件夹
  decolor-mask --mode invert           负片反转模式
  decolor-mask --mode filter --chroma-nr 0.3 --dehaze 0.2  滤镜模式
  decolor-mask D:\\photos -o D:\\out     指定导出目录
  decolor-mask . --strength 0.5        自定义强度
  decolor-mask . --wb daylight         使用日光白平衡
  decolor-mask . -r                    递归搜索子文件夹
  decolor-mask . -b 1.2 -c 1.1         调整亮度与对比度
  decolor-mask . --sharpen 0.5 --clarity 0.3 --vignette 0.4  修图选项
  decolor-mask . --mode invert --toning sepia --grain 0.15   色调+颗粒
""",
    )

    parser.add_argument(
        "folder", nargs="?", default=".",
        help="包含RAW文件的文件夹 (默认: 当前目录)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="导出目录 (默认: 输入文件夹下的 corrected/)",
    )
    parser.add_argument(
        "--mode", choices=["cross", "invert", "filter"], default="cross",
        help="处理模式: cross=正负逆冲(默认) invert=负片反转 filter=滤镜处理",
    )
    parser.add_argument(
        "--strength", "-s", type=float, default=0.8,
        help="强度 (Strength) 0.0-1.0 (cross模式:WB混合度, invert模式:反转强度). 默认: 0.8",
    )
    parser.add_argument(
        "--wb", choices=["auto", "camera", "daylight"], default="daylight",
        help="目标白平衡 (WB) 模式. 默认: daylight",
    )
    parser.add_argument(
        "--brightness", "-b", type=float, default=1.0,
        help="后处理亮度 (Brightness) 倍率. 默认: 1.0",
    )
    parser.add_argument(
        "--contrast", "-c", type=float, default=1.0,
        help="后处理对比度 (Contrast) 倍率. 默认: 1.0",
    )
    parser.add_argument(
        "--saturation", type=float, default=1.0,
        help="后处理饱和度 (Saturation) 倍率. 默认: 1.0",
    )
    parser.add_argument(
        "--recursive", "-r", action="store_true",
        help="递归搜索子文件夹",
    )

    # 滤镜参数 (mode=invert 或 mode=filter 时可用)
    parser.add_argument(
        "--chroma-nr", type=float, default=0.0,
        help="色度降噪 (CromaNR) 强度 0.0-1.0 (invert/filter模式). 推荐: 0.2-0.5",
    )
    parser.add_argument(
        "--band-nr", type=float, default=0.0,
        help="频带降噪 (BandNR) 强度 0.0-1.0 (invert/filter模式). 推荐: 0.1-0.3",
    )
    parser.add_argument(
        "--dehaze", type=float, default=0.0,
        help="去雾 (Dehaze) 强度 0.0-1.0 (invert/filter模式). 推荐: 0.2-0.5",
    )
    parser.add_argument(
        "--flat-field", type=str, default=None,
        help="平场校正 (FlatField) 白帧参考图路径 (invert/filter模式)",
    )

    # 修图参数 (所有模式通用)
    parser.add_argument(
        "--sharpen", type=float, default=0.0,
        help="智能锐化 (SmartSharp) 强度 0.0-1.0. 推荐: 0.3-0.6",
    )
    parser.add_argument(
        "--clarity", type=float, default=0.0,
        help="清晰度 (Clarity) 强度 0.0-1.0 (中频细节增强). 推荐: 0.2-0.5",
    )
    parser.add_argument(
        "--bw-filter", choices=["none", "red", "orange", "yellow", "green", "blue"],
        default="none",
        help="黑白滤镜模拟 (B&W Filter). 默认: none",
    )
    parser.add_argument(
        "--bw-filter-strength", type=float, default=1.0,
        help="B&W滤镜强度 0.0-1.0. 默认: 1.0",
    )
    parser.add_argument(
        "--toning", choices=["none", "sepia", "selenium", "cyanotype", "platinum", "copper", "cool"],
        default="none",
        help="色调映射 (Toning) 预设. 默认: none",
    )
    parser.add_argument(
        "--vignette", type=float, default=0.0,
        help="暗角 (Vignette) 强度 0.0-1.0. 推荐: 0.2-0.5",
    )
    parser.add_argument(
        "--grain", type=float, default=0.0,
        help="胶片颗粒 (Grain) 强度 0.0-1.0. 推荐: 0.1-0.3",
    )
    parser.add_argument(
        "--shadow-boost", type=float, default=0.0,
        help="暗部增强 (Shadow Boost) 强度 0.0-1.0. 推荐: 0.2-0.5",
    )
    parser.add_argument(
        "--highlights", type=float, default=0.0,
        help="高光 (Highlights) 调整 -1.0~1.0 (负数=压暗, 正数=提亮)",
    )
    parser.add_argument(
        "--shadows", type=float, default=0.0,
        help="阴影 (Shadows) 调整 -1.0~1.0 (负数=压暗, 正数=提亮)",
    )
    parser.add_argument(
        "--midtones", type=float, default=0.0,
        help="中间调 (Midtones) 调整 -1.0~1.0 (负数=压低, 正数=增强)",
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

    mode_names = {"cross": "正负逆冲 (Cross Process)", "invert": "负片反转 (Film Inversion)", "filter": "滤镜处理 (Filter Pipeline)"}
    print(f"文件夹   : {input_dir}")
    print(f"RAW文件  : {len(raw_files)} 个")
    print(f"模式     : {mode_names.get(args.mode, args.mode)}")
    print(f"强度     : {args.strength}")
    print(f"目标WB   : {args.wb}")
    if args.brightness != 1.0:
        print(f"亮度 (Brightness): {args.brightness}")
    if args.contrast != 1.0:
        print(f"对比度 (Contrast): {args.contrast}")
    if args.saturation != 1.0:
        print(f"饱和度 (Saturation): {args.saturation}")
    if args.mode in ("invert", "filter"):
        if args.chroma_nr > 0:
            print(f"色度降噪 (CromaNR): {args.chroma_nr}")
        if args.band_nr > 0:
            print(f"频带降噪 (BandNR): {args.band_nr}")
        if args.dehaze > 0:
            print(f"去雾 (Dehaze): {args.dehaze}")
        if args.flat_field:
            print(f"平场校正 (FlatField): {args.flat_field}")
    if args.sharpen > 0:
        print(f"智能锐化 (SmartSharp): {args.sharpen}")
    if args.clarity > 0:
        print(f"清晰度 (Clarity): {args.clarity}")
    if args.bw_filter != "none":
        print(f"B&W滤镜  : {args.bw_filter} (强度: {args.bw_filter_strength})")
    if args.toning != "none":
        print(f"色调映射 (Toning): {args.toning}")
    if args.vignette > 0:
        print(f"暗角 (Vignette): {args.vignette}")
    if args.grain > 0:
        print(f"胶片颗粒 (Grain): {args.grain}")
    if args.shadow_boost > 0:
        print(f"暗部增强 (Shadow Boost): {args.shadow_boost}")
    if args.highlights != 0.0 or args.shadows != 0.0 or args.midtones != 0.0:
        print(f"色调曲线 (Tone Curve): hl={args.highlights:.1f} sh={args.shadows:.1f} mid={args.midtones:.1f}")
    print(f"输出     : {output_dir}")
    print(f"GPU状态  : {get_status_text()}")
    print()

    # 构建增强管线 (所有模式通用 — sharpen/clarity/vignette/shadow_boost/tone_curve)
    post_pipeline = None
    has_enhance = (
        args.sharpen > 0.01 or args.clarity > 0.01
        or args.vignette > 0.01
        or args.shadow_boost > 0.01
        or args.highlights != 0.0 or args.shadows != 0.0 or args.midtones != 0.0
    )
    if has_enhance:
        from .kernel import ProcessingPipeline
        from .kernel.filters import (
            SmartSharpFilter, ClarityFilter,
            VignetteFilter, ToneCurveFilter, ShadowBoostFilter,
        )

        post_pipeline = ProcessingPipeline(preview_scale=1.0)

        if args.clarity > 0.01:
            post_pipeline.add_frontend(ClarityFilter(strength=args.clarity))

        if args.highlights != 0.0 or args.shadows != 0.0 or args.midtones != 0.0:
            post_pipeline.add_backend(ToneCurveFilter(
                highlights=args.highlights,
                shadows=args.shadows,
                midtones=args.midtones,
            ))
        if args.shadow_boost > 0.01:
            post_pipeline.add_backend(ShadowBoostFilter(strength=args.shadow_boost))
        if args.sharpen > 0.01:
            post_pipeline.add_backend(SmartSharpFilter(strength=args.sharpen))
        if args.vignette > 0.01:
            post_pipeline.add_backend(VignetteFilter(strength=args.vignette))

    # cross 模式下，滤镜类（B&W滤镜、颗粒、色调映射）放入 post_pipeline 且排在增强之前
    has_cross_filters = args.bw_filter != "none" or args.grain > 0.01 or args.toning != "none"
    if args.mode == "cross" and has_cross_filters:
        from .kernel import ProcessingPipeline
        from .kernel.filters import BWFilterSim, GrainFilter, ToningFilter

        cross_post = ProcessingPipeline(preview_scale=1.0)

        if args.bw_filter != "none":
            cross_post.add_frontend(BWFilterSim(
                filter_type=args.bw_filter, strength=args.bw_filter_strength))
        if args.grain > 0.01:
            cross_post.add_backend(GrainFilter(strength=args.grain))
        if args.toning != "none":
            cross_post.add_backend(ToningFilter(preset=args.toning, strength=0.7))

        # 合并增强管线到滤镜管线后面
        if post_pipeline is not None:
            for f in post_pipeline.get_stage("frontend"):
                cross_post.add_frontend(f)
            for f in post_pipeline.get_stage("core"):
                cross_post.add_core(f)
            for f in post_pipeline.get_stage("backend"):
                cross_post.add_backend(f)
        post_pipeline = cross_post

    # 构建模式管线 (invert/filter 模式 — 滤镜类在此处理，增强类在 post_pipeline)
    pipeline = None
    if args.mode in ("invert", "filter"):
        from .kernel import ProcessingPipeline
        from .kernel.filters import (
            LCCInverter, CromaNRFilter, BandNRFilter,
            DehazeFilter, FlatFieldFilter, BWFilterSim, GrainFilter, ToningFilter,
        )

        pipeline = ProcessingPipeline(preview_scale=1.0)

        if args.mode == "filter":
            if args.chroma_nr > 0.01:
                pipeline.add_frontend(CromaNRFilter(strength=args.chroma_nr))
            if args.band_nr > 0.01:
                pipeline.add_frontend(BandNRFilter(strength=args.band_nr))
            if args.flat_field:
                pipeline.add_frontend(FlatFieldFilter(flat_frame_path=args.flat_field, strength=1.0))
            if args.bw_filter != "none":
                pipeline.add_frontend(BWFilterSim(
                    filter_type=args.bw_filter, strength=args.bw_filter_strength))
            if args.dehaze > 0.01:
                pipeline.add_backend(DehazeFilter(strength=args.dehaze))
            if args.grain > 0.01:
                pipeline.add_backend(GrainFilter(strength=args.grain))
            if args.toning != "none":
                pipeline.add_backend(ToningFilter(preset=args.toning, strength=0.7))
        else:  # invert mode
            if args.chroma_nr > 0.01:
                pipeline.add_frontend(CromaNRFilter(strength=args.chroma_nr))
            if args.band_nr > 0.01:
                pipeline.add_frontend(BandNRFilter(strength=args.band_nr))
            if args.flat_field:
                pipeline.add_frontend(FlatFieldFilter(flat_frame_path=args.flat_field, strength=1.0))
            else:
                pipeline.add_frontend(FlatFieldFilter(strength=0.5))
            if args.bw_filter != "none":
                pipeline.add_frontend(BWFilterSim(
                    filter_type=args.bw_filter, strength=args.bw_filter_strength))
            pipeline.add_core(LCCInverter(strength=args.strength))
            if args.dehaze > 0.01:
                pipeline.add_backend(DehazeFilter(strength=args.dehaze))
            if args.grain > 0.01:
                pipeline.add_backend(GrainFilter(strength=args.grain))
            if args.toning != "none":
                pipeline.add_backend(ToningFilter(preset=args.toning, strength=0.7))

    results = process_folder(
        input_dir, output_dir,
        wb_mode=args.wb,
        strength=args.strength,
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
        recursive=args.recursive,
        pipeline=pipeline,
        post_pipeline=post_pipeline,
    )

    print(f"\n完成! 成功 {len(results)} 个, 失败 {len(raw_files) - len(results)} 个")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
