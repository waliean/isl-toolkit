"""RAW去色罩核心 — 双管线：简单WB混合 + 完整滤镜管线。

v1.2.0: 新增基于 DCU/ISL 引擎的滤镜管线架构。
"""

import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import rawpy
    _HAS_RAWPY = True
except ImportError:
    _HAS_RAWPY = False

from .kernel.filters import (
    GrayAxisFilmBase,
    LCCInverter,
    CromaNRFilter,
    BandNRFilter,
    DehazeFilter,
    FlatFieldFilter,
)
from .kernel import ProcessingPipeline, FilterStage

logger = logging.getLogger(__name__)

RAW_EXTENSIONS = {
    ".dng", ".pef", ".raw", ".nef", ".cr2", ".cr3", ".arw", ".orf",
    ".rw2", ".raf", ".3fr", ".dcr", ".kdc", ".mrw", ".nrw", ".sr2",
    ".srf", ".x3f",
}


def is_raw_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in RAW_EXTENSIONS


def find_raw_files(folder: str, recursive: bool = False) -> list[str]:
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")
    raw_files = []
    glob_iter = folder.rglob("*") if recursive else folder.glob("*")
    for f in glob_iter:
        if f.is_file() and is_raw_file(f.name):
            raw_files.append(str(f))
    return sorted(raw_files)


def save_image(arr: np.ndarray, path: str) -> None:
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)
    logger.debug("Saved: %s", path)


def _apply_postprocess(
    arr: np.ndarray,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
) -> np.ndarray:
    if contrast != 1.0:
        arr = (arr - 0.5) * contrast + 0.5
    arr = arr * brightness
    if saturation != 1.0:
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        gray = gray[..., np.newaxis]
        arr = gray + saturation * (arr - gray)
    return np.clip(arr, 0, 1)


def _render_raw(raw, wb_mode: str, half_size: bool = False) -> np.ndarray:
    """渲染RAW到sRGB numpy数组。"""
    pp_kwargs = dict(
        output_color=rawpy.ColorSpace.sRGB,
        gamma=(2.222, 4.5),
        no_auto_bright=True,
        bright=1.0,
    )
    if half_size:
        pp_kwargs["half_size"] = True

    if wb_mode == "camera":
        rgb = raw.postprocess(use_camera_wb=True, **pp_kwargs)
    elif wb_mode == "auto":
        rgb = raw.postprocess(use_camera_wb=False, use_auto_wb=True, **pp_kwargs)
    elif wb_mode == "daylight":
        try:
            dl_wb = raw.daylight_whitebalance
        except AttributeError:
            logger.warning("相机不支持读取日光白平衡，回退到 auto WB")
            rgb = raw.postprocess(use_camera_wb=False, use_auto_wb=True, **pp_kwargs)
        else:
            rgb = raw.postprocess(use_camera_wb=False, user_wb=dl_wb, **pp_kwargs)
    else:
        raise ValueError(f"Unknown wb_mode: {wb_mode}")
    return rgb.astype(np.float32) / 255.0


# ============================================================
#  管线工厂函数
# ============================================================

def build_pipeline(
    chroma_nr: float = 0.0,
    band_nr: float = 0.0,
    flat_field: str | None = None,
    dehaze: float = 0.0,
    inverter_strength: float = 1.0,
    preview_scale: float = 0.25,
) -> ProcessingPipeline:
    """构建滤镜管线。

    Args:
        chroma_nr: 色度降噪强度 0.0-1.0 (0=禁用)
        band_nr: 频带降噪强度 0.0-1.0 (0=禁用)
        flat_field: 白帧参考路径 (None=自动估计)
        dehaze: 去雾强度 0.0-1.0 (0=禁用)
        inverter_strength: LCC反转强度
        preview_scale: 预览降采样比例
    """
    pipe = ProcessingPipeline(preview_scale=preview_scale)

    # FrontEnd: 预处理
    if chroma_nr > 0.01:
        pipe.add_frontend(CromaNRFilter(strength=chroma_nr))
    if band_nr > 0.01:
        pipe.add_frontend(BandNRFilter(strength=band_nr))
    if flat_field:
        pipe.add_frontend(FlatFieldFilter(flat_frame_path=flat_field, strength=1.0))
    else:
        pipe.add_frontend(FlatFieldFilter(strength=0.5))

    # Core: 反转
    pipe.add_core(LCCInverter(strength=inverter_strength))

    # BackEnd: 后处理
    if dehaze > 0.01:
        pipe.add_backend(DehazeFilter(strength=dehaze))

    return pipe


# ============================================================
#  process_raw — 保持向后兼容，增加可选管线参数
# ============================================================

def process_raw(
    input_path: str,
    output_path: str | None = None,
    wb_mode: str = "auto",
    strength: float = 0.8,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    half_size: bool = False,
    use_pipeline: bool = False,
    chroma_nr: float = 0.0,
    band_nr: float = 0.0,
    flat_field: str | None = None,
    dehaze: float = 0.0,
) -> np.ndarray:
    """处理单个RAW文件。

    Parameters
    ----------
    use_pipeline : bool
        启用完整滤镜管线（LCC反转 + 降噪 + 去雾 + 平场校正）。
    chroma_nr : float
        色度降噪强度 (0-1)，仅 use_pipeline=True 时生效。
    band_nr : float
        频带降噪强度 (0-1)，仅 use_pipeline=True 时生效。
    flat_field : str or None
        白帧参考路径，仅 use_pipeline=True 时生效。
    dehaze : float
        去雾强度 (0-1)，仅 use_pipeline=True 时生效。
    """
    if not _HAS_RAWPY:
        raise ImportError("rawpy is required. Install with: pip install rawpy")

    logger.debug("Loading RAW: %s", input_path)
    raw = rawpy.imread(input_path)

    if use_pipeline:
        # 渲染目标WB（完全中性化方向）
        rgb = _render_raw(raw, wb_mode, half_size)
        pipe = build_pipeline(
            chroma_nr=chroma_nr,
            band_nr=band_nr,
            flat_field=flat_field,
            dehaze=dehaze,
            inverter_strength=strength,
            preview_scale=0.25 if half_size else 1.0,
        )
        result = pipe.run(rgb, preview=half_size)
        result = _apply_postprocess(result, brightness, contrast, saturation)
    else:
        # 原有简单WB混合管线（向后兼容）
        pp_kwargs = dict(
            output_color=rawpy.ColorSpace.sRGB,
            gamma=(2.222, 4.5),
            no_auto_bright=True,
            bright=1.0,
        )
        if half_size:
            pp_kwargs["half_size"] = True

        rgb_camera = raw.postprocess(use_camera_wb=True, **pp_kwargs)

        if wb_mode == "camera" or strength <= 0.0:
            result_rgb = rgb_camera
        else:
            if wb_mode == "auto":
                rgb_target = raw.postprocess(use_camera_wb=False, use_auto_wb=True, **pp_kwargs)
            elif wb_mode == "daylight":
                try:
                    dl_wb = raw.daylight_whitebalance
                except AttributeError:
                    logger.warning("相机不支持读取日光白平衡，回退到 auto WB")
                    rgb_target = raw.postprocess(use_camera_wb=False, use_auto_wb=True, **pp_kwargs)
                else:
                    rgb_target = raw.postprocess(use_camera_wb=False, user_wb=dl_wb, **pp_kwargs)
            else:
                raise ValueError(f"Unknown wb_mode: {wb_mode}")

            if strength >= 1.0:
                result_rgb = rgb_target
            else:
                result_rgb = (
                    rgb_camera.astype(np.float32) * (1.0 - strength)
                    + rgb_target.astype(np.float32) * strength
                )
                result_rgb = np.clip(result_rgb, 0, 255).astype(np.uint8)

        result = result_rgb.astype(np.float32) / 255.0
        result = _apply_postprocess(result, brightness, contrast, saturation)

    if output_path:
        save_image(result, output_path)
        logger.debug("Processed: %s -> %s", os.path.basename(input_path), output_path)

    return result


def process_folder(
    input_dir: str,
    output_dir: str,
    wb_mode: str = "auto",
    strength: float = 0.8,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    recursive: bool = False,
    use_pipeline: bool = False,
    chroma_nr: float = 0.0,
    band_nr: float = 0.0,
    flat_field: str | None = None,
    dehaze: float = 0.0,
) -> list[str]:
    """处理文件夹内所有RAW文件。"""
    raw_files = find_raw_files(input_dir, recursive)
    if not raw_files:
        logger.warning("No RAW files found in: %s", input_dir)
        return []

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, path in enumerate(raw_files, 1):
        name = Path(path).stem
        out = os.path.join(output_dir, f"{name}.png")
        try:
            process_raw(
                path, out,
                wb_mode=wb_mode,
                strength=strength,
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                use_pipeline=use_pipeline,
                chroma_nr=chroma_nr,
                band_nr=band_nr,
                flat_field=flat_field,
                dehaze=dehaze,
            )
            results.append(out)
            logger.info("[%d/%d] %s -> %s", i, len(raw_files), os.path.basename(path), os.path.basename(out))
        except Exception as e:
            logger.error("[%d/%d] Failed: %s - %s", i, len(raw_files), path, e)

    return results
