"""RAW去色罩核心 — 双管线：WB混合 + 滤镜管线。

v2.0.0: 功能模块分离：
  - 正负逆冲 (WB混合) → 独立插件
  - 负片反转 (LCCInverter) → 独立插件
  - 图像增强 (亮度/对比度/饱和度) → 独立插件
  - 滤镜处理 (降噪/去雾/平场) → 独立插件
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


# ── 共享 RAW 渲染参数 ──────────────────────────────────────────

def _default_pp_kwargs():
    """返回默认 postprocess 参数（延迟引用 rawpy，避免 import 时缺失）。"""
    return dict(
        output_color=rawpy.ColorSpace.sRGB,
        gamma=(2.222, 4.5),
        no_auto_bright=True,
        bright=1.0,
    )


def _render_raw_from_obj(raw, wb_mode: str, half_size: bool = False) -> np.ndarray:
    """从 rawpy 对象渲染线性 RGB（共享 daylight fallback）。

    UI 预览加载与 process_raw 共用此函数，
    确保 pp 参数和 daylight fallback exception 范围一致。
    """
    pp_kwargs = _default_pp_kwargs()
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


def _render_raw(raw, wb_mode: str, half_size: bool = False) -> np.ndarray:
    """从 rawpy 文件路径渲染（内部调用 _render_raw_from_obj）。"""
    return _render_raw_from_obj(raw, wb_mode, half_size)


# ============================================================
#  process_raw — 统一处理单张RAW文件
# ============================================================

def process_raw(
    input_path: str,
    output_path: str | None = None,
    wb_mode: str = "daylight",
    strength: float = 0.8,
    pipeline: "ProcessingPipeline | None" = None,
    post_pipeline: "ProcessingPipeline | None" = None,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    half_size: bool = False,
) -> np.ndarray:
    """处理单个RAW文件。

    Parameters
    ----------
    pipeline : ProcessingPipeline or None
        若提供，使用目标WB渲染后执行管线。
        若为None，执行WB混合（正负逆冲模式）。
    """
    if not _HAS_RAWPY:
        raise ImportError("rawpy is required. Install with: pip install rawpy")

    logger.debug("Loading RAW: %s", input_path)
    raw = rawpy.imread(input_path)
    try:
        if pipeline is not None:
            rgb = _render_raw_from_obj(raw, wb_mode, half_size)
            result = pipeline.run(rgb, preview=half_size)
        else:
            rgb_camera = _render_raw_from_obj(raw, "camera", half_size)

            if wb_mode == "camera" or strength <= 0.0:
                result = rgb_camera
            else:
                rgb_target = _render_raw_from_obj(raw, wb_mode, half_size)
                if strength >= 1.0:
                    result = rgb_target
                else:
                    result = rgb_camera * (1.0 - strength) + rgb_target * strength
                    result = np.clip(result, 0.0, 1.0)

        if post_pipeline is not None:
            result = post_pipeline.run(result, preview=half_size)

        need_post = brightness != 1.0 or contrast != 1.0 or saturation != 1.0
        if need_post:
            result = _apply_postprocess(result, brightness, contrast, saturation)

        if output_path:
            save_image(result, output_path)
            logger.debug("Processed: %s -> %s", os.path.basename(input_path), output_path)

        return result
    finally:
        raw.close()


def process_folder(
    input_dir: str,
    output_dir: str,
    wb_mode: str = "daylight",
    strength: float = 0.8,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    recursive: bool = False,
    pipeline: "ProcessingPipeline | None" = None,
    post_pipeline: "ProcessingPipeline | None" = None,
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
                pipeline=pipeline,
                post_pipeline=post_pipeline,
            )
            results.append(out)
            logger.info("[%d/%d] %s -> %s", i, len(raw_files), os.path.basename(path), os.path.basename(out))
        except Exception as e:
            logger.error("[%d/%d] Failed: %s - %s", i, len(raw_files), path, e)

    return results
