"""处理管线 — 基于 DCU ISL 引擎的 FrontEnd/Core/BackEnd 三层架构。

参考 DCU 的：
- IslEImageServerFilterFrontEnd → 预处理
- IslEImageServerFilter 系列   → 核心处理
- IslEImageServerFilterBackEnd → 后处理
"""

import numpy as np
import logging
from typing import Optional
from .filters import BaseFilter, FilterStage

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    """DCU 风格的三阶段处理管线。

    使用方式:
        pipe = ProcessingPipeline()
        pipe.add_frontend(CromaNRFilter(strength=0.3))
        pipe.add_backend(DehazeFilter(strength=0.2))
        result = pipe.run(image)

    preview模式运行降采样管线以提升速度。
    """

    def __init__(self, preview_scale: float = 0.25):
        self._stages = {
            FilterStage.FRONTEND: [],
            FilterStage.CORE: [],
            FilterStage.BACKEND: [],
        }
        self.preview_scale = max(0.1, min(1.0, preview_scale))
        self._film_base = None

    def add(self, filter_: BaseFilter, stage: str = FilterStage.CORE):
        """添加滤镜到指定阶段。"""
        if stage not in self._stages:
            raise ValueError(f"Unknown stage: {stage}. Use FilterStage.FRONTEND/CORE/BACKEND")
        self._stages[stage].append(filter_)
        logger.debug(f"Pipeline: added {filter_.name} to {stage}")
        return self

    def add_frontend(self, filter_: BaseFilter):
        return self.add(filter_, FilterStage.FRONTEND)

    def add_core(self, filter_: BaseFilter):
        return self.add(filter_, FilterStage.CORE)

    def add_backend(self, filter_: BaseFilter):
        return self.add(filter_, FilterStage.BACKEND)

    def remove(self, name: str):
        """按名称移除滤镜。"""
        for stage in self._stages:
            self._stages[stage] = [f for f in self._stages[stage] if f.name != name]

    def clear(self, stage: Optional[str] = None):
        """清除指定阶段或全部滤镜。"""
        if stage:
            self._stages[stage] = []
        else:
            for s in self._stages:
                self._stages[s] = []

    def get_stage(self, stage: str) -> list:
        return self._stages.get(stage, [])

    @property
    def film_base(self):
        return self._film_base

    def run(self, image: np.ndarray, preview: bool = False) -> np.ndarray:
        """执行完整管线。

        Args:
            image: float32 RGB (H, W, 3), [0, 1]
            preview: True 时降采样处理

        Returns:
            float32 RGB (H, W, 3), [0, 1]
        """
        full_size = image.shape[:2]

        if preview and self.preview_scale < 1.0:
            h_small = max(4, int(full_size[0] * self.preview_scale))
            w_small = max(4, int(full_size[1] * self.preview_scale))
            working = self._downsample(image, h_small, w_small)
        else:
            working = image.copy()

        # 阶段 1: 预处理
        for f in self._stages[FilterStage.FRONTEND]:
            try:
                working = f.apply(working, preview=preview)
            except Exception as e:
                logger.warning(f"FrontEnd filter '{f.name}' failed: {e}")

        # 阶段 2: 核心处理
        for f in self._stages[FilterStage.CORE]:
            try:
                working = f.apply(working, preview=preview)
            except Exception as e:
                logger.warning(f"Core filter '{f.name}' failed: {e}")

        # 阶段 3: 后处理
        for f in self._stages[FilterStage.BACKEND]:
            try:
                working = f.apply(working, preview=preview)
            except Exception as e:
                logger.warning(f"BackEnd filter '{f.name}' failed: {e}")

        # 恢复原尺寸
        if preview and self.preview_scale < 1.0 and working.shape[:2] != full_size:
            working = self._upsample(working, full_size[0], full_size[1])

        return np.clip(working, 0.0, 1.0)

    @staticmethod
    def _downsample(img: np.ndarray, h: int, w: int) -> np.ndarray:
        """整数倍降采样（平均池化）。"""
        h_src, w_src = img.shape[:2]
        step_h = h_src / h
        step_w = w_src / w
        result = np.zeros((h, w, img.shape[2]) if img.ndim == 3 else (h, w), dtype=np.float32)

        for i in range(h):
            i_start = int(i * step_h)
            i_end = int((i + 1) * step_h)
            for j in range(w):
                j_start = int(j * step_w)
                j_end = int((j + 1) * step_w)
                if img.ndim == 3:
                    result[i, j] = np.mean(img[i_start:i_end, j_start:j_end], axis=(0, 1))
                else:
                    result[i, j] = np.mean(img[i_start:i_end, j_start:j_end])

        return result.astype(np.float32)

    @staticmethod
    def _upsample(img: np.ndarray, h: int, w: int) -> np.ndarray:
        """最近邻上采样。"""
        h_src, w_src = img.shape[:2]
        y_ratio = h_src / h
        x_ratio = w_src / w
        y_idx = np.clip((np.arange(h) * y_ratio).astype(int), 0, h_src - 1)
        x_idx = np.clip((np.arange(w) * x_ratio).astype(int), 0, w_src - 1)
        return img[y_idx[:, None], x_idx[None, :]]

    def describe(self) -> str:
        """生成管线描述。"""
        lines = ["=== Processing Pipeline ==="]
        for stage_name, stage_label in [
            (FilterStage.FRONTEND, "FrontEnd (Pre-processing)"),
            (FilterStage.CORE, "Core (Main processing)"),
            (FilterStage.BACKEND, "BackEnd (Post-processing)"),
        ]:
            filters = self._stages[stage_name]
            if filters:
                lines.append(f"  [{stage_label}]")
                for f in filters:
                    lines.append(f"    {f.name} (strength={f.strength:.2f})")
        return '\n'.join(lines)
