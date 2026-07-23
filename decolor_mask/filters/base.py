"""滤镜基类 — 基于 DCU/ISL 引擎的三层管线架构设计。"""

import numpy as np
from abc import ABC, abstractmethod


class BaseFilter(ABC):
    """所有滤镜的抽象基类。

    参考 DCU 的 IslEISFilter 体系：
    - 每个滤镜独立，可自由组合
    - 支持 strength 参数控制强度
    - 预览/正式处理分离
    """

    def __init__(self, name: str = None, strength: float = 1.0):
        self.name = name or self.__class__.__name__
        self.strength = max(0.0, min(1.0, strength))

    @abstractmethod
    def apply(self, image: np.ndarray, preview: bool = False) -> np.ndarray:
        """应用滤镜。

        Args:
            image: float32 RGB 数组 (H, W, 3)，值域 0-1
            preview: True 时使用快速近似算法

        Returns:
            float32 RGB 数组 (H, W, 3)，值域 0-1
        """
        ...

    def blend(self, original: np.ndarray, processed: np.ndarray) -> np.ndarray:
        """按 strength 混合原始和结果。"""
        if self.strength >= 1.0:
            return processed
        if self.strength <= 0.0:
            return original
        return original * (1 - self.strength) + processed * self.strength


class FilterStage:
    """滤镜管线阶段：FrontEnd / Core / BackEnd。

    参考 DCU 的 IslEImageServerFilter 分层设计：
    - FrontEnd: 预处理（降噪、平场校正、色彩空间转换）
    - Core:     核心处理（反转、片基检测、密度调整）
    - BackEnd:  后处理（锐化、色调映射、输出转换）
    """
    FRONTEND = "frontend"
    CORE = "core"
    BACKEND = "backend"
