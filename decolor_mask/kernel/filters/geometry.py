"""几何校正滤镜 — 基于 DCU ISL 引擎的 Shading 平场校正。"""

import numpy as np
from PIL import Image
from .base import BaseFilter


# ============================================================
#  FlatField — 平场校正（核心改进 4/7）
# ============================================================

class FlatFieldFilter(BaseFilter):
    """平场校正：消除翻拍/扫描的照明不均匀。

    参考 DCU 的 IslEISFilterShading + IslZPentaxShadingIllumination。

    两种模式：
    1. auto: 自动估计照明场（大半径高斯模糊）
    2. file: 读取白帧参考图
    """

    def __init__(self, strength: float = 1.0, flat_frame_path: str = None,
                 blur_radius: int = 101, strength_param: float = None):
        super().__init__(name="FlatField", strength=strength_param if strength_param else strength)
        self.flat_frame_path = flat_frame_path
        self.blur_radius = blur_radius
        self._flat_frame = None

    def set_flat_frame(self, path: str):
        """设置白帧参考图。"""
        self.flat_frame_path = path
        self._flat_frame = None  # 重新加载

    def apply(self, image: np.ndarray, preview: bool = False) -> np.ndarray:
        if self.strength <= 0.0:
            return image

        if self.flat_frame_path and self._flat_frame is None:
            self._flat_frame = self._load_image(self.flat_frame_path)

        flat = self._get_flat_field(image, preview)
        if flat is None:
            return image

        # 归一化
        flat_norm = flat / np.maximum(np.mean(flat, axis=(0, 1), keepdims=True), 0.01)

        # 校正：除以照明场
        corrected = image / np.maximum(flat_norm, 0.1)
        corrected = np.clip(corrected, 0.0, 1.0)

        return self.blend(image, corrected)

    def _get_flat_field(self, image: np.ndarray, preview: bool) -> np.ndarray:
        """获取平场。"""
        if self._flat_frame is not None:
            if self._flat_frame.shape[:2] != image.shape[:2]:
                # 缩放白帧到图像尺寸
                result = np.zeros_like(image)
                # 简单双线性缩放
                h_ratio = self._flat_frame.shape[0] / image.shape[0]
                w_ratio = self._flat_frame.shape[1] / image.shape[1]
                for c in range(3):
                    y_idx = np.clip((np.arange(image.shape[0]) * h_ratio).astype(int), 
                                    0, self._flat_frame.shape[0] - 1)
                    x_idx = np.clip((np.arange(image.shape[1]) * w_ratio).astype(int),
                                    0, self._flat_frame.shape[1] - 1)
                    result[..., c] = self._flat_frame[y_idx[:, None], x_idx[None, :], c]
                return result
            return self._flat_frame.astype(np.float32)

        # 自动模式：大半径模糊作为照明场近似
        radius = max(21, self.blur_radius // 3) if preview else self.blur_radius
        radius = min(radius, min(image.shape[0], image.shape[1]) // 2 - 1)
        if radius < 1:
            return None

        flat = self._gauss_blur_separable(image, radius / 3.0)
        return flat

    @staticmethod
    def _gauss_blur_separable(img: np.ndarray, sigma: float) -> np.ndarray:
        """可分离高斯模糊。"""
        r = int(np.ceil(sigma * 3))
        if r < 1:
            return img
        x = np.arange(-r, r + 1, dtype=np.float32)
        kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
        kernel /= kernel.sum()

        result = np.zeros_like(img)
        if img.ndim == 3:
            for c in range(3):
                ch = img[..., c]
                ch = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode='same'), 1, ch)
                ch = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode='same'), 0, ch)
                result[..., c] = ch
        else:
            result = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode='same'), 1, img)
            result = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode='same'), 0, result)
        return result.astype(np.float32)

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        """加载图片为 float32 RGB。"""
        img = Image.open(path).convert('RGB')
        return np.array(img, dtype=np.float32) / 255.0
