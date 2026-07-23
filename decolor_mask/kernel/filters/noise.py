"""降噪滤镜 — 基于 DCU ISL 引擎的 CromaNR 和 BandNR 设计。"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .base import BaseFilter
from .color import rgb_to_ycc, ycc_to_rgb
from ..gpu import gauss_blur as _gpu_gauss_blur, is_available as _gpu_available


def _separable_gauss(img: np.ndarray, sigma: float) -> np.ndarray:
    """快速可分离高斯模糊 (GPU优先, CPU回退)。"""
    if _gpu_available():
        return _gpu_gauss_blur(img, sigma)

    if sigma < 0.3:
        return img
    r = int(np.ceil(sigma * 3))
    r = max(1, min(r, 50))
    x = np.arange(-r, r + 1, dtype=np.float32)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    kernel /= kernel.sum()

    if img.ndim == 3:
        padded = np.pad(img, ((0, 0), (r, r), (0, 0)), mode='edge')
        windows = sliding_window_view(padded, 2 * r + 1, axis=1)
        h = np.tensordot(windows, kernel, axes=([3], [0]))
        padded = np.pad(h, ((r, r), (0, 0), (0, 0)), mode='edge')
        windows = sliding_window_view(padded, 2 * r + 1, axis=0)
        result = np.tensordot(windows, kernel, axes=([3], [0]))
    else:
        padded = np.pad(img, ((0, 0), (r, r)), mode='edge')
        windows = sliding_window_view(padded, 2 * r + 1, axis=1)
        h = np.tensordot(windows, kernel, axes=([2], [0]))
        padded = np.pad(h, ((r, r), (0, 0)), mode='edge')
        windows = sliding_window_view(padded, 2 * r + 1, axis=0)
        result = np.tensordot(windows, kernel, axes=([2], [0]))
    return result.astype(np.float32)


# ============================================================
#  CromaNR — 色度降噪
# ============================================================

class CromaNRFilter(BaseFilter):
    """色度降噪：仅对色度通道做高斯模糊，保护亮度细节。

    RGB → YCrCb → 模糊 Cr/Cb → YCrCb → RGB。
    """

    def __init__(self, strength: float = 0.5, strength_param: float = None):
        super().__init__(name="CromaNR", strength=strength_param if strength_param else strength)

    def apply(self, image: np.ndarray, preview: bool = False) -> np.ndarray:
        if self.strength <= 0.0:
            return image

        y, cr, cb = rgb_to_ycc(image)
        sigma = self.strength * 10.0
        if preview:
            sigma /= 2.0
        if sigma < 0.5:
            return image

        cr_f = _separable_gauss(cr, sigma)
        cb_f = _separable_gauss(cb, sigma)
        result = ycc_to_rgb(y, cr_f, cb_f)
        return self.blend(image, result)


# ============================================================
#  BandNR — 频带分解降噪
# ============================================================

class BandNRFilter(BaseFilter):
    """频带分解降噪：高斯金字塔分解，不同频带独立降噪。

    - 高斯金字塔分解为 N 层
    - 低频层：强降噪
    - 高频层：弱降噪
    """

    def __init__(self, strength: float = 0.5, levels: int = 3,
                 low_strength: float = None, high_strength: float = None,
                 strength_param: float = None):
        super().__init__(name="BandNR", strength=strength_param if strength_param else strength)
        self.levels = max(2, min(5, levels))
        self.low_strength = low_strength if low_strength else self.strength * 1.5
        self.high_strength = high_strength if high_strength else self.strength * 0.3

    def apply(self, image: np.ndarray, preview: bool = False) -> np.ndarray:
        levels = max(2, self.levels - 1) if preview else self.levels

        pyramid = [image]
        current = image
        for _ in range(levels - 1):
            h, w = current.shape[0] // 2, current.shape[1] // 2
            if h < 4 or w < 4:
                break
            blurred = _separable_gauss(current, 2.0)
            current = blurred[::2, ::2]
            pyramid.append(current)

        denoised_pyramid = []
        for idx, level_img in enumerate(reversed(pyramid)):
            depth = len(pyramid) - 1 - idx
            if depth <= 1:
                sigma = self.high_strength * 3.0 + 0.5
            elif depth == 2:
                sigma = self.low_strength * 5.0 + 1.0
            else:
                sigma = self.low_strength * 8.0 + 2.0

            denoised = _separable_gauss(level_img, sigma)

            if denoised_pyramid:
                h_prev, w_prev = denoised_pyramid[-1].shape[:2]
                upsampled = self._upsample_bilinear(denoised, h_prev, w_prev)
                denoised_pyramid[-1] = denoised_pyramid[-1] * 0.7 + upsampled * 0.3
            denoised_pyramid.append(denoised)

        result = denoised_pyramid[-1]
        return self.blend(image, result)

    @staticmethod
    def _upsample_bilinear(img: np.ndarray, h: int, w: int) -> np.ndarray:
        h_src, w_src = img.shape[:2]
        y_ratio = h_src / h
        x_ratio = w_src / w
        y = (np.arange(h) * y_ratio).astype(np.float32)
        x = (np.arange(w) * x_ratio).astype(np.float32)
        y0 = np.clip(np.floor(y).astype(int), 0, h_src - 1)
        y1 = np.clip(y0 + 1, 0, h_src - 1)
        x0 = np.clip(np.floor(x).astype(int), 0, w_src - 1)
        x1 = np.clip(x0 + 1, 0, w_src - 1)
        wy = (y - y0)[:, np.newaxis, np.newaxis]
        wx = (x - x0)[np.newaxis, :, np.newaxis]

        if img.ndim == 3:
            i00 = img[y0[:, None], x0[None, :], :]
            i01 = img[y0[:, None], x1[None, :], :]
            i10 = img[y1[:, None], x0[None, :], :]
            i11 = img[y1[:, None], x1[None, :], :]
        else:
            i00 = img[y0[:, None], x0[None, :]]
            i01 = img[y0[:, None], x1[None, :]]
            i10 = img[y1[:, None], x0[None, :]]
            i11 = img[y1[:, None], x1[None, :]]

        result = ((1 - wy) * (1 - wx) * i00 +
                  (1 - wy) * wx * i01 +
                  wy * (1 - wx) * i10 +
                  wy * wx * i11)
        return result.astype(np.float32)
