"""降噪滤镜 — 基于 DCU ISL 引擎的 CromaNR 和 BandNR 设计。"""

import numpy as np
from .base import BaseFilter
from .color import rgb_to_ycc, ycc_to_rgb


# ============================================================
#  CromaNR — 色度降噪（核心改进 3/7）
# ============================================================

class CromaNRFilter(BaseFilter):
    """色度降噪：仅对色度通道降噪，保护亮度细节。

    参考 DCU 的 IslEISFilterCromaNR (CromaNRH + CromaNRV)：
    - RGB → YCrCb
    - 对 Cr/Cb 通道做双边滤波近似
    - Y 通道保持不变
    - YCrCb → RGB
    """

    def __init__(self, strength: float = 0.5, spatial_sigma: float = 7.0,
                 color_sigma: float = 20.0, strength_param: float = None):
        super().__init__(name="CromaNR", strength=strength_param if strength_param else strength)
        self.spatial_sigma = spatial_sigma
        self.color_sigma = color_sigma

    def apply(self, image: np.ndarray, preview: bool = False) -> np.ndarray:
        if self.strength <= 0.0:
            return image

        y, cr, cb = rgb_to_ycc(image)

        # 对色度通道做近似双边滤波
        # 使用可分离的高斯模糊 + 范围权重近似
        sigma_s = self.spatial_sigma / (3 if preview else 1)
        sigma_c = self.color_sigma * self.strength

        cr_d = self._bilateral_approx(cr, image[..., 0], sigma_s, sigma_c)
        cb_d = self._bilateral_approx(cb, image[..., 2], sigma_s, sigma_c)

        result = ycc_to_rgb(y, cr_d, cb_d)
        return self.blend(image, result)

    def _bilateral_approx(self, channel: np.ndarray, guide: np.ndarray,
                          sigma_s: float, sigma_c: float, kernel_radius: int = 5) -> np.ndarray:
        """近似双边滤波：空间高斯 + 引导图像的范围权重。

        纯 numpy 实现，避免 opencv 依赖。
        """
        r = min(kernel_radius, max(1, int(sigma_s) + 1))
        if r < 1:
            r = 1

        ys, xs = np.mgrid[-r:r + 1, -r:r + 1]
        spatial_kernel = np.exp(-(xs ** 2 + ys ** 2) / (2 * sigma_s ** 2))

        padded = np.pad(channel, r, mode='edge')
        padded_guide = np.pad(guide, r, mode='edge')
        output = np.zeros_like(channel)

        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                window = padded[i:i + 2 * r + 1, j:j + 2 * r + 1]
                guide_window = padded_guide[i:i + 2 * r + 1, j:j + 2 * r + 1]

                # 范围权重
                range_weight = np.exp(-((window - window[r, r]) ** 2 + 
                                       (guide_window - guide_window[r, r]) ** 2) / (2 * sigma_c ** 2))
                weights = spatial_kernel * range_weight
                output[i, j] = np.sum(window * weights) / max(np.sum(weights), 1e-8)

        return output


# ============================================================
#  BandNR — 频带分解降噪（核心改进 6/7）
# ============================================================

class BandNRFilter(BaseFilter):
    """频带分解降噪：高斯金字塔分解，不同频带独立降噪。

    参考 DCU 的 IslEISFilterBandNR (BandNRFirst + BandNRSecond)：
    - 高斯金字塔分解为 N 层
    - 低频层：强降噪（消除大块噪声/条带）
    - 高频层：弱降噪（保留细节纹理）
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

        # 构建高斯金字塔
        pyramid = [image]
        current = image
        for _ in range(levels - 1):
            h, w = current.shape[0] // 2, current.shape[1] // 2
            if h < 4 or w < 4:
                break
            blurred = self._gauss_blur(current, 2.0)
            current = blurred[::2, ::2]
            pyramid.append(current)

        # 从底层（粗）到顶层（细）降噪
        denoised_pyramid = []
        for idx, level_img in enumerate(reversed(pyramid)):
            depth = len(pyramid) - 1 - idx
            if depth <= 1:
                sigma = self.high_strength * 3.0 + 0.5
            elif depth == 2:
                sigma = self.low_strength * 5.0 + 1.0
            else:
                sigma = self.low_strength * 8.0 + 2.0

            denoised = self._gauss_blur(level_img, sigma)

            # 上采样并加到上一层
            if denoised_pyramid:
                h_prev, w_prev = denoised_pyramid[-1].shape[:2]
                upsampled = self._upsample_bilinear(denoised, h_prev, w_prev)
                denoised_pyramid[-1] = denoised_pyramid[-1] * 0.7 + upsampled * 0.3
            denoised_pyramid.append(denoised)

        result = denoised_pyramid[-1]
        return self.blend(image, result)

    @staticmethod
    def _gauss_blur(img: np.ndarray, sigma: float) -> np.ndarray:
        """分离高斯模糊（numpy实现）。"""
        if sigma < 0.3:
            return img
        r = int(np.ceil(sigma * 3))
        if r < 1:
            r = 1
        x = np.arange(-r, r + 1, dtype=np.float32)
        kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
        kernel /= kernel.sum()
        # 水平
        result = np.apply_along_axis(lambda c: np.convolve(c, kernel, mode='same'), 1, img)
        # 垂直
        result = np.apply_along_axis(lambda c: np.convolve(c, kernel, mode='same'), 0, result)
        return result.astype(np.float32)

    @staticmethod
    def _upsample_bilinear(img: np.ndarray, h: int, w: int) -> np.ndarray:
        """双线性上采样。"""
        h_src, w_src = img.shape[:2]
        y_ratio = h_src / h
        x_ratio = w_src / w
        y = (np.arange(h) * y_ratio).astype(np.float32)
        x = (np.arange(w) * x_ratio).astype(np.float32)
        y0 = np.clip(np.floor(y).astype(int), 0, h_src - 1)
        y1 = np.clip(y0 + 1, 0, h_src - 1)
        x0 = np.clip(np.floor(x).astype(int), 0, w_src - 1)
        x1 = np.clip(x0 + 1, 0, w_src - 1)
        wy = (y - y0)[:, np.newaxis]
        wx = (x - x0)[np.newaxis, :]

        if img.ndim == 3:
            result = np.zeros((h, w, img.shape[2]), dtype=np.float32)
            for c in range(img.shape[2]):
                ch = img[..., c]
                result[..., c] = ((1 - wy) * (1 - wx) * ch[y0[:, None], x0[None, :]] +
                                  (1 - wy) * wx * ch[y0[:, None], x1[None, :]] +
                                  wy * (1 - wx) * ch[y1[:, None], x0[None, :]] +
                                  wy * wx * ch[y1[:, None], x1[None, :]])
        else:
            result = ((1 - wy) * (1 - wx) * img[y0[:, None], x0[None, :]] +
                      (1 - wy) * wx * img[y0[:, None], x1[None, :]] +
                      wy * (1 - wx) * img[y1[:, None], x0[None, :]] +
                      wy * wx * img[y1[:, None], x1[None, :]])

        return result.astype(np.float32)
