"""DCU 特色滤镜 — 智能锐化、清晰度、B&W滤镜模拟、色调映射。

基于 Pentax DCU 5 ISL 引擎的 SmartSharp / MacroDetail / Mono / Sepia 设计。
"""

import numpy as np
from .base import BaseFilter


class SmartSharpFilter(BaseFilter):
    """智能锐化 — 边缘感知 USM，抑制光晕。

    参考 DCU IslEImageServerFilterSmartSharp + IslEISFilterUnSharpMask：
    1. 用梯度幅值检测边缘
    2. 在边缘区域应用 USM
    3. 平坦区域不加锐化（避免噪点放大）
    """

    def __init__(self, strength: float = 0.5, radius: float = 1.0,
                 threshold: float = 0.05, strength_param: float = None):
        super().__init__(name="SmartSharp", strength=strength_param if strength_param else strength)
        self.radius = max(0.3, min(5.0, radius))
        self.threshold = threshold

    def apply(self, image: np.ndarray, preview: bool = False) -> np.ndarray:
        if self.strength <= 0.0:
            return image

        sigma = self.radius * (1.5 if preview else 1.0)
        blurred = self._gauss(image, sigma)

        detail = image - blurred

        edge_map = self._edge_mask(image, sigma * 2)
        edge_map = np.clip(edge_map / max(self.threshold, 0.001), 0.0, 1.0)

        sharpened = image + self.strength * 2.0 * detail * edge_map[..., np.newaxis]
        sharpened = np.clip(sharpened, 0.0, 1.0)

        return self.blend(image, sharpened)

    @staticmethod
    def _edge_mask(image, sigma):
        from numpy.lib.stride_tricks import sliding_window_view
        gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        r = int(np.ceil(sigma * 2))
        r = max(1, min(r, 20))

        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / 8.0

        padded = np.pad(gray, 1, mode='edge')
        wx = sliding_window_view(padded, (3, 3))
        gx = np.sum(wx * sobel_x, axis=(-2, -1))
        gy = np.sum(wx * sobel_y, axis=(-2, -1))
        mag = np.sqrt(gx**2 + gy**2)

        r2 = max(2, r // 2)
        x = np.arange(-r2, r2 + 1, dtype=np.float32)
        k = np.exp(-x**2 / (2 * (sigma * 1.5)**2))
        k /= k.sum()
        padded = np.pad(mag, ((0, 0), (r2, r2)), mode='edge')
        w = sliding_window_view(padded, 2 * r2 + 1, axis=1)
        h = np.tensordot(w, k, axes=([2], [0]))
        padded = np.pad(h, ((r2, r2), (0, 0)), mode='edge')
        w = sliding_window_view(padded, 2 * r2 + 1, axis=0)
        return np.tensordot(w, k, axes=([2], [0])).astype(np.float32)

    @staticmethod
    def _gauss(image, sigma):
        from numpy.lib.stride_tricks import sliding_window_view
        if sigma < 0.3:
            return image
        r = int(np.ceil(sigma * 3))
        r = max(1, min(r, 30))
        x = np.arange(-r, r + 1, dtype=np.float32)
        k = np.exp(-x**2 / (2 * sigma**2))
        k /= k.sum()
        padded = np.pad(image, ((0, 0), (r, r), (0, 0)), mode='edge')
        w = sliding_window_view(padded, 2 * r + 1, axis=1)
        h = np.tensordot(w, k, axes=([3], [0]))
        padded = np.pad(h, ((r, r), (0, 0), (0, 0)), mode='edge')
        w = sliding_window_view(padded, 2 * r + 1, axis=0)
        return np.tensordot(w, k, axes=([3], [0])).astype(np.float32)


class ClarityFilter(BaseFilter):
    """清晰度 — 中频细节增强（类似 DCU MacroDetail / Lightroom Clarity）。

    使用高斯差 (DoG) 提取中频纹理，选择性增强。
    """

    def __init__(self, strength: float = 0.5, sigma_low: float = 0.8,
                 sigma_high: float = 8.0, strength_param: float = None):
        super().__init__(name="Clarity", strength=strength_param if strength_param else strength)
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high

    def apply(self, image: np.ndarray, preview: bool = False) -> np.ndarray:
        if self.strength <= 0.0:
            return image

        s_low = self.sigma_low * (1.5 if preview else 1.0)
        s_high = self.sigma_high / (1.5 if preview else 1.0)

        blurred_low = SmartSharpFilter._gauss(image, s_low)
        blurred_high = SmartSharpFilter._gauss(image, s_high)

        mid_freq = blurred_high - blurred_low

        result = image + self.strength * 3.0 * mid_freq
        result = np.clip(result, 0.0, 1.0)
        return self.blend(image, result)


class BWFilterSim(BaseFilter):
    """B&W 镜头滤镜模拟 — 模拟红/橙/黄/绿滤镜在黑白摄影中的效果。

    参考 DCU IslEImageServerFilterMono + IslEISFilterGrayAxis。
    """

    FILTERS = {
        "red":    (1.0, 0.2, 0.1, "红色滤镜 — 压暗蓝天，增强云彩对比"),
        "orange": (1.0, 0.5, 0.15, "橙色滤镜 — 介于红黄之间"),
        "yellow": (1.0, 0.8, 0.3, "黄色滤镜 — 轻微压暗蓝天"),
        "green":  (0.3, 1.0, 0.3, "绿色滤镜 — 提亮植被"),
        "blue":   (0.1, 0.3, 1.0, "蓝色滤镜 — 增强雾气效果"),
        "none":   (0.299, 0.587, 0.114, "标准灰度 — BT.601 亮度"),
    }

    def __init__(self, filter_type: str = "none", strength: float = 1.0,
                 strength_param: float = None):
        super().__init__(name="BWFilter", strength=strength_param if strength_param else strength)
        self.filter_type = filter_type if filter_type in self.FILTERS else "none"
        self._weights = np.array(self.FILTERS[self.filter_type][:3], dtype=np.float32)

    def apply(self, image: np.ndarray, preview: bool = False) -> np.ndarray:
        weights = self._weights / self._weights.sum()
        gray = (image[..., 0] * weights[0] +
                image[..., 1] * weights[1] +
                image[..., 2] * weights[2])
        result = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
        result = np.clip(result, 0.0, 1.0)
        return self.blend(image, result)


class ToningFilter(BaseFilter):
    """色调映射 — 模拟暗房调色（Sepia 棕褐色 / Selenium 冷蓝色 等）。

    参考 DCU IslEImageServerFilterSepia + IslZColorMatch。
    支持高光/阴影分离调色 (split toning)。
    """

    PRESETS = {
        "sepia":      ((0.9, 0.7, 0.5), (0.15, 0.1, 0.05), "棕褐色 — 经典怀旧"),
        "selenium":   ((0.7, 0.65, 0.7), (0.05, 0.03, 0.08), "硒调 — 冷紫褐色"),
        "cyanotype":  ((0.3, 0.4, 0.7), (0.02, 0.02, 0.1), "蓝晒 — 蓝白色调"),
        "platinum":   ((0.75, 0.7, 0.65), (0.08, 0.07, 0.06), "铂金 — 暖灰调"),
        "copper":     ((0.85, 0.55, 0.3), (0.1, 0.05, 0.02), "铜调 — 红棕色"),
        "cool":       ((0.6, 0.65, 0.8), (0.05, 0.05, 0.1), "冷调 — 蓝灰色"),
    }

    def __init__(self, preset: str = "sepia", highlight_strength: float = 0.5,
                 shadow_strength: float = 0.3, strength: float = 1.0,
                 strength_param: float = None):
        super().__init__(name="Toning", strength=strength_param if strength_param else strength)
        self.preset = preset if preset in self.PRESETS else "sepia"
        hl, sh, _ = self.PRESETS[self.preset]
        self.hl_color = np.array(hl, dtype=np.float32)
        self.sh_color = np.array(sh, dtype=np.float32)
        self.hl_str = max(0.0, min(1.0, highlight_strength))
        self.sh_str = max(0.0, min(1.0, shadow_strength))

    def apply(self, image: np.ndarray, preview: bool = False) -> np.ndarray:
        gray = (0.299 * image[..., 0] +
                0.587 * image[..., 1] +
                0.114 * image[..., 2])
        gray = np.clip(gray, 0.0, 1.0)

        hl_mask = gray
        sh_mask = 1.0 - gray

        toned = image.copy()
        for c in range(3):
            toned[..., c] = (toned[..., c] * (1.0 - self.strength) +
                             self.strength * (gray * (1.0 - gray) * toned[..., c] +
                             self.hl_str * hl_mask * self.hl_color[c] +
                             self.sh_str * sh_mask * self.sh_color[c] +
                             (1.0 - self.hl_str * hl_mask - self.sh_str * sh_mask) * gray))

        toned = np.clip(toned, 0.0, 1.0)
        return np.clip(toned, 0.0, 1.0).astype(np.float32)
