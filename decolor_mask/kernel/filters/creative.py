"""风格化滤镜 — 暗角、颗粒、色调曲线、暗部增强。"""

import numpy as np
from .base import BaseFilter


class VignetteFilter(BaseFilter):
    """暗角效果 — 图像边缘渐暗。

    通过径向渐变遮罩压暗四角，模拟镜头暗角或LOMO效果。
    中心保持明亮，边缘按强度变暗。
    """

    def __init__(self, strength: float = 0.5, falloff: float = 1.8,
                 center_x: float = 0.5, center_y: float = 0.5,
                 strength_param: float = None):
        super().__init__(name="Vignette", strength=strength_param if strength_param else strength)
        self.falloff = max(0.3, min(5.0, falloff))
        self.cx = center_x
        self.cy = center_y

    def apply(self, image: np.ndarray, preview: bool = False) -> np.ndarray:
        if self.strength <= 0.0:
            return image

        h, w = image.shape[:2]
        cx_px = int(w * self.cx)
        cy_px = int(h * self.cy)
        max_dist = np.float32(np.sqrt(max(cx_px, w - cx_px)**2 + max(cy_px, h - cy_px)**2))
        if max_dist < 1.0:
            max_dist = np.float32(1.0)

        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x.astype(np.float32) - cx_px)**2 +
                        (y.astype(np.float32) - cy_px)**2) / max_dist
        dist = dist.astype(np.float32)

        falloff = 1.0 - self.strength * (dist ** self.falloff)
        falloff = np.clip(falloff, 0.0, 1.0)

        result = image * falloff[..., np.newaxis]
        result = np.clip(result, 0.0, 1.0)
        return self.blend(image, result)


class GrainFilter(BaseFilter):
    """胶片颗粒 — 模拟高感光度胶片的随机噪点。

    生成单色（同通道）高斯噪声叠加到图像，模拟真实胶片颗粒。
    """

    def __init__(self, strength: float = 0.3, grain_size: float = 1.0,
                 strength_param: float = None):
        super().__init__(name="Grain", strength=strength_param if strength_param else strength)
        self.grain_size = max(0.5, min(4.0, grain_size))

    def apply(self, image: np.ndarray, preview: bool = False) -> np.ndarray:
        if self.strength <= 0.0:
            return image

        h, w = image.shape[:2]

        if preview and (h > 512 or w > 512):
            gh = max(4, h // 4)
            gw = max(4, w // 4)
        elif self.grain_size != 1.0:
            gh = max(4, int(h / self.grain_size))
            gw = max(4, int(w / self.grain_size))
        else:
            gh, gw = h, w

        noise = np.random.randn(gh, gw).astype(np.float32) * self.strength * 0.12

        if gh != h or gw != w:
            from numpy.lib.stride_tricks import sliding_window_view
            y_ratio = gh / h
            x_ratio = gw / w
            y_idx = np.clip((np.arange(h) * y_ratio).astype(int), 0, gh - 1)
            x_idx = np.clip((np.arange(w) * x_ratio).astype(int), 0, gw - 1)
            noise = noise[y_idx[:, None], x_idx[None, :]]

        noise_3ch = np.stack([noise, noise, noise], axis=-1)
        result = image + noise_3ch
        result = np.clip(result, 0.0, 1.0)
        return self.blend(image, result)


class ToneCurveFilter(BaseFilter):
    """色调曲线 — 独立调整高光/阴影/中间调。

    通过三个控制点 (阴影=0.0, 中间调=0.5, 高光=1.0)
    拟合二次曲线作为 LUT 映射表。
    """

    def __init__(self, highlights: float = 0.0, shadows: float = 0.0,
                 midtones: float = 0.0, strength: float = 1.0,
                 strength_param: float = None):
        super().__init__(name="ToneCurve", strength=strength_param if strength_param else strength)
        self.highlights = max(-1.0, min(1.0, highlights))
        self.shadows = max(-1.0, min(1.0, shadows))
        self.midtones = max(-1.0, min(1.0, midtones))

    def apply(self, image: np.ndarray, preview: bool = False) -> np.ndarray:
        if self.strength <= 0.0:
            return image
        hl, sh, md = self.highlights, self.shadows, self.midtones
        if hl == 0.0 and sh == 0.0 and md == 0.0:
            return image

        cp_x = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        cp_y = np.array([
            sh * 0.5,
            0.5 + md * 0.5,
            1.0 + hl * 0.5,
        ], dtype=np.float64)

        coeffs = np.polyfit(cp_x, cp_y, 2)
        x = np.linspace(0.0, 1.0, 256, dtype=np.float64)
        lut = np.polyval(coeffs, x)
        lut = np.clip(lut, 0.0, 1.0).astype(np.float32)

        idx = np.clip((image * 255).astype(np.int32), 0, 255)
        result = lut[idx]

        result = np.clip(result, 0.0, 1.0)
        return self.blend(image, result)


class ShadowBoostFilter(BaseFilter):
    """暗部增强 — 提升阴影区域亮度，同时保留高光。

    对暗区应用伽马提升，用亮度遮罩限制只作用于阴影部分。
    类似 Lightroom 中 Shadows 滑块的效果。
    """

    def __init__(self, strength: float = 0.5, radius: float = 0.3,
                 strength_param: float = None):
        super().__init__(name="ShadowBoost", strength=strength_param if strength_param else strength)
        self.radius = max(0.05, min(1.0, radius))

    def apply(self, image: np.ndarray, preview: bool = False) -> np.ndarray:
        if self.strength <= 0.0:
            return image

        gray = (0.299 * image[..., 0] +
                0.587 * image[..., 1] +
                0.114 * image[..., 2])

        shadow_mask = 1.0 - np.clip(gray / max(self.radius, 0.01), 0.0, 1.0)
        shadow_mask = shadow_mask ** 1.5

        gamma = 1.0 / (1.0 + self.strength * 3.0)
        boosted = image ** gamma

        mask_3ch = (shadow_mask * self.strength)[..., np.newaxis]
        result = image * (1.0 - mask_3ch) + boosted * mask_3ch
        result = np.clip(result, 0.0, 1.0)
        return self.blend(image, result)
