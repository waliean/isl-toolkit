"""色彩空间工具与灰度轴检测 — 基于 DCU ISL 引擎的 LCC/LAB/YCC 设计。"""

import numpy as np
from .base import BaseFilter


# ============================================================
#  色彩空间转换（纯 numpy 实现，无外部依赖）
# ============================================================

def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """RGB → CIELAB (D65)。

    简化实现：RGB → XYZ → LAB。
    参考 DCU 的 IslZColorLAB 色彩空间。
    """
    rgb = np.clip(rgb, 0.0, 1.0)

    # sRGB → 线性 RGB
    mask = rgb <= 0.04045
    linear = np.where(mask, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

    # 线性 RGB → XYZ (D65)
    x = 0.4124564 * linear[..., 0] + 0.3575761 * linear[..., 1] + 0.1804375 * linear[..., 2]
    y = 0.2126729 * linear[..., 0] + 0.7151522 * linear[..., 1] + 0.0721750 * linear[..., 2]
    z = 0.0193339 * linear[..., 0] + 0.1191920 * linear[..., 1] + 0.9503041 * linear[..., 2]

    # XYZ → LAB (D65 白点)
    xn, yn, zn = 0.95047, 1.0, 1.08883

    def f(t):
        delta = 6.0 / 29.0
        return np.where(t > delta ** 3, t ** (1.0 / 3.0), t / (3 * delta ** 2) + 4.0 / 29.0)

    fy = f(y / yn)
    L = 116.0 * fy - 16.0
    a = 500.0 * (f(x / xn) - fy)
    b = 200.0 * (fy - f(z / zn))

    return np.stack([L, a, b], axis=-1).astype(np.float32)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIELAB → RGB 逆变换。"""
    L, a, b_ch = lab[..., 0], lab[..., 1], lab[..., 2]

    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b_ch / 200.0

    delta = 6.0 / 29.0
    x = np.where(fx > delta, fx ** 3, 3 * delta ** 2 * (fx - 4.0 / 29.0)) * 0.95047
    y = np.where(fy > delta, fy ** 3, 3 * delta ** 2 * (fy - 4.0 / 29.0)) * 1.0
    z = np.where(fz > delta, fz ** 3, 3 * delta ** 2 * (fz - 4.0 / 29.0)) * 1.08883

    # XYZ → 线性 RGB
    lr = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    lg = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    lb = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z

    linear = np.stack([lr, lg, lb], axis=-1)
    linear = np.clip(linear, 0.0, 1.0)

    # 线性 RGB → sRGB
    mask = linear <= 0.0031308
    srgb = np.where(mask, 12.92 * linear, 1.055 * linear ** (1.0 / 2.4) - 0.055)

    return np.clip(srgb, 0.0, 1.0).astype(np.float32)


def rgb_to_lcc(rgb: np.ndarray) -> tuple:
    """RGB → LCC (Luminance-Chrominance-Chrominance)。

    参考 DCU 的 IslZColorLCC 空间设计。
    L = 加权亮度
    C1 = R - L (红-青轴)
    C2 = B - L (蓝-黄轴)

    返回 (L, C1, C2) 三个独立数组。
    """
    L = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    C1 = rgb[..., 0] - L  # R - L
    C2 = rgb[..., 2] - L  # B - L
    return L, C1, C2


def lcc_to_rgb(L: np.ndarray, C1: np.ndarray, C2: np.ndarray) -> np.ndarray:
    """LCC → RGB 逆变换。"""
    r = np.clip(L + C1, 0.0, 1.0)
    # G = (L - 0.299*R - 0.114*B) / 0.587
    g = np.clip((L - 0.299 * r - 0.114 * (L + C2)) / 0.587, 0.0, 1.0)
    b = np.clip(L + C2, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def rgb_to_ycc(rgb: np.ndarray) -> tuple:
    """RGB → YCrCb (BT.601)。

    参考 DCU 的 IslZColorYCC / IslEISFilterYCbCr。
    """
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    cr = (rgb[..., 0] - y) * 0.713 + 0.5
    cb = (rgb[..., 2] - y) * 0.564 + 0.5
    return y, cr, cb


def ycc_to_rgb(y: np.ndarray, cr: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """YCrCb → RGB 逆变换。"""
    r = np.clip(y + 1.402 * (cr - 0.5), 0.0, 1.0)
    g = np.clip(y - 0.344136 * (cb - 0.5) - 0.714136 * (cr - 0.5), 0.0, 1.0)
    b = np.clip(y + 1.772 * (cb - 0.5), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


# ============================================================
#  GrayAxis — 灰度轴片基检测（核心改进 1/7）
# ============================================================

class GrayAxisFilmBase:
    """使用灰度轴方法检测负片的片基色偏移。

    参考 DCU 的 IslEISFilterGrayAxis + IslZColorLCC。

    原理：
    1. 在 LAB 空间中找到最亮像素（片基区域）
    2. 分析这些像素在 AB 平面上的分布
    3. AB 中位数即为片基色偏移向量
    """

    def __init__(self, percentile: float = 95.0, min_samples: int = 100):
        self.percentile = max(50.0, min(99.9, percentile))
        self.min_samples = min_samples

    def detect(self, image: np.ndarray) -> dict:
        """检测片基色。

        Returns:
            dict with:
                'L': 片基亮度
                'a_offset': a通道偏移（红-绿）
                'b_offset': b通道偏移（黄-蓝）
                'sample_count': 采样像素数
        """
        lab = rgb_to_lab(image)
        L, a, b_ch = lab[..., 0], lab[..., 1], lab[..., 2]

        threshold = np.percentile(L, self.percentile)
        mask = L >= threshold

        count = np.sum(mask)
        if count < self.min_samples:
            # 不够样本，降阈值
            threshold = np.percentile(L, 90)
            mask = L >= threshold
            count = np.sum(mask)

        if count < 10:
            return {'L': 90.0, 'a_offset': 0.0, 'b_offset': 0.0, 'sample_count': 0}

        a_offset = float(np.median(a[mask]))
        b_offset = float(np.median(b_ch[mask]))
        L_base = float(np.median(L[mask]))

        return {
            'L': L_base,
            'a_offset': a_offset,
            'b_offset': b_offset,
            'sample_count': int(count),
        }


# ============================================================
#  LCC 空间负片反转（核心改进）
# ============================================================

class LCCInverter(BaseFilter):
    """LCC 空间中的负片反转滤镜。

    参考 DCU 的 IslZColorLCC + IslEISFilterGrayAxis 组合：
    1. RGB → LCC (亮度-色度分离)
    2. 校正色度偏移（片基色）
    3. 反转亮度通道
    4. LCC → RGB
    """

    def __init__(self, film_base: dict = None, contrast: float = 1.0, strength: float = 1.0):
        super().__init__(name="LCCInverter", strength=strength)
        self.film_base = film_base or {'a_offset': 0.0, 'b_offset': 0.0, 'L': 95.0}
        self.contrast = contrast
        self._gray_axis = GrayAxisFilmBase()

    def apply(self, image: np.ndarray, preview: bool = False) -> np.ndarray:
        if self.film_base.get('a_offset') is None:
            # 自动检测
            self.film_base = self._gray_axis.detect(image)

        # RGB → LAB
        lab = rgb_to_lab(image)
        L, a, b_ch = lab[..., 0], lab[..., 1], lab[..., 2]

        # 校正片基色偏移
        a_corrected = a - self.film_base['a_offset']
        b_corrected = b_ch - self.film_base['b_offset']

        # 反转亮度：L_inv = L_max - L
        L_max = self.film_base.get('L', np.percentile(L, 98))
        L_inv = L_max - L

        # 对比度调整
        L_inv = (L_inv - L_inv.mean()) * self.contrast + L_inv.mean()

        # 色度随亮度缩放（保持色彩自然）
        alpha = np.clip(L_inv / np.maximum(L, 1.0), 0.0, 2.0)
        a_final = a_corrected * alpha
        b_final = b_corrected * alpha

        # 重组 LAB 并转回 RGB
        lab_inv = np.stack([
            np.clip(L_inv, 0.0, 100.0),
            np.clip(a_final, -128.0, 128.0),
            np.clip(b_final, -128.0, 128.0),
        ], axis=-1).astype(np.float32)

        result = lab_to_rgb(lab_inv)
        return self.blend(image, result)


# ============================================================
#  Dehaze 去雾滤镜（核心改进 5/7）
# ============================================================

class DehazeFilter(BaseFilter):
    """暗通道先验去雾滤镜。

    参考 DCU 的 IslEISFilterDehaze + IslEISFilterDehazePreProcess。

    使用简化的暗通道先验 + 引导滤波近似。
    """

    def __init__(self, strength: float = 0.5, window_size: int = 15, strength_param: float = None):
        super().__init__(name="Dehaze", strength=strength_param if strength_param else strength)
        self.window_size = window_size

    def apply(self, image: np.ndarray, preview: bool = False) -> np.ndarray:
        ws = max(3, self.window_size // 3) if preview else self.window_size
        r = ws // 2

        # 暗通道（每像素取 RGB 最小值）
        dark = np.min(image, axis=-1)

        # 局部最小值（近似暗通道先验）
        padded = np.pad(dark, ((r, r), (r, r)), mode='edge')
        dark_channel = np.zeros_like(dark)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                dark_channel[i, j] = np.min(padded[i:i + ws, j:j + ws])

        # 大气光估计
        atm_light_val = np.percentile(dark_channel, 95)

        # 透射率
        omega = 0.95 * self.strength
        transmission = 1.0 - omega * (dark_channel / max(atm_light_val, 0.001))
        transmission = np.clip(transmission, 0.1, 1.0)

        # 简易引导滤波（Box blur 近似）
        kernel = np.ones((ws, ws), dtype=np.float32) / (ws * ws)
        padded_t = np.pad(transmission, ((r, r), (r, r)), mode='edge')
        transmission_smooth = np.zeros_like(transmission)
        for i in range(transmission.shape[0]):
            for j in range(transmission.shape[1]):
                transmission_smooth[i, j] = np.sum(padded_t[i:i + ws, j:j + ws] * kernel)
        transmission = np.clip(transmission_smooth, 0.1, 1.0)

        # 去雾
        result = np.zeros_like(image)
        for c in range(3):
            result[..., c] = (image[..., c] - atm_light_val) / np.maximum(transmission, 0.1) + atm_light_val

        result = np.clip(result, 0.0, 1.0)
        return self.blend(image, result)
