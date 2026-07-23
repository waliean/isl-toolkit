"""GPU 加速模块 — 基于 OpenCL 的滤镜加速。

所有 GPU 操作接收并返回 numpy 数组，内部处理 GPU 上下文。
提供 gpu_chain() 用于在 GPU 上一次执行多个滤波操作。
"""

import numpy as np

_AVAILABLE = False
_CTX = None
_QUEUE = None
_PROGRAMS = {}

try:
    import pyopencl as cl
    _platforms = cl.get_platforms()
    if _platforms:
        # 遍历所有 platform，防止 GPU 在非首个 platform 上被漏掉
        _gpu_devices = []
        for platform in _platforms:
            _devices = platform.get_devices()
            _gpu_devices.extend([d for d in _devices if d.type & cl.device_type.GPU])
        if _gpu_devices:
            _CTX = cl.Context(_gpu_devices)
            _QUEUE = cl.CommandQueue(_CTX)
            _AVAILABLE = True
except Exception:
    pass


def is_available() -> bool:
    return _AVAILABLE


def get_device_name() -> str:
    if not _AVAILABLE:
        return "CPU (numpy)"
    try:
        return _CTX.devices[0].name
    except Exception:
        return "Unknown GPU"


def get_status_text() -> str:
    """返回 GPU 状态文本，供 CLI/GUI 统一显示。"""
    if is_available():
        return f"已启用 ({get_device_name()})"
    return "未启用 (CPU)"


# ============================================================
#  OpenCL 内核
# ============================================================

_GAUSS_KERNEL = """
__kernel void gaussH(__global const float *in, __global float *out,
                      __constant float *gkernel, const int kr,
                      const int w, const int h, const int ch) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    int base = (y * w + x) * ch;
    float sum[4] = {0,0,0,0};
    for (int k = -kr; k <= kr; k++) {
        int sx = clamp(x + k, 0, w - 1);
        int src = (y * w + sx) * ch;
        float kw = gkernel[k + kr];
        for (int c = 0; c < ch; c++) sum[c] += in[src + c] * kw;
    }
    for (int c = 0; c < ch; c++) out[base + c] = sum[c];
}

__kernel void gaussV(__global const float *in, __global float *out,
                      __constant float *gkernel, const int kr,
                      const int w, const int h, const int ch) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    int base = (y * w + x) * ch;
    float sum[4] = {0,0,0,0};
    for (int k = -kr; k <= kr; k++) {
        int sy = clamp(y + k, 0, h - 1);
        int src = (sy * w + x) * ch;
        float kw = gkernel[k + kr];
        for (int c = 0; c < ch; c++) sum[c] += in[src + c] * kw;
    }
    for (int c = 0; c < ch; c++) out[base + c] = sum[c];
}
"""

_MIN_KERNEL = """
__kernel void minH(__global const float *in, __global float *out,
                    const int r, const int w, const int h) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float v = 1e10f;
    for (int k = -r; k <= r; k++) {
        int sx = clamp(x + k, 0, w - 1);
        v = fmin(v, in[y * w + sx]);
    }
    out[y * w + x] = v;
}

__kernel void minV(__global const float *in, __global float *out,
                    const int r, const int w, const int h) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float v = 1e10f;
    for (int k = -r; k <= r; k++) {
        int sy = clamp(y + k, 0, h - 1);
        v = fmin(v, in[sy * w + x]);
    }
    out[y * w + x] = v;
}
"""

_MEAN_KERNEL = """
__kernel void meanH(__global const float *in, __global float *out,
                     const int r, const int w, const int h) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float sum = 0, n = 2.0f * r + 1.0f;
    for (int k = -r; k <= r; k++) {
        int sx = clamp(x + k, 0, w - 1);
        sum += in[y * w + sx];
    }
    out[y * w + x] = sum / n;
}

__kernel void meanV(__global const float *in, __global float *out,
                     const int r, const int w, const int h) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float sum = 0, n = 2.0f * r + 1.0f;
    for (int k = -r; k <= r; k++) {
        int sy = clamp(y + k, 0, h - 1);
        sum += in[sy * w + x];
    }
    out[y * w + x] = sum / n;
}
"""


def _compile(name, src):
    global _AVAILABLE
    if name not in _PROGRAMS and _AVAILABLE:
        try:
            _PROGRAMS[name] = cl.Program(_CTX, src).build()
        except Exception:
            _AVAILABLE = False
            return None
    return _PROGRAMS.get(name)


# ============================================================
#  GpuPipeline — GPU 上驻留的管线
# ============================================================

class GpuPipeline:
    """一次上传 → 多次处理 → 一次下载 的 GPU 管线。"""

    def __init__(self, image: np.ndarray):
        if not _AVAILABLE:
            raise RuntimeError("GPU not available")
        self._w = image.shape[1]
        self._h = image.shape[0]
        self._ch = image.shape[2] if image.ndim == 3 else 1
        flat = np.ascontiguousarray(image.astype(np.float32))
        self._buf_a = cl.Buffer(_CTX, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=flat)
        self._buf_b = cl.Buffer(_CTX, cl.mem_flags.READ_WRITE, flat.nbytes)
        self._tmp_1ch = None
        self._tmp_1ch_b = None
        self._in_use = "a"  # track which buffer holds current data

    def _swap(self):
        self._in_use = "b" if self._in_use == "a" else "a"

    def _src(self):
        return self._buf_a if self._in_use == "a" else self._buf_b

    def _dst(self):
        return self._buf_b if self._in_use == "a" else self._buf_a

    def _1ch_size(self):
        s = self._w * self._h * 4
        if self._tmp_1ch is None or self._tmp_1ch_b is None:
            self._tmp_1ch = np.empty((self._h, self._w), dtype=np.float32)
            self._tmp_1ch_b = cl.Buffer(_CTX, cl.mem_flags.READ_WRITE, s)
        return self._tmp_1ch, self._tmp_1ch_b

    def gauss_blur(self, sigma: float):
        """原地高斯模糊。"""
        if sigma < 0.3:
            return self
        r = int(np.ceil(sigma * 3))
        r = max(1, min(r, 50))
        x = np.arange(-r, r + 1, dtype=np.float32)
        kernel = np.exp(-x ** 2 / (2 * sigma ** 2)).astype(np.float32)
        kernel /= kernel.sum()

        kbuf = cl.Buffer(_CTX, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=kernel)

        prog = _compile("gauss", _GAUSS_KERNEL)
        if prog is None:
            return self
        kh = cl.Kernel(prog, "gaussH")
        kv = cl.Kernel(prog, "gaussV")

        kw = np.int32(r)
        iw = np.int32(self._w)
        ih = np.int32(self._h)
        ich = np.int32(self._ch)

        src = self._src()
        dst = self._dst()
        kh(_QUEUE, (self._w, self._h), None, src, dst, kbuf, kw, iw, ih, ich)
        self._swap()
        src, dst = self._src(), self._dst()
        kv(_QUEUE, (self._w, self._h), None, src, dst, kbuf, kw, iw, ih, ich)
        self._swap()
        return self

    def sep_min(self, radius: int):
        """可分离最小值滤波 (单通道)。"""
        if self._ch != 1:
            raise ValueError("sep_min only supports 1-channel images")
        if radius < 1:
            return self

        prog = _compile("min", _MIN_KERNEL)
        if prog is None:
            return self
        kh = cl.Kernel(prog, "minH")
        kv = cl.Kernel(prog, "minV")

        kw = np.int32(radius)
        iw = np.int32(self._w)
        ih = np.int32(self._h)

        src = self._src()
        dst = self._dst()
        kh(_QUEUE, (self._w, self._h), None, src, dst, kw, iw, ih)
        self._swap()
        src, dst = self._src(), self._dst()
        kv(_QUEUE, (self._w, self._h), None, src, dst, kw, iw, ih)
        self._swap()
        return self

    def sep_mean(self, radius: int):
        """可分离均值滤波 (单通道)。"""
        if self._ch != 1:
            raise ValueError("sep_mean only supports 1-channel images")
        if radius < 1:
            return self

        prog = _compile("mean", _MEAN_KERNEL)
        if prog is None:
            return self
        kh = cl.Kernel(prog, "meanH")
        kv = cl.Kernel(prog, "meanV")

        kw = np.int32(radius)
        iw = np.int32(self._w)
        ih = np.int32(self._h)

        src = self._src()
        dst = self._dst()
        kh(_QUEUE, (self._w, self._h), None, src, dst, kw, iw, ih)
        self._swap()
        src, dst = self._src(), self._dst()
        kv(_QUEUE, (self._w, self._h), None, src, dst, kw, iw, ih)
        self._swap()
        return self

    def to_cpu(self) -> np.ndarray:
        """下载结果到 CPU numpy。"""
        result = np.empty((self._h, self._w, self._ch) if self._ch > 1 else (self._h, self._w), dtype=np.float32)
        cl.enqueue_copy(_QUEUE, result, self._src())
        _QUEUE.finish()
        if self._ch == 1:
            return result.reshape(self._h, self._w)
        return result


# ============================================================
#  独立 GPU 操作（每次上传/下载，适合单次调用）
# ============================================================

from numpy.lib.stride_tricks import sliding_window_view


def gauss_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """可分离高斯模糊（GPU加速）。"""
    if not _AVAILABLE or sigma < 0.3:
        return _cpu_gauss_blur(image, sigma)
    try:
        pipe = GpuPipeline(image)
        pipe.gauss_blur(sigma)
        return pipe.to_cpu()
    except Exception:
        return _cpu_gauss_blur(image, sigma)


def sep_min(image: np.ndarray, radius: int) -> np.ndarray:
    """可分离最小值滤波。"""
    if not _AVAILABLE or radius < 1:
        return _cpu_sep_min(image, radius)
    try:
        pipe = GpuPipeline(image)
        pipe.sep_min(radius)
        return pipe.to_cpu()
    except Exception:
        return _cpu_sep_min(image, radius)


def sep_mean(image: np.ndarray, radius: int) -> np.ndarray:
    """可分离均值滤波。"""
    if not _AVAILABLE or radius < 1:
        return _cpu_sep_mean(image, radius)
    try:
        pipe = GpuPipeline(image)
        pipe.sep_mean(radius)
        return pipe.to_cpu()
    except Exception:
        return _cpu_sep_mean(image, radius)


# ============================================================
#  CPU 回退
# ============================================================

def _cpu_gauss_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma < 0.3:
        return image.copy()
    r = int(np.ceil(sigma * 3))
    r = max(1, min(r, 50))
    x = np.arange(-r, r + 1, dtype=np.float32)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    kernel /= kernel.sum()
    if image.ndim == 3:
        padded = np.pad(image, ((0, 0), (r, r), (0, 0)), mode='edge')
        windows = sliding_window_view(padded, 2 * r + 1, axis=1)
        h = np.tensordot(windows, kernel, axes=([3], [0]))
        padded = np.pad(h, ((r, r), (0, 0), (0, 0)), mode='edge')
        windows = sliding_window_view(padded, 2 * r + 1, axis=0)
        result = np.tensordot(windows, kernel, axes=([3], [0]))
    else:
        padded = np.pad(image, ((0, 0), (r, r)), mode='edge')
        windows = sliding_window_view(padded, 2 * r + 1, axis=1)
        h = np.tensordot(windows, kernel, axes=([2], [0]))
        padded = np.pad(h, ((r, r), (0, 0)), mode='edge')
        windows = sliding_window_view(padded, 2 * r + 1, axis=0)
        result = np.tensordot(windows, kernel, axes=([2], [0]))
    return result.astype(np.float32)


def _cpu_sep_min(image: np.ndarray, radius: int) -> np.ndarray:
    ks = 2 * radius + 1
    padded = np.pad(image, ((0, 0), (radius, radius)), mode='edge')
    windows = sliding_window_view(padded, ks, axis=1)
    h = np.min(windows, axis=-1)
    padded = np.pad(h, ((radius, radius), (0, 0)), mode='edge')
    windows = sliding_window_view(padded, ks, axis=0)
    return np.min(windows, axis=-1)


def _cpu_sep_mean(image: np.ndarray, radius: int) -> np.ndarray:
    ks = 2 * radius + 1
    padded = np.pad(image, ((0, 0), (radius, radius)), mode='edge')
    windows = sliding_window_view(padded, ks, axis=1)
    h = np.mean(windows, axis=-1)
    padded = np.pad(h, ((radius, radius), (0, 0)), mode='edge')
    windows = sliding_window_view(padded, ks, axis=0)
    return np.mean(windows, axis=-1)
