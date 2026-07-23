"""Test GPU 模块 — 设备检测与状态文本逻辑。

验证：
- 函数接口存在且返回类型正确。
- 未启用时状态文本为"未启用 (CPU)"、设备名为"CPU (numpy)"。
- 启用时状态文本以"已启用 ("开头，设备名不是 "CPU (numpy)"。
- CPU OpenCL 设备不会被误判为 GPU 可用（通过 mock 验证过滤逻辑）。
"""

import sys
import unittest

_MODULE_PATH = "decolor_mask.kernel.gpu"


class TestGpuStatus(unittest.TestCase):
    """测试 get_status_text / get_device_name / is_available。"""

    def test_functions_exist(self):
        """三个公开函数均可调用且无异常。"""
        import importlib
        mod = importlib.import_module(_MODULE_PATH)
        self.assertTrue(callable(mod.is_available))
        self.assertTrue(callable(mod.get_device_name))
        self.assertTrue(callable(mod.get_status_text))

    def test_is_available_returns_bool(self):
        """is_available() 返回 bool。"""
        from decolor_mask.kernel.gpu import is_available
        self.assertIsInstance(is_available(), bool)

    def test_get_device_name_returns_str(self):
        """get_device_name() 返回非空字符串。"""
        from decolor_mask.kernel.gpu import get_device_name
        name = get_device_name()
        self.assertIsInstance(name, str)
        self.assertTrue(len(name) > 0)

    def test_get_status_text_returns_str(self):
        """get_status_text() 返回非空字符串。"""
        from decolor_mask.kernel.gpu import get_status_text
        text = get_status_text()
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)

    def test_status_text_not_available(self):
        """GPU 不可用时状态文本为"未启用 (CPU)"。"""
        from decolor_mask.kernel.gpu import is_available, get_status_text, get_device_name
        if not is_available():
            self.assertEqual(get_status_text(), "未启用 (CPU)")
            self.assertEqual(get_device_name(), "CPU (numpy)")

    def test_status_text_when_available(self):
        """GPU 可用时状态以"已启用 ("开头，设备名不是 CPU 回退值。"""
        from decolor_mask.kernel.gpu import is_available, get_status_text, get_device_name
        if is_available():
            text = get_status_text()
            self.assertTrue(
                text.startswith("已启用 ("),
                f"status text should start with '已启用 (', got: {text!r}",
            )
            self.assertTrue(
                text.endswith(")"),
                f"status text should end with ')', got: {text!r}",
            )
            name = get_device_name()
            self.assertNotEqual(
                name, "CPU (numpy)",
                "device name should not be CPU fallback when GPU is available",
            )


class TestGpuDeviceFiltering(unittest.TestCase):
    """验证设备过滤逻辑：只有 GPU 类型才被视为可用。"""

    def test_cpu_only_devices_not_considered_available(self):
        """模拟 pyopencl 只返回 CPU 设备时，is_available() 应为 False。"""
        try:
            from unittest.mock import patch, MagicMock
            import importlib
            import pyopencl as cl
        except ImportError:
            self.skipTest("需要 pyopencl 和 unittest.mock")

        # 构造一个假的 CPU 设备
        cpu_device = MagicMock()
        cpu_device.type = cl.device_type.CPU
        cpu_device.name = "Intel(R) Core(TM) i7 (OpenCL)"

        # 假的 platform 和 get_devices
        fake_platform = MagicMock()
        fake_platform.get_devices.return_value = [cpu_device]

        fake_cl = MagicMock()
        fake_cl.get_platforms.return_value = [fake_platform]
        fake_cl.device_type = cl.device_type  # 保留真实常量用于位运算

        # 先确保模块未缓存
        sys.modules.pop(_MODULE_PATH, None)

        with patch.dict(sys.modules, {"pyopencl": fake_cl}):
            mod = importlib.import_module(_MODULE_PATH)
            self.assertFalse(
                mod.is_available(),
                "CPU-only OpenCL device must not make is_available() return True",
            )
            self.assertEqual(
                mod.get_status_text(), "未启用 (CPU)",
            )
        # 清理以还原真实模块
        sys.modules.pop(_MODULE_PATH, None)

    def test_gpu_device_makes_available(self):
        """模拟 pyopencl 返回 GPU 设备时，is_available() 应为 True。"""
        try:
            from unittest.mock import patch, MagicMock
            import importlib
            import pyopencl as cl
        except ImportError:
            self.skipTest("需要 pyopencl 和 unittest.mock")

        gpu_device = MagicMock()
        gpu_device.type = cl.device_type.GPU
        gpu_device.name = "NVIDIA GeForce RTX 3060"

        fake_platform = MagicMock()
        fake_platform.get_devices.return_value = [gpu_device]

        fake_cl = MagicMock()
        fake_cl.get_platforms.return_value = [fake_platform]
        fake_cl.device_type = cl.device_type
        # Mock Context and CommandQueue to avoid actual OpenCL init
        fake_cl.Context = MagicMock()
        fake_cl.CommandQueue = MagicMock()

        sys.modules.pop(_MODULE_PATH, None)

        with patch.dict(sys.modules, {"pyopencl": fake_cl}):
            mod = importlib.import_module(_MODULE_PATH)
            self.assertTrue(
                mod.is_available(),
                "GPU device must make is_available() return True",
            )
            status = mod.get_status_text()
            self.assertTrue(
                status.startswith("已启用 ("),
                f"status should start with '已启用 (', got: {status!r}",
            )

        sys.modules.pop(_MODULE_PATH, None)

    def test_gpu_on_second_platform_detected(self):
        """第一个 platform 只有 CPU，后一个 platform 有 GPU 时仍应检测到。"""
        try:
            from unittest.mock import patch, MagicMock
            import importlib
            import pyopencl as cl
        except ImportError:
            self.skipTest("需要 pyopencl 和 unittest.mock")

        # Platform 0: 只有 CPU
        cpu_device = MagicMock()
        cpu_device.type = cl.device_type.CPU
        cpu_device.name = "Intel(R) Core(TM) i7 (OpenCL)"
        plat0 = MagicMock()
        plat0.get_devices.return_value = [cpu_device]

        # Platform 1: 有 GPU
        gpu_device = MagicMock()
        gpu_device.type = cl.device_type.GPU
        gpu_device.name = "NVIDIA GeForce RTX 3060"
        plat1 = MagicMock()
        plat1.get_devices.return_value = [gpu_device]

        fake_cl = MagicMock()
        fake_cl.get_platforms.return_value = [plat0, plat1]
        fake_cl.device_type = cl.device_type
        fake_cl.Context = MagicMock()
        fake_cl.CommandQueue = MagicMock()

        sys.modules.pop(_MODULE_PATH, None)

        with patch.dict(sys.modules, {"pyopencl": fake_cl}):
            mod = importlib.import_module(_MODULE_PATH)
            self.assertTrue(
                mod.is_available(),
                "GPU on platform 1 (after CPU-only platform 0) must be detected",
            )
            status = mod.get_status_text()
            self.assertTrue(
                status.startswith("已启用 ("),
                f"status should start with '已启用 (', got: {status!r}",
            )

        sys.modules.pop(_MODULE_PATH, None)


if __name__ == "__main__":
    unittest.main()
