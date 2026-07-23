"""isl-toolkit — 通用图像处理工具箱。

基于 ISL 引擎架构：
- kernel: 核心引擎（Pipeline, Filter基础设施, 色彩空间）
- plugins: 功能插件（正负逆冲, 负片反转, 图像增强, 滤镜处理）
- ui: 模块化界面（标签页式工具集）
"""

from .core import (
    is_raw_file, find_raw_files, save_image,
    process_raw, process_folder,
    RAW_EXTENSIONS,
)

__all__ = [
    "is_raw_file", "find_raw_files", "save_image",
    "process_raw", "process_folder",
    "RAW_EXTENSIONS",
]
