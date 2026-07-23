"""插件基类 — 所有图像处理功能插件的抽象基类。"""

from abc import ABC, abstractmethod
import argparse
from ..kernel import ProcessingPipeline


class PluginBase(ABC):
    """功能插件基类。

    每个插件实现一个完整的图像处理功能：
    - build_pipeline: 构建处理管线
    - attach_ui: 将 UI 面板附加到父容器 (QWidget)
    - add_cli_args: 添加 CLI 参数
    - get_cli_kwargs: 从 CLI 参数提取处理参数
    """

    name: str = "base"
    title: str = "未命名功能"
    icon: str = ""

    # ── 启停接口 (供 App 调用) ──────────────────────────────

    def is_enabled(self) -> bool:
        """返回当前是否启用。子类重写。"""
        return False

    def set_enabled(self, value: bool) -> None:
        """设置启用状态。子类重写。"""

    # ── 核心接口 ──────────────────────────────────────────

    @abstractmethod
    def build_pipeline(self, **params) -> ProcessingPipeline:
        """根据参数构建处理管线。"""
        ...

    @abstractmethod
    def attach_ui(self, parent, on_change=None) -> None:
        """将 UI 面板附加到父容器 (QWidget / QLayout 可添加)。

        Args:
            parent: QWidget 父容器
            on_change: 参数变更回调 (无参数可调用对象)
        """
        ...

    def get_params(self) -> dict:
        """从 UI 收集当前参数。"""
        return {}

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        """添加 CLI 参数到 argparse 解析器。"""
        pass

    def get_cli_kwargs(self, args) -> dict:
        """从 CLI args 提取处理参数。"""
        return {}

    def __repr__(self):
        return f"Plugin({self.name}, '{self.title}')"
