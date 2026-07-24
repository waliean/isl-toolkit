"""正负逆冲 (Cross Process) — WB混合交叉冲洗功能插件。

使用相机白平衡 (Camera WB) 与目标白平衡 (Target WB) 按强度混合，产生色彩偏移效果。
"""

import argparse

from .base import PluginBase
from ..kernel import ProcessingPipeline


class CrossProcessPlugin(PluginBase):
    """正负逆冲 (Cross Process) 插件：相机WB与目标WB按强度混合。"""

    name = "cross"
    title = "正负逆冲"
    icon = "🎞"

    def __init__(self):
        self._ui_vars = {}
        self._on_change = None
        self._enabled_cb = None

    def build_pipeline(self, **params) -> ProcessingPipeline:
        return ProcessingPipeline(preview_scale=0.25)

    def is_enabled(self) -> bool:
        if self._enabled_cb is not None:
            return self._enabled_cb.isChecked()
        return False

    def set_enabled(self, value: bool):
        if self._enabled_cb is not None:
            self._enabled_cb.setChecked(value)

    def attach_ui(self, parent, on_change=None) -> None:
        from PySide6.QtWidgets import QCheckBox, QComboBox, QVBoxLayout, QLabel, QFrame
        from PySide6.QtCore import Qt
        from ..ui.widgets import SliderRow, _Var
        self._on_change = on_change

        self._enabled_cb = QCheckBox("启用 — 正负逆冲 (Cross Process)")
        self._enabled_cb.setChecked(False)
        self._enabled_cb.toggled.connect(lambda: self._notify())
        parent.layout().addWidget(self._enabled_cb)

        # 卡片容器
        card = QFrame()
        card.setObjectName("paramCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 8, 10, 8)
        card_layout.setSpacing(6)

        # 白平衡选择行
        from PySide6.QtWidgets import QWidget, QHBoxLayout
        wb_row = QWidget()
        wb_layout = QHBoxLayout(wb_row)
        wb_layout.setContentsMargins(0, 2, 0, 2)
        wb_layout.setSpacing(4)

        wb_label = QLabel("目标白平衡 (WB):")
        wb_label.setMinimumWidth(100)
        wb_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        wb_layout.addWidget(wb_label)

        self._wb_var = _Var("daylight")
        wb = QComboBox()
        wb.addItems(["auto", "camera", "daylight"])
        wb.setCurrentText("daylight")
        wb.currentTextChanged.connect(lambda text: (
            self._wb_var.set(text), self._notify()
        ))
        wb_layout.addWidget(wb)
        wb_layout.addWidget(QLabel("  auto=自动  camera=原厂  daylight=日光"))
        wb_layout.addStretch()
        card_layout.addWidget(wb_row)
        self._ui_vars["wb_mode"] = self._wb_var

        # 强度滑块
        row = SliderRow("混合强度 (Strength)", 0.0, 1.0, 0.8, resolution=0.01)
        row.value_changed.connect(lambda v: self._notify())
        card_layout.addWidget(row)
        self._ui_vars["strength"] = row.var

        # 分隔 & 说明
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        card_layout.addWidget(sep)

        desc = QLabel("正负逆冲 (Cross Process)：\n相机白平衡 (Camera WB) 与目标白平衡 (Target WB)\n按强度混合，产生色彩偏移效果。")
        desc.setWordWrap(True)
        card_layout.addWidget(desc)

        card_layout.addStretch()
        parent.layout().addWidget(card)
        parent.layout().addStretch()

    def get_params(self) -> dict:
        return {k: (v.get() if hasattr(v, 'get') else v)
                for k, v in self._ui_vars.items()}

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--wb", choices=["auto", "camera", "daylight"], default="daylight",
                            help="目标白平衡 (WB) 模式: daylight/camera/auto")
        parser.add_argument("--strength", "-s", type=float, default=0.8,
                            help="混合强度 (Strength) 0.0-1.0")

    def get_cli_kwargs(self, args) -> dict:
        return {"wb_mode": args.wb, "strength": args.strength}

    def _notify(self):
        if self._on_change:
            self._on_change()
