"""负片反转 (Film Inversion) — 黑白负片扫描件转正片功能插件。

使用 LCC 亮度-色度分离空间进行反转，支持平场校正 (FlatField) 消除照明不均。
"""

import argparse

from .base import PluginBase
from ..kernel import ProcessingPipeline, LCCInverter, FlatFieldFilter


class FilmInversionPlugin(PluginBase):
    """负片反转 (Film Inversion) 插件 — 将负片扫描件转换为正片图像。"""

    name = "film"
    title = "负片反转"
    icon = "🎞"

    def __init__(self):
        self._ui_vars = {}
        self._on_change = None
        self._enabled_cb = None

    def build_pipeline(self, **params) -> ProcessingPipeline:
        pipe = ProcessingPipeline(preview_scale=0.25)
        strength = params.get("strength", 0.8)
        flat_field = params.get("flat_field", None)

        if flat_field:
            pipe.add_frontend(FlatFieldFilter(flat_frame_path=flat_field, strength=1.0))
        else:
            pipe.add_frontend(FlatFieldFilter(strength=0.5))

        pipe.add_core(LCCInverter(strength=strength))
        return pipe

    def is_enabled(self) -> bool:
        if self._enabled_cb is not None:
            return self._enabled_cb.isChecked()
        return False

    def set_enabled(self, value: bool):
        if self._enabled_cb is not None:
            self._enabled_cb.setChecked(value)

    def attach_ui(self, parent, on_change=None) -> None:
        from PySide6.QtWidgets import QCheckBox, QComboBox, QVBoxLayout, QLabel, QWidget, QHBoxLayout, QFrame
        from PySide6.QtCore import Qt
        from ..ui.widgets import SliderRow, _Var
        self._on_change = on_change

        self._enabled_cb = QCheckBox("启用 — 负片反转 (Film Inversion)")
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

        # 反转强度滑块
        row = SliderRow("反转强度 (Strength)", 0.0, 1.0, 0.8, resolution=0.01)
        row.value_changed.connect(lambda v: self._notify())
        card_layout.addWidget(row)
        self._ui_vars["strength"] = row.var

        # 分隔 & 说明
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        card_layout.addWidget(sep)

        desc = QLabel("负片反转 (Film Inversion)：\n将黑白负片扫描件通过 LCC 空间\n亮度-色度分离反转算法转为正片。")
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
                            help="反转强度 (Strength) 0.0-1.0")
        parser.add_argument("--flat-field", type=str, default=None,
                            help="平场校正 (FlatField) 白帧参考图路径")

    def get_cli_kwargs(self, args) -> dict:
        return {"wb_mode": args.wb, "strength": args.strength, "flat_field": args.flat_field}

    def _notify(self):
        if self._on_change:
            self._on_change()
