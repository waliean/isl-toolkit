"""图像增强 (Image Enhance) — 亮度/对比度/饱和度 + 锐化/清晰度/暗角/暗部增强/色调曲线。"""

import argparse

from .base import PluginBase
from ..kernel import ProcessingPipeline
from ..kernel.filters.dcu_legacy import SmartSharpFilter, ClarityFilter
from ..kernel.filters.creative import VignetteFilter, ToneCurveFilter, ShadowBoostFilter


class ImageEnhancePlugin(PluginBase):
    """图像增强 (Image Enhance) 插件：亮度/对比度/饱和度 + 修图效果。"""

    name = "enhance"
    title = "图像增强"
    icon = "☀"

    def __init__(self):
        self._ui_vars = {}
        self._on_change = None
        self._enabled_cb = None

    def build_pipeline(self, **params) -> ProcessingPipeline:
        pipe = ProcessingPipeline(preview_scale=0.25)

        smart_sharp = params.get("smart_sharp", 0.0)
        clarity = params.get("clarity", 0.0)
        vignette = params.get("vignette", 0.0)
        shadow_boost = params.get("shadow_boost", 0.0)
        highlights = params.get("highlights", 0.0)
        shadows = params.get("shadows", 0.0)
        midtones = params.get("midtones", 0.0)

        if clarity > 0.01:
            pipe.add_frontend(ClarityFilter(strength=clarity))
        if highlights != 0.0 or shadows != 0.0 or midtones != 0.0:
            pipe.add_backend(ToneCurveFilter(
                highlights=highlights, shadows=shadows, midtones=midtones,
            ))
        if shadow_boost > 0.01:
            pipe.add_backend(ShadowBoostFilter(strength=shadow_boost))
        if smart_sharp > 0.01:
            pipe.add_backend(SmartSharpFilter(strength=smart_sharp))
        if vignette > 0.01:
            pipe.add_backend(VignetteFilter(strength=vignette))

        return pipe

    def is_enabled(self) -> bool:
        if self._enabled_cb is not None:
            return self._enabled_cb.isChecked()
        return False

    def set_enabled(self, value: bool):
        if self._enabled_cb is not None:
            self._enabled_cb.setChecked(value)

    def attach_ui(self, parent, on_change=None) -> None:
        from PySide6.QtWidgets import (
            QCheckBox, QComboBox, QLabel, QWidget, QHBoxLayout, QFrame,
            QVBoxLayout,
        )
        from PySide6.QtCore import Qt
        from ..ui.widgets import SliderRow, _Var

        self._on_change = on_change
        lay = parent.layout()

        self._enabled_cb = QCheckBox("启用 — 图像增强 (Image Enhance)")
        self._enabled_cb.setChecked(True)
        self._enabled_cb.toggled.connect(lambda: self._notify())
        lay.addWidget(self._enabled_cb)

        # ── section card helpers ──
        _cur_card = [None]

        def _close_card():
            if _cur_card[0] is not None:
                lay.addWidget(_cur_card[0][0])
                _cur_card[0] = None

        def _open_card(title: str, hint: str | None = None):
            _close_card()
            card = QFrame()
            card.setObjectName("sectionCard")
            cl = QVBoxLayout(card)
            cl.setContentsMargins(8, 6, 8, 8)
            cl.setSpacing(4)
            title_lbl = QLabel(title)
            title_lbl.setObjectName("sectionHeader")
            cl.addWidget(title_lbl)
            if hint:
                hint_lbl = QLabel(hint)
                hint_lbl.setObjectName("sectionHint")
                hint_lbl.setWordWrap(True)
                cl.addWidget(hint_lbl)
            _cur_card[0] = (card, cl)

        def _cur_layout():
            if _cur_card[0] is not None:
                return _cur_card[0][1]
            return lay

        def _add_slider(key, label, from_, to, default):
            row = SliderRow(label, from_, to, default, resolution=0.01)
            row.value_changed.connect(lambda v: self._notify())
            _cur_layout().addWidget(row)
            self._ui_vars[key] = row.var

        def _add_combo(key, label, options, default):
            cnt = QWidget()
            hlay = QHBoxLayout(cnt)
            hlay.setContentsMargins(0, 2, 0, 2)
            hlay.setSpacing(4)
            lbl = QLabel(label)
            lbl.setMinimumWidth(50)
            hlay.addWidget(lbl)
            var = _Var(default)
            cb = QComboBox()
            cb.addItems(options)
            cb.setCurrentText(default)
            cb.currentTextChanged.connect(lambda text: (var.set(text), self._notify()))
            hlay.addWidget(cb)
            hlay.addStretch()
            _cur_layout().addWidget(cnt)
            self._ui_vars[key] = var

        # ── 基本调整 ──
        _open_card("基本调整", "亮度 / 对比度 / 饱和度")
        _add_slider("brightness", "亮度", 0.1, 3.0, 1.0)
        _add_slider("contrast", "对比度", 0.1, 3.0, 1.0)
        _add_slider("saturation", "饱和度", 0.0, 3.0, 1.0)

        # ── 智能锐化 ──
        _open_card("智能锐化 (SmartSharp)", "边缘感知USM，抑制光晕")
        _add_slider("smart_sharp", "强度", 0.0, 1.0, 0.0)

        # ── 清晰度 ──
        _open_card("清晰度 (Clarity)", "中频细节增强")
        _add_slider("clarity", "强度", 0.0, 1.0, 0.0)

        # ── 色调曲线 ──
        _open_card("色调曲线 (Tone Curve)", "高光 / 阴影 / 中间调独立调整")
        _add_slider("highlights", "高光", -1.0, 1.0, 0.0)
        _add_slider("shadows", "阴影", -1.0, 1.0, 0.0)
        _add_slider("midtones", "中间调", -1.0, 1.0, 0.0)

        # ── 暗部增强 ──
        _open_card("暗部增强 (Shadow Boost)", "提升阴影区域亮度，保留高光")
        _add_slider("shadow_boost", "强度", 0.0, 1.0, 0.0)

        # ── 暗角 ──
        _open_card("暗角 (Vignette)", "径向渐暗，模拟镜头暗角")
        _add_slider("vignette", "强度", 0.0, 1.0, 0.0)

        _close_card()
        lay.addStretch()

    def get_params(self) -> dict:
        return {k: (v.get() if hasattr(v, 'get') else v)
                for k, v in self._ui_vars.items()}

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--brightness", "-b", type=float, default=1.0)
        parser.add_argument("--contrast", "-c", type=float, default=1.0)
        parser.add_argument("--saturation", type=float, default=1.0)
        parser.add_argument("--sharpen", type=float, default=0.0, help="智能锐化 (SmartSharp) 0.0-1.0")
        parser.add_argument("--clarity", type=float, default=0.0, help="清晰度 (Clarity) 0.0-1.0")
        parser.add_argument("--highlights", type=float, default=0.0, help="高光 (Highlights) 调整 -1.0~1.0")
        parser.add_argument("--shadows", type=float, default=0.0, help="阴影 (Shadows) 调整 -1.0~1.0")
        parser.add_argument("--midtones", type=float, default=0.0, help="中间调 (Midtones) 调整 -1.0~1.0")
        parser.add_argument("--shadow-boost", type=float, default=0.0, help="暗部增强 (Shadow Boost) 0.0-1.0")
        parser.add_argument("--vignette", type=float, default=0.0, help="暗角 (Vignette) 强度 0.0-1.0")

    def get_cli_kwargs(self, args) -> dict:
        return {
            "brightness": args.brightness,
            "contrast": args.contrast,
            "saturation": args.saturation,
            "smart_sharp": args.sharpen,
            "clarity": args.clarity,
            "highlights": args.highlights,
            "shadows": args.shadows,
            "midtones": args.midtones,
            "shadow_boost": args.shadow_boost,
            "vignette": args.vignette,
        }

    def _notify(self):
        if self._on_change:
            self._on_change()
