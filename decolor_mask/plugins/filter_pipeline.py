"""滤镜处理 (Filter Pipeline) — 降噪/去雾/平场/胶片颗粒/B&W滤镜模拟/色调映射。"""

import argparse

from .base import PluginBase
from ..kernel import (
    ProcessingPipeline,
    CromaNRFilter,
    BandNRFilter,
    DehazeFilter,
    FlatFieldFilter,
    BWFilterSim,
    GrainFilter,
    ToningFilter,
)


class FilterPipelinePlugin(PluginBase):
    """滤镜处理 (Filter Pipeline) 插件：降噪/去雾/平场/颗粒/B&W滤镜。"""

    name = "filters"
    title = "滤镜处理"
    icon = "🔧"

    def __init__(self):
        self._ui_vars = {}
        self._on_change = None
        self._enabled_cb = None

    def build_pipeline(self, **params) -> ProcessingPipeline:
        pipe = ProcessingPipeline(preview_scale=0.25)

        bw_filter = params.get("bw_filter", "none")
        bw_filter_strength = params.get("bw_filter_strength", 1.0)
        chroma_nr = params.get("chroma_nr", 0.0)
        band_nr = params.get("band_nr", 0.0)
        dehaze = params.get("dehaze", 0.0)
        flat_field = params.get("flat_field", None)
        flat_field_strength = params.get("flat_field_strength", 0.0)
        grain = params.get("grain", 0.0)
        toning = params.get("toning", "none")

        if chroma_nr > 0.01:
            pipe.add_frontend(CromaNRFilter(strength=chroma_nr))
        if band_nr > 0.01:
            pipe.add_frontend(BandNRFilter(strength=band_nr))
        if flat_field:
            pipe.add_frontend(FlatFieldFilter(flat_frame_path=flat_field, strength=1.0))
        elif flat_field_strength > 0.01:
            pipe.add_frontend(FlatFieldFilter(strength=flat_field_strength))
        if bw_filter != "none":
            pipe.add_frontend(BWFilterSim(filter_type=bw_filter, strength=bw_filter_strength))

        if dehaze > 0.01:
            pipe.add_backend(DehazeFilter(strength=dehaze))
        if grain > 0.01:
            pipe.add_backend(GrainFilter(strength=grain))
        if toning != "none":
            pipe.add_backend(ToningFilter(preset=toning, strength=0.7))

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

        self._enabled_cb = QCheckBox("启用 — 滤镜处理 (Filter Pipeline)")
        self._enabled_cb.setChecked(True)
        self._enabled_cb.toggled.connect(lambda: self._notify())
        lay.addWidget(self._enabled_cb)

        # ── section card helpers ──
        _cur_card = [None]  # mutable capture: [card_frame, card_layout]

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

        # ── B&W滤镜 ──
        _open_card("黑白滤镜模拟 (B&W Filter)", "模拟彩色滤镜在黑白摄影中的效果")
        _add_combo("bw_filter", "滤镜类型:", ["none", "red", "orange", "yellow", "green", "blue"], "none")
        _add_slider("bw_filter_strength", "滤镜强度", 0.0, 1.0, 1.0)

        # ── 胶片颗粒 ──
        _open_card("胶片颗粒 (Grain)", "模拟高感光度胶片噪点")
        _add_slider("grain", "强度", 0.0, 1.0, 0.0)

        # ── 色调映射 ──
        _open_card("色调映射 (Toning)", "Sepia / Selenium / Cyanotype")
        _add_combo("toning", "预设:", ["none", "sepia", "selenium", "cyanotype", "platinum", "copper", "cool"], "none")

        # ── 降噪 ──
        _open_card("色度降噪 (CromaNR)", "仅对颜色通道降噪，保护亮度细节")
        _add_slider("chroma_nr", "强度", 0.0, 1.0, 0.0)
        _open_card("频带降噪 (BandNR)", "金字塔分解，多尺度独立降噪")
        _add_slider("band_nr", "强度", 0.0, 1.0, 0.0)

        # ── 去雾 ──
        _open_card("去雾 (Dehaze)", "暗通道先验去雾")
        _add_slider("dehaze", "强度", 0.0, 1.0, 0.0)

        # ── 平场校正 ──
        _open_card("平场校正 (FlatField)", "消除翻拍照明不均匀。\n强度0=禁用  >0=自动估计照明场")
        _add_slider("flat_field_strength", "强度", 0.0, 1.0, 0.0)

        _close_card()
        lay.addStretch()

    def get_params(self) -> dict:
        return {k: (v.get() if hasattr(v, 'get') else v)
                for k, v in self._ui_vars.items()}

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--chroma-nr", type=float, default=0.0, help="色度降噪 (CromaNR) 0.0-1.0")
        parser.add_argument("--band-nr", type=float, default=0.0, help="频带降噪 (BandNR) 0.0-1.0")
        parser.add_argument("--dehaze", type=float, default=0.0, help="去雾 (Dehaze) 0.0-1.0")
        parser.add_argument("--flat-field", type=str, default=None, help="平场校正 (FlatField) 白帧参考图路径")
        parser.add_argument("--bw-filter", choices=["none", "red", "orange", "yellow", "green", "blue"],
                            default="none", help="黑白滤镜模拟 (B&W Filter)")
        parser.add_argument("--bw-filter-strength", type=float, default=1.0, help="B&W滤镜强度 0.0-1.0")
        parser.add_argument("--grain", type=float, default=0.0, help="胶片颗粒 (Grain) 强度 0.0-1.0")
        parser.add_argument("--toning", choices=["none", "sepia", "selenium", "cyanotype", "platinum", "copper", "cool"],
                            default="none", help="色调映射 (Toning) 预设")

    def get_cli_kwargs(self, args) -> dict:
        return {
            "chroma_nr": args.chroma_nr, "band_nr": args.band_nr,
            "dehaze": args.dehaze, "flat_field": args.flat_field,
            "bw_filter": args.bw_filter,
            "bw_filter_strength": args.bw_filter_strength,
            "grain": args.grain,
            "toning": args.toning,
        }

    def _notify(self):
        if self._on_change:
            self._on_change()
