"""正负逆冲 — 负片反转功能插件。

从 DCU/ISL 引擎学习改进了核心反转算法：
- LAB 空间灰度轴片基检测
- LCC 空间亮度-色度分离反转
"""

import argparse
import tkinter as tk
from tkinter import ttk

from .base import PluginBase
from ..kernel import (
    ProcessingPipeline,
    GrayAxisFilmBase,
    LCCInverter,
    CromaNRFilter,
    BandNRFilter,
    DehazeFilter,
    FlatFieldFilter,
)
from ..core import _render_raw, _apply_postprocess


class FilmInversionPlugin(PluginBase):
    """负片反转插件 — 原 正负逆冲 核心功能。"""

    name = "film"
    title = "正负逆冲 (负片反转)"
    icon = "🎞"

    def __init__(self):
        self._ui_vars = {}
        self._on_change = None

    def build_pipeline(self, **params) -> ProcessingPipeline:
        """构建负片反转处理管线。"""
        pipe = ProcessingPipeline(preview_scale=0.25)

        chroma_nr = params.get("chroma_nr", 0.0)
        band_nr = params.get("band_nr", 0.0)
        flat_field = params.get("flat_field", None)
        dehaze = params.get("dehaze", 0.0)
        strength = params.get("strength", 0.8)

        if chroma_nr > 0.01:
            pipe.add_frontend(CromaNRFilter(strength=chroma_nr))
        if band_nr > 0.01:
            pipe.add_frontend(BandNRFilter(strength=band_nr))
        if flat_field:
            pipe.add_frontend(FlatFieldFilter(flat_frame_path=flat_field, strength=1.0))
        else:
            pipe.add_frontend(FlatFieldFilter(strength=0.5))

        pipe.add_core(LCCInverter(strength=strength))

        if dehaze > 0.01:
            pipe.add_backend(DehazeFilter(strength=dehaze))

        return pipe

    def attach_ui(self, parent: ttk.Frame, on_change=None) -> None:
        """附加负片反转 UI 面板。"""
        self._on_change = on_change

        # 白平衡
        row0 = ttk.Frame(parent)
        row0.pack(fill=tk.X, pady=1)
        ttk.Label(row0, text="目标白平衡:", width=14, anchor="e").pack(side=tk.LEFT, padx=(0, 4))
        self._ui_vars["wb_mode"] = tk.StringVar(value="auto")
        wb = ttk.Combobox(row0, textvariable=self._ui_vars["wb_mode"], state="readonly",
                          width=12, values=["auto", "camera", "daylight"])
        wb.pack(side=tk.LEFT)
        wb.bind("<<ComboboxSelected>>", lambda e: self._notify())
        ttk.Label(row0, text="  auto=自动  camera=原厂  daylight=日光").pack(side=tk.LEFT, padx=8)

        # 参数滑块
        from ..ui import _SliderRow
        for key, label, from_, to, default in [
            ("strength", "反转强度", 0.0, 1.0, 0.8),
            ("brightness", "亮度", 0.1, 3.0, 1.0),
            ("contrast", "对比度", 0.1, 3.0, 1.0),
            ("saturation", "饱和度", 0.0, 3.0, 1.0),
        ]:
            row = _SliderRow(parent, label, from_, to, default,
                             command=lambda: self._notify(), resolution=0.01)
            row.pack(fill=tk.X, pady=1)
            self._ui_vars[key] = row.var

        # 滤镜开关
        sep = ttk.Separator(parent, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, pady=4)

        ttk.Label(parent, text="滤镜管线 (可选)").pack(anchor=tk.W)

        for key, label, default in [
            ("chroma_nr", "色度降噪", 0.0),
            ("band_nr", "频带降噪", 0.0),
            ("dehaze", "去雾", 0.0),
        ]:
            row = _SliderRow(parent, label, 0.0, 1.0, default,
                             command=lambda: self._notify(), resolution=0.01)
            row.pack(fill=tk.X, pady=1)
            self._ui_vars[key] = row.var

    def get_params(self) -> dict:
        return {k: (v.get() if hasattr(v, 'get') else v)
                for k, v in self._ui_vars.items()}

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--wb", choices=["auto", "camera", "daylight"], default="auto",
                            help="目标白平衡模式. 默认: auto")
        parser.add_argument("--strength", "-s", type=float, default=0.8,
                            help="反转强度 0.0-1.0")
        parser.add_argument("--brightness", "-b", type=float, default=1.0,
                            help="亮度倍率")
        parser.add_argument("--contrast", "-c", type=float, default=1.0,
                            help="对比度倍率")
        parser.add_argument("--saturation", type=float, default=1.0,
                            help="饱和度倍率")
        parser.add_argument("--chroma-nr", type=float, default=0.0,
                            help="色度降噪 0.0-1.0")
        parser.add_argument("--band-nr", type=float, default=0.0,
                            help="频带降噪 0.0-1.0")
        parser.add_argument("--dehaze", type=float, default=0.0,
                            help="去雾 0.0-1.0")
        parser.add_argument("--flat-field", type=str, default=None,
                            help="白帧参考图路径")

    def get_cli_kwargs(self, args) -> dict:
        return {
            "wb_mode": args.wb,
            "strength": args.strength,
            "brightness": args.brightness,
            "contrast": args.contrast,
            "saturation": args.saturation,
            "chroma_nr": args.chroma_nr,
            "band_nr": args.band_nr,
            "dehaze": args.dehaze,
            "flat_field": args.flat_field,
        }

    def _notify(self):
        if self._on_change:
            self._on_change()
