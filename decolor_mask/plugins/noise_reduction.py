"""降噪 — CromaNR 色度降噪 + BandNR 频带降噪功能插件。"""

import argparse
import tkinter as tk
from tkinter import ttk

from .base import PluginBase
from ..kernel import ProcessingPipeline, CromaNRFilter, BandNRFilter


class NoiseReductionPlugin(PluginBase):
    """降噪插件：色度降噪 + 频带分解降噪。"""

    name = "denoise"
    title = "降噪处理"
    icon = "🔇"

    def __init__(self):
        self._ui_vars = {}
        self._on_change = None

    def build_pipeline(self, **params) -> ProcessingPipeline:
        pipe = ProcessingPipeline(preview_scale=0.25)

        chroma_nr = params.get("chroma_nr", 0.0)
        band_nr = params.get("band_nr", 0.0)

        if chroma_nr > 0.01:
            pipe.add_frontend(CromaNRFilter(strength=chroma_nr))
        if band_nr > 0.01:
            pipe.add_frontend(BandNRFilter(strength=band_nr))

        return pipe

    def attach_ui(self, parent: ttk.Frame, on_change=None) -> None:
        self._on_change = on_change

        from ..ui import _SliderRow

        ttk.Label(parent, text="色度降噪 (CromaNR)", font=("", 9, "bold")).pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(parent, text="仅对颜色通道降噪，保护亮度细节").pack(anchor=tk.W)
        row = _SliderRow(parent, "强度", 0.0, 1.0, 0.0,
                         command=lambda: self._notify(), resolution=0.01)
        row.pack(fill=tk.X, pady=1)
        self._ui_vars["chroma_nr"] = row.var

        sep = ttk.Separator(parent, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, pady=4)

        ttk.Label(parent, text="频带降噪 (BandNR)", font=("", 9, "bold")).pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(parent, text="金字塔分解，多尺度独立降噪").pack(anchor=tk.W)
        row = _SliderRow(parent, "强度", 0.0, 1.0, 0.0,
                         command=lambda: self._notify(), resolution=0.01)
        row.pack(fill=tk.X, pady=1)
        self._ui_vars["band_nr"] = row.var

    def get_params(self) -> dict:
        return {k: (v.get() if hasattr(v, 'get') else v)
                for k, v in self._ui_vars.items()}

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--chroma-nr", type=float, default=0.0,
                            help="色度降噪 0.0-1.0")
        parser.add_argument("--band-nr", type=float, default=0.0,
                            help="频带降噪 0.0-1.0")

    def get_cli_kwargs(self, args) -> dict:
        return {
            "chroma_nr": args.chroma_nr,
            "band_nr": args.band_nr,
        }

    def _notify(self):
        if self._on_change:
            self._on_change()
