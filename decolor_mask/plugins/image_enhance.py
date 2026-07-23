"""图像增强 — 色调调整与去雾功能插件。"""

import argparse
import tkinter as tk
from tkinter import ttk

from .base import PluginBase
from ..kernel import ProcessingPipeline, DehazeFilter, LCCInverter


class ImageEnhancePlugin(PluginBase):
    """图像增强插件：亮度/对比度/饱和度/去雾。"""

    name = "enhance"
    title = "图像增强"
    icon = "☀"

    def __init__(self):
        self._ui_vars = {}
        self._on_change = None

    def build_pipeline(self, **params) -> ProcessingPipeline:
        pipe = ProcessingPipeline(preview_scale=0.25)

        dehaze = params.get("dehaze", 0.0)
        sharpen = params.get("sharpen", 0.0)

        if dehaze > 0.01:
            pipe.add_frontend(DehazeFilter(strength=dehaze))

        return pipe

    def attach_ui(self, parent: ttk.Frame, on_change=None) -> None:
        self._on_change = on_change

        from ..ui import _SliderRow
        for key, label, from_, to, default in [
            ("brightness", "亮度", 0.1, 3.0, 1.0),
            ("contrast", "对比度", 0.1, 3.0, 1.0),
            ("saturation", "饱和度", 0.0, 3.0, 1.0),
            ("dehaze", "去雾", 0.0, 1.0, 0.0),
        ]:
            row = _SliderRow(parent, label, from_, to, default,
                             command=lambda: self._notify(), resolution=0.01)
            row.pack(fill=tk.X, pady=1)
            self._ui_vars[key] = row.var

        sep = ttk.Separator(parent, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, pady=4)

        ttk.Label(parent, text="输出格式").pack(anchor=tk.W)
        self._ui_vars["fmt"] = tk.StringVar(value="png")
        ttk.Combobox(parent, textvariable=self._ui_vars["fmt"], state="readonly",
                     width=10, values=["png", "jpg", "tiff"]).pack(anchor=tk.W, pady=2)

    def get_params(self) -> dict:
        return {k: (v.get() if hasattr(v, 'get') else v)
                for k, v in self._ui_vars.items()}

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--brightness", "-b", type=float, default=1.0)
        parser.add_argument("--contrast", "-c", type=float, default=1.0)
        parser.add_argument("--saturation", type=float, default=1.0)
        parser.add_argument("--dehaze", type=float, default=0.0)

    def get_cli_kwargs(self, args) -> dict:
        return {
            "brightness": args.brightness,
            "contrast": args.contrast,
            "saturation": args.saturation,
            "dehaze": args.dehaze,
        }

    def _notify(self):
        if self._on_change:
            self._on_change()
