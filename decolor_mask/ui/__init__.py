"""共享 UI 组件。"""

import tkinter as tk
from tkinter import ttk


class _SliderRow:
    """带标签的滑块 + 输入框行，绑定到 DoubleVar。"""

    def __init__(self, parent, label, from_, to, default, command=None, resolution=None):
        self.command = command
        self.var = tk.DoubleVar(value=default)
        self._after_id = None
        self._resolution = resolution
        self._suppress = False

        frame = ttk.Frame(parent)
        ttk.Label(frame, text=label, width=14, anchor="e").pack(side=tk.LEFT, padx=(0, 4))
        self.scale = ttk.Scale(frame, from_=from_, to=to, variable=self.var,
                               orient=tk.HORIZONTAL, command=self._on_scale)
        self.scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        vcmd = (frame.register(self._validate), "%P")
        self.entry = ttk.Entry(frame, textvariable=self.var, width=7,
                               validate="key", validatecommand=vcmd)
        self.entry.pack(side=tk.LEFT)
        self.entry.bind("<Return>", self._on_change)
        self.entry.bind("<FocusOut>", self._on_change)
        self.var.trace_add("write", self._on_var_write)
        self.frame = frame

    def _validate(self, value):
        if value in ("", "-", "."):
            return True
        try:
            float(value)
            return True
        except ValueError:
            return False

    def _on_scale(self, val):
        if self._suppress:
            return
        val = float(val)
        if self._resolution:
            val = round(val / self._resolution) * self._resolution
            self._suppress = True
            self.var.set(val)
            self._suppress = False
        self._schedule()

    def _on_var_write(self, *args):
        pass

    def _on_change(self, event=None):
        self._schedule()

    def _schedule(self):
        if self.command is None:
            return
        if self._after_id is not None:
            self.frame.after_cancel(self._after_id)
        self._after_id = self.frame.after(300, self.command)

    def pack(self, **kw):
        self.frame.pack(**kw)
