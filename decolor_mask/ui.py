"""GUI for decolor-mask: cross-process color cast correction with real-time preview.

Run: python -m decolor_mask.ui
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from decolor_mask.core import (
    load_image,
    save_image,
    correct_cross_process,
    estimate_white_balance,
)

MAX_PREVIEW_SIZE = 700


class _SliderRow:
    """A labeled slider + entry row bound to a DoubleVar."""

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
        self.entry = ttk.Entry(frame, textvariable=self.var, width=8,
                               validate="key", validatecommand=vcmd)
        self.entry.pack(side=tk.LEFT)
        self.entry.bind("<Return>", self._on_entry_return)
        self.entry.bind("<FocusOut>", self._on_entry_return)
        self.var.trace_add("write", self._on_var_write)
        self.frame = frame

    def _validate(self, value):
        if value == "" or value == "-":
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
        self._schedule_update()

    def _on_var_write(self, *args):
        pass

    def _on_entry_return(self, event=None):
        self._schedule_update()

    def _schedule_update(self):
        if self.command is None:
            return
        if self._after_id is not None:
            self.frame.after_cancel(self._after_id)
        self._after_id = self.frame.after(250, self.command)

    def pack(self, **kw):
        self.frame.pack(**kw)

    def configure_state(self, state):
        self.scale.configure(state=state)
        self.entry.configure(state=state)


class CrossProcessApp:
    """Main GUI application window."""

    def __init__(self, root):
        self.root = root
        root.title("正负逆冲 - Cross Process Correction")
        root.geometry("1280x800")
        root.minsize(900, 600)

        self.full_image = None
        self._preview_job = None
        self._updating = False

        self._build_toolbar()
        self._build_image_area()
        self._build_controls()
        self._update_manual_visibility()

        root.bind("<Control-o>", lambda e: self._open_image())
        root.bind("<Control-s>", lambda e: self._save_image())

    # ── toolbar ──────────────────────────────────────────────

    def _build_toolbar(self):
        bar = ttk.Frame(self.root)
        bar.pack(fill=tk.X, padx=8, pady=(8, 4))

        ttk.Button(bar, text="打开图像 (Ctrl+O)", command=self._open_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="保存结果 (Ctrl+S)", command=self._save_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="重置参数", command=self._reset_params).pack(side=tk.LEFT, padx=2)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=2)

        ttk.Label(bar, text="检测:").pack(side=tk.LEFT, padx=(0, 4))
        self.detect_btn = ttk.Button(
            bar, text="自动检测白平衡", command=self._auto_detect, state=tk.DISABLED,
        )
        self.detect_btn.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="就绪 — 请打开一张图像")
        ttk.Label(bar, textvariable=self.status_var).pack(side=tk.RIGHT, padx=8)

    # ── image area ───────────────────────────────────────────

    def _build_image_area(self):
        pane = ttk.Frame(self.root)
        pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        pane.columnconfigure(0, weight=1)
        pane.columnconfigure(1, weight=1)
        pane.rowconfigure(0, weight=1)

        # original
        left = ttk.LabelFrame(pane, text="原始图像")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.orig_canvas = tk.Canvas(left, bg="#2b2b2b", highlightthickness=0)
        self.orig_canvas.grid(row=0, column=0, sticky="nsew")
        self._orig_tk = None

        # preview
        right = ttk.LabelFrame(pane, text="预览 (实时)")
        right.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        self.prev_canvas = tk.Canvas(right, bg="#2b2b2b", highlightthickness=0)
        self.prev_canvas.grid(row=0, column=0, sticky="nsew")
        self._prev_tk = None

        self._bind_canvas_resize(self.orig_canvas)
        self._bind_canvas_resize(self.prev_canvas)
        self.root.after(100, self._show_placeholder)

    def _bind_canvas_resize(self, canvas):
        def _on_resize(event):
            if self.full_image is not None and not self._updating:
                self.root.after_cancel(self._preview_job) if self._preview_job else None
                self._preview_job = self.root.after(200, self._update_display)
        canvas.bind("<Configure>", _on_resize)

    def _show_placeholder(self):
        if self.full_image is None:
            w = self.orig_canvas.winfo_width()
            h = self.orig_canvas.winfo_height()
            if w > 1 and h > 1:
                self._draw_placeholder(self.orig_canvas, "拖放图像到此\n或 Ctrl+O 打开")
                self._draw_placeholder(self.prev_canvas, "处理后预览")
            else:
                self.root.after(200, self._show_placeholder)

    def _draw_placeholder(self, canvas, text):
        canvas.delete("all")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w > 1 and h > 1:
            canvas.create_text(w // 2, h // 2, text=text, fill="#888888",
                               font=("Microsoft YaHei", 13), justify=tk.CENTER)

    # ── controls ─────────────────────────────────────────────

    def _build_controls(self):
        ctrl = ttk.LabelFrame(self.root, text="参数控制", padding=8)
        ctrl.pack(fill=tk.X, padx=8, pady=(4, 8))

        row0 = ttk.Frame(ctrl)
        row0.pack(fill=tk.X, pady=1)
        ttk.Label(row0, text="白平衡方法:", width=14, anchor="e").pack(side=tk.LEFT, padx=(0, 4))
        self.method_var = tk.StringVar(value="gray_world")
        self.method_combo = ttk.Combobox(
            row0, textvariable=self.method_var, state="readonly", width=14,
            values=["gray_world", "white_patch", "percentile", "manual"],
        )
        self.method_combo.pack(side=tk.LEFT)
        self.method_combo.bind("<<ComboboxSelected>>", self._on_method_change)

        self.percentile_row = _SliderRow(ctrl, "  百分位 %", 1, 100, 95, command=self._update_preview, resolution=1)
        self.percentile_row.pack(fill=tk.X, pady=1)

        self.strength_row = _SliderRow(ctrl, "纠正强度", 0.0, 1.0, 0.6, command=self._update_preview)
        self.strength_row.pack(fill=tk.X, pady=1)

        manual_frame = ttk.Frame(ctrl)
        self.manual_frame = manual_frame
        self.white_r_row = _SliderRow(manual_frame, "  手动白点 R", 0.0, 1.0, 0.85, command=self._update_preview)
        self.white_r_row.pack(fill=tk.X, pady=1)
        self.white_g_row = _SliderRow(manual_frame, "  手动白点 G", 0.0, 1.0, 0.55, command=self._update_preview)
        self.white_g_row.pack(fill=tk.X, pady=1)
        self.white_b_row = _SliderRow(manual_frame, "  手动白点 B", 0.0, 1.0, 0.28, command=self._update_preview)
        self.white_b_row.pack(fill=tk.X, pady=1)

        self.brightness_row = _SliderRow(ctrl, "亮度", 0.1, 3.0, 1.0, command=self._update_preview)
        self.brightness_row.pack(fill=tk.X, pady=1)

        self.contrast_row = _SliderRow(ctrl, "对比度", 0.1, 3.0, 1.0, command=self._update_preview)
        self.contrast_row.pack(fill=tk.X, pady=1)

        self.saturation_row = _SliderRow(ctrl, "饱和度", 0.0, 3.0, 1.0, command=self._update_preview)
        self.saturation_row.pack(fill=tk.X, pady=1)

    # ── image loading ────────────────────────────────────────

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="选择图像",
            filetypes=[
                ("图像文件", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self.full_image = load_image(path)
            self._orig_path = path
            self.status_var.set(f"已加载: {os.path.basename(path)}  "
                                f"({self.full_image.shape[1]}×{self.full_image.shape[0]})")
            self.detect_btn.configure(state=tk.NORMAL)
            self._update_display()
        except Exception as e:
            messagebox.showerror("打开失败", str(e))

    def _save_image(self):
        if self.full_image is None:
            return
        path = filedialog.asksaveasfilename(
            title="保存结果",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("TIFF", "*.tif *.tiff"),
                ("BMP", "*.bmp"),
            ],
        )
        if not path:
            return
        try:
            result = self._compute_correction()
            save_image(result, path)
            self.status_var.set(f"已保存: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    # ── computation ──────────────────────────────────────────

    def _get_params(self):
        m = self.method_var.get()
        if m == "manual":
            wr = self.white_r_row.var.get()
            wg = self.white_g_row.var.get()
            wb = self.white_b_row.var.get()
        else:
            wr = wg = wb = None
        return dict(
            method=m,
            percentile=self.percentile_row.var.get(),
            white_r=wr,
            white_g=wg,
            white_b=wb,
            strength=self.strength_row.var.get(),
            brightness=self.brightness_row.var.get(),
            contrast=self.contrast_row.var.get(),
            saturation=self.saturation_row.var.get(),
        )

    def _compute_correction(self):
        return correct_cross_process(self.full_image, **self._get_params())

    def _update_preview(self, *_):
        if self.full_image is None:
            return
        self._updating = True
        try:
            params = self._get_params()
            result = correct_cross_process(self.full_image, **params)
            self._render_to_canvas(self.prev_canvas, result, "_prev_tk")
            self.status_var.set(f"预览已更新  |  "
                                f"方法: {params['method']}  |  强度: {params['strength']:.2f}")
        finally:
            self._updating = False

    def _update_display(self, *_):
        if self.full_image is None:
            return
        self._updating = True
        try:
            self._render_to_canvas(self.orig_canvas, self.full_image, "_orig_tk")
            result = correct_cross_process(self.full_image, **self._get_params())
            self._render_to_canvas(self.prev_canvas, result, "_prev_tk")
        finally:
            self._updating = False

    # ── display helpers ──────────────────────────────────────

    def _render_to_canvas(self, canvas, arr, tk_attr):
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        h, w = arr.shape[:2]
        scale = min(cw / w, ch / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)

        if new_w > 0 and new_h > 0:
            img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(img)
            setattr(self, tk_attr, tk_img)

            canvas.delete("all")
            canvas.create_image(cw // 2, ch // 2, image=tk_img, anchor=tk.CENTER)

    # ── actions ──────────────────────────────────────────────

    def _on_method_change(self, event=None):
        self._update_manual_visibility()
        self._update_preview()

    def _update_manual_visibility(self):
        is_manual = self.method_var.get() == "manual"
        is_percentile = self.method_var.get() == "percentile"
        if is_manual:
            self.manual_frame.pack(fill=tk.X, pady=1, before=self.brightness_row.frame)
        else:
            self.manual_frame.pack_forget()
        if is_percentile:
            self.percentile_row.pack(fill=tk.X, pady=1, before=self.strength_row.frame)
        else:
            self.percentile_row.pack_forget()

    def _auto_detect(self):
        if self.full_image is None:
            return
        method = self.method_var.get()
        if method == "manual":
            messagebox.showinfo("提示", "请先切换到自动方法（gray_world / white_patch / percentile）再检测。")
            return
        percentile = self.percentile_row.var.get()
        gains = estimate_white_balance(self.full_image, method=method, percentile=percentile)
        ref = np.clip(1.0 / (gains + 1e-8), 0, 1)
        self.white_r_row.var.set(round(float(ref[0]), 4))
        self.white_g_row.var.set(round(float(ref[1]), 4))
        self.white_b_row.var.set(round(float(ref[2]), 4))
        self.status_var.set(
            f"检测完成: 增益 R={gains[0]:.4f} G={gains[1]:.4f} B={gains[2]:.4f}"
            f"  |  等效白点 RGB=({ref[0]:.3f}, {ref[1]:.3f}, {ref[2]:.3f})"
        )

    def _reset_params(self):
        self.method_var.set("gray_world")
        self.strength_row.var.set(0.6)
        self.brightness_row.var.set(1.0)
        self.contrast_row.var.set(1.0)
        self.saturation_row.var.set(1.0)
        self.percentile_row.var.set(95)
        self.white_r_row.var.set(0.85)
        self.white_g_row.var.set(0.55)
        self.white_b_row.var.set(0.28)
        self._update_manual_visibility()
        self._update_preview()


def main():
    root = tk.Tk()
    app = CrossProcessApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
