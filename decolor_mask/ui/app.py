"""通用图像处理工具 — 主应用程序窗口。

标签页式界面，每个标签页对应一个功能插件。
"""

import logging
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

try:
    import rawpy
    _HAS_RAWPY = True
except ImportError:
    _HAS_RAWPY = False

from ..core import find_raw_files, process_raw, _apply_postprocess, RAW_EXTENSIONS, _render_raw
from ..plugins import list_all

logger = logging.getLogger(__name__)

MAX_PREVIEW = 640


class ImageToolkitApp:
    """通用图像处理工具主窗口。"""

    def __init__(self, root):
        self.root = root
        root.title("ISL Toolkit — 图像处理工具箱")
        root.geometry("1320x820")
        root.minsize(1024, 640)

        self.raw_files: list[str] = []
        self._raw_obj = None
        self._preview_path = None
        self._cam_render = None
        self._tgt_render = None
        self._processing = False
        self._plugins = list_all()
        self._current_plugin = self._plugins[0] if self._plugins else None

        self._build_toolbar()
        self._build_main_area()
        self._build_status()

        self._show_placeholders()

        root.bind("<Control-o>", lambda e: self._browse_folder())
        root.bind("<Control-s>", lambda e: self._process_current())
        root.bind("<Control-b>", lambda e: self._process_batch())

    # ── toolbar ──────────────────────────────────────────────

    def _build_toolbar(self):
        bar = ttk.Frame(self.root)
        bar.pack(fill=tk.X, padx=8, pady=(8, 4))

        ttk.Label(bar, text="文件夹:").pack(side=tk.LEFT)
        self.folder_var = tk.StringVar(value="")
        ttk.Entry(bar, textvariable=self.folder_var, width=30).pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)

        ttk.Button(bar, text="浏览", command=self._browse_folder).pack(side=tk.LEFT, padx=1)
        ttk.Button(bar, text="打开文件", command=self._open_raw_file).pack(side=tk.LEFT, padx=1)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        self.save_btn = ttk.Button(bar, text="处理当前 (Ctrl+S)", command=self._process_current, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=1)

        self.batch_btn = ttk.Button(bar, text="批量处理 (Ctrl+B)", command=self._process_batch, state=tk.DISABLED)
        self.batch_btn.pack(side=tk.LEFT, padx=1)

    def _build_status(self):
        self.status_var = tk.StringVar(value="就绪 — 打开 RAW 文件或选择文件夹")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W,
                  padding=(8, 2)).pack(fill=tk.X, padx=8, pady=(0, 4))

    # ── main area (tabs) ─────────────────────────────────────

    def _build_main_area(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        for plugin in self._plugins:
            tab = ttk.Frame(self.notebook)
            self.notebook.add(tab, text=f"  {plugin.icon} {plugin.title}  ")

            # Left: preview
            preview_frame = ttk.LabelFrame(tab, text="预览", padding=4)
            preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

            canvas = tk.Canvas(preview_frame, bg="#1e1e1e", highlightthickness=0)
            canvas.pack(fill=tk.BOTH, expand=True)
            plugin._canvas = canvas
            plugin._tkimg = None

            # Right: controls
            ctrl_frame = ttk.LabelFrame(tab, text="参数", padding=8, width=300)
            ctrl_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 0))
            ctrl_frame.pack_propagate(False)

            # Scrollable controls
            canvas_ctrl = tk.Canvas(ctrl_frame, highlightthickness=0, width=280)
            scrollbar = ttk.Scrollbar(ctrl_frame, orient=tk.VERTICAL, command=canvas_ctrl.yview)
            scrollable_frame = ttk.Frame(canvas_ctrl)

            scrollable_frame.bind("<Configure>",
                                  lambda e: canvas_ctrl.configure(scrollregion=canvas_ctrl.bbox("all")))
            canvas_ctrl.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas_ctrl.configure(yscrollcommand=scrollbar.set)

            canvas_ctrl.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            # Mousewheel scrolling
            def _on_mousewheel(event, c=canvas_ctrl):
                c.yview_scroll(int(-1 * (event.delta / 120)), "units")

            canvas_ctrl.bind_all("<MouseWheel>", _on_mousewheel)

            plugin.attach_ui(scrollable_frame, on_change=lambda p=plugin: self._on_plugin_param_change(p))

        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_change)

    def _on_tab_change(self, event=None):
        idx = self.notebook.index("current")
        if 0 <= idx < len(self._plugins):
            self._current_plugin = self._plugins[idx]
            self._update_plugin_preview()

    def _on_plugin_param_change(self, plugin=None):
        if plugin is None:
            plugin = self._current_plugin
        self._update_plugin_preview()

    # ── preview ──────────────────────────────────────────────

    def _show_placeholders(self):
        for plugin in self._plugins:
            if hasattr(plugin, '_canvas'):
                c = plugin._canvas
                w = c.winfo_width()
                h = c.winfo_height()
                if w > 1 and h > 1:
                    c.delete("all")
                    c.create_text(w // 2, h // 2, text=f"{plugin.icon} {plugin.title}\n\n打开RAW文件开始",
                                  fill="#777", font=("Microsoft YaHei", 12), justify=tk.CENTER)
        self.root.after(500, self._check_canvas_size)

    def _check_canvas_size(self):
        all_ok = True
        for plugin in self._plugins:
            if hasattr(plugin, '_canvas'):
                if plugin._canvas.winfo_width() < 10:
                    all_ok = False
        if not all_ok:
            self.root.after(200, self._show_placeholders)

    def _update_plugin_preview(self):
        if self._cam_render is None:
            return
        plugin = self._current_plugin
        if plugin is None:
            return
        canvas = getattr(plugin, '_canvas', None)
        if canvas is None:
            return

        params = plugin.get_params()
        if not params:
            return

        try:
            if plugin.name == "film":
                result = self._preview_film(params)
            elif plugin.name == "enhance":
                result = self._preview_enhance(params)
            elif plugin.name == "denoise":
                result = self._preview_denoise(params)
            else:
                result = self._cam_render.astype(np.float32) / 255.0
        except Exception as e:
            logger.debug("Preview error: %s", e)
            return

        self._render_to_canvas(canvas, result, plugin)

    def _preview_film(self, params):
        from ..kernel.filters.color import LCCInverter, DehazeFilter, GrayAxisFilmBase
        from ..kernel.filters.noise import CromaNRFilter, BandNRFilter

        rgb = self._tgt_render.astype(np.float32) / 255.0

        chroma_nr = params.get("chroma_nr", 0)
        band_nr = params.get("band_nr", 0)
        dehaze = params.get("dehaze", 0)
        strength = params.get("strength", 0.8)

        if chroma_nr > 0.01:
            rgb = CromaNRFilter(strength=chroma_nr).apply(rgb, preview=True)
        if band_nr > 0.01:
            rgb = BandNRFilter(strength=band_nr).apply(rgb, preview=True)

        rgb = LCCInverter(strength=strength).apply(rgb, preview=True)

        if dehaze > 0.01:
            rgb = DehazeFilter(strength=dehaze).apply(rgb, preview=True)

        rgb = _apply_postprocess(rgb,
                                 params.get("brightness", 1.0),
                                 params.get("contrast", 1.0),
                                 params.get("saturation", 1.0))
        return rgb

    def _preview_enhance(self, params):
        from ..kernel.filters.color import DehazeFilter

        rgb = self._tgt_render.astype(np.float32) / 255.0

        dehaze = params.get("dehaze", 0)
        if dehaze > 0.01:
            rgb = DehazeFilter(strength=dehaze).apply(rgb, preview=True)

        rgb = _apply_postprocess(rgb,
                                 params.get("brightness", 1.0),
                                 params.get("contrast", 1.0),
                                 params.get("saturation", 1.0))
        return rgb

    def _preview_denoise(self, params):
        from ..kernel.filters.noise import CromaNRFilter, BandNRFilter

        rgb = self._tgt_render.astype(np.float32) / 255.0

        chroma_nr = params.get("chroma_nr", 0)
        band_nr = params.get("band_nr", 0)

        if chroma_nr > 0.01:
            rgb = CromaNRFilter(strength=chroma_nr).apply(rgb, preview=True)
        if band_nr > 0.01:
            rgb = BandNRFilter(strength=band_nr).apply(rgb, preview=True)

        return rgb

    def _render_to_canvas(self, canvas, arr, plugin):
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
            plugin._tkimg = tk_img
            canvas.delete("all")
            canvas.create_image(cw // 2, ch // 2, image=tk_img, anchor=tk.CENTER)

    # ── file loading ─────────────────────────────────────────

    def _browse_folder(self):
        folder = filedialog.askdirectory(title="选择包含图像文件的文件夹")
        if folder:
            self.folder_var.set(folder)
            self._scan_folder()

    def _scan_folder(self):
        folder = self.folder_var.get()
        if not folder:
            return
        raw_files = find_raw_files(folder)
        if not raw_files:
            self.status_var.set(f"未找到RAW文件: {folder}")
            return
        self.raw_files = raw_files
        self._load_preview(raw_files[0])

    def _open_raw_file(self):
        exts = [f"*.{e[1:]}" for e in sorted(RAW_EXTENSIONS)]
        path = filedialog.askopenfilename(
            title="打开图像文件",
            filetypes=[("RAW文件", " ".join(exts)), ("所有文件", "*.*")],
        )
        if not path:
            return
        folder = os.path.dirname(path)
        self.folder_var.set(folder)
        self.raw_files = find_raw_files(folder)
        self._load_preview(path)

    def _load_preview(self, path):
        if not _HAS_RAWPY:
            self.status_var.set("需要 rawpy: pip install rawpy")
            return

        self.status_var.set("加载中...")
        self.root.update()

        try:
            self._preview_path = path
            raw = rawpy.imread(path)
            self._raw_obj = raw

            pp = dict(output_color=rawpy.ColorSpace.sRGB, gamma=(2.222, 4.5),
                      no_auto_bright=True, bright=1.0, half_size=True)

            self._cam_render = raw.postprocess(use_camera_wb=True, **pp)
            self._tgt_render = raw.postprocess(use_camera_wb=False, use_auto_wb=True, **pp)

            self._update_plugin_preview()

            self.save_btn.configure(state=tk.NORMAL)
            if self.raw_files:
                self.batch_btn.configure(state=tk.NORMAL, text=f"批量处理 ({len(self.raw_files)} 个)")

            self.status_var.set(f"预览: {os.path.basename(path)}  |  共 {len(self.raw_files)} 个文件")
        except Exception as e:
            self.status_var.set(f"加载失败: {e}")

    # ── processing ───────────────────────────────────────────

    def _process_current(self):
        if self._preview_path is None:
            messagebox.showinfo("提示", "请先打开文件")
            return

        import datetime
        name = os.path.splitext(os.path.basename(self._preview_path))[0]
        folder = os.path.dirname(self._preview_path)
        ts = datetime.datetime.now().strftime("%H%M%S")
        output_path = os.path.join(folder, f"{name}_{ts}.png")

        plugin = self._current_plugin
        params = plugin.get_params() if plugin else {}

        self._processing = True
        self.save_btn.configure(state=tk.DISABLED, text="处理中...")

        def _worker():
            try:
                process_raw(self._preview_path, output_path,
                            use_pipeline=(plugin.name == "film"),
                            **{k: v for k, v in params.items()
                               if k in ("wb_mode", "strength", "brightness", "contrast",
                                        "saturation", "chroma_nr", "band_nr", "dehaze")})
                self.root.after(0, self._on_single_done, output_path, "")
            except Exception as e:
                self.root.after(0, self._on_single_done, "", str(e))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_single_done(self, output_path, error):
        self._processing = False
        self.save_btn.configure(state=tk.NORMAL, text="处理当前 (Ctrl+S)")
        if error:
            messagebox.showerror("错误", error)
        else:
            self.status_var.set(f"已保存: {os.path.basename(output_path)}")
            messagebox.showinfo("完成", f"已保存: {output_path}")

    def _process_batch(self):
        if not self.raw_files:
            messagebox.showinfo("提示", "请先打开文件")
            return

        folder = self.folder_var.get()
        output_dir = os.path.join(folder, "corrected")
        os.makedirs(output_dir, exist_ok=True)

        plugin = self._current_plugin
        params = plugin.get_params() if plugin else {}
        total = len(self.raw_files)

        self._processing = True
        self.save_btn.configure(state=tk.DISABLED)
        self.batch_btn.configure(state=tk.DISABLED, text="处理中...")

        class _State:
            done = 0
            failed = 0
        state = _State()

        def _worker():
            for i, path in enumerate(self.raw_files):
                name = os.path.splitext(os.path.basename(path))[0]
                out = os.path.join(output_dir, f"{name}.png")
                try:
                    process_raw(path, out,
                                use_pipeline=(plugin.name == "film"),
                                **{k: v for k, v in params.items()
                                   if k in ("wb_mode", "strength", "brightness", "contrast",
                                            "saturation", "chroma_nr", "band_nr", "dehaze")})
                    state.done += 1
                except Exception as e:
                    state.failed += 1
                    logger.error("Failed: %s - %s", path, e)
                self.root.after(0, lambda c=i+1: self.batch_btn.configure(text=f"处理中... {c}/{total}"))
            self.root.after(0, self._on_batch_done, state.done, state.failed, output_dir)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_batch_done(self, done, failed, output_dir):
        self._processing = False
        self.save_btn.configure(state=tk.NORMAL, text="处理当前 (Ctrl+S)")
        self.batch_btn.configure(state=tk.NORMAL, text=f"批量处理 ({len(self.raw_files)} 个)")
        self.status_var.set(f"完成! 成功 {done}, 失败 {failed}")
        if failed == 0:
            messagebox.showinfo("完成", f"成功处理 {done} 个文件\n输出: {output_dir}")
        else:
            messagebox.showwarning("完成", f"成功 {done}, 失败 {failed}")
