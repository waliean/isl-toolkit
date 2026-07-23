"""ISL Toolkit 图像处理工具 — 主应用程序窗口 (PySide6)。

链式处理架构：
  - 每个功能模块有"启用"复选框
  - 所有启用模块按顺序链式处理：cross|film → filters → enhance
  - 所有标签页共享同一累积预览
  - cross 与 film 互斥

状态机:
  idle       — 未加载文件
  loading    — 文件加载中
  ready      — 已就绪，可预览/导出
  processing — 正在导出/批量导出
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np
from PIL import Image

try:
    import rawpy
    _HAS_RAWPY = True
except ImportError:
    _HAS_RAWPY = False

from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QToolBar,
    QTabWidget,
    QStatusBar,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QScrollArea,
    QFileDialog,
    QMessageBox,
    QSizePolicy,
    QComboBox,
)

from ..core import find_raw_files, process_raw, _apply_postprocess, RAW_EXTENSIONS
from ..kernel.gpu import get_status_text
from ..plugins import list_all
from ..session import load as session_load, save as session_save
from .widgets import ImageViewer

logger = logging.getLogger(__name__)

# ── 主题管理 ──────────────────────────────────────────────────

_THEME = "dark"  # 模块级当前主题


def _dark_palette():
    """构建暗色主题 QPalette。"""
    from PySide6.QtGui import QPalette, QColor
    p = QPalette()
    p.setColor(QPalette.Window, QColor(45, 45, 45))
    p.setColor(QPalette.WindowText, QColor(220, 220, 220))
    p.setColor(QPalette.Base, QColor(35, 35, 35))
    p.setColor(QPalette.AlternateBase, QColor(50, 50, 50))
    p.setColor(QPalette.ToolTipBase, QColor(30, 30, 30))
    p.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
    p.setColor(QPalette.Text, QColor(220, 220, 220))
    p.setColor(QPalette.Button, QColor(50, 50, 50))
    p.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    p.setColor(QPalette.BrightText, Qt.red)
    p.setColor(QPalette.Link, QColor(42, 130, 218))
    p.setColor(QPalette.Highlight, QColor(42, 130, 218))
    p.setColor(QPalette.HighlightedText, Qt.black)
    p.setColor(QPalette.Disabled, QPalette.WindowText, QColor(110, 110, 110))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(110, 110, 110))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(110, 110, 110))
    return p


def _light_palette():
    """构建亮色主题 QPalette。"""
    from PySide6.QtGui import QPalette, QColor
    p = QPalette()
    p.setColor(QPalette.Window, QColor(240, 240, 240))
    p.setColor(QPalette.WindowText, QColor(40, 40, 40))
    p.setColor(QPalette.Base, QColor(255, 255, 255))
    p.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    p.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    p.setColor(QPalette.ToolTipText, QColor(40, 40, 40))
    p.setColor(QPalette.Text, QColor(40, 40, 40))
    p.setColor(QPalette.Button, QColor(245, 245, 245))
    p.setColor(QPalette.ButtonText, QColor(40, 40, 40))
    p.setColor(QPalette.BrightText, QColor(180, 0, 0))
    p.setColor(QPalette.Link, QColor(26, 109, 196))
    p.setColor(QPalette.Highlight, QColor(26, 109, 196))
    p.setColor(QPalette.HighlightedText, Qt.white)
    p.setColor(QPalette.Disabled, QPalette.WindowText, QColor(160, 160, 160))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(160, 160, 160))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(160, 160, 160))
    return p


# ── QSS 样式表 ────────────────────────────────────────────────

_ACCENT = "#2a82da"

_DARK_QSS = f"""
QMainWindow {{ background-color: #2d2d2d; }}
QToolBar {{
    background-color: #353535; border-bottom: 1px solid #3d3d3d;
    padding: 4px 6px; spacing: 6px;
}}
QPushButton#primaryBtn {{
    background-color: {_ACCENT}; color: #fff; border: none;
    border-radius: 5px; padding: 6px 18px; font-weight: bold; font-size: 12px;
}}
QPushButton#primaryBtn:hover {{ background-color: #3a92ea; }}
QPushButton#primaryBtn:pressed {{ background-color: #1a72ca; }}
QPushButton#primaryBtn:disabled {{ background-color: #444; color: #888; }}
QPushButton#toolBtn {{
    background-color: #3d3d3d; color: #ddd; border: 1px solid #505050;
    border-radius: 5px; padding: 5px 14px; font-size: 12px;
}}
QPushButton#toolBtn:hover {{ background-color: #484848; border-color: #606060; }}
QPushButton#toolBtn:pressed {{ background-color: #333; }}
QPushButton#navBtn {{
    background-color: #373737; color: #ccc; border: 1px solid #484848;
    border-radius: 5px; padding: 5px 12px; font-size: 11px;
}}
QPushButton#navBtn:hover {{ background-color: #424242; border-color: #585858; }}
QPushButton#navBtn:pressed {{ background-color: #303030; }}
QPushButton#navBtn:disabled {{ color: #666; border-color: #3d3d3d; }}
QTabWidget#mainTabs::pane {{
    border: 1px solid #3d3d3d; background-color: #2d2d2d; border-radius: 6px; top: -1px;
}}
QTabWidget#mainTabs QTabBar::tab {{
    background-color: #333; color: #999; border: 1px solid #3d3d3d; border-bottom: none;
    padding: 8px 20px; margin-right: 3px; border-top-left-radius: 6px;
    border-top-right-radius: 6px; min-height: 26px;
}}
QTabWidget#mainTabs QTabBar::tab:selected {{
    background-color: #2d2d2d; color: #fff; border-bottom: 2px solid {_ACCENT};
    font-weight: bold;
}}
QTabWidget#mainTabs QTabBar::tab:hover:!selected {{
    background-color: #3d3d3d; color: #ddd;
}}
QFrame#paramCard {{
    background-color: #333; border: 1px solid #3d3d3d; border-radius: 8px;
}}
QFrame#sectionCard {{
    background-color: #363636; border: 1px solid #424242; border-radius: 6px;
    margin: 2px 0px;
}}
QLabel#sectionHeader {{
    color: #ddd; font-weight: bold; font-size: 12px;
    padding: 2px 0px 4px 0px; border-bottom: 1px solid #444; margin-bottom: 4px;
}}
QLabel#sectionHint {{ color: #888; font-size: 10px; padding: 0px 0px 4px 0px; }}
QLabel#paramTitle {{ color: #ddd; font-weight: bold; font-size: 12px; padding: 4px 0px; }}
QLabel#fileLabel {{
    color: #aaa; padding: 2px 10px; border: 1px solid #555;
    border-radius: 4px; background-color: #333;
}}
QLineEdit {{
    background-color: #404040; color: #ddd; border: 1px solid #505050;
    border-radius: 4px; padding: 5px 10px;
}}
QLineEdit:focus {{ border-color: {_ACCENT}; }}
QComboBox {{
    background-color: #404040; color: #ddd; border: 1px solid #505050;
    border-radius: 4px; padding: 4px 10px; min-width: 80px;
}}
QComboBox:hover {{ border-color: {_ACCENT}; }}
QComboBox QAbstractItemView {{
    background-color: #404040; color: #ddd; selection-background-color: {_ACCENT};
    border: 1px solid #505050;
}}
QDoubleSpinBox {{
    background-color: #404040; color: #ddd; border: 1px solid #505050;
    border-radius: 4px; padding: 4px 6px;
}}
QDoubleSpinBox:focus {{ border-color: {_ACCENT}; }}
QSlider::groove:horizontal {{
    height: 5px; background: #4a4a4a; border-radius: 3px; margin: 0px 2px;
}}
QSlider::handle:horizontal {{
    background: {_ACCENT}; width: 14px; height: 14px; margin: -5px 0px; border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: #3a92ea; }}
QSlider::sub-page:horizontal {{ background: {_ACCENT}; border-radius: 3px; }}
QCheckBox {{ color: #ddd; spacing: 8px; font-size: 12px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border: 2px solid #606060; border-radius: 4px;
    background-color: #333;
}}
QCheckBox::indicator:checked {{ background-color: {_ACCENT}; border-color: {_ACCENT}; }}
QCheckBox::indicator:hover {{ border-color: {_ACCENT}; }}
QStatusBar {{ background-color: #353535; border-top: 1px solid #3d3d3d; color: #999; font-size: 11px; }}
QScrollArea#paramScroll {{ background-color: transparent; border: none; }}
QScrollBar:vertical {{ background: #333; width: 8px; border-radius: 4px; }}
QScrollBar::handle:vertical {{ background: #555; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #666; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
QScrollBar:horizontal {{ background: #333; height: 8px; border-radius: 4px; }}
QScrollBar::handle:horizontal {{ background: #555; border-radius: 4px; min-width: 30px; }}
QScrollBar::handle:horizontal:hover {{ background: #666; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; }}
QWidget#rightPanel {{ background-color: #303030; border-left: 1px solid #3d3d3d; }}
QWidget#paramCardContainer {{ background-color: transparent; }}
"""

_LIGHT_QSS = f"""
QMainWindow {{ background-color: #f0f0f0; }}
QToolBar {{
    background-color: #fafafa; border-bottom: 1px solid #d0d0d0;
    padding: 4px 6px; spacing: 6px;
}}
QPushButton#primaryBtn {{
    background-color: #1a6dc4; color: #fff; border: none;
    border-radius: 5px; padding: 6px 18px; font-weight: bold; font-size: 12px;
}}
QPushButton#primaryBtn:hover {{ background-color: #2078d4; }}
QPushButton#primaryBtn:pressed {{ background-color: #165ea8; }}
QPushButton#primaryBtn:disabled {{ background-color: #d0d0d0; color: #999; }}
QPushButton#toolBtn {{
    background-color: #fff; color: #444; border: 1px solid #c0c0c0;
    border-radius: 5px; padding: 5px 14px; font-size: 12px;
}}
QPushButton#toolBtn:hover {{ background-color: #f0f0f0; border-color: #1a6dc4; }}
QPushButton#toolBtn:pressed {{ background-color: #e0e0e0; }}
QPushButton#navBtn {{
    background-color: #fff; color: #555; border: 1px solid #c8c8c8;
    border-radius: 5px; padding: 5px 12px; font-size: 11px;
}}
QPushButton#navBtn:hover {{ background-color: #f0f0f0; border-color: #1a6dc4; }}
QPushButton#navBtn:pressed {{ background-color: #e0e0e0; }}
QPushButton#navBtn:disabled {{ color: #bbb; border-color: #e0e0e0; background-color: #f8f8f8; }}
QTabWidget#mainTabs::pane {{
    border: 1px solid #d0d0d0; background-color: #f0f0f0; border-radius: 6px; top: -1px;
}}
QTabWidget#mainTabs QTabBar::tab {{
    background-color: #e8e8e8; color: #777; border: 1px solid #d0d0d0; border-bottom: none;
    padding: 8px 20px; margin-right: 3px; border-top-left-radius: 6px;
    border-top-right-radius: 6px; min-height: 26px;
}}
QTabWidget#mainTabs QTabBar::tab:selected {{
    background-color: #f0f0f0; color: #333; border-bottom: 2px solid #1a6dc4;
    font-weight: bold;
}}
QTabWidget#mainTabs QTabBar::tab:hover:!selected {{
    background-color: #f5f5f5; color: #444;
}}
QFrame#paramCard {{
    background-color: #fff; border: 1px solid #d8d8d8; border-radius: 8px;
}}
QFrame#sectionCard {{
    background-color: #fafafa; border: 1px solid #e0e0e0; border-radius: 6px;
    margin: 2px 0px;
}}
QLabel#sectionHeader {{
    color: #444; font-weight: bold; font-size: 12px;
    padding: 2px 0px 4px 0px; border-bottom: 1px solid #e0e0e0; margin-bottom: 4px;
}}
QLabel#sectionHint {{ color: #999; font-size: 10px; padding: 0px 0px 4px 0px; }}
QLabel#paramTitle {{ color: #444; font-weight: bold; font-size: 12px; padding: 4px 0px; }}
QLabel#fileLabel {{
    color: #555; padding: 2px 10px; border: 1px solid #c0c0c0;
    border-radius: 4px; background-color: #fff;
}}
QLineEdit {{
    background-color: #fff; color: #333; border: 1px solid #c0c0c0;
    border-radius: 4px; padding: 5px 10px;
}}
QLineEdit:focus {{ border-color: #1a6dc4; }}
QComboBox {{
    background-color: #fff; color: #333; border: 1px solid #c0c0c0;
    border-radius: 4px; padding: 4px 10px; min-width: 80px;
}}
QComboBox:hover {{ border-color: #1a6dc4; }}
QComboBox QAbstractItemView {{
    background-color: #fff; color: #333; selection-background-color: #1a6dc4;
    border: 1px solid #c0c0c0;
}}
QDoubleSpinBox {{
    background-color: #fff; color: #333; border: 1px solid #c0c0c0;
    border-radius: 4px; padding: 4px 6px;
}}
QDoubleSpinBox:focus {{ border-color: #1a6dc4; }}
QSlider::groove:horizontal {{
    height: 5px; background: #d0d0d0; border-radius: 3px; margin: 0px 2px;
}}
QSlider::handle:horizontal {{
    background: #1a6dc4; width: 14px; height: 14px; margin: -5px 0px; border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: #2078d4; }}
QSlider::sub-page:horizontal {{ background: #1a6dc4; border-radius: 3px; }}
QCheckBox {{ color: #333; spacing: 8px; font-size: 12px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border: 2px solid #b0b0b0; border-radius: 4px;
    background-color: #fff;
}}
QCheckBox::indicator:checked {{ background-color: #1a6dc4; border-color: #1a6dc4; }}
QCheckBox::indicator:hover {{ border-color: #1a6dc4; }}
QStatusBar {{ background-color: #fafafa; border-top: 1px solid #d0d0d0; color: #777; font-size: 11px; }}
QScrollArea#paramScroll {{ background-color: transparent; border: none; }}
QScrollBar:vertical {{ background: #e8e8e8; width: 8px; border-radius: 4px; }}
QScrollBar::handle:vertical {{ background: #c0c0c0; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #aaa; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
QScrollBar:horizontal {{ background: #e8e8e8; height: 8px; border-radius: 4px; }}
QScrollBar::handle:horizontal {{ background: #c0c0c0; border-radius: 4px; min-width: 30px; }}
QScrollBar::handle:horizontal:hover {{ background: #aaa; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; }}
QWidget#rightPanel {{ background-color: #f5f5f5; border-left: 1px solid #d0d0d0; }}
QWidget#paramCardContainer {{ background-color: transparent; }}
"""


def build_stylesheet(theme: str) -> str:
    """根据主题生成完整 QSS 样式表。"""
    if theme == "dark":
        return _DARK_QSS
    return _LIGHT_QSS


def apply_theme(app: QApplication, theme: str):
    """将主题应用到 QApplication，同时更新模块级 _THEME。"""
    global _THEME
    _THEME = theme
    if theme == "dark":
        app.setPalette(_dark_palette())
    else:
        app.setPalette(_light_palette())
    app.setStyleSheet(build_stylesheet(theme))


def get_theme() -> str:
    """返回当前主题名称 "dark" 或 "light"。"""
    return _THEME


# ── 状态常量 ──────────────────────────────────────────────────

STATE_IDLE = "idle"
STATE_LOADING = "loading"
STATE_READY = "ready"
STATE_PROCESSING = "processing"

# ── 处理链顺序 ────────────────────────────────────────────────

CHAIN_ORDER = ["cross", "film", "filters", "enhance"]


# ── 线程信号桥 ────────────────────────────────────────────────

class _WorkerSignals(QObject):
    """工作线程 → 主线程信号桥。"""

    preview_ready = Signal(int, object)  # version, result
    load_done = Signal(str, object, object)  # path, cam_render, tgt_render
    load_error = Signal(str)
    process_done = Signal(str, str)  # output_path, error
    batch_progress = Signal(int, int)  # current, total
    batch_done = Signal(int, int, str)  # done, failed, output_dir


class ImageToolkitApp(QMainWindow):
    """ISL Toolkit 图像处理工具主窗口 (PySide6)。"""

    def __init__(self):
        super().__init__()

        # ── 窗口属性 ──────────────────────────────────
        self.setWindowTitle("ISL Toolkit — 图像处理工具箱")
        self.resize(1320, 820)
        self.setMinimumSize(1024, 640)

        # ── 数据 ─────────────────────────────────────
        self.raw_files: list[str] = []
        self._preview_path = None
        self._file_index = -1
        self._cam_render = None
        self._tgt_render = None
        self._plugins = list_all()
        self._plugin_map = {p.name: p for p in self._plugins}
        self._current_plugin = self._plugins[0] if self._plugins else None

        # ── 状态 & 并发控制 ───────────────────────────
        self._state = STATE_IDLE
        self._preview_version = 0
        self._mutual_lock = False
        self._cached_result: np.ndarray | None = None
        self._viewers: dict[str, ImageViewer] = {}

        # ── 线程信号 ─────────────────────────────────
        self._signals = _WorkerSignals()
        self._signals.preview_ready.connect(self._on_preview_ready)
        self._signals.load_done.connect(self._on_load_done)
        self._signals.load_error.connect(self._on_load_error)
        self._signals.process_done.connect(self._on_process_done)
        self._signals.batch_progress.connect(self._on_batch_progress)
        self._signals.batch_done.connect(self._on_batch_done)

        # ── 构建 UI ──────────────────────────────────
        self._build_toolbar()
        self._build_main_area()
        self._build_statusbar()
        self._bind_shortcuts()
        self._setup_mutual_exclusion()

        # ── 主题 ────────────────────────────────────
        data = session_load()
        saved_theme = data.get("theme", "dark")
        apply_theme(QApplication.instance(), saved_theme)
        self._sync_theme_combo(saved_theme)
        self._update_theme_styles(saved_theme)

        # ── 延迟初始化 ───────────────────────────────
        QTimer.singleShot(100, self._draw_placeholders)
        QTimer.singleShot(200, self._restore_session)

    # ═══════════════════════════════════════════════════════
    #  状态管理
    # ═══════════════════════════════════════════════════════

    def _set_state(self, new_state: str):
        self._state = new_state

        if new_state == STATE_IDLE:
            self._save_btn.setEnabled(False)
            self._save_btn.setText("导出当前 (Ctrl+S)")
            self._batch_btn.setEnabled(False)
            self._batch_btn.setText("批量导出 (Ctrl+B)")
            self._prev_btn.setEnabled(False)
            self._next_btn.setEnabled(False)
        elif new_state == STATE_LOADING:
            self._save_btn.setEnabled(False)
            self._batch_btn.setEnabled(False)
            self._prev_btn.setEnabled(False)
            self._next_btn.setEnabled(False)
        elif new_state == STATE_READY:
            self._save_btn.setEnabled(True)
            self._save_btn.setText("导出当前 (Ctrl+S)")
            n = len(self.raw_files)
            self._batch_btn.setEnabled(True)
            self._batch_btn.setText(f"批量导出 ({n} 个)")
            self._update_nav_buttons()
        elif new_state == STATE_PROCESSING:
            self._save_btn.setEnabled(False)
            self._save_btn.setText("导出中...")
            self._batch_btn.setEnabled(False)
            self._batch_btn.setText("导出中...")
            self._prev_btn.setEnabled(False)
            self._next_btn.setEnabled(False)

    # ═══════════════════════════════════════════════════════
    #  工具栏
    # ═══════════════════════════════════════════════════════

    def _build_toolbar(self):
        tb = self.addToolBar("主工具栏")
        tb.setMovable(False)

        tb.addWidget(QLabel("  文件夹: "))

        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("选择包含图像文件的文件夹...")
        self._folder_edit.setMinimumWidth(200)
        tb.addWidget(self._folder_edit)

        btn_browse = QPushButton("浏览")
        btn_browse.setObjectName("toolBtn")
        btn_browse.clicked.connect(self._browse_folder)
        tb.addWidget(btn_browse)

        btn_open = QPushButton("打开文件")
        btn_open.setObjectName("toolBtn")
        btn_open.clicked.connect(self._open_raw_file)
        tb.addWidget(btn_open)

        tb.addSeparator()

        self._prev_btn = QPushButton("◀ 上一张")
        self._prev_btn.setObjectName("navBtn")
        self._prev_btn.setEnabled(False)
        self._prev_btn.clicked.connect(self._nav_prev)
        tb.addWidget(self._prev_btn)

        self._file_label = QLabel("")
        self._file_label.setObjectName("fileLabel")
        self._file_label.setMinimumWidth(60)
        self._file_label.setAlignment(Qt.AlignCenter)
        tb.addWidget(self._file_label)

        self._next_btn = QPushButton("下一张 ▶")
        self._next_btn.setObjectName("navBtn")
        self._next_btn.setEnabled(False)
        self._next_btn.clicked.connect(self._nav_next)
        tb.addWidget(self._next_btn)

        tb.addSeparator()

        self._save_btn = QPushButton("导出当前 (Ctrl+S)")
        self._save_btn.setObjectName("primaryBtn")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._process_current)
        tb.addWidget(self._save_btn)

        self._batch_btn = QPushButton("批量导出 (Ctrl+B)")
        self._batch_btn.setObjectName("primaryBtn")
        self._batch_btn.setEnabled(False)
        self._batch_btn.clicked.connect(self._process_batch)
        tb.addWidget(self._batch_btn)

        tb.addSeparator()

        tb.addWidget(QLabel("  主题: "))
        self._theme_combo = QComboBox()
        self._theme_combo.addItem("暗色", "dark")
        self._theme_combo.addItem("亮色", "light")
        self._theme_combo.currentTextChanged.connect(self._on_theme_combo_changed)
        tb.addWidget(self._theme_combo)

    # ═══════════════════════════════════════════════════════
    #  状态栏
    # ═══════════════════════════════════════════════════════

    def _build_statusbar(self):
        self._status_label = QLabel("就绪 — 打开 RAW 文件或选择文件夹开始")
        self.statusBar().addWidget(self._status_label, 1)

        self._gpu_status_label = QLabel(f"GPU: {get_status_text()}")
        self._gpu_status_label.setStyleSheet("padding-right: 8px;")
        self.statusBar().addPermanentWidget(self._gpu_status_label)

    def _set_status(self, text: str):
        self._status_label.setText(text)

    # ═══════════════════════════════════════════════════════
    #  快捷键
    # ═══════════════════════════════════════════════════════

    def _bind_shortcuts(self):
        act_open = QAction("打开文件夹 (Ctrl+O)", self)
        act_open.setShortcut(QKeySequence("Ctrl+O"))
        act_open.triggered.connect(self._browse_folder)
        self.addAction(act_open)

        act_save = QAction("导出当前", self)
        act_save.setShortcut(QKeySequence("Ctrl+S"))
        act_save.triggered.connect(self._process_current)
        self.addAction(act_save)

        act_batch = QAction("批量导出", self)
        act_batch.setShortcut(QKeySequence("Ctrl+B"))
        act_batch.triggered.connect(self._process_batch)
        self.addAction(act_batch)

        act_prev = QAction("上一张", self)
        act_prev.setShortcut(QKeySequence(Qt.Key_Left))
        act_prev.triggered.connect(self._nav_prev)
        self.addAction(act_prev)

        act_next = QAction("下一张", self)
        act_next.setShortcut(QKeySequence(Qt.Key_Right))
        act_next.triggered.connect(self._nav_next)
        self.addAction(act_next)

    # ═══════════════════════════════════════════════════════
    #  互斥：cross 与 film 不能同时启用
    # ═══════════════════════════════════════════════════════

    def _setup_mutual_exclusion(self):
        cross = self._plugin_map.get("cross")
        film = self._plugin_map.get("film")
        if not cross or not film:
            return

        def _on_cross_changed(checked):
            if self._mutual_lock:
                return
            if checked and film.is_enabled():
                self._mutual_lock = True
                film.set_enabled(False)
                self._mutual_lock = False

        def _on_film_changed(checked):
            if self._mutual_lock:
                return
            if checked and cross.is_enabled():
                self._mutual_lock = True
                cross.set_enabled(False)
                self._mutual_lock = False

        cross._enabled_cb.toggled.connect(_on_cross_changed)
        film._enabled_cb.toggled.connect(_on_film_changed)

    # ═══════════════════════════════════════════════════════
    #  主区域（标签页）
    # ═══════════════════════════════════════════════════════

    def _build_main_area(self):
        self._tabs = QTabWidget()
        self._tabs.setObjectName("mainTabs")
        self._tabs.currentChanged.connect(self._on_tab_change)
        self.setCentralWidget(self._tabs)

        for plugin in self._plugins:
            tab = QWidget()
            layout = QHBoxLayout(tab)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(4)

            # 左侧：预览
            viewer = ImageViewer()
            self._viewers[plugin.name] = viewer
            plugin._viewer = viewer
            layout.addWidget(viewer, 3)

            # 右侧：参数面板（可滚动）
            right_panel = QWidget()
            right_panel.setObjectName("rightPanel")
            right_panel.setFixedWidth(310)
            right_layout = QVBoxLayout(right_panel)
            right_layout.setContentsMargins(4, 4, 4, 4)

            scroll = QScrollArea()
            scroll.setObjectName("paramScroll")
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.NoFrame)
            scroll_content = QWidget()
            scroll_content.setObjectName("paramCardContainer")
            scroll_content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            scroll_layout = QVBoxLayout(scroll_content)
            scroll_layout.setContentsMargins(4, 4, 4, 4)
            scroll_layout.addStretch()
            scroll.setWidget(scroll_content)

            title_lbl = QLabel(f"  {plugin.icon} {plugin.title}  参数面板")
            title_lbl.setObjectName("paramTitle")
            right_layout.addWidget(title_lbl, 0)
            right_layout.addWidget(scroll, 1)

            layout.addWidget(right_panel, 0)

            plugin.attach_ui(scroll_content, on_change=lambda p=plugin: self._on_plugin_param_change(p))

            self._tabs.addTab(tab, f"  {plugin.icon} {plugin.title}  ")

    # ═══════════════════════════════════════════════════════
    #  占位符
    # ═══════════════════════════════════════════════════════

    def _draw_placeholders(self):
        if self._state != STATE_IDLE:
            return
        for plugin in self._plugins:
            viewer = getattr(plugin, '_viewer', None)
            if viewer is None:
                continue
            # 在场景中绘制占位文字
            viewer._scene.clear()
            text_item = viewer._scene.addText(
                f"{plugin.icon} {plugin.title}\n\n打开 RAW 文件开始"
            )
            text_item.setDefaultTextColor(Qt.gray)
            # 居中
            r = text_item.boundingRect()
            vw = viewer.viewport().width()
            vh = viewer.viewport().height()
            text_item.setPos((vw - r.width()) / 2, (vh - r.height()) / 2)
            # 重建 pixmap item
            from PySide6.QtWidgets import QGraphicsPixmapItem
            viewer._pixmap_item = QGraphicsPixmapItem()
            viewer._scene.addItem(viewer._pixmap_item)

    def _restore_session(self):
        """恢复上次会话的文件夹和图片。"""
        data = session_load()
        folder = data.get("last_folder", "")
        last_path = data.get("last_path", "")
        last_index = data.get("last_index", 0)
        if folder and os.path.isdir(folder):
            self._folder_edit.setText(folder)
            raw_files = find_raw_files(folder)
            if raw_files:
                self.raw_files = raw_files
                if last_path and last_path in raw_files:
                    self._file_index = raw_files.index(last_path)
                elif 0 <= last_index < len(raw_files):
                    self._file_index = last_index
                else:
                    self._file_index = 0
                self._load_preview_async(raw_files[self._file_index])

    # ═══════════════════════════════════════════════════════
    #  预览调度
    # ═══════════════════════════════════════════════════════

    def _on_tab_change(self, idx: int):
        if 0 <= idx < len(self._plugins):
            self._current_plugin = self._plugins[idx]
            if self._state == STATE_READY and self._cached_result is not None:
                self._render_to_all_viewers(self._cached_result)

    def _on_plugin_param_change(self, plugin=None):
        if self._state == STATE_READY:
            self._request_preview()

    def _request_preview(self):
        self._preview_version += 1
        version = self._preview_version

        if self._cam_render is None or self._tgt_render is None:
            return
        cam_snap = self._cam_render.copy()
        tgt_snap = self._tgt_render.copy()

        def _worker():
            try:
                result = self._compute_chain(cam_snap, tgt_snap)
                self._signals.preview_ready.emit(version, result.copy())
            except Exception as e:
                logger.debug("Preview error: %s", e)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_preview_ready(self, version: int, result: np.ndarray):
        if version != self._preview_version:
            return
        self._cached_result = result
        self._render_to_all_viewers(result)

    def _compute_chain(self, cam_snap: np.ndarray, tgt_snap: np.ndarray):
        """链式处理：cross|film → filters → enhance。"""
        cam = cam_snap.astype(np.float32) / 255.0
        tgt = tgt_snap.astype(np.float32) / 255.0
        result = tgt.copy()

        cross = self._plugin_map.get("cross")
        film = self._plugin_map.get("film")
        filters = self._plugin_map.get("filters")
        enhance = self._plugin_map.get("enhance")

        # Stage 1: cross 或 film（互斥）
        if cross and cross.is_enabled():
            cp = cross.get_params()
            strength = cp.get("strength", 0.8)
            if strength >= 1.0:
                result = tgt.copy()
            elif strength <= 0.0:
                result = cam.copy()
            else:
                result = cam * (1.0 - strength) + tgt * strength
        elif film and film.is_enabled():
            from ..kernel.filters.color import LCCInverter
            from ..kernel.filters.geometry import FlatFieldFilter
            fp = film.get_params()
            strength = fp.get("strength", 0.8)
            # Flat field strength: prefer filters' setting if enabled, else film default 0.5
            ff_strength = 0.5
            if filters and filters.is_enabled():
                fparams = filters.get_params()
                ffs = fparams.get("flat_field_strength", 0)
                if ffs > 0.01:
                    ff_strength = ffs
            if ff_strength > 0.01:
                result = FlatFieldFilter(strength=ff_strength).apply(tgt.copy(), preview=True)
            else:
                result = tgt.copy()
            result = LCCInverter(strength=strength).apply(result, preview=True)

        # Stage 2: 滤镜处理（降噪/去雾/平场/B&W滤镜/颗粒）
        if filters and filters.is_enabled():
            # 如果 film 已启用，平场已在 Stage 1 应用，避免重复
            skip_ff = film and film.is_enabled()
            result = self._apply_filters_preview(result, filters.get_params(),
                                                  skip_flat_field=skip_ff)

        # Stage 3: 图像增强（亮度/对比度/饱和度 + 锐化/清晰度/暗角/暗部增强/色调曲线）
        if enhance and enhance.is_enabled():
            ep = enhance.get_params()
            result = self._apply_enhance_preview(result, ep)
            result = _apply_postprocess(result,
                                        ep.get("brightness", 1.0),
                                        ep.get("contrast", 1.0),
                                        ep.get("saturation", 1.0))

        return result

    @staticmethod
    def _apply_filters_preview(image: np.ndarray, params: dict,
                               skip_flat_field: bool = False) -> np.ndarray:
        """预览链：仅滤镜处理（降噪/去雾/平场/B&W滤镜/颗粒/色调映射）。"""
        from ..kernel.filters.color import DehazeFilter
        from ..kernel.filters.noise import CromaNRFilter, BandNRFilter
        from ..kernel.filters.geometry import FlatFieldFilter
        from ..kernel.filters.dcu_legacy import BWFilterSim, ToningFilter
        from ..kernel.filters.creative import GrainFilter

        rgb = image
        chroma_nr = params.get("chroma_nr", 0)
        band_nr = params.get("band_nr", 0)
        flat_strength = params.get("flat_field_strength", 0)
        bw_filter = params.get("bw_filter", "none")
        bw_filter_strength = params.get("bw_filter_strength", 1.0)
        dehaze = params.get("dehaze", 0)
        grain = params.get("grain", 0)
        toning = params.get("toning", "none")

        if chroma_nr > 0.01:
            rgb = CromaNRFilter(strength=chroma_nr).apply(rgb, preview=True)
        if band_nr > 0.01:
            rgb = BandNRFilter(strength=band_nr).apply(rgb, preview=True)
        if flat_strength > 0.01 and not skip_flat_field:
            rgb = FlatFieldFilter(strength=flat_strength).apply(rgb, preview=True)
        if bw_filter != "none":
            rgb = BWFilterSim(filter_type=bw_filter, strength=bw_filter_strength).apply(rgb, preview=True)
        if dehaze > 0.01:
            rgb = DehazeFilter(strength=dehaze).apply(rgb, preview=True)
        if grain > 0.01:
            rgb = GrainFilter(strength=grain).apply(rgb, preview=True)
        if toning != "none":
            rgb = ToningFilter(preset=toning, strength=0.7).apply(rgb, preview=True)
        return rgb

    @staticmethod
    def _apply_enhance_preview(image: np.ndarray, params: dict) -> np.ndarray:
        """预览链：仅增强处理（锐化/清晰度/暗角/暗部增强/色调曲线）。"""
        from ..kernel.filters.dcu_legacy import SmartSharpFilter, ClarityFilter
        from ..kernel.filters.creative import VignetteFilter, ToneCurveFilter, ShadowBoostFilter

        rgb = image
        clarity = params.get("clarity", 0)
        smart_sharp = params.get("smart_sharp", 0)
        vignette = params.get("vignette", 0)
        shadow_boost = params.get("shadow_boost", 0)
        highlights = params.get("highlights", 0)
        shadows = params.get("shadows", 0)
        midtones = params.get("midtones", 0)

        if clarity > 0.01:
            rgb = ClarityFilter(strength=clarity).apply(rgb, preview=True)
        if highlights != 0.0 or shadows != 0.0 or midtones != 0.0:
            rgb = ToneCurveFilter(highlights=highlights, shadows=shadows,
                                  midtones=midtones).apply(rgb, preview=True)
        if shadow_boost > 0.01:
            rgb = ShadowBoostFilter(strength=shadow_boost).apply(rgb, preview=True)
        if smart_sharp > 0.01:
            rgb = SmartSharpFilter(strength=smart_sharp).apply(rgb, preview=True)
        if vignette > 0.01:
            rgb = VignetteFilter(strength=vignette).apply(rgb, preview=True)
        return rgb

    # ═══════════════════════════════════════════════════════
    #  渲染到所有预览控件
    # ═══════════════════════════════════════════════════════

    def _render_to_all_viewers(self, result: np.ndarray | None):
        for plugin in self._plugins:
            viewer = getattr(plugin, '_viewer', None)
            if viewer is not None:
                viewer.set_image(result)

    # ═══════════════════════════════════════════════════════
    #  文件加载
    # ═══════════════════════════════════════════════════════

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择包含图像文件的文件夹")
        if folder:
            self._folder_edit.setText(folder)
            self._scan_folder()

    def _scan_folder(self):
        folder = self._folder_edit.text()
        if not folder:
            return
        raw_files = find_raw_files(folder)
        if not raw_files:
            self._set_status(f"未找到RAW文件: {folder}")
            return
        self.raw_files = raw_files
        self._file_index = 0
        self._load_preview_async(raw_files[0])
        session_save(folder=folder, path=raw_files[0], index=0)

    def _open_raw_file(self):
        exts = [f"*.{e[1:]}" for e in sorted(RAW_EXTENSIONS)]
        ext_str = " ".join(exts)
        path, _ = QFileDialog.getOpenFileName(
            self, "打开图像文件", "",
            f"RAW文件 ({ext_str});;所有文件 (*.*)",
        )
        if not path:
            return
        folder = os.path.dirname(path)
        self._folder_edit.setText(folder)
        self.raw_files = find_raw_files(folder)
        try:
            self._file_index = self.raw_files.index(path)
        except ValueError:
            self._file_index = 0
        self._load_preview_async(path)
        session_save(folder=folder, path=path, index=self._file_index)

    def _update_file_label(self):
        if self.raw_files and 0 <= self._file_index < len(self.raw_files):
            self._file_label.setText(f"{self._file_index + 1}/{len(self.raw_files)}")

    def _update_nav_buttons(self):
        if not self.raw_files:
            self._prev_btn.setEnabled(False)
            self._next_btn.setEnabled(False)
            return
        self._prev_btn.setEnabled(self._file_index > 0)
        self._next_btn.setEnabled(self._file_index < len(self.raw_files) - 1)
        self._update_file_label()

    def _nav_prev(self):
        if self._state != STATE_READY or self._file_index <= 0:
            return
        self._file_index -= 1
        self._update_nav_buttons()
        path = self.raw_files[self._file_index]
        self._load_preview_async(path)
        session_save(folder=self._folder_edit.text(), path=path, index=self._file_index)

    def _nav_next(self):
        if self._state != STATE_READY or self._file_index >= len(self.raw_files) - 1:
            return
        self._file_index += 1
        self._update_nav_buttons()
        path = self.raw_files[self._file_index]
        self._load_preview_async(path)
        session_save(folder=self._folder_edit.text(), path=path, index=self._file_index)

    def _load_preview_async(self, path: str):
        if not _HAS_RAWPY:
            self._set_status("需要 rawpy: pip install rawpy")
            return

        self._set_state(STATE_LOADING)
        self._set_status("加载中...")

        def _worker():
            try:
                raw = rawpy.imread(path)
                try:
                    pp = dict(output_color=rawpy.ColorSpace.sRGB, gamma=(2.222, 4.5),
                              no_auto_bright=True, bright=1.0, half_size=True)
                    cam_render = raw.postprocess(use_camera_wb=True, **pp)
                    tgt_render = raw.postprocess(use_camera_wb=False, use_auto_wb=True, **pp)
                finally:
                    raw.close()
                self._signals.load_done.emit(path, cam_render, tgt_render)
            except Exception as e:
                self._signals.load_error.emit(str(e))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_load_done(self, path: str, cam_render: np.ndarray, tgt_render: np.ndarray):
        self._preview_path = path
        self._cam_render = cam_render
        self._tgt_render = tgt_render
        self._set_state(STATE_READY)
        self._set_status(f"预览: {os.path.basename(path)}  |  共 {len(self.raw_files)} 个文件")
        self._update_file_label()
        self._request_preview()

    def _on_load_error(self, error: str):
        self._set_state(STATE_IDLE)
        self._set_status(f"加载失败: {error}")

    # ═══════════════════════════════════════════════════════
    #  处理（单张 / 批量）
    # ═══════════════════════════════════════════════════════

    def _process_current(self):
        if self._state != STATE_READY or self._preview_path is None:
            return
        import datetime
        name = os.path.splitext(os.path.basename(self._preview_path))[0]
        folder = os.path.dirname(self._preview_path)
        ts = datetime.datetime.now().strftime("%H%M%S")
        output_path = os.path.join(folder, f"{name}_{ts}.png")

        self._set_state(STATE_PROCESSING)

        cross = self._plugin_map.get("cross")
        film = self._plugin_map.get("film")
        filters = self._plugin_map.get("filters")
        enhance = self._plugin_map.get("enhance")

        use_cross = cross and cross.is_enabled()
        use_film = film and film.is_enabled()
        use_filters = filters and filters.is_enabled()
        use_enhance = enhance and enhance.is_enabled()

        cross_params = cross.get_params() if cross else {}
        film_params = film.get_params() if film else {}
        filters_params = filters.get_params() if filters else {}
        enhance_params = enhance.get_params() if enhance else {}

        # wb_mode/strength: only read from the plugin that is actually enabled
        if use_film:
            wb_mode = film_params.get("wb_mode", "auto")
            strength = film_params.get("strength", 0.8)
        elif use_cross:
            wb_mode = cross_params.get("wb_mode", "auto")
            strength = cross_params.get("strength", 0.8)
        else:
            wb_mode = "auto"
            strength = 0.8
        brightness = enhance_params.get("brightness", 1.0) if use_enhance else 1.0
        contrast = enhance_params.get("contrast", 1.0) if use_enhance else 1.0
        saturation = enhance_params.get("saturation", 1.0) if use_enhance else 1.0

        def _worker():
            try:
                if use_cross:
                    post_pipeline = self._build_cross_post_pipeline(
                        use_filters, use_enhance, filters_params, enhance_params)
                    process_raw(self._preview_path, output_path,
                                wb_mode=wb_mode, strength=strength,
                                post_pipeline=post_pipeline,
                                brightness=brightness, contrast=contrast, saturation=saturation)
                elif use_film or use_filters:
                    pipeline = self._build_save_pipeline(use_film, use_filters,
                                                         film_params, filters_params)
                    post_pipeline = (
                        self._build_enhance_save_pipeline(enhance_params)
                        if use_enhance else None
                    )
                    proc_wb = film_params.get("wb_mode", "auto") if use_film else wb_mode
                    proc_strength = film_params.get("strength", 0.8) if use_film else 0.0
                    process_raw(self._preview_path, output_path,
                                wb_mode=proc_wb, strength=proc_strength, pipeline=pipeline,
                                post_pipeline=post_pipeline,
                                brightness=brightness, contrast=contrast, saturation=saturation)
                else:
                    post_pipeline = (
                        self._build_enhance_save_pipeline(enhance_params)
                        if use_enhance else None
                    )
                    process_raw(self._preview_path, output_path,
                                wb_mode="auto", strength=1.0,
                                post_pipeline=post_pipeline,
                                brightness=brightness, contrast=contrast, saturation=saturation)
                self._signals.process_done.emit(output_path, "")
            except Exception as e:
                self._signals.process_done.emit("", str(e))

        threading.Thread(target=_worker, daemon=True).start()

    def _build_save_pipeline(self, use_film: bool, use_filters: bool,
                             film_params: dict, filters_params: dict):
        """构建保存管线 — 仅滤镜处理（降噪/去雾/平场/B&W滤镜/颗粒/色调映射）。"""
        from ..kernel import ProcessingPipeline
        from ..kernel.filters import (LCCInverter, FlatFieldFilter,
                                      CromaNRFilter, BandNRFilter, DehazeFilter)
        from ..kernel.filters.dcu_legacy import BWFilterSim, ToningFilter
        from ..kernel.filters.creative import GrainFilter

        pipe = ProcessingPipeline(preview_scale=1.0)

        if use_film:
            # Flat field strength: prefer filters' setting if enabled, else film default 0.5
            ff_strength = 0.5
            if use_filters:
                ffs = filters_params.get("flat_field_strength", 0)
                if ffs > 0.01:
                    ff_strength = ffs
            if ff_strength > 0.01:
                pipe.add_frontend(FlatFieldFilter(strength=ff_strength))
            pipe.add_core(LCCInverter(strength=film_params.get("strength", 0.8)))

        if use_filters:
            chroma_nr = filters_params.get("chroma_nr", 0)
            band_nr = filters_params.get("band_nr", 0)
            ff_strength = filters_params.get("flat_field_strength", 0)
            bw_filter = filters_params.get("bw_filter", "none")
            bw_filter_strength = filters_params.get("bw_filter_strength", 1.0)
            dehaze = filters_params.get("dehaze", 0)
            grain = filters_params.get("grain", 0)

            if chroma_nr > 0.01:
                pipe.add_frontend(CromaNRFilter(strength=chroma_nr))
            if band_nr > 0.01:
                pipe.add_frontend(BandNRFilter(strength=band_nr))
            if ff_strength > 0.01 and not use_film:
                pipe.add_frontend(FlatFieldFilter(strength=ff_strength))
            if bw_filter != "none":
                pipe.add_frontend(BWFilterSim(filter_type=bw_filter, strength=bw_filter_strength))
            if dehaze > 0.01:
                pipe.add_backend(DehazeFilter(strength=dehaze))
            if grain > 0.01:
                pipe.add_backend(GrainFilter(strength=grain))
            toning = filters_params.get("toning", "none")
            if toning != "none":
                pipe.add_backend(ToningFilter(preset=toning, strength=0.7))

        return pipe

    @staticmethod
    def _build_enhance_save_pipeline(enhance_params: dict):
        """构建增强保存管线 — 锐化/清晰度/暗角/暗部增强/色调曲线。"""
        from ..kernel import ProcessingPipeline
        from ..kernel.filters.dcu_legacy import SmartSharpFilter, ClarityFilter
        from ..kernel.filters.creative import VignetteFilter, ToneCurveFilter, ShadowBoostFilter

        pipe = ProcessingPipeline(preview_scale=1.0)

        clarity = enhance_params.get("clarity", 0)
        smart_sharp = enhance_params.get("smart_sharp", 0)
        vignette = enhance_params.get("vignette", 0)
        shadow_boost = enhance_params.get("shadow_boost", 0)
        highlights = enhance_params.get("highlights", 0)
        shadows = enhance_params.get("shadows", 0)
        midtones = enhance_params.get("midtones", 0)

        if clarity > 0.01:
            pipe.add_frontend(ClarityFilter(strength=clarity))
        if highlights != 0.0 or shadows != 0.0 or midtones != 0.0:
            pipe.add_backend(ToneCurveFilter(
                highlights=highlights, shadows=shadows, midtones=midtones))
        if shadow_boost > 0.01:
            pipe.add_backend(ShadowBoostFilter(strength=shadow_boost))
        if smart_sharp > 0.01:
            pipe.add_backend(SmartSharpFilter(strength=smart_sharp))
        if vignette > 0.01:
            pipe.add_backend(VignetteFilter(strength=vignette))

        return pipe

    def _build_cross_post_pipeline(self, use_filters: bool, use_enhance: bool,
                                    filters_params: dict, enhance_params: dict):
        """Build combined post_pipeline for cross mode: filters → enhance.

        Cross mode uses process_raw's built-in WB blend (pipeline=None),
        so filters + enhance must be bundled into post_pipeline.
        """
        if not use_filters and not use_enhance:
            return None

        from ..kernel import ProcessingPipeline
        from ..kernel.filters import FilterStage

        post = ProcessingPipeline(preview_scale=1.0)

        if use_filters:
            fp = self._build_save_pipeline(
                False, True, film_params={}, filters_params=filters_params)
            for stage in (FilterStage.FRONTEND, FilterStage.CORE, FilterStage.BACKEND):
                for f in fp.get_stage(stage):
                    post.add(f, stage)

        if use_enhance:
            ep = self._build_enhance_save_pipeline(enhance_params)
            for stage in (FilterStage.FRONTEND, FilterStage.CORE, FilterStage.BACKEND):
                for f in ep.get_stage(stage):
                    post.add(f, stage)

        return post

    def _on_process_done(self, output_path: str, error: str):
        self._set_state(STATE_READY)
        if error:
            QMessageBox.critical(self, "错误", error)
        else:
            self._set_status(f"已保存: {os.path.basename(output_path)}")
            QMessageBox.information(self, "完成", f"已保存: {output_path}")

    def _process_batch(self):
        if not self.raw_files or self._state != STATE_READY:
            return

        folder = self._folder_edit.text()
        output_dir = os.path.join(folder, "corrected")
        os.makedirs(output_dir, exist_ok=True)

        cross = self._plugin_map.get("cross")
        film = self._plugin_map.get("film")
        filters = self._plugin_map.get("filters")
        enhance = self._plugin_map.get("enhance")

        use_cross = cross and cross.is_enabled()
        use_film = film and film.is_enabled()
        use_filters = filters and filters.is_enabled()
        use_enhance = enhance and enhance.is_enabled()

        cross_params = cross.get_params() if cross else {}
        film_params = film.get_params() if film else {}
        filters_params = filters.get_params() if filters else {}
        enhance_params = enhance.get_params() if enhance else {}

        # wb_mode/strength: only read from the plugin that is actually enabled
        if use_film:
            wb_mode = film_params.get("wb_mode", "auto")
            strength = film_params.get("strength", 0.8)
        elif use_cross:
            wb_mode = cross_params.get("wb_mode", "auto")
            strength = cross_params.get("strength", 0.8)
        else:
            wb_mode = "auto"
            strength = 0.8
        brightness = enhance_params.get("brightness", 1.0) if use_enhance else 1.0
        contrast = enhance_params.get("contrast", 1.0) if use_enhance else 1.0
        saturation = enhance_params.get("saturation", 1.0) if use_enhance else 1.0

        pipeline = None
        if use_film or use_filters:
            pipeline = self._build_save_pipeline(use_film, use_filters,
                                                 film_params, filters_params)

        post_pipeline = (
            self._build_enhance_save_pipeline(enhance_params)
            if use_enhance else None
        )

        cross_post = self._build_cross_post_pipeline(
            use_filters, use_enhance, filters_params, enhance_params)

        total = len(self.raw_files)
        self._set_state(STATE_PROCESSING)

        class _Counts:
            done = 0
            failed = 0
        counts = _Counts()

        def _worker():
            for i, path in enumerate(self.raw_files):
                name = os.path.splitext(os.path.basename(path))[0]
                out = os.path.join(output_dir, f"{name}.png")
                try:
                    if use_cross:
                        process_raw(path, out, wb_mode=wb_mode, strength=strength,
                                    post_pipeline=cross_post,
                                    brightness=brightness, contrast=contrast,
                                    saturation=saturation)
                    elif pipeline is not None and (pipeline._stages.get("core") or
                                                   pipeline._stages.get("frontend") or
                                                   pipeline._stages.get("backend")):
                        process_raw(path, out, wb_mode=wb_mode,
                                    strength=film_params.get("strength", 0.8) if use_film else 0.0,
                                    pipeline=pipeline,
                                    post_pipeline=post_pipeline,
                                    brightness=brightness, contrast=contrast,
                                    saturation=saturation)
                    else:
                        process_raw(path, out, wb_mode="auto", strength=1.0,
                                    post_pipeline=post_pipeline,
                                    brightness=brightness, contrast=contrast,
                                    saturation=saturation)
                    counts.done += 1
                except Exception as e:
                    counts.failed += 1
                    logger.error("Failed: %s - %s", path, e)
                self._signals.batch_progress.emit(i + 1, total)
            self._signals.batch_done.emit(counts.done, counts.failed, output_dir)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_batch_progress(self, current: int, total: int):
        self._batch_btn.setText(f"导出中... {current}/{total}")

    def _on_batch_done(self, done: int, failed: int, output_dir: str):
        self._set_state(STATE_READY)
        self._set_status(f"完成! 成功 {done}, 失败 {failed}")
        if failed == 0:
            QMessageBox.information(self, "完成", f"成功导出 {done} 个文件\n输出: {output_dir}")
        else:
            QMessageBox.warning(self, "完成", f"成功 {done}, 失败 {failed}")

    # ═══════════════════════════════════════════════════════
    #  主题切换
    # ═══════════════════════════════════════════════════════

    def _sync_theme_combo(self, theme: str):
        """将主题下拉栏同步到当前主题。"""
        if not hasattr(self, '_theme_combo'):
            return
        index = self._theme_combo.findData(theme)
        if index >= 0 and index != self._theme_combo.currentIndex():
            old = self._theme_combo.blockSignals(True)
            self._theme_combo.setCurrentIndex(index)
            self._theme_combo.blockSignals(old)

    def _on_theme_combo_changed(self, _text: str):
        """下拉栏主题选择变化时应用主题。"""
        theme = self._theme_combo.currentData()
        if theme not in {"dark", "light"}:
            return
        if theme == get_theme():
            return
        apply_theme(QApplication.instance(), theme)
        session_save(theme=theme)
        self._update_theme_styles(theme)

    def _update_theme_styles(self, theme: str):
        """根据主题更新局部样式（zoom_label 等无法通过全局 QSS 覆盖的控件）。"""
        if theme == "dark":
            zoom_qss = (
                "background: rgba(30,30,30,180); color: #aaa; padding: 2px 6px;"
                "border-radius: 3px; font-family: Consolas; font-size: 10px;"
            )
        else:
            zoom_qss = (
                "background: rgba(240,240,240,200); color: #444; padding: 2px 6px;"
                "border-radius: 3px; font-family: Consolas; font-size: 10px;"
            )

        self._sync_theme_combo(theme)

        # 更新所有预览 viewer 的缩放标签
        for viewer in self._viewers.values():
            if hasattr(viewer, '_zoom_label'):
                viewer._zoom_label.setStyleSheet(zoom_qss)


def main():
    """启动 PySide6 GUI。"""
    import sys

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("需要 PySide6: pip install PySide6", file=sys.stderr)
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 提前应用主题，避免窗口显示时的闪烁
    data = session_load()
    saved_theme = data.get("theme", "dark")
    apply_theme(app, saved_theme)

    window = ImageToolkitApp()
    window.show()
    sys.exit(app.exec())
