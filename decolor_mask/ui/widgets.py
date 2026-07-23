"""PySide6 共享 UI 控件。

提供：
  - _Var: 简单可变值封装，兼容 tkinter Variable 的 .get()/.set() API
  - SliderRow: 标签 + 滑块 + 数值输入框 的参数行控件
  - ImageViewer: QGraphicsView 预览控件，支持滚轮缩放、拖拽平移、双击重置
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter
from PySide6.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QGraphicsPixmapItem,
    QHBoxLayout,
    QLabel,
    QSlider,
    QDoubleSpinBox,
    QWidget,
    QSizePolicy,
)

import numpy as np
from PIL import Image


# ── 可变值封装 ──────────────────────────────────────────────

class _Var:
    """可变值：兼容 tkinter Variable 的 .get()/.set() 接口。

    用于在插件 _ui_vars 中存储滑块值和下拉框值，
    使得 get_params() 逻辑无需修改。
    """

    __slots__ = ('_value',)

    def __init__(self, value=None):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


# ── 参数滑块行 ──────────────────────────────────────────────

class SliderRow(QWidget):
    """带标签的滑块 + 数值输入框行。

    用法::

        row = SliderRow("强度", 0.0, 1.0, 0.8)
        row.value_changed.connect(my_handler)
        self._ui_vars[key] = row  # row.var 有 .get()
    """

    value_changed = Signal(float)

    def __init__(
        self,
        label: str,
        from_: float,
        to: float,
        default: float,
        resolution: float | None = 0.01,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("sliderRow")
        self._range = to - from_
        self._from = from_
        self._resolution = resolution
        self._suppress = False
        self.var = _Var(default)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(4)

        # 标签
        lbl = QLabel(label)
        lbl.setMinimumWidth(100)
        lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(lbl)

        # 滑块（整数步进）
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, self._slider_steps())
        self._slider.setValue(self._to_step(default))
        self._slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self._slider, 1)

        # 数值输入
        self._spin = QDoubleSpinBox()
        self._spin.setRange(from_, to)
        self._spin.setValue(default)
        dec = max(0, len(str(resolution).split('.')[-1]) if resolution else 2)
        self._spin.setDecimals(dec)
        if resolution:
            self._spin.setSingleStep(resolution)
        self._spin.valueChanged.connect(self._on_spin)
        self._spin.setMinimumWidth(72)
        layout.addWidget(self._spin)

        # 防抖定时器
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(250)
        self._debounce.timeout.connect(self._fire)

    def _slider_steps(self) -> int:
        if self._resolution:
            return max(1, int(round(self._range / self._resolution)))
        return 1000

    def _to_step(self, value: float) -> int:
        steps = self._slider_steps()
        frac = (value - self._from) / self._range
        return int(round(frac * steps))

    def _to_value(self, step: int) -> float:
        steps = self._slider_steps()
        frac = step / steps
        val = self._from + frac * self._range
        if self._resolution:
            val = round(val / self._resolution) * self._resolution
        return val

    def _on_slider(self, step: int):
        if self._suppress:
            return
        val = self._to_value(step)
        self._suppress = True
        self._spin.setValue(val)
        self._suppress = False
        self.var.set(val)
        self._debounce.start()

    def _on_spin(self, val: float):
        if self._suppress:
            return
        self._suppress = True
        self._slider.setValue(self._to_step(val))
        self._suppress = False
        self.var.set(val)
        self._debounce.start()

    def _fire(self):
        self.value_changed.emit(self.var.get())

    @property
    def value(self) -> float:
        return self.var.get()

    @value.setter
    def value(self, val: float):
        self._suppress = True
        self._slider.setValue(self._to_step(val))
        self._spin.setValue(val)
        self._suppress = False
        self.var.set(val)


# ── 图像预览控件 ──────────────────────────────────────────────

class ImageViewer(QGraphicsView):
    """支持缩放和平移的图像预览控件。

    交互：
      - 滚轮：缩放 (1.1x / 0.9x)
      - 左键拖拽：平移（仅在放大时）
      - 双击：重置缩放
    """

    _MIN_ZOOM = 0.05
    _MAX_ZOOM = 32.0

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        self._zoom = 1.0
        self._panning = False
        self._last_pan_pos = None
        self._image_array: np.ndarray | None = None
        self._viewport_size = (0, 0)

        # 外观 — 预览区保持偏深色，适合图像工具
        self.setRenderHints(
            QPainter.Antialiasing
            | QPainter.SmoothPixmapTransform
        )
        self.setStyleSheet("QGraphicsView { background-color: #1a1a1a; border: none; }")
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 缩放标签（覆盖在右下角）
        self._zoom_label = QLabel(self)
        self._zoom_label.setStyleSheet(
            "background: rgba(30,30,30,180); color: #aaa; padding: 2px 6px;"
            "border-radius: 3px; font-family: Consolas; font-size: 10px;"
        )
        self._zoom_label.setAlignment(Qt.AlignCenter)
        self._update_zoom_label_pos()

    # ── 公共接口 ──────────────────────────────────────────

    def set_image(self, arr: np.ndarray | None):
        """从 numpy 数组 (H, W, 3) float32 [0,1] 设置预览图像。"""
        self._image_array = arr
        if arr is None:
            self._scene.clear()
            self._pixmap_item = QGraphicsPixmapItem()
            self._scene.addItem(self._pixmap_item)
            self._zoom_label.hide()
            return

        h, w = arr.shape[:2]
        img = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        # QImage 需要连续内存且是 RGB 格式
        q_image = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
        self._pixmap_item.setPixmap(QPixmap.fromImage(q_image))
        self._scene.setSceneRect(0, 0, w, h)
        self._apply_transform()
        self._zoom_label.show()

    def reset_zoom(self):
        """重置缩放为适合视口。"""
        self._zoom = 1.0
        self._apply_transform()

    # ── 内部变换 ──────────────────────────────────────────

    def _apply_transform(self):
        if self._image_array is None:
            return

        h, w = self._image_array.shape[:2]
        vw = self.viewport().width()
        vh = self.viewport().height()
        if vw < 2 or vh < 2:
            return

        # Save current proportional scroll positions before reset
        h_bar = self.horizontalScrollBar()
        v_bar = self.verticalScrollBar()
        old_h_max = max(h_bar.maximum(), 1)
        old_v_max = max(v_bar.maximum(), 1)
        h_frac = h_bar.value() / old_h_max
        v_frac = v_bar.value() / old_v_max

        self.resetTransform()

        base = min(vw / w, vh / h, 1.0)
        scale = base * self._zoom
        self.scale(scale, scale)

        # Restore proportional scroll positions
        new_h_max = max(h_bar.maximum(), 1)
        new_v_max = max(v_bar.maximum(), 1)
        h_bar.setValue(int(h_frac * new_h_max))
        v_bar.setValue(int(v_frac * new_v_max))

        self._update_zoom_label()

    def _update_zoom_label(self):
        self._zoom_label.setText(f"  {int(self._zoom * 100)}%  ")
        self._zoom_label.adjustSize()
        self._update_zoom_label_pos()

    def _update_zoom_label_pos(self):
        vw = self.viewport().width()
        vh = self.viewport().height()
        lw = self._zoom_label.width()
        lh = self._zoom_label.height()
        self._zoom_label.move(vw - lw - 6, vh - lh - 4)

    # ── 事件处理 ──────────────────────────────────────────

    def wheelEvent(self, event):
        factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        self._zoom = max(self._MIN_ZOOM, min(self._MAX_ZOOM, self._zoom * factor))
        self._apply_transform()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._zoom > 1.01:
            self._panning = True
            self._last_pan_pos = event.position()
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and self._last_pan_pos is not None:
            delta = event.position() - self._last_pan_pos
            self._last_pan_pos = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._panning:
            self._panning = False
            self._last_pan_pos = None
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.reset_zoom()
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_transform()
        self._update_zoom_label_pos()
