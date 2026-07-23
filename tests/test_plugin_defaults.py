"""Test GUI default enabled states of all four plugins.

Verifies:
- ImageEnhancePlugin  → enabled by default (setChecked(True))
- FilterPipelinePlugin → enabled by default (setChecked(True))
- CrossProcessPlugin  → disabled by default (setChecked(False))
- FilmInversionPlugin → disabled by default (setChecked(False))
"""

import sys
import unittest

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout

from decolor_mask.plugins.image_enhance import ImageEnhancePlugin
from decolor_mask.plugins.filter_pipeline import FilterPipelinePlugin
from decolor_mask.plugins.cross_process import CrossProcessPlugin
from decolor_mask.plugins.film_inversion import FilmInversionPlugin


class TestPluginDefaultStates(unittest.TestCase):
    """Verify each plugin's default enabled/disabled state after attach_ui."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    # ── enhance ──

    def test_enhance_enabled_by_default(self):
        parent = QWidget()
        parent.setLayout(QVBoxLayout())
        plugin = ImageEnhancePlugin()
        plugin.attach_ui(parent)
        self.assertTrue(plugin.is_enabled(),
                        "ImageEnhancePlugin should be enabled by default")

    # ── filters ──

    def test_filters_enabled_by_default(self):
        parent = QWidget()
        parent.setLayout(QVBoxLayout())
        plugin = FilterPipelinePlugin()
        plugin.attach_ui(parent)
        self.assertTrue(plugin.is_enabled(),
                        "FilterPipelinePlugin should be enabled by default")

    # ── cross ──

    def test_cross_disabled_by_default(self):
        parent = QWidget()
        parent.setLayout(QVBoxLayout())
        plugin = CrossProcessPlugin()
        plugin.attach_ui(parent)
        self.assertFalse(plugin.is_enabled(),
                         "CrossProcessPlugin should be disabled by default")

    # ── film ──

    def test_film_disabled_by_default(self):
        parent = QWidget()
        parent.setLayout(QVBoxLayout())
        plugin = FilmInversionPlugin()
        plugin.attach_ui(parent)
        self.assertFalse(plugin.is_enabled(),
                         "FilmInversionPlugin should be disabled by default")


if __name__ == "__main__":
    unittest.main()
