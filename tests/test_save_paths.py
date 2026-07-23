"""Test save-path logic: cross post_pipeline, filters-only wb_mode, batch fallback.

Verifies:
- Cross mode calls process_raw with a post_pipeline containing both
  filters and enhance pipe filters (non-empty).
- Filters-only path does NOT pick up cross's wb_mode (stays "auto").
- Batch else branch (enhance-only / empty pipeline fallback) passes
  wb_mode="auto", strength=1.0 to process_raw.

Each test creates a fresh ImageToolkitApp with fresh plugin instances
to avoid singleton plugin registry contamination.
"""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from decolor_mask.plugins.cross_process import CrossProcessPlugin
from decolor_mask.plugins.film_inversion import FilmInversionPlugin
from decolor_mask.plugins.image_enhance import ImageEnhancePlugin
from decolor_mask.plugins.filter_pipeline import FilterPipelinePlugin

# ── helpers ──────────────────────────────────────────────────────────

def _fresh_plugins():
    """Create fresh plugin instances for test isolation."""
    return [
        CrossProcessPlugin(),
        FilmInversionPlugin(),
        ImageEnhancePlugin(),
        FilterPipelinePlugin(),
    ]


class _SyncThread:
    """Drop-in for threading.Thread that runs target() synchronously."""

    def __init__(self, target, daemon=False):
        self._target = target

    def start(self):
        self._target()


# ── test case ────────────────────────────────────────────────────────

class TestSavePaths(unittest.TestCase):
    """Integration-style tests for _process_current / _process_batch logic."""

    @classmethod
    def setUpClass(cls):
        cls._qt_app = QApplication.instance() or QApplication(sys.argv)
        QApplication.setQuitOnLastWindowClosed(False)

    def setUp(self):
        # recorded process_raw calls
        self._process_raw_calls: list[dict] = []
        self._patches = []

        # -- 1. mock process_raw --
        def _mock_process_raw(input_path, output_path=None, *,
                              wb_mode="auto", strength=0.8,
                              pipeline=None, post_pipeline=None,
                              brightness=1.0, contrast=1.0,
                              saturation=1.0, half_size=False):
            self._process_raw_calls.append(dict(
                wb_mode=wb_mode,
                strength=strength,
                pipeline=pipeline,
                post_pipeline=post_pipeline,
            ))
            return np.zeros((10, 10, 3), dtype=np.float32)

        self._patches.append(patch(
            "decolor_mask.ui.app.process_raw", side_effect=_mock_process_raw))

        # -- 2. sync thread --
        self._patches.append(patch("threading.Thread", new=_SyncThread))

        # -- 3. suppress delayed init & popups --
        self._patches.append(patch.object(QTimer, "singleShot", return_value=None))
        self._patches.append(patch.object(QMessageBox, "information", return_value=None))
        self._patches.append(patch.object(QMessageBox, "critical", return_value=None))
        self._patches.append(patch.object(QMessageBox, "warning", return_value=None))
        self._patches.append(patch("decolor_mask.ui.app.session_load", return_value={}))
        self._patches.append(patch("os.makedirs"))

        # -- 4. isolate from singleton plugin registry --
        self._fresh = _fresh_plugins()
        self._patches.append(patch(
            "decolor_mask.ui.app.list_all", return_value=self._fresh))

        for p in self._patches:
            p.start()

        # -- create app *after* patches are active --
        from decolor_mask.ui.app import ImageToolkitApp, STATE_READY
        self.ImageToolkitApp = ImageToolkitApp
        self.STATE_READY = STATE_READY
        self.app = ImageToolkitApp()

        # convenience refs to plugin instances
        self._cross = self.app._plugin_map["cross"]
        self._film = self.app._plugin_map["film"]
        self._filters = self.app._plugin_map["filters"]
        self._enhance = self.app._plugin_map["enhance"]

        # ensure baseline: only enhance + filters on, cross & film off
        self._cross.set_enabled(False)
        self._film.set_enabled(False)
        self._filters.set_enabled(True)
        self._enhance.set_enabled(True)

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        self._patches.clear()
        self._process_raw_calls.clear()

    def _make_ready(self, preview_path="fake.dng", raw_files=None):
        """Put app into READY state with a fake preview path and raw_files list."""
        self.app._state = self.STATE_READY
        self.app._preview_path = preview_path
        self.app.raw_files = raw_files or [preview_path]
        self.app._file_index = 0

    # ── test 2: cross mode carries filters + enhance in post_pipeline ──

    def test_cross_save_includes_filters_and_enhance_in_post_pipeline(self):
        """When cross is enabled + filters + enhance, process_raw gets a
        non-empty post_pipeline containing filter stages from both."""
        self._cross.set_enabled(True)
        # give filters some active params
        self._filters.get_params  = lambda: {
            "chroma_nr": 0.5, "dehaze": 0.3,
            "bw_filter": "red",
            "band_nr": 0.0, "flat_field_strength": 0.0,
            "grain": 0.0, "bw_filter_strength": 1.0, "toning": "none",
        }
        # give enhance some active params
        self._enhance.get_params = lambda: {
            "clarity": 0.5, "smart_sharp": 0.3,
            "brightness": 1.0, "contrast": 1.0, "saturation": 1.0,
            "vignette": 0.0, "shadow_boost": 0.0,
            "highlights": 0.0, "shadows": 0.0, "midtones": 0.0,
        }

        self._make_ready()
        self.app._process_current()

        self.assertGreaterEqual(len(self._process_raw_calls), 1,
                                "process_raw should have been called at least once")
        call = self._process_raw_calls[0]
        post = call["post_pipeline"]
        self.assertIsNotNone(post, "post_pipeline must not be None in cross mode")

        # collect filter names from all stages
        from decolor_mask.kernel.filters import FilterStage
        filter_names = set()
        for stage in (FilterStage.FRONTEND, FilterStage.CORE, FilterStage.BACKEND):
            for f in post.get_stage(stage):
                filter_names.add(f.name)

        # should contain filters from both filters and enhance plugins
        self.assertIn("CromaNR", filter_names,
                      "post_pipeline should contain CromaNR from filters")
        self.assertIn("Dehaze", filter_names,
                      "post_pipeline should contain Dehaze from filters")
        self.assertIn("Clarity", filter_names,
                      "post_pipeline should contain Clarity from enhance")
        self.assertIn("SmartSharp", filter_names,
                      "post_pipeline should contain SmartSharp from enhance")

    # ── test 3: filters-only does not read cross wb_mode ──────────────

    def test_filters_only_ignores_cross_wb_mode(self):
        """When cross is NOT enabled but its wb_mode is 'camera',
        filters-only save path still passes wb_mode='auto' to process_raw."""
        # cross stays disabled but its dropdown shows "camera"
        self._cross._wb_var.set("camera")

        self._make_ready()
        self.app._process_current()

        self.assertGreaterEqual(len(self._process_raw_calls), 1)
        call = self._process_raw_calls[0]
        self.assertEqual(call["wb_mode"], "auto",
                         "filters-only path should use wb_mode='auto', "
                         "not cross's 'camera'")
        # pipeline exists because filters is enabled
        self.assertIsNotNone(call["pipeline"],
                             "filters-only path should have a pipeline")

    # ── test 4: batch else path uses auto baseline ────────────────────

    def test_batch_fallback_uses_auto_baseline(self):
        """Batch else path (enhance-only / empty pipeline) calls process_raw
        with wb_mode='auto' and strength=1.0."""
        # disable cross, film, filters – only enhance remains
        self._filters.set_enabled(False)
        self._cross.set_enabled(False)
        self._film.set_enabled(False)
        self._enhance.set_enabled(True)

        self._make_ready(raw_files=["fake1.dng"])
        self.app._folder_edit.setText(".")  # prevents empty folder path
        self.app._process_batch()

        self.assertGreaterEqual(len(self._process_raw_calls), 1,
                                "batch processing should call process_raw")
        call = self._process_raw_calls[0]
        self.assertEqual(call["wb_mode"], "auto",
                         "batch else path must use wb_mode='auto'")
        self.assertEqual(call["strength"], 1.0,
                         "batch else path must use strength=1.0")


if __name__ == "__main__":
    unittest.main()
