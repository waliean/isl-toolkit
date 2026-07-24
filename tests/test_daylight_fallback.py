"""Test shared daylight fallback and base-image consistency.

Covers:
- _render_raw_from_obj fallback path on daylight_whitebalance AttributeError
- _compute_chain default base = tgt (daylight) when cross/film disabled
- process_raw uses shared _render_raw_from_obj consistently
"""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np


class TestRenderRawFromObj(unittest.TestCase):
    """_render_raw_from_obj daylight fallback and parameter sharing."""

    def setUp(self):
        # Build a minimal rawpy mock with postprocess returning known data
        self._mock_raw = MagicMock()
        self._mock_raw.postprocess.return_value = np.full(
            (10, 10, 3), 128, dtype=np.uint8)
    def test_camera_mode_calls_postprocess_with_camera_wb(self):
        from decolor_mask.core import _render_raw_from_obj
        result = _render_raw_from_obj(self._mock_raw, "camera")
        self.assertEqual(result.shape, (10, 10, 3))
        self.assertEqual(result.dtype, np.float32)
        self._mock_raw.postprocess.assert_called_once()
        call_kwargs = self._mock_raw.postprocess.call_args[1]
        self.assertTrue(call_kwargs.get("use_camera_wb"))

    def test_auto_mode_calls_postprocess_with_auto_wb(self):
        from decolor_mask.core import _render_raw_from_obj
        result = _render_raw_from_obj(self._mock_raw, "auto")
        self._mock_raw.postprocess.assert_called_once()
        call_kwargs = self._mock_raw.postprocess.call_args[1]
        self.assertFalse(call_kwargs.get("use_camera_wb", True))
        self.assertTrue(call_kwargs.get("use_auto_wb"))

    def test_daylight_mode_uses_daylight_whitebalance(self):
        self._mock_raw.daylight_whitebalance = (1.5, 1.0, 1.2, 1.0)
        from decolor_mask.core import _render_raw_from_obj
        result = _render_raw_from_obj(self._mock_raw, "daylight")
        self._mock_raw.postprocess.assert_called_once()
        call_kwargs = self._mock_raw.postprocess.call_args[1]
        self.assertIn("user_wb", call_kwargs)

    def test_daylight_fallback_on_attribute_error(self):
        """When daylight_whitebalance raises AttributeError, fallback to auto."""
        from decolor_mask.core import _render_raw_from_obj

        # Build a plain object that has postprocess but NO daylight_whitebalance
        class FakeRaw:
            daylight_whitebalance = property()  # descriptor exists but no getter → AttributeError

            def postprocess(self, **kwargs):
                FakeRaw._last_call_kwargs = kwargs
                return np.full((10, 10, 3), 128, dtype=np.uint8)

        fake_raw = FakeRaw()
        result = _render_raw_from_obj(fake_raw, "daylight")
        call_kwargs = FakeRaw._last_call_kwargs
        self.assertFalse(call_kwargs.get("use_camera_wb", True))
        self.assertTrue(call_kwargs.get("use_auto_wb"))

    def test_unknown_wb_mode_raises(self):
        from decolor_mask.core import _render_raw_from_obj
        with self.assertRaises(ValueError):
            _render_raw_from_obj(self._mock_raw, "invalid_wb")

    def test_half_size_flag_passed(self):
        from decolor_mask.core import _render_raw_from_obj
        _render_raw_from_obj(self._mock_raw, "camera", half_size=True)
        call_kwargs = self._mock_raw.postprocess.call_args[1]
        self.assertTrue(call_kwargs.get("half_size"))

    def test_returns_float32_zero_one(self):
        from decolor_mask.core import _render_raw_from_obj
        result = _render_raw_from_obj(self._mock_raw, "camera")
        self.assertEqual(result.dtype, np.float32)
        self.assertGreaterEqual(result.min(), 0.0)
        self.assertLessEqual(result.max(), 1.0)


class TestComputeChainBaseImage(unittest.TestCase):
    """_compute_chain uses daylight (tgt) as base when cross/film disabled."""

    def _make_app_mock(self, cross_enabled=False, film_enabled=False,
                       filters_enabled=False, enhance_enabled=False,
                       cross_params=None, film_params=None,
                       filters_params=None, enhance_params=None):
        """Build a mock ImageToolkitApp with plugin map and compute_chain."""
        from unittest.mock import MagicMock

        app = MagicMock()

        # Plugin mocks
        def _mk_plugin(enabled):
            p = MagicMock()
            p.is_enabled.return_value = enabled
            p.get_params.return_value = {}
            return p

        cross = _mk_plugin(cross_enabled)
        film = _mk_plugin(film_enabled)
        filters = _mk_plugin(filters_enabled)
        enhance = _mk_plugin(enhance_enabled)

        if cross_params:
            cross.get_params.return_value = cross_params
        if film_params:
            film.get_params.return_value = film_params
        if filters_params:
            filters.get_params.return_value = filters_params
        if enhance_params:
            enhance.get_params.return_value = enhance_params

        app._plugin_map = {
            "cross": cross,
            "film": film,
            "filters": filters,
            "enhance": enhance,
        }
        # Mock builders to return empty pipelines
        from decolor_mask.kernel import ProcessingPipeline
        app._build_save_pipeline.return_value = ProcessingPipeline(preview_scale=1.0)
        app._build_enhance_save_pipeline.return_value = ProcessingPipeline(preview_scale=1.0)

        return app

    def test_no_plugins_uses_tgt_as_base(self):
        """Without cross/film, base should be tgt (daylight), not cam."""
        from decolor_mask.ui.app import ImageToolkitApp
        # We can't instantiate the full app, so we bind the method manually
        app = self._make_app_mock()

        cam = np.full((8, 8, 3), 0.3, dtype=np.float32) / 255.0  # dark
        tgt = np.full((8, 8, 3), 0.7, dtype=np.float32) / 255.0  # bright

        # Use _compute_chain via unbound method
        cam_float = cam.astype(np.float32) / 255.0
        tgt_float = tgt.astype(np.float32) / 255.0

        # Bind the real _compute_chain to our mock app
        result = ImageToolkitApp._compute_chain(app, cam, tgt)
        # result should be close to tgt, not cam
        mean = result.mean()
        cam_mean = cam.astype(np.float32).mean() / 255.0
        tgt_mean = tgt.astype(np.float32).mean() / 255.0
        self.assertAlmostEqual(mean, tgt_mean, delta=0.05,
                               msg="non-cross/film should use tgt as base")
        self.assertGreater(abs(mean - cam_mean), abs(mean - tgt_mean) * 0.5,
                           msg="result should be closer to tgt than cam")

    def test_filters_only_uses_tgt_as_base(self):
        """Filters-only without cross/film: apply filters on daylight base."""
        from decolor_mask.ui.app import ImageToolkitApp

        app = self._make_app_mock(
            cross_enabled=False, film_enabled=False,
            filters_enabled=True, enhance_enabled=False,
            filters_params={"chroma_nr": 0.0, "band_nr": 0.0,
                            "flat_field_strength": 0.0, "bw_filter": "none",
                            "dehaze": 0.0, "grain": 0.0, "toning": "none"},
        )
        cam = np.full((8, 8, 3), 100, dtype=np.uint8)
        tgt = np.full((8, 8, 3), 200, dtype=np.uint8)

        result = ImageToolkitApp._compute_chain(app, cam, tgt)
        tgt_mean = tgt.astype(np.float32).mean() / 255.0
        self.assertAlmostEqual(result.mean(), tgt_mean, delta=0.05,
                               msg="filters-only should use daylight as base")

    def test_enhance_only_uses_tgt_as_base(self):
        """Enhance-only without cross/film: apply enhance on daylight base."""
        from decolor_mask.ui.app import ImageToolkitApp

        app = self._make_app_mock(
            cross_enabled=False, film_enabled=False,
            filters_enabled=False, enhance_enabled=True,
            enhance_params={"clarity": 0.0, "smart_sharp": 0.0,
                            "vignette": 0.0, "shadow_boost": 0.0,
                            "highlights": 0.0, "shadows": 0.0, "midtones": 0.0,
                            "brightness": 1.0, "contrast": 1.0, "saturation": 1.0},
        )
        cam = np.full((8, 8, 3), 100, dtype=np.uint8)
        tgt = np.full((8, 8, 3), 200, dtype=np.uint8)

        result = ImageToolkitApp._compute_chain(app, cam, tgt)
        tgt_mean = tgt.astype(np.float32).mean() / 255.0
        self.assertAlmostEqual(result.mean(), tgt_mean, delta=0.05,
                               msg="enhance-only should use daylight as base")

    def test_cross_mode_still_blends_cam_tgt(self):
        """Cross mode should still blend cam and tgt, not just use tgt."""
        from decolor_mask.ui.app import ImageToolkitApp

        app = self._make_app_mock(
            cross_enabled=True, film_enabled=False,
            filters_enabled=False, enhance_enabled=False,
            cross_params={"strength": 0.5, "wb_mode": "daylight"},
        )
        cam = np.full((8, 8, 3), 100, dtype=np.uint8)
        tgt = np.full((8, 8, 3), 200, dtype=np.uint8)

        result = ImageToolkitApp._compute_chain(app, cam, tgt)
        cam_mean = cam.astype(np.float32).mean() / 255.0
        tgt_mean = tgt.astype(np.float32).mean() / 255.0
        expected_mean = cam_mean * 0.5 + tgt_mean * 0.5
        self.assertAlmostEqual(result.mean(), expected_mean, delta=0.05,
                               msg="cross mode should blend cam and tgt")


class TestProcessRawDaylightDefault(unittest.TestCase):
    """process_raw defaults and pipeline routing."""

    def test_process_raw_default_wb_is_daylight(self):
        """process_raw defaults to wb_mode='daylight'."""
        import inspect
        from decolor_mask.core import process_raw
        sig = inspect.signature(process_raw)
        self.assertEqual(sig.parameters["wb_mode"].default, "daylight")

    def test_process_raw_default_strength(self):
        """process_raw defaults to strength=0.8."""
        import inspect
        from decolor_mask.core import process_raw
        sig = inspect.signature(process_raw)
        self.assertEqual(sig.parameters["strength"].default, 0.8)


if __name__ == "__main__":
    unittest.main()
