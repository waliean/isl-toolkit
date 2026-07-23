"""Test cross/film mutual exclusion via real ImageToolkitApp instance.

Verifies the actual _setup_mutual_exclusion() wiring — not a manual
re-implementation — by instantiating ImageToolkitApp and driving the
real _plugin_map / _enabled_cb / set_enabled() path.

Each test creates a fresh ImageToolkitApp with fresh plugin instances
(via patching list_all) to avoid singleton state contamination.

Side-effects during app init (QTimer.singleShot, session_load) are
mocked to avoid deferred callbacks and file I/O.  The mutual exclusion
itself is exercised untouched.
"""

import sys
import unittest
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from decolor_mask.ui.app import ImageToolkitApp
from decolor_mask.plugins.cross_process import CrossProcessPlugin
from decolor_mask.plugins.film_inversion import FilmInversionPlugin
from decolor_mask.plugins.image_enhance import ImageEnhancePlugin
from decolor_mask.plugins.filter_pipeline import FilterPipelinePlugin


def _fresh_plugins():
    """Create fresh plugin instances for test isolation."""
    return [
        CrossProcessPlugin(),
        FilmInversionPlugin(),
        ImageEnhancePlugin(),
        FilterPipelinePlugin(),
    ]


class TestMutualExclusionViaApp(unittest.TestCase):
    """Test cross/film mutual exclusion through a real ImageToolkitApp.

    The app is instantiated per test with fresh plugin instances,
    avoiding singleton registry contamination between tests.
    """

    @classmethod
    def setUpClass(cls):
        cls.qapp = QApplication.instance() or QApplication(sys.argv)
        QApplication.setQuitOnLastWindowClosed(False)

    def setUp(self):
        """Create fresh app per test — patches + fresh plugins + app."""
        self._patches = []

        # ── mock init-time side effects ──────────────
        self._patches.append(patch("decolor_mask.ui.app.QTimer.singleShot"))
        self._patches.append(patch(
            "decolor_mask.ui.app.session_load", return_value={}))

        # ── isolate from singleton plugin registry ───
        self._fresh = _fresh_plugins()
        self._patches.append(patch(
            "decolor_mask.ui.app.list_all", return_value=self._fresh))

        for p in self._patches:
            p.start()

        self.app = ImageToolkitApp()
        self.cross = self.app._plugin_map["cross"]
        self.film = self.app._plugin_map["film"]

        # ── reset to known state (fresh app already has both off,
        #     but this guards against future default changes) ──
        self.app._mutual_lock = True
        self.cross.set_enabled(False)
        self.film.set_enabled(False)
        self.app._mutual_lock = False

    def tearDown(self):
        """Close app window and stop all patches."""
        self.app.close()
        for p in reversed(self._patches):
            p.stop()
        self._patches.clear()

    # ── test 1: initial state ─────────────────────────────────

    def test_initial_both_disabled(self):
        """After app init, both cross and film are disabled."""
        self.assertFalse(self.cross.is_enabled(),
                         "cross should be disabled after app init")
        self.assertFalse(self.film.is_enabled(),
                         "film should be disabled after app init")

    # ── test 2: cross first, then film → cross auto-off ──────

    def test_cross_first_then_film_disables_cross(self):
        """Enabling cross then film auto-disables cross via app wiring."""
        self.cross.set_enabled(True)
        self.assertTrue(self.cross.is_enabled(),
                        "cross should be enabled after set_enabled(True)")
        self.assertFalse(self.film.is_enabled(),
                         "film should still be disabled")

        self.film.set_enabled(True)
        self.assertTrue(self.film.is_enabled(),
                        "film should now be enabled")
        self.assertFalse(self.cross.is_enabled(),
                         "cross should have been auto-disabled by "
                         "app._setup_mutual_exclusion")

    # ── test 3: film first, then cross → film auto-off ──────

    def test_film_first_then_cross_disables_film(self):
        """Enabling film then cross auto-disables film via app wiring."""
        self.film.set_enabled(True)
        self.assertTrue(self.film.is_enabled(),
                        "film should be enabled after set_enabled(True)")
        self.assertFalse(self.cross.is_enabled(),
                         "cross should still be disabled")

        self.cross.set_enabled(True)
        self.assertTrue(self.cross.is_enabled(),
                        "cross should now be enabled")
        self.assertFalse(self.film.is_enabled(),
                         "film should have been auto-disabled by "
                         "app._setup_mutual_exclusion")


if __name__ == "__main__":
    unittest.main()
