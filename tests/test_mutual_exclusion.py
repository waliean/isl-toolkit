"""Test GUI cross/film mutual exclusion toggle behavior.

Verifies:
- Initially both cross and film are disabled.
- Enabling cross then film: cross gets auto-disabled.
- Enabling film then cross: film gets auto-disabled.

Uses the same signal-slot mutual exclusion pattern as
ImageToolkitApp._setup_mutual_exclusion(), tested at the plugin
level to avoid full app instantiation.
"""

import sys
import unittest

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout

from decolor_mask.plugins.cross_process import CrossProcessPlugin
from decolor_mask.plugins.film_inversion import FilmInversionPlugin


class TestMutualExclusion(unittest.TestCase):
    """Verify cross/film mutual exclusion through checkbox toggle signals."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        # ── Create fresh plugin pair, attach UI ──
        # Keep parent alive as instance attr so child C++ widgets survive.
        self._parent = QWidget()
        self._parent.setLayout(QVBoxLayout())

        self._cross = CrossProcessPlugin()
        self._film = FilmInversionPlugin()
        self._cross.attach_ui(self._parent)
        self._film.attach_ui(self._parent)

        # ── Wire mutual exclusion (mirrors app._setup_mutual_exclusion) ──
        self._mutual_lock = False

        def _on_cross_changed(checked):
            if self._mutual_lock:
                return
            if checked and self._film.is_enabled():
                self._mutual_lock = True
                self._film.set_enabled(False)
                self._mutual_lock = False

        def _on_film_changed(checked):
            if self._mutual_lock:
                return
            if checked and self._cross.is_enabled():
                self._mutual_lock = True
                self._cross.set_enabled(False)
                self._mutual_lock = False

        self._cross._enabled_cb.toggled.connect(_on_cross_changed)
        self._film._enabled_cb.toggled.connect(_on_film_changed)

    # ── test 1: initial state ──────────────────────────────────────

    def test_initial_both_disabled(self):
        """Both cross and film are disabled before any user action."""
        self.assertFalse(self._cross.is_enabled(),
                         "cross should be disabled by default")
        self.assertFalse(self._film.is_enabled(),
                         "film should be disabled by default")

    # ── test 2: cross first, then film → cross auto-off ──

    def test_cross_first_then_film_disables_cross(self):
        """Enabling cross, then enabling film, must auto-disable cross."""
        self._cross.set_enabled(True)
        self.assertTrue(self._cross.is_enabled(),
                        "cross should be enabled after set_enabled(True)")
        self.assertFalse(self._film.is_enabled(),
                         "film should still be disabled")

        self._film.set_enabled(True)
        self.assertTrue(self._film.is_enabled(),
                        "film should now be enabled")
        self.assertFalse(self._cross.is_enabled(),
                         "cross should have been auto-disabled by film toggle")

    # ── test 3: film first, then cross → film auto-off ──

    def test_film_first_then_cross_disables_film(self):
        """Enabling film, then enabling cross, must auto-disable film."""
        self._film.set_enabled(True)
        self.assertTrue(self._film.is_enabled(),
                        "film should be enabled after set_enabled(True)")
        self.assertFalse(self._cross.is_enabled(),
                         "cross should still be disabled")

        self._cross.set_enabled(True)
        self.assertTrue(self._cross.is_enabled(),
                        "cross should now be enabled")
        self.assertFalse(self._film.is_enabled(),
                         "film should have been auto-disabled by cross toggle")


if __name__ == "__main__":
    unittest.main()
