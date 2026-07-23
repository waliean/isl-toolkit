"""会话记忆 — 记住上次打开的文件夹和图片路径。"""
import json
import os
from pathlib import Path

_SETTINGS_DIR = Path(os.environ.get("APPDATA", Path.home())) / "isl-toolkit"
_SETTINGS_FILE = _SETTINGS_DIR / "session.json"


def _ensure_dir():
    _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)


def load() -> dict:
    try:
        if _SETTINGS_FILE.exists():
            with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def save(folder: str = None, path: str = None, index: int = None, theme: str = None):
    data = load()
    if folder is not None:
        data["last_folder"] = folder
    if path is not None:
        data["last_path"] = path
    if index is not None:
        data["last_index"] = index
    if theme is not None:
        data["theme"] = theme
    _ensure_dir()
    with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
