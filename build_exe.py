"""Build standalone executables for isl-toolkit using PyInstaller.

Requires: pip install pyinstaller pyopencl

The resulting .exe bundles pyopencl — no Python/pip install needed on target machine.
GPU acceleration uses the system's OpenCL.dll (provided by GPU driver).
If no GPU driver found, falls back to CPU automatically.

Output:
    dist/isl-toolkit.exe      CLI tool (console)
    dist/isl-toolkit-gui.exe  GUI tool (no console)
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build(name, entry, noconsole=False):
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", name,
        "--distpath", str(ROOT / "dist"),
        "--paths", str(ROOT),
        "--hidden-import", "pyopencl",
        "--hidden-import", "pyopencl._cl",
        "--hidden-import", "pyopencl.cache",
        str(ROOT / entry),
    ]
    if noconsole:
        cmd.insert(3, "--noconsole")
    print(f"\n{'='*60}")
    print(f"Building {name}.exe ...")
    print(f"{'='*60}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main():
    build("isl-toolkit", "entry_cli.py", noconsole=False)
    build("isl-toolkit-gui", "entry_gui.py", noconsole=True)
    print("\nBuild complete! Output in dist/")
    print(f"  {ROOT / 'dist' / 'isl-toolkit.exe'}")
    print(f"  {ROOT / 'dist' / 'isl-toolkit-gui.exe'}")


if __name__ == "__main__":
    main()
