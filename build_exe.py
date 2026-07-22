"""Build standalone executables for decolor-mask using PyInstaller.

Requires: pip install pyinstaller

Output:
    dist/decolor-mask.exe      CLI tool (console)
    dist/decolor-mask-gui.exe  GUI tool (no console)
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
        str(ROOT / entry),
    ]
    if noconsole:
        cmd.insert(3, "--noconsole")
    print(f"\n{'='*60}")
    print(f"Building {name}.exe ...")
    print(f"{'='*60}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main():
    build("decolor-mask", "decolor_mask/cli.py", noconsole=False)
    build("decolor-mask-gui", "decolor_mask/ui.py", noconsole=True)
    print("\nBuild complete! Output in dist/")
    print(f"  {ROOT / 'dist' / 'decolor-mask.exe'}")
    print(f"  {ROOT / 'dist' / 'decolor-mask-gui.exe'}")


if __name__ == "__main__":
    main()
