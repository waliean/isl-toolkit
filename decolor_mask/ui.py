"""RAW去色罩 GUI — 新架构重定向。

运行: python -m decolor_mask.ui
"""

from .ui.app import ImageToolkitApp
import tkinter as tk


def main():
    root = tk.Tk()
    ImageToolkitApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
