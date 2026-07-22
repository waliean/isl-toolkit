"""一键启动 — 无参数启动 GUI，带参数启动 CLI。

用法：
    双击 run.py            → 打开 GUI 图形界面
    python run.py photo.jpg output.jpg --strength 0.6  → 命令行处理
"""
import sys


def main():
    if len(sys.argv) > 1:
        from decolor_mask.cli import main as cli_main
        cli_main()
    else:
        from decolor_mask.ui import main as ui_main
        ui_main()


if __name__ == "__main__":
    main()
