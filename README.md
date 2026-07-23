# 正负逆冲

去除照片中任意颜色的色罩（Color Mask），恢复自然色彩。支持自动检测色罩颜色，也支持手动指定。

## 原理

色罩相当于叠加在图像上的一层颜色滤镜。去除色罩的核心操作是：**在线性空间中除以该颜色**。

1. 检测或手动指定色罩颜色（各种方法见下方）
2. 在 sRGB 线性空间中做除法，消去色罩
3. 用 `--strength` 控制去除强度，不完全消除时可保留风格特征

## 安装

```bash
git clone https://github.com/waliean/positive-negative-reverse.git
cd positive-negative-reverse
pip install -e .
```

或者直接安装依赖：

```bash
pip install numpy Pillow
```

## 使用

### 一键启动

```bash
python run.py                          # 无参数→打开 GUI
python run.py photo.jpg output.jpg     # 带参数→走 CLI
```

双击 `run.bat` 或 `run.py` 启动 GUI。

### GUI 界面

```bash
python -m decolor_mask.ui
```

- 左侧原图 / 右侧实时预览
- 参数支持滑块拖动 + 文本输入双向调整
- 自动检测色罩按钮一键估算
- Ctrl+O 打开图像，Ctrl+S 保存结果

### 命令行

```bash
# 自动检测色罩（灰度世界法，默认）
python -m decolor_mask.cli photo.jpg output.jpg

# 白块法
python -m decolor_mask.cli photo.jpg output.jpg --method white_patch

# 百分位法
python -m decolor_mask.cli photo.jpg output.jpg --method percentile --percentile 90

# 暗像素法（适合胶卷扫描件）
python -m decolor_mask.cli scan.jpg output.jpg --method dark_pixel

# 边框法（适合有边框的扫描件）
python -m decolor_mask.cli scan.jpg output.jpg --method border --border-size 0.05

# 手动指定色罩颜色
python -m decolor_mask.cli photo.jpg output.jpg --method manual --mask 0.85 0.55 0.28

# 轻微纠正（保留更多风格）
python -m decolor_mask.cli photo.jpg output.jpg --strength 0.35

# 仅检测色罩颜色
python -m decolor_mask.cli photo.jpg --detect-only --method gray_world

# 批量处理
python -m decolor_mask.cli "photos/*.jpg" output/ --batch
```

## 参数说明

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--method` | `-m` | 检测方法：`gray_world` / `white_patch` / `percentile` / `dark_pixel` / `border` / `manual` | `gray_world` |
| `--mask` | | 手动色罩 RGB (0-1)，配合 `--method manual` | - |
| `--percentile` | | 百分位法的百分位 (0-100) | `95.0` |
| `--border-size` | | 边框法的边框比例 | `0.05` |
| `--strength` | | 去除强度 0.0-1.0，0=不动 1=完全去除 | `0.6` |
| `--brightness` | `-b` | 亮度乘数 | `1.0` |
| `--contrast` | `-c` | 对比度乘数 | `1.0` |
| `--saturation` | `-s` | 饱和度乘数 | `1.0` |
| `--detect-only` | | 仅检测色罩颜色 | - |
| `--batch` | | 批量模式 | - |
| `--verbose` | `-v` | 详细输出 | - |
| `--quiet` | `-q` | 静默输出 | - |

### 检测方法说明

| 方法 | 原理 | 适用场景 |
|------|------|----------|
| `gray_world` | 假设全图平均色=色罩颜色 | 通用，大多数场景 |
| `white_patch` | 假设最亮像素=色罩颜色 | 有高光的场景 |
| `percentile` | 取最亮 N% 像素的均值 | 比 white_patch 更稳健 |
| `dark_pixel` | 取最暗 2% 像素的中位数 | 胶卷扫描件 |
| `border` | 取图像边框区域的中位数 | 带黑框/齿孔的扫描件 |
| `manual` | 手动指定 | 精确控制 |

## 封装版

```bash
# 构建独立的 .exe 文件（无需安装 Python）
pip install pyinstaller
python build_exe.py
# 输出在 dist/decolor-mask.exe 和 dist/decolor-mask-gui.exe
```

## 依赖

- Python >= 3.10
- numpy >= 1.21.0
- Pillow >= 9.0.0

## License

MIT
