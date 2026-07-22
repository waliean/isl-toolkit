# 正负逆冲

去除胶片负片扫描中的橙色色罩（Color Mask），恢复真实色彩。

## 原理

胶片负片扫描图像带有橙色的片基色罩（C-41 工艺的橙色色罩）。本工具通过分析图像中最暗像素（未曝光的片基区域）或图像边框区域来估算色罩颜色，然后在 sRGB 线性空间中除以该色罩，再应用亮度、对比度、饱和度调整，最终还原自然色彩。

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

### 单张处理

```bash
# 自动检测色罩（默认）
python -m decolor_mask.cli scan.jpg output.jpg

# 边框分析模式（适合有边框的扫描件）
python -m decolor_mask.cli scan.jpg output.jpg --mode border

# 手动指定色罩颜色（RGB，范围 0-1）
python -m decolor_mask.cli scan.jpg output.jpg --mode manual --mask 0.85 0.55 0.28

# 仅检测色罩颜色，不处理
python -m decolor_mask.cli scan.jpg --detect-only

# 调整输出效果
python -m decolor_mask.cli scan.jpg output.jpg -b 1.1 -c 1.05 -s 1.2
```

### 批量处理

```bash
# 使用 glob 模式
python -m decolor_mask.cli "scans/*.jpg" output/ --batch

# 递归匹配
python -m decolor_mask.cli "scans/**/*.tif" output/ --batch
```

### 假如是"正负逆冲"风格的数码照片

```bash
# 不加反转，直接去除色罩
python -m decolor_mask.cli photo.jpg output.jpg --type digital
```

## 参数说明

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--type` | `-t` | 处理类型：`negative`（反转+去色罩）或 `digital`（仅去色罩） | `negative` |
| `--mode` | `-m` | 色罩检测模式：`auto`（暗像素分析）、`border`（边框分析）、`manual`（手动指定） | `auto` |
| `--mask` | | 手动色罩 RGB 值（0-1），配合 `--mode manual` 使用 | - |
| `--border-size` | | 边框分析区域占比 | `0.05` |
| `--brightness` | `-b` | 亮度乘数 | `1.0` |
| `--contrast` | `-c` | 对比度乘数 | `1.0` |
| `--saturation` | `-s` | 饱和度乘数 | `1.0` |
| `--detect-only` | | 仅检测色罩颜色 | - |
| `--batch` | | 批量处理模式 | - |
| `--verbose` | `-v` | 详细输出 | - |
| `--quiet` | `-q` | 静默输出 | - |

## 生成测试图片

```bash
python gen_test_images.py
```

会生成 `test_original.png`（原始参考图）和 `test_negative_scan.png`（模拟负片扫描图），用于验证工具效果。

## 依赖

- Python >= 3.8
- numpy >= 1.21.0
- Pillow >= 9.0.0

## License

MIT
