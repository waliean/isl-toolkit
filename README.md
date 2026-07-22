# 正负逆冲

修正宾得正负逆冲（Cross Process）风格照片的色偏，使色彩更自然，同时保留模拟胶片质感。

## 原理

宾得相机的「正负逆冲」创意滤镜模拟了正片负冲（Cross Process）效果，会在照片上叠加非均匀的色偏（通常偏暖/偏绿）。本工具通过以下步骤进行轻度纠正：

1. **白平衡检测**：估算场景的标准白点（三种自动方法或手动指定）
2. **线性空间校正**：在 sRGB 线性空间中按比例纠正色偏
3. **强度混合**：用 `strength` 参数控制纠正强度，不完全消除风格特征

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

### GUI 界面（推荐）

```bash
decolor-mask-gui
# 或
python -m decolor_mask.ui
```

- 左侧显示原图，右侧实时预览纠正效果
- 所有参数支持**滑块拖动 + 文本输入**双向调整
- 切换白平衡方法时自动显示/隐藏对应控件
- `自动检测白平衡` 按钮一键估算并填入手动白点参数
- Ctrl+O 打开图像，Ctrl+S 保存结果

### 命令行（单张处理）

```bash
# 自动纠正（灰度世界法，强度 0.6）
python -m decolor_mask.cli photo.jpg output.jpg

# 白块法（亮区为白）
python -m decolor_mask.cli photo.jpg output.jpg --method white_patch

# 百分位法（更稳健）
python -m decolor_mask.cli photo.jpg output.jpg --method percentile --percentile 90

# 手动指定白点（图像中什么颜色应该变白）
python -m decolor_mask.cli photo.jpg output.jpg --method manual --white 0.8 0.7 0.6

# 轻度纠正（保留更多风格）
python -m decolor_mask.cli photo.jpg output.jpg --strength 0.35

# 仅检测白平衡参数
python -m decolor_mask.cli photo.jpg --detect-only

# 调整亮度/对比度/饱和度
python -m decolor_mask.cli photo.jpg output.jpg -b 1.1 -c 1.05 -s 1.15
```

### 批量处理

```bash
python -m decolor_mask.cli "photos/*.jpg" output/ --batch
python -m decolor_mask.cli "photos/**/*.tif" output/ --batch
```

## 参数说明

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--method` | `-m` | 白平衡方法：`gray_world`、`white_patch`、`percentile`、`manual` | `gray_world` |
| `--percentile` | | 百分位法的百分位（0-100） | `95.0` |
| `--white` | | 手动参考白点 RGB 值（0-1），配合 `--method manual` | - |
| `--strength` | | 纠正强度 0.0-1.0。0=不改动，1=完全纠正 | `0.6` |
| `--brightness` | `-b` | 亮度乘数 | `1.0` |
| `--contrast` | `-c` | 对比度乘数 | `1.0` |
| `--saturation` | `-s` | 饱和度乘数 | `1.0` |
| `--detect-only` | | 仅检测白平衡增益 | - |
| `--batch` | | 批量处理模式 | - |
| `--verbose` | `-v` | 详细输出 | - |
| `--quiet` | `-q` | 静默输出 | - |

### 白平衡方法说明

- `gray_world`：假设场景平均色应为中性灰，计算各通道均值 → 增益
- `white_patch`：假设图像最亮像素应为纯白，实用但易受高光干扰
- `percentile`：取最亮 N% 像素的平均值作为白点，比 white_patch 更稳健
- `manual`：手动指定图像中某个区域的颜色作为白点参考

## 生成测试图片

```bash
python gen_test_images.py
```

生成 `test_original.png`（原始参考图）和 `test_negative_scan.png`（模拟正负逆冲风格的测试图）。

## 依赖

- Python >= 3.10
- numpy >= 1.21.0
- Pillow >= 9.0.0

## License

MIT
