# ISL Toolkit — 图像处理工具箱

基于 Pentax DCU 5 / ISL (Ichikawa Soft Laboratory) 引擎架构的通用图像处理工具。

> 内核 + 插件架构，当前内置 3 个功能模块，支持扩展。

## 功能模块

| 模块 | 说明 | 核心技术 |
|------|------|----------|
| **正负逆冲** | RAW负片反转 | LCC空间反转 + GrayAxis片基检测 + WB混合 |
| **图像增强** | 亮度/对比度/去雾 | 暗通道先验去雾 + 色调映射 |
| **降噪处理** | 色度降噪 + 频带降噪 | CromaNR (保护亮度) + BandNR (金字塔分解) |

## 架构

```
isl-toolkit/
├── kernel/         核心引擎（Pipeline/Filter基础设施/色彩空间）
├── plugins/        功能插件（film_inversion / image_enhance / noise_reduction）
└── ui/             模块化界面（标签页式）
```

基于 DCU/ISL 逆向学习的设计原则：
- **FrontEnd → Core → BackEnd** 三层滤镜管线
- **预览/正式分离** 低分辨率预览即时响应
- **亮度/色度分离** 降噪只动色度，保护细节
- **插件可扩展** 继承 `PluginBase` 即可新增功能

## 安装

```bash
git clone https://github.com/waliean/positive-negative-reverse.git
cd positive-negative-reverse
pip install -e .
```

## 使用

### GUI

```bash
isl-toolkit-gui
# 或 python -m decolor_mask.ui
```

### CLI

```bash
# 负片反转（向下兼容）
decolor-mask ./photos --strength 0.8 --wb auto

# 完整管线
decolor-mask . --pipeline --chroma-nr 0.3 --dehaze 0.2

# 新命令
isl-toolkit ./photos --strength 0.8
```

### 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--strength` | 反转强度 0.0–1.0 | 0.8 |
| `--wb` | 白平衡: auto/camera/daylight | auto |
| `--brightness` | 亮度倍率 | 1.0 |
| `--contrast` | 对比度倍率 | 1.0 |
| `--saturation` | 饱和度倍率 | 1.0 |
| `--pipeline` | 启用完整滤镜管线 | - |
| `--chroma-nr` | 色度降噪 0.0–1.0 | 0.0 |
| `--band-nr` | 频带降噪 0.0–1.0 | 0.0 |
| `--dehaze` | 去雾 0.0–1.0 | 0.0 |
| `--flat-field` | 白帧参考路径 | - |
| `-r` | 递归子文件夹 | - |

## 依赖

- Python >= 3.10
- numpy >= 1.21.0
- Pillow >= 9.0.0
- rawpy >= 0.17.0（RAW文件支持）

## License

MIT
