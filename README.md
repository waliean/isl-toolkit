# ISL Toolkit — 图像处理工具箱

> v1.0.0 | [下载封装版](https://github.com/waliean/isl-toolkit/releases)
>
> **v1.0.0 为首个正式版，此前版本均为测试版/预发布。**

基于 Pentax DCU 5 / ISL (Ichikawa Soft Laboratory) 引擎架构的通用图像处理工具 (Image Toolkit)。

> 内核 + 插件架构，当前内置 4 个功能模块，支持扩展。

## 功能模块

| 模块 | 说明 | 核心技术 |
|------|------|----------|
| **正负逆冲 (Cross Process)** | Camera WB + Target WB 按强度混合 | WB混合交叉冲洗 |
| **负片反转 (Film Inversion)** | RAW负片→正片导出 | LCC亮度-色度分离反转 + 平场校正 (FlatField) |
| **图像增强 (Image Enhance)** | 亮度/对比度/饱和度 + 修图 | 智能锐化 (SmartSharp) / 清晰度 (Clarity) / 色调曲线 (Tone Curve) |
| **滤镜处理 (Filter Pipeline)** | 降噪/去雾/平场/颗粒/B&W滤镜 | 色度降噪 (CromaNR) + 频带降噪 (BandNR) + 去雾 (Dehaze) |

## 架构

```
isl-toolkit/
├── kernel/         核心引擎（Pipeline/Filter基础设施/色彩空间）
├── plugins/        功能插件（cross_process / film_inversion / image_enhance / filter_pipeline）
└── ui/             模块化界面（标签页式）
```

基于 DCU/ISL 逆向学习的设计原则：
- **FrontEnd → Core → BackEnd** 三层滤镜管线架构
- **预览/正式导出分离** 低分辨率预览即时响应
- **亮度/色度分离 (LCC)** 降噪只动色度，保护亮度细节
- **插件可扩展** 继承 `PluginBase` 即可新增功能

## 安装

```bash
git clone https://github.com/waliean/isl-toolkit.git
cd isl-toolkit
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
# 正负逆冲 (Cross Process) 模式（默认）
decolor-mask ./photos --strength 0.8 --wb auto

# 负片反转 (Film Inversion) 模式
decolor-mask ./photos --mode invert --strength 0.8

# 滤镜处理 (Filter Pipeline) 模式
decolor-mask . --mode filter --chroma-nr 0.3 --dehaze 0.2

# 通用参数
decolor-mask . --mode invert -b 1.2 -c 1.1  # 亮度+对比度
decolor-mask . -r                              # 递归子文件夹
```

### 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 模式: cross / invert / filter | cross |
| `--strength` | 强度 0.0–1.0 (cross=混合度, invert=反转强度) | 0.8 |
| `--wb` | 目标白平衡 (WB): auto / camera / daylight | auto |
| `--brightness` | 亮度 (Brightness) 倍率 | 1.0 |
| `--contrast` | 对比度 (Contrast) 倍率 | 1.0 |
| `--saturation` | 饱和度 (Saturation) 倍率 | 1.0 |
| `--chroma-nr` | 色度降噪 (CromaNR) 0.0–1.0 (invert/filter) | 0.0 |
| `--band-nr` | 频带降噪 (BandNR) 0.0–1.0 (invert/filter) | 0.0 |
| `--dehaze` | 去雾 (Dehaze) 0.0–1.0 (invert/filter) | 0.0 |
| `--flat-field` | 平场校正 (FlatField) 白帧参考路径 (invert/filter) | - |
| `-r` | 递归子文件夹 | - |

## 依赖

- Python >= 3.10
- numpy >= 1.21.0
- Pillow >= 9.0.0
- rawpy >= 0.17.0（RAW文件支持）

## 反馈与捐赠

- 发现问题请提交 [Issue](https://github.com/waliean/isl-toolkit/issues)
- 或发送邮件：3221636117@qq.com

欢迎投喂 ☕

<img src="mm_facetoface_collect_qrcode_1784791431933.png" width="200"> <img src="alipay_qrcode.png" width="200">

## License

MIT
