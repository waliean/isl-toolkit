"""通用图像处理引擎内核。

提供滤镜基础设施和标准 Pipeline，所有功能插件基于此构建。
"""

from .pipeline import ProcessingPipeline, FilterStage
from .filters import (
    BaseFilter,
    rgb_to_lab, lab_to_rgb,
    rgb_to_lcc, lcc_to_rgb,
    rgb_to_ycc, ycc_to_rgb,
    GrayAxisFilmBase,
    LCCInverter,
    DehazeFilter,
    CromaNRFilter,
    BandNRFilter,
    FlatFieldFilter,
    SmartSharpFilter,
    ClarityFilter,
    BWFilterSim,
    ToningFilter,
    VignetteFilter,
    GrainFilter,
    ToneCurveFilter,
    ShadowBoostFilter,
)
