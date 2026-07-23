"""处理管线 — 重定向到 kernel.pipeline 以保持向后兼容。"""
from .kernel.pipeline import ProcessingPipeline, FilterStage
__all__ = ["ProcessingPipeline", "FilterStage"]
