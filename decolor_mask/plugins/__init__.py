"""插件注册表。"""
from .base import PluginBase
from .cross_process import CrossProcessPlugin
from .film_inversion import FilmInversionPlugin
from .image_enhance import ImageEnhancePlugin
from .filter_pipeline import FilterPipelinePlugin

_registry: dict[str, PluginBase] = {}


def register(plugin: PluginBase):
    _registry[plugin.name] = plugin
    return plugin


def get(name: str) -> PluginBase | None:
    return _registry.get(name)


def list_all() -> list[PluginBase]:
    return list(_registry.values())


# 注册内置插件
register(CrossProcessPlugin())
register(FilmInversionPlugin())
register(ImageEnhancePlugin())
register(FilterPipelinePlugin())
