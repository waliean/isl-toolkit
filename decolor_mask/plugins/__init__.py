"""插件注册表。"""
from .base import PluginBase
from .film_inversion import FilmInversionPlugin
from .image_enhance import ImageEnhancePlugin
from .noise_reduction import NoiseReductionPlugin

_registry: dict[str, PluginBase] = {}


def register(plugin: PluginBase):
    _registry[plugin.name] = plugin
    return plugin


def get(name: str) -> PluginBase | None:
    return _registry.get(name)


def list_all() -> list[PluginBase]:
    return list(_registry.values())


# 注册内置插件
register(FilmInversionPlugin())
register(ImageEnhancePlugin())
register(NoiseReductionPlugin())
