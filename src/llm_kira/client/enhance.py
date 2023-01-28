# -*- coding: utf-8 -*-
# @Time    : 1/23/23 8:11 PM
# @FileName: enhance.py
# @Software: PyCharm
# @Github    ：sudoskys
from loguru import logger

from .module.platform import PluginConfig
from .module.plugin.search import Search


# TODO 分离插件层，更好地内置 + 探测器，供外部单独调用。

class Support(object):
    async def run(self
                  ) -> str:
        return ""


class PluginSystem(Support):
    def __init__(self, plugin_table: dict, prompt: str):
        self.table = plugin_table
        self.prompt = prompt

    async def run(self) -> str:
        _append = "-"
        _return = []
        if not all([self.table, self.prompt]):
            return _append
        from .module.platform import ChatPlugin, PluginParam
        processor = ChatPlugin()
        for plugin in self.table.keys():
            processed = await processor.process(param=PluginParam(text=self.prompt, server=self.table),
                                                plugins=[plugin])
            _return.extend(processed)
        reply = "\n".join(_return) if _return else ""
        logger.debug(f"AllPluginReturn:{reply}")
        return reply


class WebSearch(Support):
    def __init__(self, config: PluginConfig):
        self.config = config

    async def run(self) -> str:
        _return = await Search().process(params=self.config)
        reply = "\n".join(_return) if _return else ""
        logger.debug(f"AllPluginReturn:{reply}")
        return reply
