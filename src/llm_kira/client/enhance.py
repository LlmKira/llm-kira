# -*- coding: utf-8 -*-
# @Time    : 1/23/23 8:11 PM
# @FileName: enhance.py
# @Software: PyCharm
# @Github    ：sudoskys
from typing import List

from loguru import logger

from ..client.types import Interaction, PromptItem


class Support(object):
    async def run(self
                  ) -> str:
        return ""


class PluginSystem(Support):
    def __init__(self, plugin_table: dict, prompt: str):
        self.table = plugin_table
        self.prompt = prompt

    async def run(self) -> List[Interaction]:
        _return = []
        if not all([self.table, self.prompt]):
            return []
        from .module.platform import ChatPlugin, PluginParam
        processor = ChatPlugin()
        for plugin in self.table.keys():
            processed = await processor.process(param=PluginParam(text=self.prompt, server=self.table),
                                                plugins=[plugin])
            if processed:
                reply = "\n".join(processed)
                _return.append(Interaction(single=True, ask=PromptItem(start=plugin, text=reply)))
        logger.debug(f"AllPluginReturn:{_return}")
        return _return
