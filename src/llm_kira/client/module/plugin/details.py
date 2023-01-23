# -*- coding: utf-8 -*-
# @Time    : 1/1/23 12:50 PM
# @FileName: details.py
# @Software: PyCharm
# @Github    ：sudoskys

from ..platform import ChatPlugin, PluginConfig
from ._plugin_tool import PromptTool
import os
from loguru import logger

modulename = os.path.basename(__file__).strip(".py")


@ChatPlugin.plugin_register(modulename)
class Details(object):
    def __init__(self):
        self._server = None
        self._text = None
        # 绝望列表
        self._keywords = PromptTool.help_words_list()

    async def check(self, params: PluginConfig) -> bool:
        if PromptTool.isStrIn(prompt=params.text, keywords=self._keywords):
            return True
        return False

    def requirements(self):
        return []

    async def process(self, params: PluginConfig) -> list:
        _return = []
        self._text = params.text
        # 校验
        if not all([self._text]):
            return []
        # GET
        _return.append(f"仔细思考 and show your step.")
        logger.trace(_return)
        return _return
