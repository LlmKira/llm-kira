# -*- coding: utf-8 -*-
# @Time    : 2/16/23 2:36 PM
# @FileName: __init__.py
# @Software: PyCharm
# @Github    ：sudoskys

# TODO
from pydantic import BaseModel


class chatEngineConfig(BaseModel):
    user = "Ryan"
    bot = "Gordon"


class ChatEngine(object):
    def __init__(self,
                 description: str,
                 examples: dict,
                 flowResetText: str,
                 promptConfig: chatEngineConfig = None
                 ):
        self.promptConfig = promptConfig
        self.flowResetText = flowResetText
        self.examples = examples
        self.description = description
        self.dialog = []

    def buildPrompt(self, query: str):
        pass

    def addInteraction(self, query, response):
        pass
