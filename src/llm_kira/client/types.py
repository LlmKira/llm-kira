# -*- coding: utf-8 -*-
# @Time    : 1/8/23 11:00 AM
# @FileName: types.py
# @Software: PyCharm
# @Github    ：sudoskys
import time
import uuid
from typing import Optional
from typing import List
from pydantic import BaseModel, validator, root_validator
import shortuuid


class PromptItem(BaseModel):
    id: str = str(shortuuid.uuid())
    start: str = ""
    text: str
    connect_words: str = ":"

    @property
    def prompt(self):
        return f"{self.start}{self.connect_words}{self.text}"

    @root_validator
    def start_check(cls, values):
        start, connect_words = values.get('start'), values.get('connect_words')
        if len(start) > 50:
            raise ValueError('start name too long')
        if not start:
            values["start"] = "*"
        if connect_words:
            values["start"] = start.strip().rstrip(connect_words)
        return values

    @validator('text')
    def text_check(cls, v):
        if not v:
            return "None"
        return v


class Interaction(BaseModel):
    single: bool = False  # 是否单条标示
    ask: PromptItem
    reply: Optional[PromptItem]
    time: int = int(time.time() * 1000)

    @property
    def content(self):
        if self.single:
            return [self.ask.prompt]
        else:
            return [self.ask.prompt, self.reply.prompt]

    @property
    def raw(self):
        return "\n".join(self.content)

    @property
    def message(self):
        if self.single:
            return [[self.ask.start, self.ask.text]]
        else:
            return [[self.ask.start, self.ask.text], [self.reply.start, self.reply.text]]


class InteractionWeight(BaseModel):
    interaction: Interaction
    weight: List[float] = []

    @property
    def score(self):
        return sum(self.weight) / (len(self.weight) * 100 + 0.1)

    @property
    def sum(self):
        return sum(self.weight)


class LlmReturn(BaseModel):
    model_flag: Optional[str]
    prompt: str
    reply: List[str]
    usage: Optional[int]
    time: int = int(time.time())
    raw: Optional[dict]


class ChatBotReturn(BaseModel):
    conversation_id: str
    llm: LlmReturn
    ask: str
    reply: str


class LlmException(Exception):
    pass
