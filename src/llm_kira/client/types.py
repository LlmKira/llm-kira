# -*- coding: utf-8 -*-
# @Time    : 1/8/23 11:00 AM
# @FileName: types.py
# @Software: PyCharm
# @Github    ：sudoskys
import time
from typing import Optional
from typing import List
from pydantic import BaseModel


class MemeryItem(BaseModel):
    ask: str
    reply: str
    weight: list = []


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


class PromptItem(BaseModel):
    types: str = ""
    start: str
    text: str
    method: str = ""


class MemoryItem(BaseModel):
    weight: list = []
    ask: str = ""
    reply: str = ""


# {"weight": [], "ask": f"{ask}", "reply": f"{reply}"}
class Memory_Flow(BaseModel):
    content: MemoryItem
    time: int = int(time.time() * 1000)
