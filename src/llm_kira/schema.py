# -*- coding: utf-8 -*-
# @Time    : 2023/4/10 下午11:46
# @Author  : sudoskys
# @File    : schema.py
# @Software: PyCharm
import hashlib
import time
from typing import Any, Optional, Tuple
from typing import List, Dict

from loguru import logger
from pydantic import BaseModel
from pydantic import root_validator, validator


class MetaData(BaseModel):
    start: str = ""
    connect_words: str = ":"
    text: str
    timestamp: int = int(time.time())

    @root_validator
    def start_check(cls, values):
        """
        校验起始函数
        """
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
        """
        检查原始文本
        """
        if not v:
            return "None"
        return v


class Message(BaseModel):
    metadata: MetaData = MetaData(text="Hello, world!")
    ttl: int = -1
    timestamp: int = int(time.time())

    def get_vector(self, area: str = "default"):
        if self.encoder is None:
            return None
        if area not in self.vector:
            self.vector[area] = self.encoder(self.metadata.text)
        return self.vector[area]

    @property
    def text(self):
        return f"{self.metadata.start}{self.metadata.connect_words}{self.metadata.text}"


class InteractionWeight(BaseModel):
    interaction: Message
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


class LlmTransfer(BaseModel):
    index: List[str]
    data: Any
    raw: Tuple[Any, Any]


def get_hex(string):
    bytes_str = string.encode('utf-8')
    md5 = hashlib.md5()
    md5.update(bytes_str)
    h16 = md5.hexdigest()
    return int(h16, 16)


class Conversation(object):
    """基础身份类型，供其他模块使用"""

    def __init__(self, start_name: str,
                 restart_name: str,
                 conversation_id: int = 1,
                 init_usage: int = 0
                 ):
        """
        start_name: 说话者的名字
        restart_name: 回答时候使用的名字
        conversation_id: 对话 ID，很重要，如果不存在会计算 start_name 的 唯一ID 作为 ID
        init_usage: int 初始计费
        """
        self.hash_secret = "LLM"
        if not conversation_id:
            conversation_id = get_hex(start_name)
            logger.warning("conversation_id empty!!!")
        self.conversation_id = str(conversation_id)
        self.start_name = start_name.strip(":").strip("：")
        self.restart_name = restart_name.strip(":").strip("：")
        self.__usage = init_usage if init_usage > 0 else 0

    def get_conversation_hash(self):
        uid = f"{str(self.hash_secret)}{str(self.conversation_id)}"
        hash_object = hashlib.sha256(uid.encode())
        return hash_object.hexdigest()

    def get_round_usage(self):
        return self.__usage

    def update_usage(self, usage: int = 0, override: bool = False):
        if override:
            self.__usage = usage
        else:
            self.__usage += usage
