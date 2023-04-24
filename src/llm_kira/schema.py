# -*- coding: utf-8 -*-
# @Time    : 2023/4/10 下午11:46
# @Author  : sudoskys
# @File    : schema.py
# @Software: PyCharm
import time
from abc import ABC, abstractmethod
from typing import Union, Any, List, Optional, Tuple

import shortuuid
from pydantic import BaseModel, root_validator, validator


class Vector(BaseModel):
    vector: List[float]


class PromptItem(BaseModel):
    id: str = str(shortuuid.uuid())
    start: str = ""
    text: str
    connect_words: str = ":"
    vector: Optional[Vector] = None

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


class LlmTransfer(BaseModel):
    index: List[str]
    data: Any
    raw: Tuple[Any, Any]


####

class LlmBaseParam(BaseModel):
    pass


class LlmBase(ABC):

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "unknown"

    @abstractmethod
    def get_token_limit(self) -> int:
        return 2000

    @abstractmethod
    def tokenizer(self, text, raw=False) -> Union[int, list]:
        if raw:
            return []
        return len(text)

    @abstractmethod
    def resize_sentence(self, text: str, token: int) -> str:
        return text[:token]

    @abstractmethod
    def task_context(self, task: str, prompt: str, predict_tokens: int = 500) -> Any:
        return None

    @abstractmethod
    def resize_context(self, head: list, body: list, foot: list = None, token: int = 0) -> str:
        if foot is None:
            foot = []
        _all = ''.join(head + body + foot)
        return self.resize_sentence(_all, token=token)

    @staticmethod
    def model_context_size(model_name: str) -> int:
        pass

    @abstractmethod
    def parse_reply(self, reply: List[str]) -> str:
        if reply:
            return str(reply[0])
        else:
            return ""

    @staticmethod
    def parse_response(response) -> list:
        return [""]

    @staticmethod
    def parse_usage(response) -> Optional[int]:
        return None

    @abstractmethod
    async def transfer(self,
                       prompt: Any,
                       predict_tokens: int = 2000
                       ) -> LlmTransfer:
        pass

    @abstractmethod
    async def run(self,
                  prompt: Any,
                  validate: Union[List[str], None] = None,
                  predict_tokens: int = 500,
                  llm_param: LlmBaseParam = None,
                  stop_words: list = None,
                  ) -> Optional[LlmReturn]:
        return None


class MemoryBaseLoader(object):
    def __init__(
            self,
            session_id: str,
            key_prefix: str = "llm_kira_message_store:",
    ):
        self._memory: list = []
        self.key_prefix = key_prefix
        self.session_id = session_id

    @property
    def key(self) -> str:
        return self.key_prefix + self.session_id

    @property
    def message(self) -> List[Interaction]:
        items = self._memory
        messages = [Interaction(**items) for items in items]
        return messages

    def append(self, message: List[Interaction]) -> None:
        for item in message:
            self._memory.append(item.dict())

    def clear(self) -> None:
        self._memory = []
