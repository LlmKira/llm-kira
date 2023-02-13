# -*- coding: utf-8 -*-
# @Time    : 1/24/23 11:43 AM
# @FileName: base.py
# @Software: PyCharm
# @Github    ：sudoskys
import json
import time
import os
from abc import abstractmethod, ABC
from typing import Union, Optional, Callable, Any, Dict, Tuple, Mapping, List

from loguru import logger
from pydantic import BaseModel

from ..types import LlmReturn


def mix_result(_item):
    """
    使用遍历方法的混淆器
    """
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "vocab.json")
    )
    with open(path, encoding="utf8") as f:
        target = json.loads(f.read())
    # 遍历
    for key, value in target.items():
        _item = _item.replace(key, value)
    return _item


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
    async def run(self,
                  prompt: str,
                  validate: Union[List[str], None] = None,
                  predict_tokens: int = 500,
                  llm_param: LlmBaseParam = None,
                  ) -> Optional[LlmReturn]:
        return None
