# -*- coding: utf-8 -*-
# @Time    : 1/23/23 7:00 PM
# @FileName: llm.py
# @Software: PyCharm
# @Github    ：sudoskys
import json
import time
import os
import random
from abc import abstractmethod, ABC
from typing import Union, Optional, Callable, Any
from transformers import GPT2TokenizerFast

from .agent import Conversation
from .types import LlmReturn
from ..openai import Completion
from ..utils.chat import Detect


def mix_result(_item):
    """
    使用遍历方法的混淆器
    """
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), ".", "vocab.json")
    )
    with open(path, encoding="utf8") as f:
        target = json.loads(f.read())
    # 遍历
    for key, value in target.items():
        _item = _item.replace(key, value)
    return _item


class LlmBase(ABC):
    _annotation: type

    @abstractmethod
    def token_limit(self) -> int:
        return 2000

    @abstractmethod
    def tokenizer(self, text):
        return len(text)

    def parse_reply(self, reply: list) -> str:
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
                  predict_tokens: int = 500,
                  **kwargs) -> Optional[LlmReturn]:
        return None


class OpenAi(LlmBase):
    def __init__(self,
                 profile: Conversation,
                 api_key: Union[str, list] = None,
                 token_limit: int = 3700,
                 no_penalty: bool = False,
                 call_func: Callable[[dict, str], Any] = None
                 ):
        """
        chatGPT 的实现由上下文实现，所以我会做一个存储器来获得上下文
        :param api_key: api key
        :param token_limit: 总限制
        :param call_func: 回调
        :param no_penalty: 不使用自动惩罚参数调整
        """
        self.no_penalty = no_penalty
        self.profile = profile
        # if api_key is None:
        #     api_key = setting.openaiApiKey
        if isinstance(api_key, list):
            api_key: list
            if not api_key:
                raise RuntimeError("NO KEY")
            api_key = random.choice(api_key)
            api_key: str
        self.__api_key = api_key
        if not api_key:
            raise RuntimeError("NO KEY")
        self.__start_sequence = self.profile.start_name
        self.__restart_sequence = self.profile.restart_name
        self.__call_func = call_func
        self.__token_limit = token_limit

    def token_limit(self) -> int:
        return self.__token_limit

    def tokenizer(self, text) -> int:
        gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        _token = len(gpt_tokenizer.encode(text))
        return _token

    @staticmethod
    def parse_response(response) -> list:
        REPLY = []
        Choice = response.get("choices")
        if Choice:
            for item in Choice:
                _text = item.get("text")
                REPLY.append(_text)
        if not REPLY:
            REPLY = [""]
        return REPLY

    @staticmethod
    def parse_usage(response) -> Optional[int]:
        usage = None
        usage_dict = response.get("usage")
        if usage_dict:
            usage = usage_dict["total_tokens"]
        return usage

    async def run(self,
                  prompt: str,
                  predict_tokens: int = 500,
                  model: str = "text-davinci-003",
                  **kwargs
                  ) -> LlmReturn:
        """
        异步的，得到对话上下文
        :param predict_tokens: 限制返回字符数量
        :param model: 模型选择
        :param prompt: 提示词

        :return:
        """
        _request_arg = {
            "temperature": 0.9,
            "logit_bias": {}
        }

        if not self.no_penalty:
            # THINK ABOUT HOT CAKE
            _frequency_penalty, _presence_penalty, _temperature = Detect().gpt_tendency_arg(prompt=prompt)
            # SOME HOT CAKE
            _request_arg.update({
                "frequency_penalty": _frequency_penalty,
                "presence_penalty": _presence_penalty,
                "temperature": _temperature,
                "logit_bias": {}
            })

        # Kwargs
        _arg_config = {key: item for key, item in kwargs.items() if key in _request_arg.keys()}
        _request_arg.update(_arg_config)

        # Req
        response = await Completion(api_key=self.__api_key, call_func=self.__call_func).create(
            model=model,
            prompt=str(prompt),
            max_tokens=predict_tokens,
            top_p=1,
            n=1,
            user=str(self.profile.get_conversation_hash()),
            stop=[f"{self.profile.start_name}:",
                  f"{self.profile.restart_name}:",
                  f"{self.profile.start_name}：",
                  f"{self.profile.restart_name}："],
            **_request_arg
        )
        reply = self.parse_response(response)
        usage = self.parse_usage(response)
        return LlmReturn(model_flag=model,
                         raw=response,
                         prompt=prompt,
                         usage=usage,
                         time=int(time.time()),
                         reply=reply,
                         )
