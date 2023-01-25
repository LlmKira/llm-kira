# -*- coding: utf-8 -*-
# @Time    : 1/23/23 7:00 PM
# @FileName: openai.py
# @Software: PyCharm
# @Github    ：sudoskys
import os
import time
import json
import random
import tiktoken
from typing import Union, Optional, Callable, Any, Dict, Tuple, Mapping, List

from loguru import logger
from pydantic import BaseModel, Field

from ..agent import Conversation
from ..llms.base import LlmBase, LlmBaseParam
from ..types import LlmReturn
from ...openai import Completion
from ...utils.chat import Detect


class OpenAiParam(LlmBaseParam, BaseModel):
    model_name: str = "text-davinci-003"
    """Model name to use."""
    temperature: float = 0.8
    """What sampling temperature to use."""
    max_tokens: int = 256
    """The maximum number of tokens to generate in the completion.
    -1 returns as many tokens as possible given the prompt and
    the models maximal context size."""
    top_p: float = 1
    """Total probability mass of tokens to consider at each step."""
    frequency_penalty: float = 0
    """Penalizes repeated tokens according to frequency."""
    presence_penalty: float = 0
    """Penalizes repeated tokens."""
    n: int = 1
    """How many completions to generate for each prompt."""
    best_of: int = 1
    """Generates best_of completions server-side and returns the "best"."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for create call not explicitly specified."""
    batch_size: int = 20
    """Batch size to use when passing multiple documents to generate."""
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout for requests to OpenAI completion API. Default is 600 seconds."""
    logit_bias: Optional[Dict[str, float]] = Field(default_factory=dict)
    """Adjust the probability of specific tokens being generated."""

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        normal_params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "n": self.n,
            "best_of": self.best_of,
            "request_timeout": self.request_timeout,
            "logit_bias": self.logit_bias,
        }
        return {**normal_params, **self.model_kwargs}

    @property
    def invocation_params(self) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        return self._default_params

    @property
    def identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}


class OpenAi(LlmBase):

    def __init__(self, profile: Conversation,
                 api_key: Union[str, list] = None,
                 token_limit: int = 3700,
                 auto_penalty: bool = False,
                 call_func: Callable[[dict, str], Any] = None,
                 ):
        """
        chatGPT 的实现由上下文实现，所以我会做一个存储器来获得上下文
        :param api_key: api key
        :param token_limit: 总限制
        :param call_func: 回调
        :param auto_penalty: 不使用自动惩罚参数调整
        """
        self.auto_penalty = auto_penalty
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
        self.token_limit = token_limit

    def get_token_limit(self) -> int:
        return self.token_limit

    def tokenizer(self, text) -> int:
        gpt_tokenizer = tiktoken.get_encoding("gpt2")
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
            REPLY = ["."]
        return REPLY

    @staticmethod
    def parse_usage(response) -> Optional[int]:
        usage = None
        usage_dict = response.get("usage")
        if usage_dict:
            usage = usage_dict["total_tokens"]
        return usage

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "unknown"

    def parse_reply(self, reply: List[str]) -> str:
        """处理解码后的列表"""
        return "".join(reply)

    def resize_context(self, text: str, token: int) -> str:
        token = token if token > 5 else 5
        while self.tokenizer(text) > token:
            text = text[4:]
        return text

    @staticmethod
    def model_context_size(model_name: str) -> int:
        if model_name == "text-davinci-003":
            return 4000
        elif model_name == "text-curie-001":
            return 2048
        elif model_name == "text-babbage-001":
            return 2048
        elif model_name == "text-ada-001":
            return 2048
        elif model_name == "code-davinci-002":
            return 8000
        elif model_name == "code-cushman-001":
            return 2048
        else:
            return 4000

    async def run(self,
                  prompt: str,
                  predict_tokens: int = 500,
                  llm_param: OpenAiParam = None
                  ) -> LlmReturn:
        """
        异步的，得到对话上下文
        :param predict_tokens: 限制返回字符数量
        :param prompt: 提示词
        :param llm_param: 参数表
        :return:
        """

        _request_arg = {
            "temperature": float(0.9),
            "logit_bias": {},
            "top_p": float(1),
            "n": int(1)
        }
        # Kwargs
        if llm_param:
            _request_arg.update(llm_param.invocation_params)

        _request_arg.update(model=str(llm_param.model_name),
                            prompt=str(prompt),
                            max_tokens=int(predict_tokens),
                            user=str(self.profile.get_conversation_hash()),
                            stop=[f"{self.profile.start_name}:",
                                  f"{self.profile.restart_name}:",
                                  f"{self.profile.start_name}：",
                                  f"{self.profile.restart_name}："],
                            )

        # Penalty
        if self.auto_penalty:
            # THINK ABOUT HOT CAKE
            _frequency_penalty, _presence_penalty, _temperature = Detect().gpt_tendency_arg(prompt=prompt)
            # SOME HOT CAKE
            _request_arg.update({
                "frequency_penalty": float(_frequency_penalty),
                "presence_penalty": float(_presence_penalty),
                "temperature": float(_temperature),
            })

        # Req
        response = await Completion(api_key=self.__api_key, call_func=self.__call_func).create(
            **_request_arg
        )

        # Reply
        reply = self.parse_response(response)
        usage = self.parse_usage(response)
        return LlmReturn(model_flag=llm_param.model_name,
                         raw=response,
                         prompt=prompt,
                         usage=usage,
                         time=int(time.time()),
                         reply=reply,
                         )
