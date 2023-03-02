# -*- coding: utf-8 -*-
# @Time    : 1/23/23 7:00 PM
# @FileName: openai.py
# @Software: PyCharm
# @Github    ：sudoskys
import math
import time
import random
import tiktoken
from typing import Union, Optional, Callable, Any, Dict, Tuple, Mapping, List

from loguru import logger
# from loguru import logger

from ...error import RateLimitError, ServiceUnavailableError
from ...tool import openai as openai_api
from pydantic import BaseModel, Field
from tenacity import retry_if_exception_type, retry, stop_after_attempt, wait_exponential
from ..agent import Conversation
from ...creator.engine import PromptEngine
from ..llms.base import LlmBase, LlmBaseParam
from ..types import LlmReturn, LlmException
from ...utils.chat import Sim
from ...utils.data import DataUtils
from ...utils.setting import llmRetryAttempt, llmRetryTime, llmRetryTimeMax, llmRetryTimeMin


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
                 **kwargs
                 ):
        """
        Openai LLM 的方法类集合
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

    def tokenizer(self, text, raw: bool = False) -> Union[int, list]:
        gpt_tokenizer = tiktoken.get_encoding("gpt2")
        _token = gpt_tokenizer.encode(text)
        if raw:
            return _token
        return len(_token)

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

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "unknown"

    def parse_reply(self, reply: List[str]) -> str:
        """处理解码后的列表"""
        _reply = "".join(reply)
        _reply = DataUtils.remove_suffix(input_string=_reply, suffix="<|im_end|>")
        return _reply

    def resize_sentence(self, text: str, token: int) -> str:
        """
        改进后的梯度缓存裁剪算法
        """
        token = token if token > 0 else 0
        step = 4
        _cache = {}

        def _cache_cutter(_text):
            if _cache.get(_text):
                return _cache.get(_text)
            _value = self.tokenizer(_text)
            _cache[_text] = _value
            return _value

        while len(text) > step and _cache_cutter(text) > token:
            _rank = math.floor((_cache_cutter(text) - token) / 100) + 4
            step = _rank * 8
            text = text[step:]
        _cache = {}
        return text

    async def task_context(self, task: str, prompt: str, predict_tokens: int = 500) -> LlmReturn:
        prompt = self.resize_sentence(prompt, 1200)
        _prompt = f"Text:{prompt}\n{task}: "
        llm_result = await self.run(prompt=_prompt,
                                    predict_tokens=predict_tokens,
                                    llm_param=OpenAiParam(model_name="text-davinci-003"),
                                    stop_words=["Text:", "\n\n"]
                                    )
        return llm_result

    def resize_context(self, head: list, body: list, foot: list = None, token: int = 5) -> str:
        if foot is None:
            foot = []
        # 去空
        body = [item for item in body if item]
        # 强制测量
        token = token if token > 5 else 5

        # 弹性计算
        def _connect(_head, _body, _foot):
            _head = '\n'.join(_head) + "\n"
            _body = "\n".join(_body) + "\n"
            _foot = ''.join(_foot)
            # Resize
            return _head + _body + _foot

        _all = _connect(head, body, foot)
        while len(body) > 2 and self.tokenizer(_all) >= token:
            body.pop(0)
            _all = _connect(head, body, foot)
        _all = self.resize_sentence(_all, token=token)
        return _all

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

    @retry(retry=retry_if_exception_type((RateLimitError,
                                          ServiceUnavailableError)),
           stop=stop_after_attempt(llmRetryAttempt),
           wait=wait_exponential(multiplier=llmRetryTime, min=llmRetryTimeMin, max=llmRetryTimeMax),
           reraise=True)
    async def run(self,
                  prompt: Union[PromptEngine, str],
                  validate: Union[List[str], None] = None,
                  predict_tokens: int = 500,
                  llm_param: OpenAiParam = None,
                  stop_words: list = None,
                  anonymous_user: bool = False,
                  rank_name: bool = False,
                  **kwargs
                  ) -> LlmReturn:
        """
        异步的，得到对话上下文
        :param predict_tokens: 限制返回字符数量
        :param validate: 惩罚验证列表
        :param prompt: 提示词
        :param llm_param: 参数表
        :param anonymous_user:
        :param stop_words:
        :param rank_name:
        :return:
        """
        _request_arg = {
            "top_p": 1,
            "n": 1
        }
        _request_arg: dict
        if not isinstance(prompt, str):
            # Prompt
            _prompt_list = []
            # Get
            _llm_result_limit = self.get_token_limit() - predict_tokens
            _llm_result_limit = _llm_result_limit if _llm_result_limit > 0 else 1
            _prompt_input, _prompt = prompt.build_prompt(predict_tokens=_llm_result_limit)
            if not _prompt_input:
                raise LlmException("Input Is Empty")
            for item in _prompt:
                _prompt_list.extend(item.content)
            prompt_build = "\n".join(_prompt_list) + f"\n{self.profile.restart_name}:"
            prompt_build = self.resize_sentence(prompt_build, token=self.get_token_limit())
            stop_words = None if not rank_name else self.__person(prompt=_prompt_input, prompt_list=_prompt)
        else:
            prompt_build = self.resize_sentence(prompt, token=predict_tokens)

        # 停止符号
        if stop_words is None:
            stop_words = [f"{self.profile.start_name}:",
                          f"{self.profile.restart_name}:",
                          f"{self.profile.start_name}：",
                          f"{self.profile.restart_name}："]

        # 补全参数
        if llm_param:
            _request_arg.update(llm_param.invocation_params)
        if validate is None:
            validate = []

        # 构造覆盖信息
        _request_arg.update(model=str(llm_param.model_name),
                            prompt=prompt_build,  # IMPORTANT
                            max_tokens=int(predict_tokens),
                            user=str(self.profile.get_conversation_hash()),
                            stop=stop_words[:4],
                            )

        # Anonymous
        if anonymous_user:
            _request_arg.pop("user", None)

        # Adjust Penalty
        """
        if self.auto_penalty and validate:
            # Cook
            _frequency_penalty, _presence_penalty, _temperature = Detect().gpt_tendency_arg(prompt=prompt,
                                                                                            memory=validate,
                                                                                            tokenizer=self.tokenizer
                                                                                            )
            # Some Update
            _request_arg.update({
                "frequency_penalty": float(_frequency_penalty),
                "presence_penalty": float(_presence_penalty),
                "temperature": float(_temperature),
            })
        """
        if _request_arg.get("frequency_penalty") == 0:
            _request_arg.pop("frequency_penalty", None)
        if _request_arg.get("presence_penalty") == 0:
            _request_arg.pop("presence_penalty", None)

        # 校准字节参数
        if not _request_arg.get("logit_bias"):
            _request_arg["logit_bias"] = {}
            _request_arg.pop("logit_bias", None)

        # 校准温度和惩罚参数
        if _request_arg.get("frequency_penalty"):
            _frequency_penalty = _request_arg["frequency_penalty"]
            _frequency_penalty = _frequency_penalty if -2.0 < _frequency_penalty else -1.9
            _frequency_penalty = _frequency_penalty if _frequency_penalty < 2.0 else 1.9
            _request_arg["frequency_penalty"] = _frequency_penalty

        if _request_arg.get("presence_penalty"):
            _presence_penalty = _request_arg["presence_penalty"]
            _presence_penalty = _presence_penalty if -2.0 < _presence_penalty else -1.9
            _presence_penalty = _presence_penalty if _presence_penalty < 2.0 else 1.9
            _request_arg["presence_penalty"] = _presence_penalty

        if _request_arg.get("temperature"):
            _temperature = _request_arg["temperature"]
            _request_arg["temperature"] = _temperature if 0 < _temperature < 1 else 0.9

        # 自维护 Api 库
        response = await openai_api.Completion(api_key=self.__api_key, call_func=self.__call_func).create(
            **_request_arg
        )

        # Reply
        reply = self.parse_response(response)
        self.profile.update_usage(usage=self.parse_usage(response))
        return LlmReturn(model_flag=llm_param.model_name,
                         raw=response,
                         prompt=prompt_build,
                         usage=self.profile.get_round_usage(),
                         time=int(time.time()),
                         reply=reply
                         )

    @staticmethod
    def __rank_name(prompt: str, users: List[str]):
        __temp = {}
        for item in users:
            __temp[item] = 0
        users = list(__temp.keys())
        _ranked = list(sorted(users, key=lambda i: Sim.cosion_similarity(pre=str(prompt), aft=str(i)), reverse=True))
        return _ranked

    def __person(self, prompt, prompt_list: list):
        _person_list = [f"{self.profile.start_name}:",
                        f"{self.profile.restart_name}:",
                        f"{self.profile.start_name}：",
                        f"{self.profile.restart_name}：",
                        ]
        for item in prompt_list:
            if item.ask.connect_words.strip() in [":", "："]:
                _person_list.append(f"{item.ask.start}{item.ask.connect_words}")
        _person_list = self.__rank_name(prompt=prompt, users=_person_list)
        return _person_list
