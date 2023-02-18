# -*- coding: utf-8 -*-
# @Time    : 1/23/23 11:18 PM
# @FileName: anchor.py
# @Software: PyCharm
# @Github    ：sudoskys
import random
from typing import Union, Callable, List, Optional
from loguru import logger

from .llms.base import LlmBaseParam
from .llms.openai import LlmBase
from .types import LlmReturn, Interaction, PromptItem
from ..creator.engine import PromptEngine
from ..error import LLMException

# Utils
from ..utils.chat import Detect, Sim

# Completion
from .types import ChatBotReturn
from .agent import Conversation, MemoryManager


class ChatBot(object):
    def __init__(self,
                 profile: Conversation,
                 llm_model: LlmBase = None
                 ):
        """
        对话机器人代理端
        """
        self.profile = profile
        self.memory_manager = MemoryManager(profile=profile)
        self.prompt = None
        self.llm = llm_model
        if llm_model is None:
            raise LLMException("Whats your llm model?")

    def __person(self, prompt, prompt_list):
        _person_list = [f"{self.profile.start_name}:",
                        f"{self.profile.restart_name}:",
                        f"{self.profile.start_name}：",
                        f"{self.profile.restart_name}：",
                        ]
        for item in prompt_list:
            if item.ask.connect_words.strip() in [":", "："]:
                _person_list.append(f"{item.ask.start}{item.ask.connect_words}")
        _person_list = self.__rank_name(prompt=prompt.prompt, users=_person_list)
        return _person_list

    @staticmethod
    def __rank_name(prompt: str, users: List[str]):
        __temp = {}
        for item in users:
            __temp[item] = 0
        users = list(__temp.keys())
        _ranked = list(sorted(users, key=lambda i: Sim.cosion_similarity(pre=str(prompt), aft=str(i)), reverse=True))
        return _ranked

    async def predict(self,
                      prompt: PromptEngine,
                      predict_tokens: Union[int] = 100,
                      llm_param: LlmBaseParam = None,
                      parse_reply: Callable[[list], str] = None,
                      rank_name: bool = True,
                      ) -> ChatBotReturn:
        """
        :param prompt: PromptEngine
        :param predict_tokens: 预测 Token 位
        :param llm_param: 大语言模型参数
        :param parse_reply: Callable[[list], str] 覆写解析方法
        :param rank_name: 自动排序停止词减少第三人称的冲突出现
        """
        self.prompt = prompt
        # ReWrite
        if parse_reply:
            self.llm.parse_reply = parse_reply
        if predict_tokens > self.llm.get_token_limit():
            # Or Auto Cut?
            raise LLMException("Why your predict token > set token limit?")
        _llm_result_limit = self.llm.get_token_limit() - predict_tokens
        _llm_result_limit = _llm_result_limit if _llm_result_limit > 0 else 1
        # Get
        _prompt_index, _prompt = self.prompt.build_prompt(predict_tokens=predict_tokens)
        _prompt_list = []
        _person_list = None if not rank_name else self.__person(prompt=_prompt_index, prompt_list=_prompt)

        # Prompt 构建
        for item in _prompt:
            _prompt_list.extend(item.content)

        prompt_build = "\n".join(_prompt_list) + f"\n{self.profile.restart_name}:"
        prompt_build = self.llm.resize_sentence(prompt_build, token=_llm_result_limit)

        # Get
        llm_result: LlmReturn = await self.llm.run(
            prompt=prompt_build,
            predict_tokens=predict_tokens,
            llm_param=llm_param,
            stop_words=_person_list
        )
        self.prompt.build_interaction(
            ask=_prompt_index,
            response=PromptItem(
                start=self.profile.restart_name,
                text=self.llm.parse_reply(llm_result.reply)
            ),
            single=False
        )

        # Re-Save
        self.prompt.save_interaction()

        # Return
        return ChatBotReturn(
            conversation_id=f"{self.profile.conversation_id}",
            llm=llm_result,
            ask=_prompt_index.text,
            reply=self.llm.parse_reply(llm_result.reply)
        )
