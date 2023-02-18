# -*- coding: utf-8 -*-
# @Time    : 1/23/23 11:18 PM
# @FileName: anchor.py
# @Software: PyCharm
# @Github    ：sudoskys
import random
from typing import Union, Callable, List, Optional
from loguru import logger

# Tool
from .enhance import Support
from .llms.base import LlmBaseParam
from .llms.openai import LlmBase
from .types import LlmReturn, Interaction, PromptItem
from ..creator.engine import PromptEngine
from ..error import LLMException

# Utils
from ..utils.chat import Detect

# Completion
from .types import ChatBotReturn
from .agent import Conversation, MemoryManager

# Antennae
from ..radio.decomposer import PromptTool
from ..radio.setting import HELP_WORDS
from ..utils.data import MsgFlow


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

    async def predict(self,
                      prompt: PromptEngine,
                      predict_tokens: Union[int] = 100,
                      llm_param: LlmBaseParam = None,
                      parse_reply: Callable[[list], str] = None,
                      ) -> ChatBotReturn:
        """
        :param prompt: PromptEngine
        :param predict_tokens: 预测 Token 位
        :param llm_param: 大语言模型参数
        :param parse_reply: Callable[[list], str] 覆写解析方法
        """
        self.prompt = prompt
        # ReWrite
        if parse_reply:
            self.llm.parse_reply = parse_reply
        if predict_tokens > self.llm.get_token_limit():
            # Or Auto Cut?
            raise LLMException("Why your predict token > set token limit?")

        # Get
        prompt_index, prompt = self.prompt.build_prompt(predict_tokens=predict_tokens)
        logger.warning(prompt_index)
        logger.warning(prompt)
        # Get
        llm_result: LlmReturn = await self.llm.run(
            prompt=prompt,
            predict_tokens=predict_tokens,
            llm_param=llm_param
        )
        self.prompt.insert_interaction(
            ask=prompt_index,
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
            ask=prompt_index.text,
            reply=self.llm.parse_reply(llm_result.reply)
        )
