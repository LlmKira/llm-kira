# -*- coding: utf-8 -*-
# @Time    : 1/23/23 11:18 PM
# @FileName: anchor.py
# @Software: PyCharm
# @Github    ：sudoskys

from typing import Union, Callable

# Completion
from .agent import Conversation
from ..creator.engine import PromptEngine
from ..error import LlmException
# from loguru import logger
from ..schema import LlmBaseParam, LlmBase, PromptItem, LlmReturn, ChatBotReturn


class ChatBot(object):
    def __init__(self,
                 profile: Conversation,
                 llm_model: LlmBase = None
                 ):
        """
        对话机器人代理端
        :param profile: Conversation
        :param llm_model: LlmBase 类型，大语言模型
        """
        self.profile = profile
        self.prompt = None
        self.llm = llm_model
        if llm_model is None:
            raise LlmException("llm model missing!")

    async def predict(self,
                      prompt: PromptEngine,
                      predict_tokens: Union[int] = 100,
                      llm_param: LlmBaseParam = None,
                      parse_reply: Callable[[list], str] = None,
                      clean_prompt: bool = True,
                      ) -> ChatBotReturn:
        """
        :param prompt: PromptEngine
        :param predict_tokens: 预测 Token 位
        :param llm_param: 大语言模型对应的参数样式
        :param parse_reply: Callable[[list], str] 覆写解析方法
        :param clean_prompt:
        """
        self.prompt = prompt
        # ReWrite
        if parse_reply:
            self.llm.parse_reply = parse_reply
        if predict_tokens > self.llm.get_token_limit():
            # Or Auto Cut?
            raise LlmException("Why your predict token > set token limit?")

        # Get Question Index
        _prompt_index = self.prompt.prompt
        # Get
        _transfer = await self.llm.transfer(prompt=prompt, predict_tokens=predict_tokens)
        llm_result: LlmReturn = await self.llm.run(
            prompt=_transfer,
            predict_tokens=predict_tokens,
            llm_param=llm_param
        )
        if clean_prompt:
            prompt.clean(clean_prompt=True)

        # Save
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
