# -*- coding: utf-8 -*-
# @Time    : 2/17/23 9:27 AM
# @FileName: engine.py
# @Software: PyCharm
# @Github    ：sudoskys
import time
from typing import List, Tuple, Optional
from loguru import logger

from . import Optimizer
from .base import BaseEngine
from ..client.agent import Conversation, MemoryManager
from ..schema import LlmBase, PromptItem, Interaction
from llm_kira.creator.bucket import MsgFlow


class PromptEngine(BaseEngine):
    """
    设计用于维护提示系统和接入外骨骼
    """

    def __init__(self,
                 profile: Conversation,
                 memory_bucket: MemoryManager,
                 knowledge_bucket: MemoryManager = None,
                 llm_model: LlmBase = None,
                 description: str = None,
                 connect_words: str = "\n",
                 reference_ratio: float = 0.5,
                 forget_words: List[str] = None,
                 optimizer: Optimizer = Optimizer.SinglePoint,
                 reverse_prompt_buffer: bool = False
                 ):
        """
        :param profile: 身份类型，同时承担计费管理
        :param memory_bucket: 记忆管理，可继承重写
        :param llm_model: LLM代理，可扩增
        :param description: Prompt 的恒久状态头
        :param connect_words: 连接Prompt的连接词
        :param reference_ratio: 分配给知识库的 token位 比例
        :param forget_words: 阻断列表，如果 input 中有，则不加入 Prompt
        :param optimizer: 优化器，可以覆写，按照模板继承即可
        """
        self.profile = profile
        self.llm = llm_model
        self.memory_manger = memory_bucket
        self.knowledge_manger = knowledge_bucket
        self.reference_ratio = reference_ratio
        self.optimizer = optimizer
        self.reverse_prompt_buffer = reverse_prompt_buffer
        self.forget_words = forget_words if forget_words else []
        # self.skeleton = skeleton
        self.__connect_words: str = connect_words

        self.__uid = self.profile.conversation_id
        self.__start_name = profile.start_name
        self.__restart_name = profile.restart_name

        self.description: str = description  # 头部状态标识器
        self.prompt_buffer: List[PromptItem] = []  # 外骨骼用
        self.interaction_pool: List[Interaction] = self.memory_manger.read_bucket()
        self.knowledge_pool: List[Interaction] = []
        if self.knowledge_manger:
            self.knowledge_pool = self.knowledge_manger.read_bucket()
        if optimizer is None:
            self.optimizer = Optimizer.SinglePoint
        self._MsgFlow = MsgFlow(uid=self.profile.conversation_id)

    @property
    def restart_name(self):
        return self.__restart_name

    @property
    def start_name(self):
        return self.__start_name

    @property
    def uid(self):
        return self.__uid

    @property
    def prompt(self):
        _buffer = self.prompt_buffer.copy()
        if self.reverse_prompt_buffer:
            _buffer = list(reversed(_buffer))
        if self.prompt_buffer:
            return _buffer[-1]
        else:
            return None

    def _build_prompt_buffer(self):
        _buffer = self.prompt_buffer.copy()
        if self.reverse_prompt_buffer:
            _buffer = list(reversed(_buffer))
        if not _buffer:
            return None
        _index = _buffer.pop(-1)
        for item in _buffer:
            self.build_interaction(ask=item, single=True)
        return _index

    def read_interaction(self):
        return self.memory_manger.read_bucket()

    def save_interaction(self):
        return self.memory_manger.save_bucket(self.interaction_pool, override=True)

    def read_knowledge(self):
        if self.knowledge_manger:
            return self.knowledge_manger.read_bucket()

    def save_knowledge(self):
        if self.knowledge_manger:
            return self.knowledge_manger.save_bucket(self.interaction_pool, override=True)

    def clean(self, clean_prompt: bool = False, clean_memory: bool = False, clean_knowledge: bool = False):
        if clean_knowledge:
            self.knowledge_pool = []
        if clean_memory:
            self.interaction_pool = []
        if clean_prompt:
            self.prompt_buffer = []
        return True

    def build_interaction(self, ask: PromptItem, response: Optimizer = None, single: bool = False):
        if not response and not single:
            raise Exception("Not Allowed Method")
        interaction = Interaction(ask=ask, reply=response, single=single, time=time.time() * 1000)
        self.insert_interaction(interaction)

    def insert_knowledge(self, knowledge: Interaction):
        """基础知识参考添加"""
        if isinstance(knowledge, str):
            logger.warning("Knowledge Should Be Interaction Class")
            knowledge = Interaction(single=True, ask=PromptItem(start="*", text=str(knowledge)))
        self.knowledge_pool.append(knowledge)

    def insert_interaction(self, interaction: Interaction):
        if isinstance(interaction, str):
            logger.warning("interaction Should Be Interaction Class")
            interaction = Interaction(single=True, ask=PromptItem(start="*", text=str(interaction)))
        self.interaction_pool.append(interaction)

    def build_knowledge(self, ask: PromptItem, response: PromptItem):
        """基础知识参考构建"""
        knowledge = Interaction(ask=ask, reply=response)
        self.insert_knowledge(knowledge)

    def insert_prompt(self, prompt: PromptItem):
        """基础Prompt Buffer添加方法"""
        return self.prompt_buffer.append(prompt)

    def build_context(self, prompt: PromptItem, predict_tokens) -> List[Interaction]:
        # Resize
        _llm_result_limit = self.llm.get_token_limit() - predict_tokens
        _llm_result_limit = _llm_result_limit if _llm_result_limit > 0 else 1
        if _llm_result_limit < 10:
            logger.warning("llm free mem lower than 10...may limit too low or predict token too high")

        # Optimizing Prompt and Cut Prompt
        _optimized_prompt = self.optimizer(
            prompt=prompt,
            desc=self.description,
            interaction=self.interaction_pool,
            knowledge=self.knowledge_pool,
            forget_words=self.forget_words,
            token_limit=_llm_result_limit,
            tokenizer=self.llm.tokenizer,
            reference_ratio=self.reference_ratio,
        ).run()
        _optimized_prompt.append(Interaction(single=True, ask=prompt))
        return _optimized_prompt

    def build_prompt(self, predict_tokens: int = 500) -> Tuple[Optional[PromptItem], List[Interaction]]:
        """
        Optimising context and re-cutting
        """
        user_input = self._build_prompt_buffer()
        if not user_input:
            logger.warning("No Buffer")
            return None, []
        prompt = self.build_context(user_input, predict_tokens=predict_tokens)
        return user_input, prompt
