# -*- coding: utf-8 -*-
# @Time    : 2/17/23 9:27 AM
# @FileName: engine.py
# @Software: PyCharm
# @Github    ：sudoskys
import time
from typing import List, Union, Tuple, Optional

from loguru import logger

from .base import BaseEngine
from ..client import Optimizer
from ..client.agent import Conversation, MemoryManager
from ..client.llms.base import LlmBase
from ..client.types import Interaction, PromptItem
from ..radio.anchor import Antennae
from ..utils.data import MsgFlow


class PromptEngine(BaseEngine):
    """
    设计用于维护提示系统和接入外骨骼
    """

    def __init__(self,
                 profile: Conversation,
                 memory_manger: MemoryManager,
                 knowledge_manger: MemoryManager = None,
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
        :param memory_manger: 记忆管理，可继承重写
        :param llm_model: LLM代理，可扩增
        :param description: Prompt 的恒久状态头
        :param connect_words: 连接Prompt的连接词
        :param reference_ratio: 分配给知识库的 token位 比例
        :param forget_words: 阻断列表，如果 input 中有，则不加入 Prompt
        :param optimizer: 优化器，可以覆写，按照模板继承即可
        """
        self.profile = profile
        self.llm = llm_model
        self.memory_manger = memory_manger
        self.knowledge_manger = knowledge_manger
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
        self.interaction_pool: List[Interaction] = self.memory_manger.read_context()
        self.knowledge_pool: List[Interaction] = []
        if self.knowledge_manger:
            self.knowledge_pool = self.knowledge_manger.read_context()
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
        return self.memory_manger.read_context()

    def save_interaction(self):
        return self.memory_manger.save_context(self.interaction_pool, override=True)

    def read_knowledge(self):
        if self.knowledge_manger:
            return self.knowledge_manger.read_context()

    def save_knowledge(self):
        if self.knowledge_manger:
            return self.knowledge_manger.save_context(self.interaction_pool, override=True)

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

    async def build_skeleton(self,
                             skeleton: Antennae,
                             query: Union[PromptItem, str],
                             llm_task: str = None) -> List[Interaction]:
        """
        异步的外骨骼，用于启动第三方接口提供的知识参考
        :return 列表类型的互动数据
        """
        prompt = query
        if not isinstance(query, str):
            prompt = query.text
        if llm_task:
            llm_result = await self.llm.task_context(task=llm_task,
                                                     predict_tokens=30,
                                                     prompt=prompt)
            prompt = llm_result.reply[0]
        knowledge: List[Interaction]
        knowledge = await skeleton.run(prompt=prompt)
        return knowledge

    def build_context(self, prompt: PromptItem, predict_tokens) -> List[Interaction]:
        # Resize
        _llm_result_limit = self.llm.get_token_limit() - predict_tokens
        _llm_result_limit = _llm_result_limit if _llm_result_limit > 0 else 1
        if _llm_result_limit < 10:
            logger.warning("llm free mem lower than 10...may limit too low or predict token too high")

        # 基准点 prompt.prompt
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


class Preset(object):
    """
    预设和角色
    """

    def __init__(self, profile: Conversation):
        self.profile = profile

    @staticmethod
    def add_tail(switch: bool = False, sentence: str = "", tail: str = ""):
        if switch:
            return f"{sentence}{tail}"
        else:
            return f"{sentence}"

    def character(self,
                  character: list = None,
                  lang: str = "ZH"
                  ) -> list:
        if character:
            return character
        lang = lang.upper()
        if lang == "ZH":
            return [
                "幽默地"
            ]
        elif lang == "EN":
            return [
                "helpful",
            ]
        elif lang == "JA":
            return ["教育された",
                    "ユーモラスに",
                    "興味深い"
                    ]
        else:
            return [
                "helpful",
                "Interesting",
            ]

    def role(self, role: str = "",
             restart_name: str = "",
             character: str = "",
             is_need_help: bool = True,
             lang: str = "ZH"
             ) -> str:
        if role:
            return role
        lang = lang.upper()
        role = ""
        if lang == "ZH":
            role = f"{restart_name} 是 {character}"
            role = self.add_tail(is_need_help, sentence=role, tail=" 的助手.")
        elif lang == "EN":
            role = f"{restart_name} is  {character}.."
            role = self.add_tail(is_need_help, sentence=role, tail=" Assistant")
        elif lang == "JA":
            role = f"{restart_name} は {character}. "
            role = self.add_tail(is_need_help, sentence=role, tail="指導提供")
        return f"{role} "

    def head(self,
             head: str = "",
             prompt_iscode: bool = False,
             lang: str = "ZH"
             ) -> str:
        if head:
            return head
        lang = lang.upper()
        head = ""
        start_name = self.profile.start_name
        restart_name = self.profile.restart_name
        if lang == "ZH":
            head = f"下面是聊天内容,"
            head = self.add_tail(prompt_iscode, sentence=head, tail="提供编程指导,")
        elif lang == "EN":
            head = f"Here is {restart_name}`s Chat,"
            head = self.add_tail(prompt_iscode, sentence=head, tail="Provide programming guidance,")
        elif lang == "JA":
            head = f"{start_name}{restart_name}の会話,"
            head = self.add_tail(prompt_iscode, sentence=head, tail="プログラミング指導を提供する,")
        return f"{head}"


class MiddlePrompt(object):
    def __init__(self, prompt: PromptEngine = None, limit_token: int = 2000):
        self.prompt = prompt
        self.limit_token: int = limit_token
