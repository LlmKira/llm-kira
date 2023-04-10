# -*- coding: utf-8 -*-
# @Time    : 2023/4/4 下午9:24
# @Author  : sudoskys
# @File    : bucket.py
# @Software: PyCharm
from typing import List, Union

from llm_kira.client import Conversation
from llm_kira.schema import Interaction


# 这里是继承身份类的记忆组件，用于存储和读取记忆。

class MemoryManager(object):
    def __init__(self,
                 profile: Union[Conversation, int],
                 area: str = ""
                 ):
        """
        记忆管理器
        :param profile: Conversation 类型，或者 conversation_id
        :param area: 后缀，用于区分不同的记忆空间
        """
        if not isinstance(profile, int):
            profile = profile.conversation_id
        self._DataManager = MsgFlow(uid=f"{profile}{area}")

    def reset_bucket(self):
        return self._DataManager.forget()

    def read_bucket(self) -> List[Interaction]:
        return self._DataManager.read()

    def save_bucket(self, message: List[Interaction], override: bool = True):
        self._DataManager.save(interaction_flow=message, override=override)
        return message
