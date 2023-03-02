# -*- coding: utf-8 -*-
# @Time    : 1/23/23 11:16 PM
# @FileName: agent.py
# @Software: PyCharm
# @Github    ：sudoskys
import hashlib
from typing import List
from loguru import logger

from .types import Interaction
from ..utils.data import MsgFlow


def getStrId(string):
    bytes_str = string.encode('utf-8')
    md5 = hashlib.md5()
    md5.update(bytes_str)
    h16 = md5.hexdigest()
    return int(h16, 16)


class Conversation(object):
    """基础身份类型，供其他模块使用"""

    def __init__(self, start_name: str,
                 restart_name: str,
                 conversation_id: int = 1,
                 init_usage: int = 0
                 ):
        """
        start_name: 说话者的名字
        restart_name: 回答时候使用的名字
        conversation_id: 对话 ID，很重要，如果不存在会计算 start_name 的 唯一ID 作为 ID
        init_usage: int 初始计费
        """
        self.hash_secret = "LLM"
        if not conversation_id:
            conversation_id = getStrId(start_name)
            logger.warning("conversation_id empty!!!")
        self.conversation_id = str(conversation_id)
        self.start_name = start_name.strip(":").strip("：")
        self.restart_name = restart_name.strip(":").strip("：")
        self.__usage = init_usage if init_usage > 0 else 0

    def get_conversation_hash(self):
        uid = f"{str(self.hash_secret)}{str(self.conversation_id)}"
        hash_object = hashlib.sha256(uid.encode())
        return hash_object.hexdigest()

    def get_round_usage(self):
        return self.__usage

    def update_usage(self, usage: int = 0, override: bool = False):
        if override:
            self.__usage = usage
        else:
            self.__usage += usage


class MemoryManager(object):
    def __init__(self,
                 profile: Conversation,
                 ):
        """
        记忆管理器
        """
        self.profile = profile
        self._DataManager = MsgFlow(uid=self.profile.conversation_id)

    def reset_chat(self):
        return self._DataManager.forget()

    def read_context(self) -> List[Interaction]:
        return self._DataManager.read()

    def save_context(self, message: List[Interaction], override: bool = True):
        self._DataManager.save(interaction_flow=message, override=override)
        return message

