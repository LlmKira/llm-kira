# -*- coding: utf-8 -*-
# @Time    : 1/23/23 11:16 PM
# @FileName: agent.py
# @Software: PyCharm
# @Github    ：sudoskys
import hashlib
from loguru import logger


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
                 ):
        """
        start_name: 说话者的名字
        restart_name: 回答时候使用的名字
        """
        self.hash_secret = "LLM"
        if not conversation_id:
            conversation_id = getStrId(start_name)
            logger.warning("conversation_id empty!!!")
        self.conversation_id = str(conversation_id)
        self.start_name = start_name.strip(":").strip("：")
        self.restart_name = restart_name.strip(":").strip("：")

    def get_conversation_hash(self):
        uid = f"{str(self.hash_secret)}{str(self.conversation_id)}"
        hash_object = hashlib.sha256(uid.encode())
        return hash_object.hexdigest()
