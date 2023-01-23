# -*- coding: utf-8 -*-
# @Time    : 1/23/23 11:16 PM
# @FileName: agent.py
# @Software: PyCharm
# @Github    ：sudoskys

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
        self.conversation_id = str(conversation_id)
        self.start_name = start_name.strip(":").strip("：")
        self.restart_name = restart_name.strip(":").strip("：")

    def get_conversation_hash(self):
        import hashlib
        my_string = str(self.conversation_id)
        # 使用 hashlib 模块中的 sha256 算法创建一个散列对象
        hash_object = hashlib.sha256(my_string.encode())
        return hash_object.hexdigest()
