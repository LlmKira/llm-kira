# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 上午12:12
# @Author  : sudoskys
# @File    : elara.py
# @Software: PyCharm
import json

import elara


# TODO 继承我们的抽象类
class ElaraWorker:
    """
    Redis 数据基类
    不想用 redis 可以自动改动此类，换用其他方法。应该比较简单。
    """

    def __init__(self, filepath, prefix='llm_kira_'):
        self.redis = elara.exe(filepath)
        self.prefix = prefix

    def set_key(self, key, obj):
        self.redis.set(self.prefix + str(key), json.dumps(obj, ensure_ascii=False))
        self.redis.commit()
        return True

    def delete_key(self, key):
        self.redis.rem(key)
        return True

    def get_key(self, key):
        result = self.redis.get(self.prefix + str(key))
        if result:
            return json.loads(result)
        else:
            return {}

        # TODO
