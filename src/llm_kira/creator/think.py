# -*- coding: utf-8 -*-
# @Time    : 2/18/23 11:52 AM
# @FileName: think.py
# @Software: PyCharm
# @Github    ：sudoskys

# 仿生系统

"""
我们采用一种 溢满 方法决定当前状态。采用时间+事件驱动方法分别对不同的 profile 对象改动状态类。
同样采用内置记忆类型
特征组和规则组

我们根据影响因子 + 影响效果勾连各个情感类型。然后用 rank 计算当前的情绪状态。

外部的 hook,内部的状态接口：活力，频次，睡眠
"""
import random
import time
from typing import List, Optional, Union

from loguru import logger
from pydantic import BaseModel, root_validator

from llm_kira.client import Conversation
from llm_kira.utils.data import Bucket


class Hook(BaseModel):
    name: str
    trigger: str = ""
    value: int = 2
    time: int = int(time.time())
    last: int = 60

    # safe_rank: range = range(-10, 10)

    @root_validator
    def start_check(cls, values):
        name, trigger = values.get('name'), values.get('trigger')
        if not trigger:
            values["trigger"] = name
        return values


# 钩子迭代机
class HookPool(BaseModel):
    hook: List[Hook] = []
    last_hook_result: List[Hook] = []
    time: int = int(time.strftime("%Y%m%d%H", time.localtime()))


class HookRank(BaseModel):
    hook: Hook
    rank: int = 0
    value: int = 0


class ThinkEngine(object):
    def __init__(self,
                 profile: Union[Conversation, int],
                 ):
        uid = profile
        if isinstance(profile, Conversation):
            uid = profile.conversation_id
        self._hook = {}
        self.bucket = Bucket(uid=uid, area="think_hook")
        _read_hook = self.bucket.get()
        self._hook_pool = HookPool(**_read_hook)
        self.__life_end()

    @property
    def time_stamp(self):
        return int(time.time())

    @property
    def hour_stamp(self):
        return int(time.strftime("%Y%m%d%H", time.localtime()))

    @property
    def hook_pool(self):
        return self._hook_pool

    def __life_end(self):
        """
        统计状态
        """
        # 销毁过时 Hook
        _hook = []
        for item in self._hook_pool.hook:
            if (self.time_stamp - item.time) < item.last:
                _hook.append(item)
        self._hook_pool.hook = _hook
        # 惊喜
        _seed = random.randint(0, 100)
        if _seed > 96:
            self._hook_pool.last_hook_result = []
        # 小时重装
        if self._hook_pool.time != self.hour_stamp:
            # 状态更新
            _old = self.__rank_bucket()
            if _old:
                self._hook_pool.last_hook_result.append(_old[0].hook)
                self._hook_pool.hook = []
            # 标记点更新
            self._hook_pool.time = self.hour_stamp
        self.__save_hook()

    def __save_hook(self):
        self.bucket.set(self._hook_pool.dict())

    def __rank_bucket(self) -> List[HookRank]:
        """
        计算最多的 Hook
        """
        _rank = self._hook_pool.hook
        hook_dict = {}
        for hook in _rank:
            if not hook_dict.get(hook.name):
                hook_dict[hook.name] = {}
                hook_dict[hook.name]["value"] = 0
                hook_dict[hook.name]["rank"] = 0
            hook_dict[hook.name]["hook"] = hook.dict()
            hook_dict[hook.name]["value"] += hook.value
            hook_dict[hook.name]["rank"] += abs(hook.value)
        _hook = list(hook_dict.values())
        _hook_rank = (sorted(_hook, key=lambda i: i['rank'], reverse=True))
        _return = []
        for item in _hook_rank:
            _return.append(HookRank(**item))
        return _return

    @property
    def is_night(self):
        if time.localtime().tm_hour < 5 or time.localtime().tm_hour > 23:
            return True
        return False

    def register_hook(self, hook: Hook):
        self._hook[hook.name] = hook
        return hook

    @property
    def last_time_status(self) -> Optional[Hook]:
        if not self._hook_pool.last_hook_result:
            return None
        return self._hook_pool.last_hook_result[-1]

    def hook(self, name: str) -> None:
        if not self._hook.get(name):
            logger.warning(f"Hook:{name} Do not Exist!?")
            return
        # 更新Hook池
        self._hook_pool.hook.append(self._hook.get(name))
        self.__save_hook()

    def build_status(self, trans: bool = False, rank: int = 50) -> List[str]:
        self.__life_end()
        _before = self.__rank_bucket()
        _after = self.last_time_status
        _trans = []
        if _before:
            if _before[0].rank > rank:
                _trans.append(f"become {_before[0].hook.trigger}")
                if _after and trans:
                    _trans.append(f"after being {_after.trigger}")
        return _trans
