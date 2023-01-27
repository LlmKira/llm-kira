# -*- coding: utf-8 -*-
# @Time    : 12/9/22 5:47 PM
# @FileName: Summer.py
# @Software: PyCharm
# @Github    ：sudoskys
"""
优化器
"""
from typing import List

import numpy

from .types import Memory_Flow
from ..utils.chat import Utils, Detect
from ..utils.data import MsgFlow


def random_string(length):
    """
    生成随机字符串
    :param length:
    :return:
    """
    import string
    import random
    all_chars = string.ascii_letters + string.digits
    result = ''
    for i in range(length):
        result += random.choice(all_chars)
    return result


def convert_msgflow_to_list(msg_list: List[Memory_Flow],
                            sign: bool = True
                            ) -> list:
    """
    提取以单条 msgflow 组成的列表的回复。
    :param msg_list:消息列表
    :param sign: 是否签名
    :return:
    """
    _result = []
    for ir in msg_list:
        ask, reply = MsgFlow.get_content(memory_flow=ir, sign=sign)
        _result.append(ask)
        _result.append(reply)
    return _result


class Point(object):

    def run(self):
        pass


class MatrixPoint(Point):
    def __init__(self,
                 tokenizer,
                 prompt: str = "",
                 memory: List[dict] = None,
                 attention: int = 3,
                 # start_token: int = 0,
                 extra_token: int = 0,
                 token_limit: int = 2000,
                 ):
        """
                群组多点聊天，适用于 Group 的记忆对象
                数据清洗采用权重设定，而不操作元素删减
                :param token_limit: 提示词限制
                # :param start_token: 中间件传过来的 token,作为限制的初始值
                :param attention: 注意力
                :param prompt: 记忆提示
                :param extra_token: 需要预留的位置
                :param memory: 记忆桶数据
                :return: 新的列表
        """
        self.memory = memory
        self.attention = attention
        # self.start_token = start_token
        self.extra_token = extra_token
        self.token_limit = token_limit
        self.prompt = prompt
        self.tokenizer = tokenizer

    def run(self) -> list:
        # 单条消息的内容 {"ask": self._restart_sequence+prompt, "reply": self._start_sequence+REPLY[0]}
        memory = self.memory
        attention = self.attention
        prompt = self.prompt
        # start_token = self.start_token
        if self.memory is None:
            return []
        _create_token = self.token_limit - self.extra_token
        # 入口检查
        if len(memory) - attention < 0:
            return convert_msgflow_to_list(memory)

        def forgetting_curve(x):
            _weight = numpy.exp(-x / 5) * 100 + 10
            # 低谷值
            _weight = _weight if _weight > 0 else 0
            # 高度线
            _weight = _weight if _weight < 100 else 100
            return _weight

        # 转换为
        for item in range(len(memory)):
            memory[item] = memory[item].dict()
        memory: list

        # 计算初始保留比并初始化
        memory = list(reversed(memory))
        for i in range(0, len(memory)):
            _forget = forgetting_curve(i)
            if _forget > 10:
                memory[i]["content"]["weight"] = [_forget]
            else:
                memory[i]["content"]["weight"] = []
        memory = list(reversed(memory))

        # 相似度检索
        for i in range(0, len(memory)):
            ask, reply = MsgFlow.get_content(memory[i], sign=False)
            _diff1 = Utils.cosion_sismilarity(pre=prompt, aft=f"{ask}{reply}")
            _diff = _diff1
            score = _diff * 100
            score = score if score < 95 else 1
            if score != 0:
                memory[i]["content"]["weight"].append(score)

        # 主题检索
        _key = Utils.tfidf_keywords(prompt, topK=5)
        full_score = len(_key)
        if full_score > 5:
            for i in range(0, len(memory)):
                score = 0
                ask, reply = MsgFlow.get_content(memory[i], sign=False)
                for ir in _key:
                    if ir in f"{ask}{reply}":
                        score += 1
                _get = (score / full_score) * 100
                if _get != 0:
                    memory[i]["content"]["weight"].append(_get)  # 基准数据，置信为 0.5 百分比

        # 预处理
        for i in range(0, len(memory) - attention):
            ask, reply = MsgFlow.get_content(memory[i], sign=False)
            if self.tokenizer(f"{ask}{reply}") > 240:
                if Detect.get_text_language(sentence=f"{ask}{reply}") == "CN":
                    _sum = Utils.tfidf_summarization(sentence=f"{ask}{reply}", ratio=0.5)
                    if len(_sum) > 7:
                        memory[i]["content"]["ask"] = "info"
                        memory[i]["content"]["reply"] = _sum

        # 进行筛选，计算限制
        _msg_flow = []
        _msg_return = []
        _now_token = 0
        memory = sorted(memory, key=lambda x: x['time'], reverse=True)
        for i in range(0, len(memory)):
            total = len(memory[i]["content"]["weight"])
            full_score = total * 100
            score = sum(memory[i]["content"]["weight"])
            level = (score / full_score) * 100
            ask, reply = MsgFlow.get_content(memory[i], sign=True)
            if level > 30:
                _now_token += self.tokenizer(f"{ask}{reply}")
                if _now_token > _create_token:
                    break
                _msg_flow.append(memory[i])
        _msg_flow = sorted(_msg_flow, key=lambda x: x['time'], reverse=False)
        # print(_msg_flow)
        _msg_flow_list = convert_msgflow_to_list(_msg_flow)
        _msg_return.extend(_msg_flow_list)
        return _msg_flow_list


class SinglePoint(Point):
    def __init__(self,
                 tokenizer,
                 prompt: str = "",
                 memory: List[dict] = None,
                 attention: int = 3,
                 # start_token: int = 0,
                 extra_token: int = 0,
                 token_limit: int = 2000,
                 ):
        """
                单点聊天，更准确
                数据清洗采用权重设定，而不操作元素删减
                :param token_limit: 提示词限制
                # :param start_token: 中间件传过来的 token,作为限制的初始值
                :param attention: 注意力
                :param prompt: 记忆提示
                :param extra_token: 需要预留的位置
                :param memory: 记忆桶数据
                :return: 新的列表
        """
        self.memory = memory
        self.attention = attention
        # self.start_token = start_token
        self.extra_token = extra_token
        self.token_limit = token_limit
        self.prompt = prompt
        self.tokenizer = tokenizer

    def run(self) -> list:
        # 单条消息的内容 {"ask": self._restart_sequence+prompt, "reply": self._start_sequence+REPLY[0]}
        memory = self.memory
        attention = self.attention
        prompt = self.prompt
        # start_token = self.start_token
        if self.memory is None:
            memory = []
        _create_token = self.token_limit - self.extra_token

        # 入口检查
        if len(memory) - attention < 0:
            return convert_msgflow_to_list(memory)

        # 转换为
        for item in range(len(memory)):
            if isinstance(memory[item], Memory_Flow):
                memory[item] = memory[item].dict()
        memory: list

        def forgetting_curve(x):
            _weight = numpy.exp(-x / 5) * 100
            # 低谷值
            _weight = _weight if _weight > 0 else 0
            # 高度线
            _weight = _weight if _weight < 100 else 100
            # 推底值，防止无法唤起
            _weight = _weight if _weight > 15 else 15
            return _weight

        # 计算初始保留比并初始化
        memory = list(reversed(memory))
        for i in range(0, len(memory)):
            _forget = forgetting_curve(i)
            if _forget > 5:
                memory[i]["content"]["weight"] = [_forget]
            else:
                memory[i]["content"]["weight"] = []
        memory = list(reversed(memory))

        # 筛选标准发言
        _index = []
        for i in range(0, len(memory) - attention):
            ask, reply = MsgFlow.get_content(memory[i], sign=False)
            if len(ask) < 1 or len(reply) < 1:
                memory[i]["content"]["weight"].append(-1000)

        # 相似度检索
        for i in range(0, len(memory)):
            ask, reply = MsgFlow.get_content(memory[i], sign=False)
            _diff1 = Utils.cosion_sismilarity(pre=prompt, aft=ask)
            _diff2 = Utils.cosion_sismilarity(pre=prompt, aft=reply)
            _diff = _diff1 if _diff1 > _diff2 else _diff2
            score = _diff * 100
            score = score if score < 90 else 0
            if score != 0:
                memory[i]["content"]["weight"].append(score)
            if ask == reply:
                memory[i]["content"]["ask"] = ""

        # 主题检索
        _key = Utils.tfidf_keywords(prompt, topK=5)
        full_score = len(_key)
        if full_score > 5:
            for i in range(0, len(memory)):
                score = 0
                ask, reply = MsgFlow.get_content(memory[i], sign=False)
                for ir in _key:
                    if ir in f"{ask}{reply}":
                        score += 1
                _get = (score / full_score) * 100
                _get = _get if _get < 95 else 50
                if _get != 0:
                    memory[i]["content"]["weight"].append(_get)  # 基准数据，置信为 0.5 百分比

        # 进行筛选，计算限制
        _msg_flow = []
        _msg_return = []
        _now_token = 0
        memory = sorted(memory, key=lambda x: x['time'], reverse=True)
        for i in range(0, len(memory)):
            total = len(memory[i]["content"]["weight"])
            full_score = total * 100
            score = sum(memory[i]["content"]["weight"])
            level = (score / full_score) * 100
            ask, reply = MsgFlow.get_content(memory[i], sign=True)
            if level > 50:
                _now_token += self.tokenizer(f"{ask}{reply}")
                if _now_token > _create_token:
                    break
                _msg_flow.append(memory[i])
        _msg_flow = sorted(_msg_flow, key=lambda x: x['time'], reverse=False)
        # print(_msg_flow)
        _msg_flow_list = convert_msgflow_to_list(_msg_flow)
        _msg_return.extend(_msg_flow_list)
        return _msg_flow_list
