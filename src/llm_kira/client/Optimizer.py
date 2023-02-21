# -*- coding: utf-8 -*-
# @Time    : 12/9/22 5:47 PM
# @FileName: Summer.py
# @Software: PyCharm
# @Github    ：sudoskys
"""
优化器
"""
import math
from operator import attrgetter
from typing import List, Tuple
import datetime
from .types import Interaction, PromptItem, InteractionWeight
from ..utils.chat import Utils, Sim
from ..utils.data import MsgFlow


def cal_time_seconds(stamp1, stamp2):
    time1 = datetime.datetime.fromtimestamp(stamp1)
    time2 = datetime.datetime.fromtimestamp(stamp2)
    return (time2 - time1).total_seconds()


def sim_forget(sim, hour, rank=0.5):
    S = sim * rank
    if S == 0:
        S = 0.001
    # 一天的时间
    R = math.exp(hour * math.log(0.9) / S)
    return R


def get_head_foot(prompt: str, cap: int = 12):
    body = prompt
    head = ""
    if ":" in prompt[:cap]:
        _split = prompt.split(":", 1)
        if len(_split) > 1:
            body = _split[1]
            head = _split[0]
    if not body:
        body = "."
    return head, body


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


def convert_msgflow_to_list(message: List[Interaction],
                            sign: bool = True
                            ) -> list:
    """
    提取以单条 msgflow 组成的列表的回复。
    :param message:消息列表
    :param sign: 是否签名
    :return:
    """
    _result = []
    for item in message:
        ask, reply = MsgFlow.parse(interaction=item, sign=sign)
        _result.append(ask)
        _result.append(reply)
    return _result


def build_weight(convert: List[Interaction]):
    """
    构造权重表
    [
        [some,[1, 12]],
        [some,[1, 12]],
    ]
    """
    _returner = []
    _covert = []
    for item in convert:
        try:
            _ = item.time
        except Exception as e:
            pass
        else:
            _covert.append(item)
    _covert: List[Interaction]
    _covert.sort(key=attrgetter('time'), reverse=True)
    for item in _covert:
        _returner.append(InteractionWeight(interaction=item, weight=[]))
    return _returner


class Point(object):
    def run(self):
        pass


class SinglePoint(Point):
    def __init__(self,
                 tokenizer,
                 prompt: PromptItem = "",
                 desc: str = "",
                 attention: int = 3,
                 interaction: List[Interaction] = None,
                 knowledge: List[Interaction] = None,
                 reference_ratio: float = 0.2,
                 token_limit: int = 2000,
                 forget_words: List[str] = None,
                 ):
        self.token_limit = token_limit
        self.prompt = prompt
        self.desc = desc
        self.tokenizer = tokenizer
        self.attention = attention
        self.interaction = interaction
        self.knowledge = knowledge
        self.reference_ratio = reference_ratio if 0 <= reference_ratio <= 1 else 0.5
        self.forget_words = forget_words if forget_words else []

    @staticmethod
    def forgetting_curve(x):
        _weight = math.exp(-x / 5) * 100
        _weight = _weight if _weight > 0 else 0
        _weight = _weight if _weight < 100 else 100
        # 推底值
        _weight = _weight if _weight > 12 else 12
        return _weight

    def _filler(self, _message: List[InteractionWeight], token: int) -> Tuple[List[Interaction], int]:
        __now = 0
        __returner = []
        for __item in _message:
            if __item.score > 0.5 and __now < token:
                __now += self.tokenizer(__item.interaction.raw)
                __returner.append(__item.interaction)
        __returner = list(reversed(__returner))
        return __returner, token - __now

    def run(self) -> List[Interaction]:
        # 单条消息的内容 {"ask": self._restart_sequence+prompt, "reply": self._start_sequence+REPLY[0]}
        prompt = self.prompt
        interaction = build_weight(self.interaction)
        knowledge = build_weight(self.knowledge)
        _knowledge_token_limit = int(self.token_limit * self.reference_ratio)
        _interaction_token_limit = self.token_limit - _knowledge_token_limit
        _key = Utils.tfidf_keywords(prompt.prompt, topK=7)
        _returner = [Interaction(single=True, ask=PromptItem(start="*", text=self.desc))]
        _old_prompt = interaction[:1]
        # Desc
        if self.tokenizer(self.desc) > self.token_limit:
            return _returner

        # interaction attention
        _attention = self.attention if len(interaction) >= self.attention else len(interaction)
        for ir in range(0, _attention):
            interaction[ir].weight.append(70)

        # interaction 遗忘函数
        for i in range(0, len(interaction)):
            _forget = self.forgetting_curve(i)
            interaction[i].weight.append(_forget)

        # interaction 相似度检索
        for item in interaction:
            _content = "".join(item.interaction.content)
            _ask_diff = Sim.cosion_similarity(pre=prompt.prompt, aft=_content)
            _ask_diff = _ask_diff * 100
            _edit_diff = Sim.edit_similarity(pre=prompt.text, aft=_content)
            if _edit_diff < 4:
                score = _ask_diff * 0.2
            else:
                score = _ask_diff if _ask_diff < 90 else 1
            item.weight.append(score)

        # interaction 主题检索
        full_score = len(_key)
        if full_score > 4:
            for item in interaction:
                score = 0
                _content = "".join(item.interaction.content)
                for ir in _key:
                    if ir in _content:
                        score += 1
                _get = (score / full_score) * 100
                _get = _get if _get < 95 else 50
                if _get != 0:
                    item.weight.append(_get)

        # Knowledge Search
        # interaction 遗忘函数
        for item in knowledge:
            item.weight.append(45)
        # 追溯搜索
        # knowledge 相似度检索
        for item in knowledge:
            _content = "".join(item.interaction.content)
            _come_diff = 0
            if _old_prompt:
                _old = _old_prompt[0].interaction.raw
                _come_diff = Sim.cosion_similarity(pre=_old, aft=_content)
            _ask_diff = Sim.cosion_similarity(pre=prompt.prompt, aft=_content)
            _ask_diff = _ask_diff if _ask_diff > _come_diff else _come_diff
            score = _ask_diff * 100
            item.weight.append(score)

        # Fill
        _optimized, _rest = self._filler(_message=knowledge, token=_knowledge_token_limit)
        _returner.extend(_optimized)
        _forget_all = False
        for item in self.forget_words:
            if item in prompt.text:
                _forget_all = True
        if not _forget_all:
            _optimized2, _rest = self._filler(_message=interaction, token=_interaction_token_limit + _rest)
            _returner.extend(_optimized2)
        return _returner
