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


class Scorer(object):
    @staticmethod
    def cal_time_seconds(stamp1, stamp2):
        time1 = datetime.datetime.fromtimestamp(stamp1)
        time2 = datetime.datetime.fromtimestamp(stamp2)
        return (time2 - time1).total_seconds()

    @staticmethod
    def sim_forget(sim, hour, rank=0.5):
        S = sim * rank
        if S == 0:
            S = 0.001
        # 一天的时间
        R = math.exp(hour * math.log(0.9) / S)
        return R

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def convert_msgflow_to_list(
            message: List[Interaction],
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

    @staticmethod
    def topic_score(topics: List[str], sentence: str) -> float:
        """
        range:0~1
        """
        _full_score = len(topics)
        if _full_score == 0:
            return 0
        _score = 0
        for ir in topics:
            if ir in sentence:
                _score += 1
        _get = (_score / _full_score)
        return _get


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
                 reference_ratio: float = 0.15,
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

    def _filler(self,
                _message: List[InteractionWeight],
                token: int,
                sort_by_weight: bool = False,
                reversed_sort: bool = True,
                just_score: float = 0.5,
                ) -> Tuple[List[Interaction], int]:
        __now = 0
        __returner = []
        if sort_by_weight:
            _message.sort(key=attrgetter('score'), reverse=True)
        for __item in _message:
            if __item.score > just_score and __now < token:
                __now += self.tokenizer(__item.interaction.raw)
                __returner.append(__item.interaction)
        if reversed_sort:
            __returner = list(reversed(__returner))
        return __returner, token - __now

    def run(self) -> List[Interaction]:
        # 单条消息的内容 {"ask": self._restart_sequence+prompt, "reply": self._start_sequence+REPLY[0]}
        prompt = self.prompt
        interaction = Scorer.build_weight(self.interaction)
        knowledge = Scorer.build_weight(self.knowledge)
        _knowledge_token_limit = int(self.token_limit * self.reference_ratio)
        _interaction_token_limit = self.token_limit - _knowledge_token_limit

        # Desc
        _returner = [Interaction(single=True, ask=PromptItem(start="system", text=self.desc))]

        _old_prompt = interaction[:1]
        # Desc
        if self.tokenizer(self.desc) > self.token_limit:
            return _returner

        # interaction attention
        _attention = self.attention if len(interaction) > self.attention else len(interaction)
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
                # 超相近惩罚措施
                score = _ask_diff * 0.22
            else:
                score = _ask_diff if _ask_diff < 90 else 1
            item.weight.append(score)

        # interaction 主题检索
        _key = Utils.tfidf_keywords(prompt.prompt, topK=7)
        if len(_key) > 4:
            for item in interaction:
                _content = "".join(item.interaction.content)
                _score = Scorer.topic_score(topics=_key, sentence=_content) * 100  # IMPORTANT
                _score = _score if _score < 90 else 90
                if _score != 0:
                    item.weight.append(_score)

        # Knowledge Search
        # knowledge 搜索引擎优待函数
        # _attention = 3 if len(knowledge) > 3 else len(knowledge)
        # for i in range(0, _attention):
        #    knowledge[i].weight.append(80)

        # knowledge 相似度检索
        for item in knowledge:
            _content = "".join(item.interaction.content)
            _come_diff = 0
            if _old_prompt:
                _old = _old_prompt[0].interaction.raw
                _come_diff = Sim.cosion_similarity(pre=_old, aft=_content)
            _ask_diff = Sim.cosion_similarity(pre=prompt.prompt, aft=_content)
            _ask_diff = _ask_diff if _ask_diff > _come_diff else _come_diff
            score = _ask_diff * 100 + 31
            item.weight.append(score)

        # knowledge 梯度初始权重
        for i in range(0, len(knowledge)):
            _forget = self.forgetting_curve(i)
            knowledge[i].weight.append(_forget)

        # Fill
        _optimized, _rest = self._filler(_message=knowledge,
                                         token=_knowledge_token_limit,
                                         sort_by_weight=True,
                                         reversed_sort=False)
        _returner.extend(_optimized)
        _forget_all = False
        for item in self.forget_words:
            if item in prompt.text:
                _forget_all = True
        if not _forget_all:
            _optimized2, _rest = self._filler(_message=interaction, token=_interaction_token_limit + _rest)
            _returner.extend(_optimized2)
        return _returner
