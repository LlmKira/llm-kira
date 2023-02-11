# -*- coding: utf-8 -*-
# @Time    : 12/27/22 8:00 PM
# @FileName: plugins.py
# @Software: PyCharm
# @Github    ：sudoskys
import tiktoken
from loguru import logger

from ....utils import setting
from ....utils import network
from ....utils.chat import Utils, Sim

gpt_tokenizer = tiktoken.get_encoding("gpt2")

netTool = network


class PromptTool(object):
    @staticmethod
    def help_words_list():
        return ["怎么做",
                "How",
                "how",
                "如何做",
                "帮我",
                "帮助我",
                "请给我",
                "给出建议",
                "给建议",
                "给我建议",
                "给我一些",
                "请教",
                "建议",
                "步骤",
                "怎样",
                "如何",
                "怎么样",
                "为什么",
                "帮朋友",
                "怎么",
                "需要什么",
                "注意什么",
                "怎么办"] + ['怎麼做', '如何做', '幫我', '幫助我', '請給我', '給出建議', '給建議', '給我建議',
                             '給我一些', '請教', '建議', '步驟', '怎樣', '如何', '怎麼樣', '為什麼', '幫朋友',
                             '怎麼', '需要什麼', '註意什麼', '怎麼辦'] + ['助け', '何を', 'なぜ', '教えて', '提案',
                                                                          '何が', '何に']

    @staticmethod
    def isStrIn(prompt: str, keywords: list):
        isIn = False
        for i in keywords:
            if i in prompt:
                isIn = True
        return isIn

    @staticmethod
    def isStrAllIn(prompt: str, keywords: list):
        isIn = True
        for i in keywords:
            if i not in prompt:
                isIn = False
        return isIn

    @staticmethod
    def match_enhance(prompt):
        import re
        match = re.findall(r"\[(.*?)\]", prompt)
        match2 = re.findall(r"\"(.*?)\"", prompt)
        match3 = re.findall(r"\((.*?)\)", prompt)
        match.extend(match2)
        match.extend(match3)
        return match


class NlP(object):
    @staticmethod
    def get_webServerStopSentence():
        return setting.webServerStopSentence

    @staticmethod
    def get_is_filter_url():
        return setting.webServerUrlFilter

    @staticmethod
    def summary(text: str, ratio: float = 0.5) -> str:
        return Utils.tfidf_summarization(sentence=text, ratio=ratio)

    @staticmethod
    def keyPhrase(text: str, ) -> str:
        return Utils.keyPhraseExtraction(sentence=text)

    @staticmethod
    def nlp_filter_list(prompt, material: list):
        if not material or not isinstance(material, list):
            return []
        logger.trace(f"NLP")
        # 双匹配去重
        while len(material) > 2:
            prev_len = len(material)
            _pre = material[0]
            _afe = material[1]
            sim = Sim.simhash_similarity(pre=_pre, aft=_afe)
            if sim < 12:
                _remo = _afe if len(_afe) > len(_pre) else _pre
                # 移除过于相似的
                material.remove(_remo)
            if len(material) == prev_len:
                break

        while len(material) > 2:
            prev_len = len(material)
            material_len = len(material)
            for i in range(0, len(material), 2):
                if i + 1 >= material_len:
                    continue
                _pre = material[i]
                _afe = material[i + 1]
                sim = Sim.cosion_similarity(pre=_pre, aft=_afe)
                if sim > 0.7:
                    _remo = _afe if len(_afe) > len(_pre) else _pre
                    # 移除过于相似的
                    material.remove(_remo)
                    material_len = material_len - 1
            if len(material) == prev_len:
                break

        # 去重排序+删除无关
        material_ = {item: -1 for item in material}
        material = list(material_.keys())
        _top_table = {}
        for item in material:
            _top_table[item] = Sim.cosion_similarity(pre=prompt, aft=item)
        material = {k: v for k, v in _top_table.items() if v > 0.15}
        # 搜索引擎比相似度算法靠谱所以注释掉了
        # material = OrderedDict(sorted(material.items(), key=lambda t: t[1]))
        # logger.trace(material)

        # 关联度指数计算
        _key = Utils.tfidf_keywords(prompt, topK=7)
        _score = 0
        _del_keys = []
        for k, i in material.items():
            for ir in _key:
                if ir in k:
                    _score += 1
            if _score / len(_key) < 0.3:
                _del_keys.append(k)
        for k in _del_keys:
            material.pop(k)
        return list(material.keys())
