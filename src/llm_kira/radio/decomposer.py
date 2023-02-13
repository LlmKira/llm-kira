# -*- coding: utf-8 -*-
# @Time    : 2/12/23 11:20 PM
# @FileName: decomposer.py
# @Software: PyCharm
# @Github    ：sudoskys
from ..utils.chat import Utils, Sim

from .setting import STOP_SENTENCE
from typing import List, Tuple, Optional
from loguru import logger

import re
from goose3 import Goose
from inscriptis import get_text


class Filter(object):
    @staticmethod
    def english_sentence_cut(text) -> list:
        list_ = list()
        for s_str in text.split('.'):
            if '?' in s_str:
                list_.extend(s_str.split('?'))
            elif '!' in s_str:
                list_.extend(s_str.split('!'))
            else:
                list_.append(s_str)
        return list_

    @staticmethod
    def chinese_sentence_cut(text) -> list:
        text = re.sub('([。！？\?])([^’”])', r'\1\n\2', text)
        # 普通断句符号且后面没有引号
        text = re.sub('(\.{6})([^’”])', r'\1\n\2', text)
        # 英文省略号且后面没有引号
        text = re.sub('(\…{2})([^’”])', r'\1\n\2', text)
        # 中文省略号且后面没有引号
        text = re.sub('([.。！？\?\.{6}\…{2}][’”])([^’”])', r'\1\n\2', text)
        # 断句号+引号且后面没有引号
        return text.split("\n")

    @staticmethod
    def __filter_sentence(sentence: str, filter_url: bool = False) -> str:
        import re
        stop_sentence = STOP_SENTENCE
        skip = False
        for ir in stop_sentence:
            if ir in sentence:
                skip = True
        if filter_url:
            pas = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
            _link = re.findall(pas, sentence)
            if _link:
                for i in _link:
                    sentence = sentence.replace(i, "")
            _link = re.findall("(?:[\w-]+\.)+[\w-]+", sentence)
            if _link:
                if len("".join(_link)) / len(sentence) > 0.7:
                    skip = True
                for i in _link:
                    sentence = sentence.replace(i, "")
        if skip:
            return ""
        # 处理数据
        sentence = sentence.strip(".").strip("…").replace('\xa0', '').replace('   ', '').replace("/r", '')
        sentence = sentence.replace("/v", '').replace("/s", '').replace("/p", '').replace("/a", '').replace("/d", '')
        sentence = sentence.replace("，", ",").replace("。", ".").replace("\n", ".")
        if 18 < len(sentence):
            return sentence.strip(".")
        else:
            return ""

    def filter(self, sentences: List[str], limit: Tuple[int, int] = (0, 500)):
        _return_str = {}
        for item in sentences:
            _fixed = self.__filter_sentence(item)
            if _fixed:
                _return_str[_fixed] = 0
        _return_ = []
        for item in list(_return_str.keys()):
            if len(item) in range(*limit):
                _return_.extend(self.chinese_sentence_cut(item))
        _return_str = [item for item in _return_ if item and len(item) in range(*limit)]
        return _return_str


class Extract(object):
    @staticmethod
    def english_sentence_cut(text) -> list:
        list_ = list()
        for s_str in text.split('.'):
            if '?' in s_str:
                list_.extend(s_str.split('?'))
            elif '!' in s_str:
                list_.extend(s_str.split('!'))
            else:
                list_.append(s_str)
        return list_

    @staticmethod
    def chinese_sentence_cut(text) -> list:
        text = re.sub('([。！？\?])([^’”])', r'\1\n\2', text)
        # 普通断句符号且后面没有引号
        text = re.sub('(\.{6})([^’”])', r'\1\n\2', text)
        # 英文省略号且后面没有引号
        text = re.sub('(\…{2})([^’”])', r'\1\n\2', text)
        # 中文省略号且后面没有引号
        text = re.sub('([.。！？\?\.{6}\…{2}][’”])([^’”])', r'\1\n\2', text)
        # 断句号+引号且后面没有引号
        return text.split("\n")

    @staticmethod
    def goose_extract(html) -> List[str]:
        _return_raw = []
        try:
            article = Goose().extract(raw_html=html)
            meta = article.meta_description
        except Exception as e:
            logger.trace(e)
            meta = ""
        if meta:
            _return_raw.append(meta)
        return _return_raw

    @staticmethod
    def trafilatura_extract(downloaded):
        import trafilatura
        return Extract.chinese_sentence_cut(trafilatura.extract(downloaded))

    @staticmethod
    def sumy_extract(url, html) -> List[str]:
        _return_raw = []
        from sumy.parsers.html import HtmlParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.lsa import LsaSummarizer as Summarizer
        from sumy.nlp.stemmers import Stemmer
        from sumy.utils import get_stop_words
        LANGUAGE = "chinese"
        SENTENCES_COUNT = 25
        parser = HtmlParser.from_string(string=html, url=url, tokenizer=Tokenizer(LANGUAGE))
        stemmer = Stemmer(LANGUAGE)
        summarizer = Summarizer(stemmer)
        summarizer.stop_words = get_stop_words(LANGUAGE)
        for sentence in summarizer(parser.document, SENTENCES_COUNT):
            _return_raw.append(str(sentence))
        return _return_raw

    @staticmethod
    def inscriptis_extract(html) -> Optional[str]:
        try:
            text = get_text(html)
        except Exception as e:
            logger.trace(e)
            return None
        else:
            return text

    def process_html(self, url, html) -> List[str]:
        _return_raw = []
        _return_raw.extend(self.goose_extract(html))
        _raw_text = self.inscriptis_extract(html)
        _return_raw = self.trafilatura_extract(html)
        if not _raw_text:
            return []
        _summary = self.sumy_extract(url=url, html=_raw_text)
        _return_raw.extend(_summary)
        return _return_raw


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
