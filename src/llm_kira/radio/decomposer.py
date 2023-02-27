# -*- coding: utf-8 -*-
# @Time    : 2/12/23 11:20 PM
# @FileName: decomposer.py
# @Software: PyCharm
# @Github    ：sudoskys
import re
from loguru import logger
from typing import List, Tuple, Optional
from goose3 import Goose
from inscriptis import get_text
from pydantic import BaseModel

from ..utils.chat import Utils, Sim, Cut, DeEmphasis
from .setting import STOP_SENTENCE, HELP_WORDS


class Filter(object):
    @staticmethod
    def url_filter(sentence):
        pas = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        _link = re.findall(pas, sentence)
        if _link:
            for i in _link:
                sentence = sentence.replace(i, "")
        _link = re.findall("(?:[\w-]+\.)+[\w-]+", sentence)
        if _link:
            for i in _link:
                sentence = sentence.replace(i, "")
        return sentence

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
        text = re.sub('([。！？\?])([^’”])', r'\1\n\2', text)  # 普通断句符号且后面没有引号
        text = re.sub('(\.{6})([^’”])', r'\1\n\2', text)  # 英文省略号且后面没有引号
        text = re.sub('(\…{2})([^’”])', r'\1\n\2', text)  # 中文省略号且后面没有引号
        text = re.sub('([.。！？\?\.{6}\…{2}][’”])([^’”])', r'\1\n\2', text)  # 断句号+引号且后面没有引号
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
        pas = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        _link = re.findall(pas, sentence)
        sentence = sentence.strip(".").strip("…").replace('\xa0', '').replace('   ', '').replace("/r/r", '')
        sentence = sentence.replace("/v/v", '').replace("/s/s", '').replace("/p/p", '').replace("/a/a", '').replace(
            "/d/d",
            '')
        sentence = sentence.replace("\u2002", "")
        if not _link:
            sentence = sentence.strip(".").strip("…").replace('\xa0', '').replace('   ', '').replace("/r", '')
            sentence = sentence.replace("/v", '').replace("/s", '').replace("/p", '').replace("/a", '').replace("/d",
                                                                                                                '')
        sentence = sentence.replace("，", ",").replace("。", ".").replace("\n", ".")
        if 18 < len(sentence):
            return sentence.strip(".")
        else:
            return ""

    def filter(self, sentences: List[str], limit: Tuple[int, int] = (0, 500), filter_url: bool = True):
        _return_str = {}
        for item in sentences:
            _fixed = self.__filter_sentence(item, filter_url=filter_url)
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
        text = re.sub('([。！？\?])([^’”])', r'\1\n\2', text)  # 普通断句符号且后面没有引号
        text = re.sub('(\.{6})([^’”])', r'\1\n\2', text)  # 英文省略号且后面没有引号
        text = re.sub('(\…{2})([^’”])', r'\1\n\2', text)  # 中文省略号且后面没有引号
        text = re.sub('([.。！？\?\.{6}\…{2}][’”])([^’”])', r'\1\n\2', text)  # 断句号+引号且后面没有引号
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

    """
    @staticmethod
    def trafilatura_extract(downloaded):
        import trafilatura
        return Extract.chinese_sentence_cut(trafilatura.extract(downloaded))
    """

    """
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
    """

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
        """
        负责切分提取网页内容，本来是 Sumy 库负责的，临时改成正则
        # TODO
        """
        _return_raw = []
        _return_raw.extend(self.goose_extract(html))
        _raw_text = self.inscriptis_extract(html)

        # !!!WARNING!!! AGPL 3.0 API!!!
        # _return_raw = self.trafilatura_extract(html)

        if not _raw_text:
            return []

        # _summary = Utils.textrank_summarization(sentence=_raw_text, ratio=0.5)
        # _summary = self.sumy_extract(url=url, html=_raw_text)

        _summary = Cut().cut_sentence(_raw_text)
        _return_raw.extend(_summary)
        return _return_raw


class NlpUtils(object):
    @staticmethod
    def help_words_list():
        cn = ["怎么做", "How", "how",
              "如何做", "帮我", "帮助我",
              "请给我", "给出建议", "给建议",
              "给我建议", "给我一些", "请教",
              "建议", "步骤", "怎样", "如何",
              "怎么样", "为什么", "帮朋友", "怎么",
              "需要什么", "注意什么",
              "怎么办"]
        cn2 = ['怎麼做',
               '如何做', '幫我', '幫助我',
               '請給我', '給出建議', '給建議',
               '給我建議', '給我一些', '請教', '建議',
               '步驟', '怎樣', '如何', '怎麼樣', '為什麼',
               '幫朋友', '怎麼', '需要什麼', '註意什麼',
               '怎麼辦']
        jp = ['助け',
              '何を', 'なぜ',
              '教えて', '提案',
              '何が', '何に']
        return cn + cn2 + jp

    @staticmethod
    def isStrIn(prompt: str, keywords: list):
        isIn = False
        for i in keywords:
            if i in prompt:
                isIn = True
        return isIn

    @staticmethod
    def compression(prompt, material: list):
        if not material or not isinstance(material, list):
            return []
        material = list(DeEmphasis().by_sim(sentence_list=material))
        # 去重排序
        material_ = {item: 1 for item in material}
        material = list(material_.keys())
        _top_table = {}
        for item in material:
            _top_table[item] = Sim.cosion_similarity(pre=prompt, aft=item)
        material = {k: v for k, v in _top_table.items() if v > 0.1}

        # 搜索引擎比相似度算法靠谱所以注释掉了
        # material = OrderedDict(sorted(material.items(), key=lambda t: t[1]))

        # 二倍问题过滤测量
        """
        _del_keys = []
        for k, i in material.items():
            # 调整对于标题的惩罚参数
            _k_real_len = len(Filter.url_filter(k))
            if _k_real_len < len(prompt[:20]) * 2:
                _del_keys.append(k)
        for ks in _del_keys:
            material.pop(ks)
        """

        # 过滤
        USELESS_WORDS = ["怎么", "吗？", "什么", "怎样", "么"]
        _del_keys = []
        for k, i in material.items():
            # 调整对于标题的惩罚参数
            _real_len = len(Filter.url_filter(k[:20]))
            if NlpUtils.isStrIn(keywords=USELESS_WORDS, prompt=k) and _real_len < 30:
                _del_keys.append(k)
        for ks in _del_keys:
            material.pop(ks)

        # 编辑计算
        _del_keys = []
        for k, i in material.items():
            if Sim.edit_similarity(pre=k, aft=prompt) < 5:
                _del_keys.append(k)
        for ks in _del_keys:
            material.pop(ks)

        # 聚类
        material = DeEmphasis().by_tfidf(sentence_list=list(material.keys()), threshold=0.5)
        return material
