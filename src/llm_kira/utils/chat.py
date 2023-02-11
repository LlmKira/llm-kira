# -*- coding: utf-8 -*-
# @Time    : 12/16/22 2:34 PM
# @FileName: Utils.py
# @Software: PyCharm
# @Github    ：sudoskys
import re
import random
from typing import Union, Callable, List

from llm_kira.error import LLMException
from .data import singleton
from ..client.text_analysis_tools.api.keywords.tfidf import TfidfKeywords
from ..client.text_analysis_tools.api.sentiment.sentiment import SentimentAnalysis
from ..client.text_analysis_tools.api.summarization.textrank_summarization import TextRankSummarization
from ..client.text_analysis_tools.api.summarization.tfidf_summarization import TfidfSummarization
from ..client.text_analysis_tools.api.text_similarity.simhash import SimHashSimilarity
from ..client.text_analysis_tools.api.text_similarity.cosion import CosionSimilarity
from ..client.text_analysis_tools.api.text_similarity.edit import EditSimilarity
from ..client.text_analysis_tools.api.keyphrase.keyphrase import KeyPhraseExtraction
import tiktoken

gpt_tokenizer = tiktoken.get_encoding("gpt2")


def default_gpt_tokenizer(text, raw: bool = False) -> Union[int, list]:
    _token = gpt_tokenizer.encode(text)
    if raw:
        return _token
    return len(_token)


class Detect(object):
    @staticmethod
    def isNeedHelp(sentence) -> bool:
        _check = ['怎么做', 'How', 'how', 'what', 'What', 'Why', 'why', '复述', '复读', '要求你', '原样', '例子',
                  '解释', 'exp', '推荐', '说出', '写出', '如何实现', '代码', '写', 'give', 'Give',
                  '请把', '请给', '请写', 'help', 'Help', '写一', 'code', '如何做', '帮我', '帮助我', '请给我', '什么',
                  '为何', '给建议', '给我', '给我一些', '请教', '建议', '怎样', '如何', '怎么样',
                  '为什么',
                  '帮朋友', '怎么', '需要什么', '注意什么', '怎么办', '助け', '何を', 'なぜ', '教えて', '提案', '何が', '何に',
                  '何をす', '怎麼做', '複述', '復讀', '原樣', '解釋', '推薦', '說出', '寫出', '如何實現', '代碼', '寫',
                  '請把', '請給', '請寫', '寫一', '幫我', '幫助我', '請給我', '什麼', '為何', '給建議', '給我',
                  '給我一些', '請教', '建議', '步驟', '怎樣', '怎麼樣', '為什麼', '幫朋友', '怎麼', '需要什麼',
                  '註意什麼', '怎麼辦']
        _do_things = ['翻译', "翻訳", "函数", "函数", "base64", "encode", "encode", "cript", '脚本', 'code', '步骤',
                      'sdk', 'api',
                      'key', ]
        for item in _check + _do_things:
            if item in sentence:
                return True
        return False

    @staticmethod
    def isCode(sentence) -> bool:
        code = False
        _reco = [
            '("',
            '")',
            ").",
            "()",
            "!=",
            "==",
        ]
        _t = len(_reco)
        _r = 0
        for i in _reco:
            if i in sentence:
                _r += 1
        if _r > _t / 2:
            code = True
        rms = [
            "print_r(",
            "var_dump(",
            'NSLog( @',
            'println(',
            '.log(',
            'print(',
            'printf(',
            'WriteLine(',
            '.Println(',
            '.Write(',
            'alert(',
            'echo(',
        ]
        for i in rms:
            if i in sentence:
                code = True
        return code

    @staticmethod
    def get_text_language(sentence: str):
        try:
            from .fatlangdetect import detect
            lang_type = detect(text=sentence.replace("\n", "").replace("\r", ""), low_memory=True).get("lang").upper()
        except Exception as e:
            from .langdetect import detect
            lang_type = detect(text=sentence.replace("\n", "").replace("\r", ""))[0][0].upper()
        return lang_type

    def gpt_tendency_arg(self, prompt: str,
                         memory: list = None,
                         tokenizer: Callable[[str, bool], Union[int, list]] = default_gpt_tokenizer,
                         lang: str = "CN") -> tuple:

        if memory is None:
            memory = []
        temperature = 0.9
        frequency_penalty = 0
        presence_penalty = 0

        if self.isCode(sentence=prompt):
            return frequency_penalty, presence_penalty, temperature

        if self.isNeedHelp(sentence=prompt):
            temperature -= 0.2
            frequency_penalty -= 0.1
            presence_penalty -= 0.1

        # 控制随机数的精度round(数值，精度)
        # presence_penalty += round(random.uniform(-1, 1) / 10, 2)
        # frequency_penalty += round(random.uniform(-1, 1) / 10, 2)
        _sentiment_score = Utils.sentiment(sentence=prompt).get("score")
        while _sentiment_score > 1.5 or _sentiment_score < -1.5:
            _sentiment_score = _sentiment_score / 10
        _sentiment_score = 0.1 if 0.05 < _sentiment_score < 0.1 else _sentiment_score
        _sentiment_score = -0.1 if -0.1 < _sentiment_score < -0.05 else _sentiment_score
        # 不谈论新话题
        presence_penalty -= _sentiment_score * 0.4
        # 拒绝重复
        frequency_penalty += _sentiment_score * 0.4

        # 验证记忆体
        if len(memory) > 3:
            # 计算回复指数指标
            _token = tokenizer("".join(memory[-4:]), True)
            _repeat_score = 2 * (0.8 - len(set(_token)) / len(_token))
            frequency_penalty = frequency_penalty + _repeat_score
            print(_repeat_score)

        # Fix
        temperature = round(temperature, 1)
        presence_penalty = round(presence_penalty, 1)
        frequency_penalty = round(frequency_penalty, 1)

        # Check
        return frequency_penalty, presence_penalty, temperature


class Utils(object):

    @staticmethod
    def keyPhraseExtraction(sentence: str):
        return KeyPhraseExtraction().key_phrase_extraction(text=sentence)

    @staticmethod
    def sentiment(sentence: str):
        return SentimentAnalysis().analysis(sentence=sentence)

    @staticmethod
    def textrank_summarization(sentence: str, ratio=0.2):
        """
        采用 textrank 进行摘要抽取
        :param sentence: 待处理语句
        :param ratio: 摘要占文本长度的比例
        :return:
        """
        _sum = TextRankSummarization(ratio=ratio)
        _sum = _sum.analysis(sentence)
        return _sum

    @staticmethod
    def tfidf_summarization(sentence: str, ratio=0.5):
        """
        采用tfidf进行摘要抽取
        :param sentence:
        :param ratio: 摘要占文本长度的比例
        :return:
        """
        _sum = TfidfSummarization(ratio=ratio)
        _sum = _sum.analysis(sentence)
        return _sum

    @staticmethod
    def tfidf_keywords(keywords, delete_stopwords=True, topK=5, withWeight=False):
        """
        tfidf 提取关键词
        :param keywords:
        :param delete_stopwords: 是否删除停用词
        :param topK: 输出关键词个数
        :param withWeight: 是否输出权重
        :return: [(word, weight), (word1, weight1)]
        """
        tfidf = TfidfKeywords(delete_stopwords=delete_stopwords, topK=topK, withWeight=withWeight)
        return tfidf.keywords(keywords)

    @staticmethod
    def get_gpt2_tokenizer():
        return gpt_tokenizer


@singleton
class SimilarityUtils(object):
    def __init__(self):
        from similarities import Similarity
        self.SimilarityModel = Similarity(
            model_name_or_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def similarity(self, query: str, corpus: List[str], topn: int = 20) -> dict:
        self.SimilarityModel.corpus = {}
        self.SimilarityModel.corpus_ids_map = {}
        self.SimilarityModel.corpus_embeddings = []
        self.SimilarityModel.add_corpus(corpus)
        res = self.SimilarityModel.most_similar(queries=query, topn=topn)
        _result = {}
        for q_id, c in res.items():
            for corpus_id, score in c.items():
                _result[f"{self.SimilarityModel.corpus[corpus_id]}"] = float(score)
        return _result

    def corpus_similarity(self, sentences1, sentences2):
        model = self.Similarity(model_name_or_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        similarity_score = model.similarity(sentences1, sentences2)
        return float(similarity_score)


class Sim(object):

    @staticmethod
    def cosion_similarity(pre, aft):
        """
        基于余弦计算文本相似性 0 - 1 (1为最相似)
        :return: 余弦值
        """
        _cos = CosionSimilarity()
        _sim = _cos.similarity(pre, aft)
        return _sim

    @staticmethod
    def edit_similarity(pre, aft):
        """
        基于编辑计算文本相似性
        :return: 差距
        """
        _cos = EditSimilarity()
        _sim = _cos.edit_dist(pre, aft)
        return _sim

    @staticmethod
    def simhash_similarity(pre, aft):
        """
        采用simhash计算文本之间的相似性
        :return:
        """
        simhash = SimHashSimilarity()
        sim = simhash.run_simhash(pre, aft)
        # print("simhash result: {}\n".format(sim))
        return sim


class Cut(object):
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

    def cut_chinese_sentence(self, text):
        p = re.compile("“.*?”")
        listr = []
        index = 0
        for i in p.finditer(text):
            temp = ''
            start = i.start()
            end = i.end()
            for j in range(index, start):
                temp += text[j]
            if temp != '':
                temp_list = self.chinese_sentence_cut(temp)
                listr += temp_list
            temp = ''
            for k in range(start, end):
                temp += text[k]
            if temp != ' ':
                listr.append(temp)
            index = end
        return listr

    def cut_sentence(self, sentence: str) -> list:
        language = Detect.get_text_language(sentence)
        if language == "CN":
            _reply_list = self.cut_chinese_sentence(sentence)
        elif language == "EN":
            # from nltk.tokenize import sent_tokenize
            _reply_list = self.english_sentence_cut(sentence)
        else:
            _reply_list = [sentence]
        if len(_reply_list) < 1:
            return [sentence]
        return _reply_list

    def cut_ai_prompt(self, prompt: str) -> list:
        """
        切薄负载机
        :param prompt:
        :return:
        """
        _some = prompt.split(":", 1)
        _head = ""
        if len(_some) > 1:
            _head = f"{_some[0]}:"
            prompt = _some[1]
        _reply = self.cut_sentence(prompt)
        _prompt_list = []
        for item in _reply:
            _prompt_list.append(f"{_head}{item.strip()}")
        _prompt_list = list(filter(None, _prompt_list))
        return _prompt_list
