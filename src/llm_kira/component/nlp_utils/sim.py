# -*- coding: utf-8 -*-
# @Time    : 2023/4/5 上午10:17
# @Author  : sudoskys
# @File    : sim.py
# @Software: PyCharm
from llm_kira.utils.api.text_similarity.cosion import CosionSimilarity
from llm_kira.utils.api.text_similarity.edit import EditSimilarity
from llm_kira.utils.api.text_similarity.simhash import SimHashSimilarity


class Sim(object):
    """
    文本相似度计算，基于基础 jieba 分词 101 向量
    """

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
