# -*- coding: utf-8 -*-
# @Time    : 2023/4/5 上午10:19
# @Author  : sudoskys
# @File    : cluster.py
# @Software: PyCharm
from typing import List

import jieba
import numpy as np
# from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import Birch


class Cluster(object):
    def __init__(self):
        self.weight = None
        self.cluster = None
        self.title_dict: dict = {}

    def init(self, sentence_list):
        """
        初始化
        """
        # corpus = [] #文档预料 空格连接
        corpus = []
        # f_write = open("jieba_result.dat","w")
        self.title_dict = {}
        index = 0
        for line in sentence_list:
            title = line.strip()
            self.title_dict[index] = title
            output = ' '.join(['%s' % x for x in list(jieba.cut(title, cut_all=False))]).encode('utf-8')  # 空格拼接
            # print(list(list(jieba.cut(title, cut_all=False))))
            index += 1
            corpus.append(output.strip())
        # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
        _vectorizer = CountVectorizer()
        # 该类会统计每个词语的tf-idf权值
        transformer = TfidfTransformer()
        # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
        tfidf = transformer.fit_transform(_vectorizer.fit_transform(corpus))
        # 获取词袋模型中的所有词语
        word = _vectorizer.get_feature_names_out()
        # 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
        self.weight = tfidf.toarray()

    def birch_cluster(self, sentence_list: List[str], threshold: float = 0.6) -> None:
        """
        Birch聚类
        :param sentence_list: 文本列表
        :param threshold: 聚类阈值
        :return:
        """
        self.init(sentence_list=sentence_list)
        self.cluster = Birch(threshold=threshold, n_clusters=None)
        self.cluster.fit_predict(self.weight)

    def build(self):
        # self.cluster.labels_ 为聚类后corpus中文本index 对应 类别 {index: 类别} 类别值int值 相同值代表同一类
        # cluster_dict key为Birch聚类后的每个类，value为 title对应的index
        cluster_dict = {}
        for index, value in enumerate(self.cluster.labels_):
            if value not in cluster_dict:
                cluster_dict[value] = [index]
            else:
                cluster_dict[value].append(index)
        # print("-----before cluster Birch count title:", len(self.title_dict))
        # result_dict key为Birch聚类后距离中心点最近的title，value为sum_similar求和
        result_dict = {}
        for _index in cluster_dict.values():
            latest_index = _index[0]
            similar_num = len(_index)
            if len(_index) >= 2:
                min_s = np.sqrt(np.sum(np.square(
                    self.weight[_index[0]] - self.cluster.subcluster_centers_[self.cluster.labels_[_index[0]]])))
                for index in _index:
                    s = np.sqrt(np.sum(
                        np.square(self.weight[index] - self.cluster.subcluster_centers_[self.cluster.labels_[index]])))
                    if s < min_s:
                        min_s = s
                        latest_index = index
            title = self.title_dict[latest_index]
            result_dict[title] = similar_num
        # print("-----after cluster Birch count title:", len(result_dict))
        return result_dict


"""
用法
cluster = Cluster()
cluster.birch_cluster(sentence_list)
result_dict = cluster.build()
"""
