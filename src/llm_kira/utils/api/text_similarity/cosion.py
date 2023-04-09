# -*- coding: utf-8 -*-
from typing import Tuple, Union

import jieba
from sklearn.metrics.pairwise import cosine_similarity
from ...api.keywords import STOPWORDS


class CosionSimilarity(object):
    """
    根据余弦函数计算相似性
    one-hot编码
    """

    def load_stopwords(self, stopwords_path):
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]

    def cut_words(self, text, stopwords):
        return [word for word in jieba.cut(text) if word not in stopwords]

    def str_to_vector(self, text1: str, text2: str) -> Tuple[list, list]:
        stopwords = self.load_stopwords(STOPWORDS)
        text1_words = set(self.cut_words(text1, stopwords))
        text2_words = set(self.cut_words(text2, stopwords))
        all_words = list(text1_words | text2_words)
        text1_vector = [1 if word in text1_words else 0 for word in all_words]
        text2_vector = [1 if word in text2_words else 0 for word in all_words]
        return text1_vector, text2_vector

    def similarity(self, text1: Union[str, list], text2: Union[str, list]):
        stopwords = self.load_stopwords(STOPWORDS)
        text1_words = set(self.cut_words(text1, stopwords))
        text2_words = set(self.cut_words(text2, stopwords))
        all_words = list(text1_words | text2_words)
        text1_vector = [1 if word in text1_words else 0 for word in all_words]
        text2_vector = [1 if word in text2_words else 0 for word in all_words]
        if not text1_vector or not text2_vector:
            return 0
        return cosine_similarity([text1_vector], [text2_vector])[0][0]

    @staticmethod
    def vector_similarity(self, text1_vector: list, text2_vector: list):
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity([text1_vector], [text2_vector])[0][0]


if __name__ == '__main__':
    text1 = "小明，你妈妈喊你回家吃饭啦"
    text2 = "回家吃饭啦，小明"
    similarity = CosionSimilarity()
    print(similarity.similarity(text1, text2))
