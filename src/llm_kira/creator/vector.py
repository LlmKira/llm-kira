# -*- coding: utf-8 -*-
# @Time    : 3/7/23 10:20 PM
# @FileName: Vector.py
# @Software: PyCharm
# @Github    ：sudoskys

import tiktoken
from typing import List
from openai.embeddings_utils import get_embedding, cosine_similarity


class embeddings(object):

    def encode(self, string: str, encoding_name: str) -> List[int]:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        return encoding.encode(string)


import pandas as pd

df = pd.read_csv('output/embedded_1k_reviews.csv')
df['ada_embedding'] = df.ada_embedding.apply(eval).apply(np.array)


def search_reviews(df, product_description, n=3):
    embedding = get_embedding(product_description, model='text-embedding-ada-002')
    df['similarities'] = df.ada_embedding.apply(lambda x: cosine_similarity(x, embedding))
    res = df.sort_values('similarities', ascending=False).head(n)
    return res


res = search_reviews(df, 'delicious beans', n=3)
