# -*- coding: utf-8 -*-
# @Time    : 2023/3/27 下午11:23
# @Author  : sudoskys
# @File    : sola.py
# @Software: PyCharm
def singleton(cls):
    """
    单例模式
    :param cls:
    :return:
    """
    _instance = {}

    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]

    return inner
