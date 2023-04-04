# -*- coding: utf-8 -*-
# @Time    : 2/18/23 4:00 PM
# @FileName: filter_str.py
# @Software: PyCharm
# @Github    ：sudoskys
import re

area = "sads asda 121 _你好 ————+ ++ __"


def __safe(_sentence):
    _sentence = _sentence.replace("_", "玹")
    _sentence = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", _sentence)
    _sentence = _sentence.replace("玹", "_")
    return _sentence


print(__safe(area))
