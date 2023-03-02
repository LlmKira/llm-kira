# -*- coding: utf-8 -*-
# @Time    : 2/11/23 4:07 PM
# @FileName: forget.py
# @Software: PyCharm
# @Github    ：sudoskys
import math


# 稳定性
def forget(sim, hour, rank=0.5):
    S = sim * rank
    # 一天的时间
    R = math.exp(hour * math.log(0.9) / S)
    return R


print(forget(sim=0.5, hour=1.55) * 100)
