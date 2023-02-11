# -*- coding: utf-8 -*-
# @Time    : 1/2/23 1:06 AM
# @FileName: setting.py
# @Software: PyCharm
# @Github    ：sudoskys
from pydantic import BaseModel


class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = None


llmRetryAttempt = 3
# 2^x * multiplier 秒，x为重试次数，最小4秒，最多10秒
llmRetryTime = 1
llmRetryTimeMin = 3
llmRetryTimeMax = 60
redisSetting = RedisConfig()
dbFile = "kira_llm.db"
proxyUrl = ""
webServerUrlFilter = False
webServerStopSentence = ["下面就让我们",
                         "小编", "一起来看一下", "小伙伴们",
                         "究竟是什么意思", "看影片", "看人次", "？", "是什么", "什么意思", "意思介绍", " › ",
                         "游侠", "为您提供", "今日推荐", "線上看", "线上看",
                         "高清观看", "点击下载", "带来不一样的", "..去看看",
                         "最新章节", "电影网", "资源下载：", "高清全集在线",
                         "在线观看地址"]  # "?","_哔哩哔哩_bilibili","知乎",
