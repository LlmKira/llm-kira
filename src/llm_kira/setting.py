# -*- coding: utf-8 -*-
# @Time    : 1/2/23 1:06 AM
# @FileName: setting.py
# @Software: PyCharm
# @Github    ：sudoskys
from pydantic import BaseModel, RedisDsn, BaseSettings


class RetrySettings(BaseModel):
    """
    重试设置
    """
    # 重试次数
    retry_attempt: int = 3
    # 重试时间
    retry_time: int = 1
    # 重试最小时间
    retry_time_min: int = 3
    # 重试最大时间
    retry_time_max: int = 60


class ProxySettings(BaseModel):
    """
    代理设置
    """
    # 限定代理类型
    proxy_status: bool = False
    # 代理地址
    proxy_address: str = "all://127.0.0.1:7890"


class CacheSettings(BaseSettings):
    """
    缓存设置
    """
    # 缓存地址
    redisDsn: RedisDsn = "redis://localhost:6379/0"
    # 缓存文件
    dbFile: str = "llm_kira.db"


retrySetting = RetrySettings()
cacheSetting = CacheSettings()
proxySetting = ProxySettings()
