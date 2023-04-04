# -*- coding: utf-8 -*-
# @Time    : 1/2/23 1:06 AM
# @FileName: setting.py
# @Software: PyCharm
# @Github    ：sudoskys
from pydantic import BaseModel, RedisDsn, BaseSettings, Extra, Field
from dotenv import load_dotenv

# 提前导入加载变量
load_dotenv()


class RetrySettings(BaseModel):
    """
    重试设置
    :param retry_attempt: 重试次数
    :param retry_time: 重试时间
    :param retry_time_min: 重试最小时间
    :param retry_time_max: 重试最大时间
    """
    # 重试次数
    retry_attempt: int = 3
    # 重试时间
    retry_time: int = 1
    # 重试最小时间
    retry_time_min: int = 3
    # 重试最大时间
    retry_time_max: int = 60


class ProxySettings(BaseSettings):
    """
    代理设置
    :param proxy_status: 代理状态
    :param proxy_address: 代理地址
    :param openai_proxy_backend: OPENAI 反代API
    """
    # 限定代理类型
    proxy_status: bool = False
    # 代理地址
    proxy_address: str = Field("all://127.0.0.1:7890", env="HTTPS_PROXY")
    # OPENAI 反代API
    openai_proxy_backend: str = Field("https://api.openai.com/v1/", env='OPENAI_PROXY_API')

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'


class CacheSettings(BaseSettings):
    """
    缓存设置
    :param redisDsn: 缓存地址
    :param dbFile: 备用缓存文件
    """
    # 缓存地址
    redisDsn: RedisDsn = Field("redis://localhost:6379/0", env='REDIS_DSN')
    # 备用缓存文件
    dbFile: str = "llm_kira.db"

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'


retrySetting = RetrySettings()
cacheSetting = CacheSettings()
proxySetting = ProxySettings()
