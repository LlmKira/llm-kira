# -*- coding: utf-8 -*-
# @Time    : 2023/4/10 下午11:25
# @Author  : sudoskys
# @File    : redis.py
# @Software: PyCharm
import json
from typing import List, Optional

from loguru import logger

from ..schema import Interaction, MemoryBaseLoader
from ..setting import cacheSetting


class RedisMessageLoader(MemoryBaseLoader):
    def __init__(self, session_id: str, url: str = cacheSetting.redisDsn, key_prefix: str = "llm_kira_message_store:",
                 ttl: Optional[int] = None):
        super().__init__(session_id, key_prefix)
        try:
            import redis
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )

        try:
            self.redis_client = redis.Redis.from_url(url=url)
        except redis.exceptions.ConnectionError as error:
            logger.error(error)
        self.session_id = session_id
        self.key_prefix = key_prefix
        self.ttl = ttl

    @property
    def key(self) -> str:
        """Construct the record key to use"""
        return self.key_prefix + self.session_id

    @property
    def message(self) -> List[Interaction]:  # type: ignore
        """Retrieve the messages from Redis"""
        _items = self.redis_client.lrange(self.key, 0, -1)
        messages = [Interaction(**json.loads(m.decode("utf-8"))) for m in _items[::-1]]
        return messages

    def append(self, message: Interaction) -> None:
        """Append the message to the record in Redis"""
        self.redis_client.lpush(self.key, message.json())
        if self.ttl:
            self.redis_client.expire(self.key, self.ttl)

    def clear(self) -> None:
        """Clear session memory from Redis"""
        self.redis_client.delete(self.key)
