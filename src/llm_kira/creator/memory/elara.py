import elara
import json

from typing import List, Optional

from loguru import logger

from ..schema import Interaction, MemoryBaseLoader
from ..setting import cacheSetting


class ElaraMessageLoader(MemoryBaseLoader):
    def __init__(self, session_id: str, path: str, key_prefix: str = "llm_kira_message_store:",
                 ttl: Optional[int] = None):
        super().__init__(session_id, key_prefix)
        self.db = elara.exe(path)
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
        _items = self.db.lrange(self.key, 0, -1)
        messages = [Interaction(**json.loads(m.decode("utf-8"))) for m in _items[::-1]]
        return messages

    def append(self, message: Interaction) -> None:
        """Append the message to the record in Redis"""
        self.db.lpush(self.key, message.json())
        if self.ttl:
            self.db.expire(self.key, self.ttl)

    def clear(self) -> None:
        """Clear session memory from Redis"""
        self.db.delete(self.key)
        