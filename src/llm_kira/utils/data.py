# -*- coding: utf-8 -*-
# @Time    : 12/6/22 6:55 PM
# @FileName: data.py
# @Software: PyCharm
# @Github    ：sudoskys
import ast
import json
import re
from typing import Union, Optional, List
from loguru import logger
from ..client.types import Interaction
from ..utils import setting
from pydantic import BaseModel

_redis_config = setting.redisSetting
_db_file = setting.dbFile

redis_installed = True

try:
    from redis import Redis, ConnectionPool
    from redis import StrictRedis
except Exception as e:
    redis_installed = False

filedb = True
try:
    import elara
except Exception:
    filedb = False

if not filedb and not redis_installed:
    raise Exception("Db/redis all Unusable")


def safe_sentence(_sentence):
    _sentence = _sentence.replace("_", "玹")
    _sentence = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", _sentence)
    _sentence = _sentence.replace("玹", "_")
    return _sentence


class RedisWorker(object):
    """
    Redis 数据基类
    不想用 redis 可以自动改动此类，换用其他方法。应该比较简单。
    """

    def __init__(self, host='localhost', port=6379, db=0, password=None, prefix='llm_kira_'):
        self.redis = ConnectionPool(host=host, port=port, db=db, password=password)

        # self.con = Redis(connection_pool=self.redis) -> use this when necessary
        # {chat_id: {user_id: {'state': None, 'data': {}}, ...}, ...}
        self.prefix = prefix
        if not redis_installed:
            raise Exception("Redis is not installed. Install it via 'pip install redis'")

    def ping(self):
        connection = Redis(connection_pool=self.redis)
        connection.ping()

    def setKey(self, key, obj, exN=None):
        connection = Redis(connection_pool=self.redis)
        connection.set(self.prefix + str(key), json.dumps(obj), ex=exN)
        connection.close()
        return True

    def deleteKey(self, key):
        connection = Redis(connection_pool=self.redis)
        connection.delete(self.prefix + str(key))
        connection.close()
        return True

    def getKey(self, key):
        connection = Redis(connection_pool=self.redis)
        result = connection.get(self.prefix + str(key))
        connection.close()
        if result:
            return json.loads(result)
        else:
            return {}

    def addToList(self, key, listData: list):
        data = self.getKey(key)
        if isinstance(data, str):
            listGet = ast.literal_eval(data)
        else:
            listGet = []
        listGet = listGet + listData
        listGet = list(set(listGet))
        if self.setKey(key, str(listGet)):
            return True

    def getList(self, key):
        listGet = ast.literal_eval(self.getKey(key))
        if not listGet:
            listGet = []
        return listGet


class ElaraWorker(object):
    """
    Redis 数据基类
    不想用 redis 可以自动改动此类，换用其他方法。应该比较简单。
    """

    def __init__(self, filepath, prefix='llm_kira_'):
        self.redis = elara.exe(filepath)
        self.prefix = prefix
        # self.con = Redis(connection_pool=self.redis) -> use this when necessary
        # {chat_id: {user_id: {'state': None, 'data': {}}, ...}, ...}

    def setKey(self, key, obj):
        self.redis.set(self.prefix + str(key), json.dumps(obj, ensure_ascii=False))
        self.redis.commit()
        return True

    def deleteKey(self, key):
        self.redis.rem(key)
        return True

    def getKey(self, key):
        result = self.redis.get(self.prefix + str(key))
        if result:
            return json.loads(result)
        else:
            return {}

    def addToList(self, key, listData: list):
        data = self.getKey(key)
        if isinstance(data, str):
            listGet = ast.literal_eval(data)
        else:
            listGet = []
        listGet = listGet + listData
        listGet = list(set(listGet))
        if self.setKey(key, str(listGet)):
            return True

    def getList(self, key):
        listGet = ast.literal_eval(self.getKey(key))
        if not listGet:
            listGet = []
        return listGet


class DataUtils(object):
    @staticmethod
    def processString5(txt, ori: str, rep: str, dels: str = None):
        if len(ori) != len(rep):
            raise Exception("NO")
        transTable = txt.maketrans(ori, rep, dels)
        txt = txt.translate(transTable)
        return txt

    @staticmethod
    def remove_suffix(input_string: str, suffix: str) -> str:
        """
        Remove suffix in python < 3.9
        """
        if suffix and input_string.endswith(suffix):
            return input_string[: -len(suffix)]
        return input_string


class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = None


def GetDataManager(redis_config: RedisConfig,
                   filedb_path: str,
                   prefix: str = "llm_kira_memory_") -> Union[RedisWorker, ElaraWorker]:
    MsgFlowData = None
    if filedb:
        MsgFlowData = ElaraWorker(filepath=filedb_path)
    if redis_installed:
        try:
            MsgFlowData_ = RedisWorker(
                host=redis_config.host,
                port=redis_config.port,
                db=redis_config.db,
                password=redis_config.password,
                prefix=prefix)
            MsgFlowData_.ping()
        except Exception as e:
            pass
        else:
            MsgFlowData = MsgFlowData_
    return MsgFlowData


class Bucket(object):
    def __init__(self,
                 uid: int,
                 area: str = "cache"
                 ):
        """
        消息流存储器
        :param uid: 独立 id ，是一个消息桶
        """
        self.uid = str(uid)
        area = safe_sentence(area)[:50]
        # 工具数据类型
        self.MsgFlowData = GetDataManager(_redis_config, _db_file, prefix=f"llm_kira_{area}_")

    def get(self):
        _get = self.MsgFlowData.getKey(self.uid)
        if not _get:
            _get = {}
        return _get

    def set(self, data):
        if not data:
            data = {}
        return self.MsgFlowData.setKey(self.uid, data)


class MsgFlow(object):
    """
    数据存储桶，用于上下文分析时候提取桶的内容
    """

    def __init__(self, uid: str):
        """
        消息流存储器
        :param uid: 独立 id ，是一个消息桶
        """
        if not uid:
            raise Exception("MsgFlow Miss UID...")
        self.uid = str(uid)
        self.memory: int = 500
        # 工具数据类型
        self.MsgFlowData = GetDataManager(_redis_config, _db_file)

    @staticmethod
    def composing_uid(user_id, chat_id):
        return f"{user_id}:{chat_id}"

    def _get_uid(self, uid):
        return self.MsgFlowData.getKey(uid)

    def _set_uid(self, uid, message_streams):
        return self.MsgFlowData.setKey(uid, message_streams)

    @staticmethod
    def parse(interaction: Interaction, sign: bool = False) -> List[str]:
        """
        得到互动的内容
        :param sign: 是否署名
        :param interaction: 消息对象提取内容
        :return: ask,reply
        """
        _returner = []
        if sign:
            _returner.append(interaction.ask.text)
        else:
            _returner.append(interaction.ask.prompt)
        if not interaction.single:
            if sign:
                _returner.append(interaction.reply.text)
            else:
                _returner.append(interaction.reply.prompt)
        return _returner

    def save(self, interaction_flow: List[Interaction], override: bool = False) -> None:
        # 获取 Json 数据
        _message_streams = self._get_uid(self.uid)
        _message = []
        # 读取前世记忆
        if "message" in _message_streams:
            _message = _message_streams["message"]
            _message = sorted(_message, key=lambda x: x['time'], reverse=True)
            _message = _message[:int(self.memory)]
        if override:
            _message = []
        # 填充
        for item in interaction_flow:
            _message.append(item.dict())
        # 回存
        # _message = sorted(_message, key=lambda x: x['time'], reverse=True)
        _message_streams["message"] = _message
        self._set_uid(self.uid, _message_streams)

    def read(self) -> Optional[List[Interaction]]:
        _message_streams = self._get_uid(self.uid)
        if "message" not in _message_streams:
            return []
        _streams = _message_streams["message"]
        _returner = []
        # 倒序
        _message = sorted(_streams, key=lambda x: x['time'], reverse=True)
        for item in _streams:
            try:
                _returner.append(Interaction(**item))
            except Exception as error:
                logger.warning(f"Failed insert {error}")
        return _returner

    def forget(self):
        _message_streams = self._get_uid(self.uid)
        if "message" in _message_streams:
            _message_streams["message"] = []
            self._set_uid(self.uid, _message_streams)
        return True


def singleton(cls):
    _instance = {}

    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]

    return inner
