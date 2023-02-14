# -*- coding: utf-8 -*-
# @Time    : 12/15/22 9:54 PM
# @FileName: __init__.py
# @Software: PyCharm
# @Github    ：sudoskys
import random
from typing import Union
from ...openai_utils.api.api_utils import load_api
from ...openai_utils.api.network import request
from ...utils import setting

API = load_api()


class Completion(object):
    def __init__(self, api_key: Union[str, list] = None, proxy_url: str = "", call_func=None):
        # if api_key is None:
        #     api_key = setting.openaiApiKey
        if isinstance(api_key, list):
            api_key: list
            if not api_key:
                raise RuntimeError("Use Out")
            random.shuffle(api_key)
            api_key = random.choice(api_key)
            api_key: str
        if not api_key:
            raise RuntimeError("NO KEY")
        self.__api_key = api_key
        if not proxy_url:
            proxy_url = setting.proxyUrl
        self.__proxy = proxy_url
        self.__call_func = call_func

    def get_api_key(self):
        return self.__api_key

    async def create(self,
                     model: str = "text-davinci-003",
                     prompt: str = "Say this is a test",
                     temperature: float = 0,
                     max_tokens: int = 7,
                     **kwargs
                     ):
        """
        得到一个对话，预设了一些参数，其实还有很多参数，如果你有api文档
        :param model: 模型
        :param prompt: 提示
        :param temperature: unknown
        :param max_tokens: 返回数量
        :return:
        """
        """
        curl https://api.openai.com/v1/completions \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer YOUR_API_KEY" \
        -d '{"model": "text-davinci-003", "prompt": "Say this is a test", "temperature": 0, "max_tokens": 7}'
        """
        api = API["v1"]["completions"]
        # 参数决定
        params = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        _api_config = {
            param: api["params"][param]["Defaults"]
            for param in api["params"].keys()
            if (param in kwargs) or (param in params)
        }
        _api_config.update(params)
        _api_config.update(kwargs)
        _api_config = {key: item
                       for key, item in _api_config.items()
                       if key in api["params"].keys()
                       }
        # 返回请求
        return await request(
            method="POST",
            url=api["url"],
            data=_api_config,
            auth=self.__api_key,
            proxy=self.__proxy,
            json_body=True,
            call_func=self.__call_func
        )
