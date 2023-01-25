# -*- coding: utf-8 -*-
# @Time    : 1/11/23 1:16 PM
# @FileName: moderations.py
# @Software: PyCharm
# @Github    ：sudoskys
import random
from typing import Union

from ..api.api_utils import load_api
from ..api.network import request
from ...utils import setting

API = load_api()


class Moderations(object):
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
                     input: str = "Sample",
                     **kwargs
                     ):
        """
        :param input: 审查内容
        :return:
        """
        """
        curl https://api.openai.com/v1/moderations \
          -X POST \
          -H "Content-Type: application/json" \
          -H "Authorization: Bearer $OPENAI_API_KEY" \
          -d '{"input": "Sample text goes here"}'
        """
        api = API["v1"]["moderations"]
        # 参数决定
        params = {
            "input": input,
        }
        api_config = {
            param: api["params"][param]["Defaults"]
            for param in api["params"].keys()
            if (param in kwargs) or (param in params)
        }
        api_config.update(params)
        api_config.update(kwargs)
        # 返回请求
        return await request(
            "POST",
            api["url"],
            data=api_config,
            auth=self.__api_key,
            proxy=self.__proxy,
            json_body=True,
            call_func=self.__call_func
        )
