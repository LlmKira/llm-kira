import random
from typing import Union
from ...azure_openai_sdk.api.network import request
from .... import setting


class Completion(object):
    def __init__(self, resource_name: str, deployment_id: str, api_version: str, api_key: Union[str, list] = None,
                 proxy_url: str = "", call_func=None):
        self.__resource_name = resource_name
        self.__deployment_id = deployment_id
        self.__api_version = api_version
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
        得到一个对话，预设了一些参数，其实还有很多参数，详情请阅读：
        https://learn.microsoft.com/zh-cn/azure/cognitive-services/openai/reference
        :param model: 模型
        :param prompt: 提示
        :param temperature: unknown
        :param max_tokens: 返回数量
        :return:
        """
        """
        curl https://YOUR_RESOURCE_NAME.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT_NAME/completions?api-version=2022-12-01\
          -H "Content-Type: application/json" \
          -H "api-key: YOUR_API_KEY" \
          -d "{
          \"prompt\": \"Once upon a time\",
          \"max_tokens\": 5
        }"
        """
        # 参数决定
        _api_config = (
            {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            | kwargs
        )
        # 返回请求
        return await request(
            method="POST",
            url=f"https://{self.__resource_name}.openai.azure.com/openai/deployments/{self.__deployment_id}"
                f"/completions?api-version={self.__api_version}",
            data=_api_config,
            auth=self.__api_key,
            proxy=self.__proxy,
            json_body=True,
            call_func=self.__call_func
        )
