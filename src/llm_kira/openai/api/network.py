# -*- coding: utf-8 -*-
# @Time    : 12/5/22 9:58 PM
# @FileName: network.py
# @Software: PyCharm
# @Github    ：sudoskys
"""
参考 bilibili_api.utils.network_httpx 制造的工具类
"""

from typing import Any
import json
#
import pycurl
from io import BytesIO

__session_pool = {}


async def request(
        method: str,
        url: str,
        params: dict = None,
        data: Any = None,
        auth: str = None,
        json_body: bool = False,
        proxy: str = "",
        call_func=None,
        encodings: str = "",
        **kwargs,
):
    """
    请求
    :param call_func: 回调函数，用于调整结构
    :param method:
    :param url:
    :param params:
    :param data:
    :param auth:
    :param json_body:
    :param proxy:
    :param kwargs: 参数
    :return:
    """
    if auth is None:
        return Exception("API KEY MISSING")
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Authorization": f"Bearer {auth}"
    }
    if params is None:
        params = {}
    if not data:
        raise Exception("Openai Network:Empty Data")
    config = {
        "method": method.upper(),
        "url": url,
        "params": params,
        "data": data,
        "headers": headers,
    }
    # 更新
    config.update(kwargs)
    if json_body:
        config["headers"]["Content-Type"] = "application/json"
        config["data"] = json.dumps(config["data"])
    if method == "POST":
        code, raw_data = await curl_post(url=url,
                                         headers=config["headers"],
                                         params=config["params"],
                                         data=config["data"],
                                         encodings=encodings,
                                         proxy=proxy
                                         )
    else:
        code, raw_data = await curl_get(url=url,
                                        headers=config["headers"],
                                        params=config["params"],
                                        encodings=encodings,
                                        proxy=proxy
                                        )
    req_data: dict
    req_data = json.loads(raw_data)
    ERROR = req_data.get("error")
    if ERROR:
        # if ERROR.get('type') == "insufficient_quota":
        if call_func:
            call_func(req_data, auth)
        raise RuntimeError(f"{ERROR.get('type')}:{ERROR.get('message')}")
    return req_data


async def curl_get(url,
                   headers: dict,
                   params,
                   encodings: str = "",
                   proxy: str = "",
                   connect_timeout: int = 120,
                   timeout: int = 60,
                   **kwargs
                   ):
    _headers = []
    for key, values in headers.items():
        _headers.append(f"{key}:{values}")
    _params = []
    for key, values in params.items():
        _params.append(f"{key}={values}&")
    if _params:
        url = f"{url}?" + "".join(_params).strip("&")
    c = pycurl.Curl()  # 通过curl方法构造一个对象
    c.setopt(pycurl.FOLLOWLOCATION, True)  # 自动进行跳转抓取
    c.setopt(pycurl.MAXREDIRS, 3)  # 设置最多跳转多少次
    c.setopt(pycurl.CONNECTTIMEOUT, connect_timeout)  # 设置链接超时
    c.setopt(pycurl.TIMEOUT, timeout)  # 下载超时
    if encodings:
        c.setopt(pycurl.ENCODING, encodings)  # 处理gzip内容
    if proxy:
        c.setopt(pycurl.PROXY, proxy)  # 代理
    c.fp = BytesIO()
    c.setopt(pycurl.URL, url)  # 设置要访问的URL
    # c.setopt(pycurl.USERAGENT, ua)  # 传入User-Agent
    c.setopt(pycurl.HTTPHEADER, _headers)  # 传入请求头
    c.setopt(pycurl.WRITEFUNCTION, c.fp.write)  # 回调写入字符串缓存
    c.perform()
    code = c.getinfo(pycurl.RESPONSE_CODE)
    raw_data = c.fp.getvalue()  # 返回源代码
    c.close()
    return code, raw_data


async def curl_post(url,
                    headers: dict,
                    params,
                    data,
                    encodings: str = "",
                    proxy: str = "",
                    connect_timeout: int = 120,
                    timeout: int = 60,
                    **kwargs
                    ):
    _headers = []
    for key, values in headers.items():
        _headers.append(f"{key}:{values}")
    _params = []
    for key, values in params.items():
        _params.append(f"{key}={values}&")
    if _params:
        url = f"{url}?" + "".join(_params).strip("&")
    c = pycurl.Curl()  # 通过curl方法构造一个对象
    c.setopt(pycurl.FOLLOWLOCATION, True)  # 自动进行跳转抓取
    c.setopt(pycurl.MAXREDIRS, 3)  # 设置最多跳转多少次
    c.setopt(pycurl.CONNECTTIMEOUT, connect_timeout)  # 设置链接超时
    c.setopt(pycurl.TIMEOUT, timeout)  # 下载超时
    if encodings:
        c.setopt(pycurl.ENCODING, encodings)  # 处理gzip内容
    if proxy:
        c.setopt(pycurl.PROXY, proxy)  # 代理
    c.fp = BytesIO()
    c.setopt(pycurl.URL, url)  # 设置要访问的URL
    # c.setopt(pycurl.USERAGENT, ua)  # 传入User-Agent
    c.setopt(pycurl.HTTPHEADER, _headers)  # 传入请求头
    c.setopt(pycurl.POST, 1)
    c.setopt(pycurl.POSTFIELDS, data)
    c.setopt(pycurl.WRITEFUNCTION, c.fp.write)  # 回调写入字符串缓存
    c.perform()
    code = c.getinfo(pycurl.RESPONSE_CODE)
    raw_data = c.fp.getvalue()  # 返回源代码
    c.close()
    return code, raw_data
