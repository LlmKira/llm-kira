# -*- coding: utf-8 -*-
# @Time    : 2/13/23 9:25 AM
# @FileName: anchor.py
# @Software: PyCharm
# @Github    ：sudoskys
import random
from abc import ABC, abstractmethod
from typing import List, Union

from llm_kira.utils.chat import Sim

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .decomposer import Filter, NlpUtils
from .crawer import raw_content, Duckgo
from ..client.types import PromptItem, Interaction
from ..utils.data import Bucket


def warp_interaction(start: Union[str, bool] = "Google", content: List[str] = None):
    _returner = []
    _index = 0
    for item in content:
        _start = f"[^{_index}]"
        if isinstance(start, str):
            _start = start
        if isinstance(item, str):
            _returner.append(Interaction(ask=PromptItem(start=_start, text=item), single=True))
        elif isinstance(item, Interaction):
            _returner.append(item)
        _index += 1
    return _returner


class Multiplexers(object):
    def __init__(self):
        self.bucket = Bucket(uid=10086)

    def index(self, prompt: str, limit: int = 10) -> List[str]:
        _data = self.bucket.get()
        _return = []
        for key, item in _data.items():
            _cos_sim = Sim.cosion_similarity(aft=prompt, pre=str(key))
            _len = 0.3 * len(prompt) < len(str(key)) < 7 * len(prompt)
            if 0.86 < _cos_sim and _len:
                _return.extend(item)
            if len(_return) > limit:
                return _return
        return _return

    def insert(self, key: Union[str, list], result: list):
        if isinstance(key, list):
            key = ",".join(key)
        _data = self.bucket.get()
        _data[str(key)] = result
        self.bucket.set(data=_data)


class Antennae(ABC):
    @abstractmethod
    async def run(self, prompt, prompt_raw: str = None, **kwargs) -> List[Interaction]:
        return []


class SearchCraw(Antennae):
    def __init__(self,
                 deacon: List[str] = None,
                 start_name: str = None
                 ):
        if not deacon:
            deacon = ["https://www.bing.com/search?q={}&form=QBLH",
                      "https://www.google.com/search?q={}&source=hp&"
                      ]
        self.deacon = deacon
        self.start_name = start_name

    @retry(retry=retry_if_exception_type((LookupError)), stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=5, max=10), reraise=True)
    async def run(self, prompt: str, prompt_raw: str = None) -> List[Interaction]:
        if prompt_raw is None:
            prompt_raw = prompt
        if prompt is None:
            prompt = prompt_raw
        if not prompt:
            return []
        _content = Multiplexers().index(prompt=prompt)
        if len(_content) < 3:
            url = random.choice(self.deacon)
            _content = await raw_content(url=url, query=prompt, raise_empty=False)
        _content = Filter().filter(sentences=_content, limit=(0, 300), filter_url=True)
        _content = NlpUtils.compression(prompt=prompt_raw, material=_content)
        if len(_content) > 3:
            Multiplexers().insert(key=prompt, result=_content)
        _content = [item for item in _content if item]
        _returner = warp_interaction(start=self.start_name, content=_content)
        return _returner


class DuckgoCraw(Antennae):
    def __init__(self,
                 deacon: List[str] = None,
                 start_name: str = None
                 ):
        self.deacon = deacon
        self.start_name = start_name

    @retry(retry=retry_if_exception_type((LookupError)), stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=1, max=5), reraise=True)
    async def run(self, prompt: str, prompt_raw: str = None) -> List[Interaction]:
        if prompt_raw is None:
            prompt_raw = prompt
        if prompt is None:
            prompt = prompt_raw
        if not prompt:
            return []
        _content = Multiplexers().index(prompt=prompt)
        if len(_content) < 3:
            _results = await Duckgo().get_result(keywords=prompt)
            if _results:
                _content = [f"{i['title']}-{i['body']}\n{i['href']}" for i in _results]
                _link = [f"{i['href']}" for i in _results]
                # for item in _link[:3]:
                # ### _content.extend(await raw_content(url=item, query=prompt))
        _content = Filter().filter(sentences=_content, limit=(0, 300), filter_url=False)
        _content = NlpUtils.compression(prompt=prompt_raw, material=_content)
        if len(_content) > 3:
            Multiplexers().insert(key=prompt, result=_content)
        _content = [item for item in _content if item]
        _returner = warp_interaction(start=self.start_name, content=_content)
        return _returner
