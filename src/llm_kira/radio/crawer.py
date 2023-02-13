# -*- coding: utf-8 -*-
# @Time    : 2/12/23 11:20 PM
# @FileName: crawer.py
# @Software: PyCharm
# @Github    ：sudoskys
from typing import Optional, List
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from loguru import logger

from .decomposer import Extract
from ..utils import network


class UniMatch(object):
    @staticmethod
    def get_tld(url):
        """
        获取顶级域名
        :param url:
        :return:
        """
        parsed_url = urlparse(url)
        return parsed_url.netloc

    async def get_raw_html(self,
                           url: str,
                           query: str = "KKSK 是什么意思？"
                           ) -> Optional[str]:
        if url:
            _url = url
        if query:
            _url = url.format(query)
        else:
            return None
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, defalte",
            "Connection": "keep-alive",
            "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
            "Host": f"{self.get_tld(url)}",
            "Referer": f"https://www.{self.get_tld(url)}/",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:106.0) Gecko/20100101 Firefox/106.0"
        }
        html = await network.request("GET", url=_url, headers=headers, timeout=5)
        if html.status_code == 200:
            return html.text
        else:
            return None


class Duckgo(object):
    cache = set()
    PAGINATION_STEP = 25

    async def get_page(self, payload, page):
        from duckduckgo_search.utils import _normalize
        page_results = []
        page_data = []
        payload["s"] = max(self.PAGINATION_STEP * (page - 1), 0)
        try:
            resp = await network.request("POST", "https://links.duckduckgo.com/d.js", params=payload, timeout=5)
            resp.raise_for_status()
            page_data = resp.json().get("results", None)
        except Exception as e:
            logger.error(f"Duckgo Client Error{e}")
        if not page_data:
            return page_results
        for row in page_data:
            if "n" not in row and row["u"] not in self.cache:
                self.cache.add(row["u"])
                body = _normalize(row["a"])
                if body:
                    page_results.append(
                        {
                            "title": _normalize(row["t"]),
                            "href": row["u"],
                            "body": body,
                        }
                    )
        return page_results

    async def get_result(self,
                         keywords,
                         region="wt-wt",
                         safesearch="moderate",
                         max_results=None,
                         time=None,
                         page=1
                         ):
        from duckduckgo_search.utils import _get_vqd
        if not keywords:
            return None
        vqd = _get_vqd(keywords)
        if not vqd:
            return None

        # prepare payload
        safe_search_base = {"On": 1, "Moderate": -1, "Off": -2}
        payload = {
            "q": keywords,
            "l": region,
            "p": safe_search_base[safesearch.capitalize()],
            "s": 0,
            "df": time,
            "o": "json",
            "vqd": vqd,
        }
        results = await self.get_page(page=page, payload=payload)
        results = results[:max_results]
        return results


async def raw_content(url, query: str = None, div_span: bool = True, raise_empty: bool = True) -> List[str]:
    _html = await UniMatch().get_raw_html(url=url, query=query)
    if not _html:
        if not raise_empty:
            return []
        raise LookupError("No Match Content")
    try:
        if div_span:
            _text = []
            rs = BeautifulSoup(_html, "html.parser")
            for i in rs.select("div"):
                if i.parent.select("a[href]"):
                    continue
                _text.append(i.parent.text)
        else:
            _text = Extract().process_html(url, html=_html)
    except Exception as e:
        raise LookupError(f"Error in Extract().process_html(){e}")
    return _text
