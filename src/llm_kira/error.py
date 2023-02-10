# -*- coding: utf-8 -*-
# @Time    : 2/10/23 10:54 AM
# @FileName: error.py
# @Software: PyCharm
# @Github    ：sudoskys
class LLMException(BaseException):
    pass


class AuthenticationError(Exception):
    pass


class RateLimitError(Exception):
    pass


class ServiceUnavailableError(Exception):
    pass
