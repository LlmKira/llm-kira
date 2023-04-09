# -*- coding: utf-8 -*-
# @Time    : 2/10/23 10:54 AM
# @FileName: error.py
# @Software: PyCharm
# @Github    ：sudoskys

class LlmException(BaseException):
    """
    Base class for all exceptions raised by this library.
    """
    pass


class AuthenticationError(Exception):
    """
    Raised when the API key is invalid.
    """
    pass


class RateLimitError(Exception):
    """
    Raised when the API key has exceeded its rate limit.
    """
    pass


class ServiceUnavailableError(Exception):
    """
    Raised when the API is unavailable.
    """
    pass


class InvalidArgumentError(Exception):
    """ Raised when an invalid argument is passed to a function.
    """
    pass


class InvalidResponseError(Exception):
    """ Raised when the API returns an invalid response.
    """
    pass
