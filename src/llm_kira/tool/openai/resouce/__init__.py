# -*- coding: utf-8 -*-
# @Time    : 12/15/22 11:11 AM
# @FileName: __init__.py
# @Software: PyCharm
# @Github    ：sudoskys
from .chat import ChatCompletion, ChatPrompt
from .completion import Completion
from .moderations import Moderations

API_ERROR_TYPE = ["invalid_request_error",
                  "billing_not_active",
                  "billing_not_active",
                  "insufficient_quota"
                  ]
