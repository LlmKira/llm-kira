# -*- coding: utf-8 -*-
# @Time    : 3/2/23 4:13 PM
# @FileName: base.py
# @Software: PyCharm
# @Github    ：sudoskys
from abc import ABC, abstractmethod


class BaseEngine(ABC):
    """
    Base class for all engines
    """

    @abstractmethod
    def build_prompt(self, predict_tokens: int = 500):
        """
        Build prompt for engine
        """
        pass
