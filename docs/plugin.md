# Removed in New Era.被外骨骼系统代替

## Plugin

**Table**

| plugins   | desc              | value/server                                          | use                                   |
|-----------|-------------------|-------------------------------------------------------|---------------------------------------|
| `time`    | now time          | `""`,no need                                          | `明昨今天`....                            |
| `week`    | week time         | `""`,no need                                          | `周几` .....                            |
| `search`  | Web Search        | `["some.com?searchword={}"]`,must need                | `查询` `你知道` len<80 / end with`?`len<15 |
| `duckgo`  | Web Search        | `""`,no need,but need `pip install duckduckgo_search` | `查询` `你知道` len<80 / end with`?`len<15 |
| `details` | answer with steps | `""`,no need                                          | Ask for help `how to`                 |

## Plugin dev

There is a middleware between the memory pool and the analytics that provides some networked retrieval support and
operational support. It can be spiked with services that interface to other Api's.

**Prompt Injection**

Use `""` `[]` to emphasise content.

### Exp

First create a file in `llm_kira/Chat/module/plugin` without underscores (`_`) in the file name.

**Template**

```python
from ..platform import ChatPlugin, PluginConfig
from ._plugin_tool import PromptTool
import os
from loguru import logger

modulename = os.path.basename(__file__).strip(".py")


@ChatPlugin.plugin_register(modulename)
class Week(object):
    def __init__(self):
        self._server = None
        self._text = None
        self._time = ["time", "多少天", "几天", "时间", "几点", "今天", "昨天", "明天", "几月", "几月", "几号",
                      "几个月",
                      "天前"]

    def requirements(self):
        return []

    async def check(self, params: PluginConfig) -> bool:
        if PromptTool.isStrIn(prompt=params.text, keywords=self._time):
            return True
        return False

    async def process(self, params: PluginConfig) -> list:
        _return = []
        self._text = params.text
        # 校验
        if not all([self._text]):
            return []
        # GET
        from datetime import datetime, timedelta, timezone
        utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
        bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
        now = bj_dt.strftime("%Y-%m-%d %H:%M")
        _return.append(f"Current Time UTC8 {now}")
        # LOGGER
        logger.trace(_return)
        return _return
```

`llm_kira/Chat/module/plugin/_plugin_tool.py` provides some tool classes, PR is welcome

**Testing**

You cannot test directly from within the module package, please run the `llm_kira/Chat/test_module.py` file to test
the module, with the prompt matching check.

Alternatively, you can safely use `from loguru import logger` + `logger.trace(_return)` in the module to debug the
module variables and the trace level logs will not be output by the production environment.
