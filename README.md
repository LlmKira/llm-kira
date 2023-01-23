# llm-kira

A refactored version of the `openai-kira` specification. Use redis or a file database.

Building ChatBot with LLMs.Using `async` requests.

There are comprehensive examples of use in `test/usage_tmp.py`.

> Contributors welcomed.

## Basic Use

`pip install -U llm-kira`

**Init**

```python
import llm_kira

llm_kira.setting.redisSetting = llm_kira.setting.RedisConfig(host="localhost",
                                                             port=6379,
                                                             db=0,
                                                             password=None)
llm_kira.setting.dbFile = "client_memory.db"
llm_kira.setting.proxyUrl = None  # "127.0.0.1"

# Plugin
llm_kira.setting.webServerUrlFilter = False
llm_kira.setting.webServerStopSentence = ["广告", "营销号"]
```

## Demo

SEE `./test` for More Exp!

Take `openai` as an example

```python
import asyncio
import random
import llm_kira
from llm_kira.client import Optimizer
from llm_kira.client.types import PromptItem

openaiApiKey = ["key1", "key2"]
openaiApiKey: list[str]

receiver = llm_kira.client
conversation = receiver.Conversation(
    start_name="Human:",
    restart_name="AI:",
    conversation_id=10093,  # random.randint(1, 10000000),
)

llm = receiver.llm.OpenAi(profile=conversation,
                          api_key=openaiApiKey,
                          token_limit=3700,
                          no_penalty=True,
                          call_func=None)
mem = receiver.MemoryManger(profile=conversation)
chat_client = receiver.ChatBot(profile=conversation,
                               memory_manger=mem,
                               optimizer=Optimizer.SinglePoint,
                               llm_model=llm)


async def chat():
    promptManger = receiver.PromptManger(profile=conversation,
                                         connect_words="\n",
                                         )
    promptManger.insert(item=PromptItem(start=conversation.start_name, text="My number is 1596321"))
    response = await chat_client.predict(model="text-davinci-003",
                                         prompt=promptManger,
                                         predict_tokens=500,
                                         increase="外部增强:每句话后面都要带 “喵”"
                                         )
    print(f"id {response.conversation_id}")
    print(f"ask {response.ask}")
    print(f"reply {response.reply}")
    print(f"usage:{response.llm.usage}")
    print(f"---{response.llm.time}---")

    promptManger.clean()
    promptManger.insert(item=PromptItem(start=conversation.start_name, text="whats my number?"))
    response = await chat_client.predict(model="text-davinci-003",
                                         prompt=promptManger,
                                         predict_tokens=500,
                                         increase="External enhancement:each sentence followed by meow",
                                         info="The parse_reply callback handles the reply fields of llm, such as list, etc. Pass in list and pass out str for the reply.",
                                         info2="The rest of the extra parameters are passed into the extra parameters of llm, as per the Api documentation, such as the top parameter of openai or whatever"
                                         )
    print(f"id {response.conversation_id}")
    print(f"ask {response.ask}")
    print(f"reply {response.reply}")
    print(f"usage:{response.llm.usage}")
    print(f"---{response.llm.time}---")


asyncio.run(chat())
```

## Frame

```
├── client
│      ├── agent.py  //profile class
│      ├── anchor.py // client etc.
│      ├── enhance.py // web search etc.
│      ├── __init__.py
│      ├── llm.py // llm func.
│      ├── module  // plugin for enhance
│      ├── Optimizer.py // memory Optimizer (cutter
│      ├── pot.py // test cache
│      ├── test_module.py // test plugin
│      ├── text_analysis_tools // nlp support
│      ├── types.py // data class
│      └── vocab.json // cache?
├── __init__.py
├── openai  // func
│      ├── api // data
│      ├── __init__.py
│      └── resouce  // func
├── requirements.txt
└── utils  // utils... tools...
    ├── chat.py
    ├── data.py
    ├── fatlangdetect //lang detect
    ├── langdetect
    ├── network.py
    └── setting.py

```
