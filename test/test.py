# -*- coding: utf-8 -*-
# @Time    : 1/23/23 10:42 PM
# @FileName: usage_tmp.py
# @Software: PyCharm
# @Github    ：sudoskys
import asyncio
from typing import List

# 最小单元测试
import src.llm_kira as llm_kira
import setting
from src.llm_kira.client import Optimizer
from src.llm_kira.client.llms.openai import OpenAiParam
from src.llm_kira.client.types import PromptItem

print(llm_kira.RedisConfig())

openaiApiKey = setting.ApiKey
openaiApiKey: List[str]


def random_string(length):
    import random
    string = ""
    for i in range(length):
        string += chr(random.randint(97, 122))  # 生成小写字母

    return string


async def completion():
    try:
        response = await llm_kira.openai.Completion(api_key=openaiApiKey, proxy_url="").create(
            model="text-davinci-003",
            prompt="Say this is a test",
            temperature=0,
            max_tokens=20)
        # TEST
        print(response)
        print(type(response))
    except Exception as e:
        print(e)
        if "Incorrect API key provided" in str(e):
            print("OK", e)
        else:
            print("NO", e)


receiver = llm_kira.client
conversation = receiver.Conversation(
    start_name="Human:",
    restart_name="AI:",
    conversation_id=10093,  # random.randint(1, 10000000),
)

llm = llm_kira.client.llms.OpenAi(
    profile=conversation,
    api_key=openaiApiKey,
    token_limit=3700,
    auto_penalty=False,
    call_func=None,
)

mem = receiver.MemoryManager(profile=conversation)
chat_client = receiver.ChatBot(profile=conversation,
                               memory_manger=mem,
                               optimizer=Optimizer.SinglePoint,
                               llm_model=llm)


async def chat():
    promptManager = receiver.PromptManager(profile=conversation,
                                         connect_words="\n",
                                         )
    # 大型数据对抗测试
    # promptManager.insert(item=PromptItem(start="Neko", text=random_string(4000)))
    # promptManager.insert(item=PromptItem(start="Neko", text=random_string(500)))

    # 多 prompt 对抗测试
    # promptManager.insert(item=PromptItem(start="Neko", text="喵喵喵"))

    promptManager.insert(item=PromptItem(start=conversation.start_name, text="我的账号是 2216444"))
    response = await chat_client.predict(
        llm_param=OpenAiParam(model_name="text-davinci-003", temperature=0.8, presence_penalty=0.1, n=2, best_of=2),
        prompt=promptManager,
        predict_tokens=500,
        increase="外部增强:每句话后面都要带 “喵”",
    )
    print(f"id {response.conversation_id}")
    print(f"ask {response.ask}")
    print(f"reply {response.reply}")
    print(f"usage:{response.llm.usage}")
    print(f"usage:{response.llm.raw}")
    print(f"---{response.llm.time}---")
    promptManager.clean()
    promptManager.insert(item=PromptItem(start=conversation.start_name, text="说出我的账号？"))
    response = await chat_client.predict(llm_param=OpenAiParam(model_name="text-davinci-003", logit_bias=None),
                                         prompt=promptManager,
                                         predict_tokens=500,
                                         increase="外部增强:每句话后面都要带 “喵”",
                                         # parse_reply=None
                                         )
    _info = "parse_reply 回调会处理 llm 的回复字段，比如 list 等，传入list，传出 str 的回复。必须是 str。",
    print(f"id {response.conversation_id}")
    print(f"ask {response.ask}")
    print(f"reply {response.reply}")
    print(f"usage:{response.llm.usage}")
    print(f"usage:{response.llm.raw}")
    print(f"---{response.llm.time}---")


async def Moderation():
    response = await llm_kira.openai.Moderations(api_key=openaiApiKey).create(input=random_string(5000))  # "Kill You！")
    print(response)


async def Sentiment():
    _sentence_list = [
        "你是？",
        "我没那么多时间也懒得自己",
        "什么是？",
        "玉玉了，紫砂了",
        "我知道了",
        "主播抑郁了，自杀了",
        "公主也能看啊",
        "换谁都被吓走吧"
    ]
    for item in _sentence_list:
        print(item)
        response = llm_kira.utils.chat.Utils.sentiment(item)
        print(response)


async def KeyParse():
    _sentence_list = [
        "《压缩毛巾》是部怎样的作品？",
        "我没那么多时间也懒得自己",
    ]
    for item in _sentence_list:
        print(item)
        response = llm_kira.utils.chat.Utils.keyPhraseExtraction(item)
        print(response)


async def GPT2():
    _sentence_list = [
        "《压缩毛巾》是部怎样的作品？",
        "我没那么多时间也懒得自己",
    ]
    for item in _sentence_list:
        print(item)
        response = llm_kira.utils.chat.Utils.get_gpt2_tokenizer().encode(item)
        print(response)


async def Web():
    config = receiver.enhance.PluginConfig(server=["https://www.google.com/search?q={}"],
                                           text="任何邪恶，终将？")
    h0 = await receiver.enhance.WebSearch(
        config=config).run()
    h1 = await receiver.enhance.PluginSystem(plugin_table={"time": ""}, prompt="what time now?").run()
    print(h0)
    print(h1)


# asyncio.run(completion())
asyncio.run(chat())
# asyncio.run(Moderation())
# asyncio.run(Sentiment())
# asyncio.run(KeyParse())
# asyncio.run(GPT2())
# asyncio.run(Web())
# print(float(1))
# print(int(1.2))
