# -*- coding: utf-8 -*-
# @Time    : 1/23/23 10:42 PM
# @FileName: usage_tmp.py
# @Software: PyCharm
# @Github    ：sudoskys
import asyncio
import random

# 最小单元测试
import src.llm_kira as openai_kira
import setting
from src.llm_kira.client import Optimizer
from src.llm_kira.client.types import PromptItem

print(openai_kira.RedisConfig())
openaiApiKey = setting.ApiKey
openaiApiKey: list[str]


async def completion():
    try:
        response = await openai_kira.openai.Completion(api_key=openaiApiKey, proxy_url="").create(
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


receiver = openai_kira.client
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
    promptManger.insert(item=PromptItem(start=conversation.start_name, text="我的号码是 1596321"))
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
    promptManger.insert(item=PromptItem(start=conversation.start_name, text="我的号码是多少？"))
    response = await chat_client.predict(model="text-davinci-003",
                                         prompt=promptManger,
                                         predict_tokens=500,
                                         increase="外部增强:每句话后面都要带 “喵”",
                                         info="parse_reply 回调会处理 llm 的回复字段，比如 list 等，传入list，传出 str 的回复。必须是 str。",
                                         info2="其余多余参数会传入 llm 的额外参数中，按照Api 文档传入，比如 openai 的 top 参数什么的"
                                         )
    print(f"id {response.conversation_id}")
    print(f"ask {response.ask}")
    print(f"reply {response.reply}")
    print(f"usage:{response.llm.usage}")
    print(f"---{response.llm.time}---")


async def Moderation():
    response = await openai_kira.openai.Moderations(api_key=openaiApiKey).create(input="Kill You！")
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
        response = openai_kira.utils.chat.Utils.sentiment(item)
        print(response)


async def KeyParse():
    _sentence_list = [
        "《压缩毛巾》是部怎样的作品？",
        "我没那么多时间也懒得自己",
    ]
    for item in _sentence_list:
        print(item)
        response = openai_kira.utils.chat.Utils.keyPhraseExtraction(item)
        print(response)


async def GPT2():
    _sentence_list = [
        "《压缩毛巾》是部怎样的作品？",
        "我没那么多时间也懒得自己",
    ]
    for item in _sentence_list:
        print(item)
        response = openai_kira.utils.chat.Utils.get_gpt2_tokenizer().encode(item)
        print(response)


# asyncio.run(completion())
asyncio.run(chat())
# asyncio.run(Moderation())
# asyncio.run(Sentiment())
# asyncio.run(KeyParse())
# asyncio.run(GPT2())
