# -*- coding: utf-8 -*-
# @Time    : 1/23/23 10:42 PM
# @FileName: usage_tmp.py
# @Software: PyCharm
# @Github    ：sudoskys
import asyncio
import random
import time
from typing import List

# 最小单元测试
import src.llm_kira as llm_kira
import setting
from src.llm_kira.creator.think import ThinkEngine, Hook
from src.llm_kira.client import Optimizer
from src.llm_kira.client.llms.openai import OpenAiParam
from src.llm_kira.client.types import PromptItem, Interaction

openaiApiKey = setting.ApiKey
openaiApiKey: List[str]
print(llm_kira.RedisConfig())

import openai as open_clinet

open_clinet.api_key = random.choice(openaiApiKey)  # supply your API key however you choose


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
    conversation_id=12094,  # random.randint(1, 10000000),
)

llm = llm_kira.client.llms.OpenAi(
    profile=conversation,
    api_key=openaiApiKey,
    token_limit=4000,
    auto_penalty=False,
    call_func=None,
)

mem = receiver.MemoryManager(profile=conversation)
chat_client = receiver.ChatBot(profile=conversation,
                               llm_model=llm
                               )


async def mood_hook():
    _think = ThinkEngine(profile=conversation)
    _think.register_hook(Hook(name="happy", trigger="happy", value=2, last=60, time=int(time.time())))  # 60s
    _think.hook("happy")
    print(_think.hook_pool)
    print(_think.build_status(rank=5))


async def chat():
    promptManager = llm_kira.creator.PromptEngine(
        reverse_prompt_buffer=False,
        profile=conversation,
        connect_words="\n",
        memory_manger=mem,
        llm_model=llm,
        description="这是一段对话",
        reference_ratio=0.5,
        forget_words=["忘掉对话"],
        optimizer=Optimizer.SinglePoint,
    )
    # 大型数据对抗测试
    # promptManager.insert_prompt(prompt=PromptItem(start="Neko", text=random_string(8000)))
    # promptManager.insert_prompt(prompt=PromptItem(start="Neko", text=random_string(500)))

    # 多 prompt 对抗测试
    promptManager.insert_prompt(prompt=PromptItem(start="Neko", text="喵喵喵"))
    promptManager.insert_interaction(Interaction(single=True, ask=PromptItem(start="alice", text="MewMewMewMew")))
    # 测试
    promptManager.insert_prompt(prompt=PromptItem(start=conversation.start_name, text=input("TestPrompt:")))
    response = await chat_client.predict(
        prompt=promptManager,
        llm_param=OpenAiParam(model_name="text-davinci-003", temperature=0.8, presence_penalty=0.1, n=1, best_of=1),
        predict_tokens=1000,
    )
    print(f"id {response.conversation_id}")
    print(f"ask {response.ask}")
    print(f"reply {response.reply}")
    print(f"usage:{response.llm.usage}")
    print(f"raw:{response.llm.raw}")
    print(f"---{response.llm.time}---")
    promptManager.clean(clean_prompt=True, clean_knowledge=False, clean_memory=False)
    return "End"
    promptManager.insert_prompt(prompt=PromptItem(start=conversation.start_name, text='今天天气怎么样'))
    response = await chat_client.predict(
        llm_param=OpenAiParam(model_name="text-davinci-003", temperature=0.8, presence_penalty=0.1, n=2, best_of=2),
        prompt=promptManager,
        predict_tokens=500,
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
        "抑郁了，自杀了",
        "公主也能看啊",
        "换谁都被吓走吧",
        "错了"
    ]
    for item in _sentence_list:
        print(item)
        response = llm_kira.utils.chat.Utils.sentiment(item)
        print(response)


async def Sim():
    # response = llm_kira.utils.chat.Utils.edit_similarity(pre="4552", aft="1224")
    # print(response)
    response = llm_kira.utils.chat.Sim.cosion_similarity(pre="", aft="你是不是啊")
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


t1 = time.time()
# asyncio.run(completion())
# asyncio.run(mood_hook())
asyncio.run(chat())
# asyncio.run(Moderation())
# asyncio.run(Sentiment())
# asyncio.run(KeyParse())
# asyncio.run(GPT2())
# asyncio.run(Sim())
# print(float(1))
# print(int(1.2))
t2 = time.time()

print(t2 - t1)
