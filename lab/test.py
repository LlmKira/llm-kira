# -*- coding: utf-8 -*-
# @Time    : 1/23/23 10:42 PM
# @FileName: usage_tmp.py
# @Software: PyCharm
# @Github    ：sudoskys
import asyncio
import random
import time
from typing import List

from llm_kira.client.llms import ChatGptParam
from llm_kira.radio.anchor import SearchCraw
from loguru import logger
from llm_kira import radio

# 最小单元测试
import src.llm_kira as llm_kira
import setting
from src.llm_kira.creator.think import ThinkEngine, Hook
from src.llm_kira.client import Optimizer
from src.llm_kira.client.llms.openai import OpenAiParam
from src.llm_kira.client.types import PromptItem, Interaction

import sys

logger.remove()
handler_id = logger.add(sys.stderr, level="TRACE")

#
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


async def mood_hook():
    _think = ThinkEngine(profile=conversation)
    _think.register_hook(Hook(name="happy", trigger="happy", value=2, last=60, time=int(time.time())))  # 60s
    _think.hook("happy")
    print(_think.hook_pool)
    print(_think.build_status(rank=5))


async def chat_str():
    llm = llm_kira.client.llms.ChatGpt(
        profile=conversation,
        api_key=openaiApiKey,
        token_limit=4000,
        auto_penalty=False,
        call_func=None,
    )
    _res = await llm.run(prompt="你好", predict_tokens=200, llm_param=ChatGptParam())
    print(_res)


async def chatGpt():
    llm = llm_kira.client.llms.ChatGpt(
        profile=conversation,
        api_key=openaiApiKey,
        token_limit=4000,
        auto_penalty=False,
        call_func=None,
    )

    mem = llm_kira.client.MemoryManager(profile=conversation)
    chat_client = llm_kira.client.ChatBot(
        profile=conversation,
        llm_model=llm
    )
    promptManager = llm_kira.creator.engine.PromptEngine(
        reverse_prompt_buffer=False,
        profile=conversation,
        connect_words="\n",
        memory_manger=mem,
        llm_model=llm,
        description="晚上了，这里是河边",
        reference_ratio=0.5,
        forget_words=["忘掉对话"],
        optimizer=Optimizer.SinglePoint,
    )
    # 大型数据对抗测试
    # promptManager.insert_prompt(prompt=PromptItem(start="Neko", text=random_string(8000)))
    # promptManager.insert_prompt(prompt=PromptItem(start="Neko", text=random_string(500)))

    # 多 prompt 对抗测试
    testPrompt = input("TestPrompt:")
    promptManager.insert_prompt(prompt=PromptItem(start="Neko", text="喵喵喵"))
    promptManager.insert_interaction(Interaction(single=True, ask=PromptItem(start="alice", text="MewMewMewMew")))
    _result = await promptManager.build_skeleton(query=testPrompt,
                                                 llm_task="Summary Text" if len(
                                                     testPrompt) > 20 else None,
                                                 skeleton=random.choice([SearchCraw(
                                                     deacon=["https://www.bing.com/search?q={}&form=QBLH"])])
                                                 )
    _index = 1
    for item in _result:
        logger.trace(item.content)
        item.ask.start = f"[{_index}]"
        promptManager.insert_knowledge(knowledge=item)
        _index += 1
    promptManager.insert_knowledge(Interaction(single=True, ask=PromptItem(start="alice", text="MewMewMewMew")))
    # 测试
    promptManager.insert_prompt(prompt=PromptItem(start=conversation.start_name, text=testPrompt))
    response = await chat_client.predict(
        prompt=promptManager,
        llm_param=ChatGptParam(model_name="gpt-3.5-turbo", temperature=0.8, presence_penalty=0.1, n=1, best_of=1),
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


async def chat():
    llm = llm_kira.client.llms.OpenAi(
        profile=conversation,
        api_key=openaiApiKey,
        token_limit=4000,
        auto_penalty=False,
        call_func=None,
    )

    mem = llm_kira.client.MemoryManager(profile=conversation)
    chat_client = llm_kira.client.ChatBot(
        profile=conversation,
        llm_model=llm
    )
    promptManager = llm_kira.creator.engine.PromptEngine(
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
    testPrompt = input("TestPrompt:")
    promptManager.insert_prompt(prompt=PromptItem(start="Neko", text="喵喵喵"))
    promptManager.insert_interaction(Interaction(single=True, ask=PromptItem(start="alice", text="MewMewMewMew")))
    _result = await promptManager.build_skeleton(query=testPrompt,
                                                 llm_task="Summary Text" if len(
                                                     testPrompt) > 20 else None,
                                                 skeleton=random.choice([SearchCraw(
                                                     deacon=["https://www.bing.com/search?q={}&form=QBLH"])])
                                                 )
    _index = 1
    for item in _result:
        logger.trace(item.content)
        item.ask.start = f"[{_index}]"
        promptManager.insert_knowledge(knowledge=item)
        _index += 1

    promptManager.insert_knowledge(Interaction(single=True, ask=PromptItem(start="alice", text="MewMewMewMew")))
    # 测试
    promptManager.insert_prompt(prompt=PromptItem(start=conversation.start_name, text=testPrompt))
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
    test1 = """
    早苗（さなえ）
耕种水稻时刚刚种植的幼苗
原型是守矢早苗（もりやさなえ，生于1945年），守矢家第七十八代当主，是实际存在的人物。
守矢家是洩矢神的子孙，现任诹访神社下社神长官。洩矢神的祭祀司守矢家代代口传的祭神秘法。那个秘传是一脉相承的，在半夜没有火光的祈祷殿之中秘密传授。但是随着时代的变迁，世袭神官制度在明治五年被取消了。到明治六年，家传之宝（包括：印（印文「卖神祝印」）与镜、太刀等）从诹访大社上社被移走家里只残留下用佐奈伎铃（在大御立座祭神中所使用的）祭祀御左口神的方法。在明治时代，守矢实久（第七十六代当主）被取消了神长官一职，可惜当时口传秘法已失，实久只告诉了守矢真幸（第七十七代当主，实久之弟，诹访大社的祢宜宫司）剩下的部分。到守矢早苗（第七十八代当主，真幸之孙，平成18年(注)3月末从校长(注)[6]一职退下之后，一直致力于环境保护的演讲）这一代，已经不再继承代代相传的已消失的秘法了。到现在，再也没有人知道守矢祭神秘法了。
    """
    response = llm_kira.utils.chat.Sim.cosion_similarity(pre=test1, aft="守矢家第七十八代当主")
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
    # 一阶线性微分方程的条件？
    # 什么是线性定常系统？
    # mysql中regexp的用法有哪些
    #
    _sentence_list = [
        "什么是线性定常系统？"
    ]
    from llm_kira.radio.anchor import SearchCraw, DuckgoCraw
    for item in _sentence_list:
        logger.trace(item)
        response = await SearchCraw(deacon=["https://www.bing.com/search?q={}&form=QBLH"]).run(prompt=item,
                                                                                               prompt_raw=item)
        # response = DuckgoCraw().run(prompt=item, prompt_raw=item)
        for items in response:
            # logger.info(type(items))
            logger.info(items.ask.prompt)


t1 = time.time()
# asyncio.run(completion())
# asyncio.run(mood_hook())
# asyncio.run(Web())
# asyncio.run(chat())
asyncio.run(chatGpt())
# asyncio.run(chat_str())
# asyncio.run(Moderation())
# asyncio.run(Sentiment())
# asyncio.run(KeyParse())
# asyncio.run(GPT2())
# asyncio.run(Sim())
# print(float(1))
# print(int(1.2))
t2 = time.time()

print(f"Run Cost:{t2 - t1}")
