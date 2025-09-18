# -*- coding: UTF-8 -*-

# 创建偏好微调数据文件
# 该模块中的代码需要提前安装ollama软件并启动

import ollama
import json
import random
from rich.progress import Progress


PROMPT_FAKE_QUESTION = """
假设你要测试一个经过指令微调后的大语言模型的回答能力，请你生成{:}个问题用于测试。
你的回答应当只包含问题信息，每个问题之间应当用换行符分隔。不应该包含任何多余的内容。
"""

PROMPT_FAKE_ANSWER = """
假设你是一个仅经过指令微调且能力不是很优秀的大语言模型，现在有如下问题：
{:}
请你对此问题作出回答。
你的回答应当只包含答案信息，不应该包含任何多余的内容。
"""


def generate_fake_data(data_size: int, fp: str):
    """
    生成虚假测试数据，仅供测试
    """
    response = ollama.chat(model="deepseek-r1", messages=[
        {
            "role": "user",
            "content": PROMPT_FAKE_QUESTION.format(data_size)
        }
    ], think=False)["message"]["content"]
    questions = response.split('\n')
    data = []
    for i, question in enumerate(questions):
        response = ollama.chat(model="deepseek-r1", messages=[
            {
                "role": "user",
                "content": PROMPT_FAKE_ANSWER.format(question)
            },
        ], think=False)["message"]["content"]
        data.append(
            {
                "input": question,
                "output": response
            }
        )
    with open(fp, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def generate_dataset_by_llm(src: str, dst: str, prompt: str, keywords: list, data_size: int, model="deepseek-r1"):
    """
    依据src、prompt生成微调数据集，存储在dst中
    prompt必须包含三个空闲字段，顺序依次是输入、输出、待选关键词
    keywords长度必须是2，且前者为赞同的关键词，后者为反对的关键词
    """
    with open(src, "r", encoding="utf-8") as file:
        data = json.load(file)
    data = data[:data_size]
    generate_data = [dict() for _ in range(len(data))]

    with Progress() as progress:
        task = progress.add_task("Generating dataset", total=len(data))

        for i, entry in enumerate(data):
            index = random.randint(0, 1)
            response = ollama.chat(model=model, messages=[
                {
                    "role": "user",
                    "content": prompt.format(
                        entry["instruction"] + ", Input:" + entry["input"],
                        entry["output"],
                        keywords[index]
                    )
                }
            ], think=False)["message"]["content"]
            generate_data[i]["instruction"] = entry["instruction"]
            generate_data[i]["input_text"] = entry["input"]
            if index == 0:
                generate_data[i]["chosen"] = response
                generate_data[i]["rejected"] = entry["output"]
            else:
                generate_data[i]["chosen"] = entry["output"]
                generate_data[i]["rejected"] = response
            progress.advance(task, 1)

    with open(dst, "w", encoding="utf-8") as file:
        json.dump(generate_data, file, indent=4, ensure_ascii=False)

