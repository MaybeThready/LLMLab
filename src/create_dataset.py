# -*- coding: UTF-8 -*-
#
# 生成数据集
# 食用方法：
# 1.将CURRENT_VERSION设为1运行
# 2.将CURRENT_VERSION设为2运行
# 3.按需将CURRENT_VERSION设为3运行
# 参数详解已标明在代码中
# ！！注意运行前要启动Ollama！！

from Models.dataset.create_preference_dataset import generate_dataset_by_llm, generate_dataset_by_llm_v2
from os.path import abspath
import json
import os
from dotenv import load_dotenv

load_dotenv()


# PROMPT是用于提示大模型生成数据的，三个格式化空位依次是提示词、正确输出、关键词
PROMPT = """
Given the input
{:}
and correct output
{:}
slightly rewrite the output to be more {:}.
Keep the modification minimal.Only return the generated response and nothing else.
"""

# KEYWORDS是用于调整数据集的关键词，第一个词是支持的关键词，第二个词是拒绝的关键词
KEYWORDS_1 = ["polite", "rude"]  # 数据会更加礼貌
KEYWORDS_2 = ["rude", "polite"]  # 数据会更加粗鲁
KEYWORDS_3 = ["like a catgirl", "serious"]  # 整活用，按需运行

DST_PATH_1 = abspath("../data/datasets/preference/data_v1.json")
DST_PATH_2 = abspath("../data/datasets/preference/data_v2.json")
DST_PATH_3 = abspath("../data/datasets/preference/data_v3.json")
DATASET_PATH = abspath("../data/datasets/instruct/alpaca_data.json")

MODEL_NAME = "llama3"  # 使用哪个模型
DATA_SIZE = 10000  # 数据集大小。这个值不能超过alpaca_data.json的数据条数

CURRENT_VERSION = 1  # 当前生成的数据版本


def generate_data(version: int):
    api_key = os.getenv("API_KEY")
    match version:
        case 1:
            generate_dataset_by_llm_v2(DATASET_PATH, DST_PATH_1, PROMPT, KEYWORDS_1, DATA_SIZE, api_key=api_key)

        case 2:
            with open(DST_PATH_1, "r", encoding="utf-8") as file:
                data = json.load(file)
            for i in range(len(data)):
                data[i]["chosen"], data[i]["rejected"] = data[i]["rejected"], data[i]["chosen"]
            with open(DST_PATH_2, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4, ensure_ascii=False)

        case 3:
            generate_dataset_by_llm_v2(DATASET_PATH, DST_PATH_3, PROMPT, KEYWORDS_3, DATA_SIZE, api_key=api_key)


if __name__ == '__main__':
    generate_data(CURRENT_VERSION)

