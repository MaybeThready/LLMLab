# -*- coding: UTF-8 -*-

# 主程序入口

from os.path import abspath

import torch
from rich import print

from Models.models.tokenizer import JsonTokenizer
from Models.models.gpt import GPTModel
from config import TEST_GPT_NETWORK_CONFIG


MAX_TEXT_LENGTH = 50
TOKENIZER_PATH = abspath("../data/models/tokenizer/Chinese-LLaMA.json")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Running on GPU")
else:
    DEVICE = torch.device("cpu")
    print("Running on CPU")


def main():
    tokenizer = JsonTokenizer(TOKENIZER_PATH)
    model = GPTModel(tokenizer, TEST_GPT_NETWORK_CONFIG, DEVICE)
    print("这是一个示例程序。程序实例化了一个GPT网络。你可以在接下来的测试中向模型输入文本，它会尝试自动补全这个文本。\n"
          "由于这个网络没有进行任何训练，因此它的回答一定是混乱的。\n"
          "输入 EXIT 以退出程序。")
    while True:
        text = input("提示文本：")
        if text == "EXIT":
            break
        response = model.generate_from_text(text, MAX_TEXT_LENGTH)
        print(f"GPT：{response}")



if __name__ == '__main__':
    main()
