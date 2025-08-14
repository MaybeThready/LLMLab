# -*- coding: UTF-8 -*-

# 主程序入口

from os.path import abspath

import torch
from rich import print

from Models.models.tokenizer import TikTokenizer, JsonTokenizer
from Models.models.pretrained import PretrainedModel
from config import GPT2_124M_CONFIG, QWEN2P5_1P5B_CONFIG, QWEN2P5_1P5B_TOKENIZER_EOS_ID


MAX_TEXT_LENGTH = 50
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Running on GPU")
else:
    DEVICE = torch.device("cpu")
    print("Running on CPU")


def main_gpt():
    tokenizer = TikTokenizer("gpt2")
    model = PretrainedModel(tokenizer, GPT2_124M_CONFIG, DEVICE)
    model.load(abspath("../data/models/pretrained/gpt2-124M.pth"))
    print("这是一个示例程序。程序实例化了一个GPT网络，并加载了GPT2-124M模型权重。\n"
          "你可以在接下来的测试中向模型输入文本，它会尝试自动补全这个文本。\n"
          "由于没有经过指令微调，请不要对模型的回答抱有期许。\n"
          "输入 EXIT 以退出程序。")
    while True:
        text = input("提示文本：")
        if text == "EXIT":
            break
        response = model.generate_from_text(text, MAX_TEXT_LENGTH)
        print(f"GPT：{response}")


def main_qwen():
    tokenizer = JsonTokenizer(
        abspath("../data/models/tokenizer/qwen2p5-1p5B.json"),
        eos_id=QWEN2P5_1P5B_TOKENIZER_EOS_ID
    )
    model = PretrainedModel(tokenizer, QWEN2P5_1P5B_CONFIG, DEVICE)
    model.load(abspath("../data/models/pretrained/qwen2p5-1p5B.pth"))
    print("这是一个示例程序。程序实例化了一个Qwen网络，并加载了Qwen2.5-1.5B模型权重。\n"
          "你可以在接下来的测试中向模型输入文本，它会尝试自动补全这个文本。\n"
          "由于没有经过指令微调，请不要对模型的回答抱有期许。\n"
          "输入 EXIT 以退出程序。")
    while True:
        text = input("提示文本：")
        if text == "EXIT":
            break
        response = model.generate_from_text(text, MAX_TEXT_LENGTH)
        print(f"Qwen：{response}")



if __name__ == '__main__':
    main_gpt()
