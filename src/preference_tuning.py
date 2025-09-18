# -*- coding: UTF-8 -*-
# 偏好微调程序入口
# 食用方法：选择相应的数据集，调整参数，直接运行即可。注：更改数据集需要修改两处代码，已标明在下文中
# ！！请确保已经生成数据集后再运行此代码！！
# ！！运行前确保日志目录已经手动创建过，创建的具体位置在config.py中有声明！！


import torch
from Models.models import PretrainedModel, TikTokenizer, InstructModel, PreferenceModel, PreferenceTrainer
from Models.dataset.preference import create_dataloader
from config import GPT2_124M_CONFIG, PREFERENCE_V1_CONFIG, PREFERENCE_V2_CONFIG, PREFERENCE_V3_CONFIG
from os.path import abspath
from rich import print


PREFERENCE_VX_CONFIG = PREFERENCE_V1_CONFIG  # 数据集修改代码第一处：修改CONFIG
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Running on GPU")
else:
    DEVICE = torch.device("cpu")
    print("Running on CPU")
DATASET_PATH = abspath("../data/datasets/preference/data_v1.json")  # 数据集修改代码第二处：修改数据集位置
INSTRUCT_MODEL_PATH = abspath("../data/models/instruct/v1/inst-model-v1-epoch3.pth")

TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 4
VAL_DATA_SIZE = 64

TEST_TEXT = "What is the apple?"
MAX_OUTPUT_TOKENS = 50

RANK = 16
ALPHA = 16
LORA_MODULE = ["W_query", "W_key", "W_value", "out_proj"]


def main():
    tokenizer = TikTokenizer("gpt2")

    # Step 1 构建数据集
    train_dataloader, val_dataloader = create_dataloader(
        path=DATASET_PATH,
        train_batch_size=TRAIN_BATCH_SIZE,
        val_batch_size=VAL_BATCH_SIZE,
        tokenizer=tokenizer,
        template=PREFERENCE_VX_CONFIG.prompt_template,
        max_length=GPT2_124M_CONFIG.context_length,
        device=DEVICE,
        val_size=VAL_DATA_SIZE,
        mask_prompt_tokens=True
    )
    print(f"Successfully Build Dataset:"
          f"\n- Train Dataset: Size {len(train_dataloader) * TRAIN_BATCH_SIZE}, Iter {len(train_dataloader)}"
          f"\n- Val Dataset: Size {len(val_dataloader) * VAL_BATCH_SIZE}, Iter {len(val_dataloader)}")

    # Step 2 加载指令微调模型
    pretrained_model = PretrainedModel(
        tokenizer=tokenizer,
        network_config=GPT2_124M_CONFIG,
        device=DEVICE
    )
    instruct_model = InstructModel(
        pretrained_model=pretrained_model,
        rank=RANK,
        alpha=ALPHA,
        template=PREFERENCE_VX_CONFIG.prompt_template,
        module_names=LORA_MODULE
    )
    instruct_model.load(INSTRUCT_MODEL_PATH)
    print("Successfully Load Instruct Model")

    # Step 3 创建偏好微调模型
    preference_model = PreferenceModel(
        instruct_model=instruct_model
    )
    print("Successfully Build Preference Model")

    # Step 4 训练模型
    preference_model = PreferenceTrainer(
        model=preference_model,
        config=PREFERENCE_VX_CONFIG,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader
    )
    tokens_seen = preference_model.train(test_text=TEST_TEXT, max_output_tokens=MAX_OUTPUT_TOKENS)
    print(f"Training compelete! {tokens_seen} tokens have been seen.")


if __name__ == '__main__':
    main()