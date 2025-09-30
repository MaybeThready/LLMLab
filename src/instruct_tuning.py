# -*- coding: UTF-8 -*-
# 指令微调程序入口

import torch
from Models.models import PretrainedModel, TikTokenizer, InstructModel, InstructTrainer
from Models.dataset.instruct import create_dataloader
from config import GPT2_124M_CONFIG, INSTRUCT_TRAINER_CONFIG
from os.path import abspath
from rich import print


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Running on GPU")
else:
    DEVICE = torch.device("cpu")
    print("Running on CPU")
DATASET_PATH = abspath("../data/datasets/instruct/alpaca_data.json")
PRETRAINED_MODEL_PATH = abspath("../data/models/pretrained/gpt2-124M.pth")

TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 16
VAL_DATA_SIZE = 64

RANK = 16
ALPHA = 16
LORA_MODULE_V1 = None
LORA_MODULE_V2 = ["W_query", "W_key", "W_value", "out_proj"]
LORA_MODULE_V3 = ["0", "2"]
LORA_MODULE_V4 = ["W_query", "W_key", "W_value", "out_proj", "0", "2"]
LORA_MODULE = LORA_MODULE_V2  # 修改此处以尝试不同的微调方法

TEST_TEXT = "What is the apple?"
MAX_OUTPUT_TOKENS = 50


def main():
    tokenizer = TikTokenizer("gpt2")

    # Step 1 构建数据集
    train_dataloader, val_dataloader = create_dataloader(
        path=DATASET_PATH,
        train_batch_size=TRAIN_BATCH_SIZE,
        val_batch_size=VAL_BATCH_SIZE,
        tokenizer=tokenizer,
        template=INSTRUCT_TRAINER_CONFIG.prompt_template,
        max_length=GPT2_124M_CONFIG.context_length,
        device=DEVICE,
        val_size=VAL_DATA_SIZE
    )
    print(f"Successfully Build Dataset:"
          f"\n- Train Dataset: Size {len(train_dataloader) * TRAIN_BATCH_SIZE}, Iter {len(train_dataloader)}"
          f"\n- Val Dataset: Size {len(val_dataloader) * VAL_BATCH_SIZE}, Iter {len(val_dataloader)}")

    # Step 2 加载预训练模型
    pretrained_model = PretrainedModel(
        tokenizer=tokenizer,
        network_config=GPT2_124M_CONFIG,
        device=DEVICE
    )
    pretrained_model.load(PRETRAINED_MODEL_PATH)
    print("Successfully Load Pretrained Model")

    # Step 3 创建指令微调模型
    instruct_model = InstructModel(
        pretrained_model=pretrained_model,
        rank=RANK,
        alpha=ALPHA,
        template=INSTRUCT_TRAINER_CONFIG.prompt_template,
        module_names=LORA_MODULE
    )
    print("Successfully Build Instruct Model")

    print("# Trainable Parameters:", sum(p.numel() for p in instruct_model.network.parameters() if p.requires_grad))

    # Step 4 训练模型
    instruct_trainer = InstructTrainer(
        model=instruct_model,
        config=INSTRUCT_TRAINER_CONFIG,
        train_loader=train_dataloader,
        val_loader=val_dataloader
    )
    instruct_trainer.train(test_text=TEST_TEXT, max_output_tokens=MAX_OUTPUT_TOKENS)
    print("Training Compelete")


if __name__ == '__main__':
    main()