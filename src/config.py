# -*- coding: UTF-8 -*-
# 在这里实例化配置类型，或写入一些常数

from Models.config import GPTNetworkConfig, QwenNetworkConfig, InstructTrainerConfig
from os.path import abspath


TEST_GPT_NETWORK_CONFIG = GPTNetworkConfig(
    context_length=512,
    embedding_dim=768,
    num_heads=12,
    num_layers=12,
    drop_rate=0.1,
    qkv_bias=False,
)

GPT2_124M_CONFIG = GPTNetworkConfig(
    context_length=1024,
    embedding_dim=768,
    num_heads=12,
    num_layers=12,
    drop_rate=0.0,
    qkv_bias=True,
    vocab_size=50257
)

GPT2_355M_CONFIG = GPTNetworkConfig(
    context_length=1024,
    embedding_dim=1024,
    num_heads=16,
    num_layers=24,
    drop_rate=0.0,
    qkv_bias=True,
    vocab_size=50257
)

GPT2_774M_CONFIG = GPTNetworkConfig(
    context_length=1024,
    embedding_dim=1280,
    num_heads=20,
    num_layers=36,
    drop_rate=0.0,
    qkv_bias=True,
    vocab_size=50257
)

GPT2_1558M_CONFIG = GPTNetworkConfig(
    context_length=1024,
    embedding_dim=1600,
    num_heads=25,
    num_layers=48,
    drop_rate=0.0,
    qkv_bias=True,
    vocab_size=50257
)

QWEN2P5_1P5B_CONFIG = QwenNetworkConfig(
    context_length=131072,
    embedding_dim=1536,
    hidden_dim=8960,
    num_query_heads=12,
    num_key_value_heads=2,
    rope_theta=1000000.0,
    num_layers=28,
    drop_rate=0.0,
    qkv_bias=True,
    vocab_size=151936
)

QWEN2P5_1P5B_TOKENIZER_EOS_ID = 151643

INSTRUCT_TRAINER_CONFIG = InstructTrainerConfig(
    max_learning_rate=2e-4,
    min_learning_rate=1e-10,
    warmup_rate=0.01,
    epochs=3,
    eval_freq=50,
    model_save_path=abspath("../data/models/instruct/v1/"),
    model_save_name="inst-model-v1",
    log_dir=abspath("../data/models/instruct/v1/log/"),
    prompt_template="alpaca"
)
