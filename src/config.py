# -*- coding: UTF-8 -*-
# 在这里实例化配置类型，或写入一些常数

from Models.config import GPTNetworkConfig, QwenNetworkConfig, InstructTrainerConfig, PreferenceTrainerConfig
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
    drop_rate=0.1,
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

INSTRUCT_TRAINER_V1_CONFIG = InstructTrainerConfig(
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

PREFERENCE_V1_CONFIG = PreferenceTrainerConfig(
    beta=0.1,
    log_dir=abspath("../data/models/preference/v1/log/"),
    epochs=1,
    eval_freq=50,
    lr=1e-5,
    model_save_path=abspath("../data/models/preference/v1/"),
    model_save_name="preference-model-v1",
    prompt_template="alpaca"
)

PREFERENCE_V2_CONFIG = PreferenceTrainerConfig(
    beta=0.1,
    log_dir=abspath("../data/models/preference/v2/log/"),
    epochs=1,
    eval_freq=50,
    lr=1e-5,
    model_save_path=abspath("../data/models/preference/v2/"),
    model_save_name="preference-model-v2",
    prompt_template="alpaca"
)

PREFERENCE_V3_CONFIG = PreferenceTrainerConfig(
    beta=0.1,
    log_dir=abspath("../data/models/preference/v3/log/"),
    epochs=1,
    eval_freq=50,
    lr=1e-5,
    model_save_path=abspath("../data/models/preference/v3/"),
    model_save_name="preference-model-v3",
    prompt_template="alpaca"
)
