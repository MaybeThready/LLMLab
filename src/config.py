# -*- coding: UTF-8 -*-

# 在这里实例化配置类型，或写入一些常数

from Models.config import GPTNetworkConfig


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
