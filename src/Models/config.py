# -*- coding: UTF-8 -*-

# 配置类。在这里定义网络的配置类，用户调用该类时可以直接实例化一个对象反复利用，方便构建多个模型
# 训练模型的配置类也可以写在这里

from dataclasses import dataclass


@dataclass
class GPTNetworkConfig:
    context_length: int                         # 上下文长度
    embedding_dim: int                          # 词嵌入维度
    num_heads: int                              # 注意力头数
    num_layers: int                             # transformer层数
    drop_rate: float                            # dropout率
    qkv_bias: bool = False                      # 是否启用qkv偏置
    vocab_size: int = -1                        # 词表大小。这个值通常不用用户自己指定，内置模块会自动配置


@dataclass
class QwenNetworkConfig:
    context_length: int                         # 上下文长度
    embedding_dim: int                          # 词嵌入维度
    hidden_dim: int                             # MLP隐藏层维度
    num_query_heads: int                        # query头数
    num_key_value_heads: int                    # kv头数
    rope_theta: float                           # RoPE旋转角度
    num_layers: int                             # transformer层数
    drop_rate: float                            # dropout率
    qkv_bias: bool = False                      # 是否启用qkv偏置
    vocab_size: int = -1                        # 词表大小。这个值通常不用用户自己指定，内置模块会自动配置
