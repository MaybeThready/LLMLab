# -*- coding: UTF-8 -*-

# 在这里定义一些构建复杂模型可能会用到的基本网络模型/层

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .config import GPTNetworkConfig


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            dim_in: int,
            dim_out: int,
            context_length: int,
            drop_rate: float,
            num_heads: int,
            qkv_bias: bool = False
    ):
        """
        多头注意力模型
        :param dim_in: 输入维度
        :param dim_out: 输出维度
        :param context_length: 上下文长度
        :param drop_rate: dropout率
        :param num_heads: 头数，应当是输出维度的因数
        :param qkv_bias: 是否添加偏置
        """
        super().__init__()
        if dim_out % num_heads != 0:
            raise ValueError("dim_out must be divisible by num_heads")

        self.dim_out = dim_out
        self.num_heads = num_heads
        self.head_dim = self.dim_out // self.num_heads

        # 初始化权重
        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.out_proj = nn.Linear(dim_out, dim_out)
        self.dropout = nn.Dropout(drop_rate)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x: torch.Tensor):
        batch_size, num_tokens, dim_in = x.shape

        # 生成键、值、查找关键字
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 依据头数变换形状
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算注意力
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, - torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.dim_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


class FeedForward(nn.Module):
    def __init__(self, dim: int):
        """
        前馈神经网络
        :param dim: 维度
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            context_length: int,
            num_heads: int,
            drop_rate: float,
            qkv_bias: bool = False
    ):
        """
        transformer单元
        :param embedding_dim: 嵌入维度
        :param context_length: 上下文长度
        :param num_heads: 注意力头数
        :param drop_rate: dropout率
        :param qkv_bias: 是否启用qkv偏置
        """
        super().__init__()
        self.attention = MultiHeadAttention(
            dim_in=embedding_dim,
            dim_out=embedding_dim,
            context_length=context_length,
            num_heads=num_heads,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias
        )
        self.feedforward = FeedForward(embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout_shortcut = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor):
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.feedforward(x)
        x = self.dropout_shortcut(x)
        x = x + shortcut

        return x


class GPTNetwork(nn.Module):
    def __init__(self, config: GPTNetworkConfig):
        """
        GPT神经网络
        :param config:
        """
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.pos_emb = nn.Embedding(config.context_length, config.embedding_dim)
        self.drop_emb = nn.Dropout(config.drop_rate)
        self.transformers = nn.Sequential(
            *[TransformerBlock(
                config.embedding_dim,
                config.context_length,
                config.num_heads,
                config.drop_rate,
                config.qkv_bias
            ) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.out_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.config = config

    def forward(self, x: torch.Tensor):
        batch_size, seq_len = x.shape
        tok_embs = self.tok_emb(x)
        pos_embs = self.pos_emb(torch.arange(seq_len, device=x.device))
        x = tok_embs + pos_embs
        x = self.drop_emb(x)
        x = self.transformers(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
