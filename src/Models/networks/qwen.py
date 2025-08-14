# -*- coding: UTF-8 -*-

# 在这里定义Qwen网络

import torch
from torch import nn

from ..config import QwenNetworkConfig


class RoPE:
    def __init__(self, dim: int, max_context_length: int, theta: float=10000.):
        """
        RoPE位置编码
        这玩意数学推导我是真看不懂，想了解的可以看看https://www.zhihu.com/tardis/bd/art/647109286
        """
        self.dim = dim
        self.theta = theta
        # 生成 token 序列索引 t = [0, 1,..., seq_len-1], freqs.shape = [seq_len, group_dim // 2]
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

        t = torch.arange(max_context_length)
        freqs = torch.outer(t, freqs).float()  # 计算m * \theta

        # 计算结果是个复数向量
        # 假设 freqs = [x, y]
        # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
        self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs).unsqueeze(1).unsqueeze(1)

    def rotary_emb(self, num_tokens: int, query: torch.Tensor, key: torch.Tensor):
        device = query.device
        self.freqs_cis = self.freqs_cis.to(device)

        # query.shape =     [batch_size, num_tokens, g, kvh,    group_dim]
        # query_.shape =    [batch_size, num_tokens, g, kvh,    group_dim // 2, 2]
        # key.shape =       [batch_size, num_tokens, 1, kvh,    group_dim]
        # key_.shape =      [batch_size, num_tokens, 1, kvh,    group_dim // 2, 2]
        query_ = query.float().reshape(*query.shape[:-1], -1, 2)
        key_ = key.float().reshape(*key.shape[:-1], -1, 2)

        # 转为复数域
        # query_.shape =>   [batch_size, num_tokens,    g,  kvh,    group_dim // 2  ]
        # key_.shape =>     [batch_size, num_tokens,    1,  kvh,    group_dim // 2  ]
        # freqs_cis.shape =             [num_tokens,    1,  1,      group_dim // 2  ]
        query_ = torch.view_as_complex(query_)
        key_ = torch.view_as_complex(key_)

        # 应用旋转操作，然后将结果转回实数域
        # query_.shape  =>  [batch_size,    num_tokens, g,  kvh,    group_dim // 2, 2]
        #               =>  [batch_size,    num_tokens, g,  kvh,    group_dim]
        # key_.shape    =>  [batch_size,    num_tokens, 1,  kvh,    group_dim // 2, 2]
        #               =>  [batch_size,    num_tokens, 1,  kvh,    group_dim]
        query_out = torch.view_as_real(query_ * self.freqs_cis[:num_tokens]).flatten(4)
        key_out = torch.view_as_real(key_ * self.freqs_cis[:num_tokens]).flatten(4)
        return query_out.type_as(query), key_out.type_as(key)


class GroupedQueryAttention(nn.Module):
    def __init__(
            self,
            dim_in: int,
            dim_out: int,
            drop_rate: float,
            num_query_heads: int,
            num_key_value_heads: int,
            rope: RoPE = None,
            qkv_bias: bool = False
    ):
        """
        分组查询注意力
        与参考书上提到的GPT所使用的多头注意力模型（MHA）不同，Qwen采用的是分组查询注意力模型（GQA）
        :param dim_in: 输入维度
        :param dim_out: 输出维度
        :param drop_rate: dropout率
        :param num_query_heads: 查询头数
        :param num_key_value_heads: 键值头数
        :param rope: RoPE位置编码
        :param qkv_bias: 是否添加偏置
        """
        super().__init__()
        if num_query_heads % num_key_value_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_key_value_heads")

        if dim_out % num_query_heads != 0:
            raise ValueError("dim_out must be divisible by num_query_heads")

        self.rope = rope
        self.dim_out = dim_out
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.group_size = self.num_query_heads // self.num_key_value_heads  # 每组kv对应多少个q
        self.group_dim = self.dim_out // self.num_query_heads
        key_value_dim_out = self.group_dim * self.num_key_value_heads

        # 初始化权重
        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_key = nn.Linear(dim_in, key_value_dim_out, bias=qkv_bias)
        self.W_value = nn.Linear(dim_in, key_value_dim_out, bias=qkv_bias)
        self.out_proj = nn.Linear(dim_out, dim_out, bias=False)
        self.dropout = nn.Dropout(drop_rate)
        # self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x: torch.Tensor):
        batch_size, num_tokens, dim_in = x.shape

        # 生成键、值、查找关键字
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 依据头数变换形状
        # 与GPT不同的是，这里进行了分组。尤其注意一下queries的代码
        keys = keys.view(batch_size, num_tokens, 1, self.num_key_value_heads, self.group_dim)
        queries = queries.view(batch_size, num_tokens, self.group_size, self.num_key_value_heads, self.group_dim)
        values = values.view(batch_size, num_tokens, 1, self.num_key_value_heads, self.group_dim)

        if self.rope is not None:
            queries, keys = self.rope.rotary_emb(num_tokens, queries, keys)

        keys = keys.permute(0, 3, 2, 1, 4)          # (b, n, 1, kvh, dim) -> (b, kvh, 1, n, dim)
        queries = queries.permute(0, 3, 2, 1, 4)    # (b, n, g, kvh, dim) -> (b, kvh, g, n, dim)
        values = values.permute(0, 3, 2, 1, 4)

        # 计算注意力
        # (b, kvh, g, n, dim) @ (b, kvh, 1, n, dim) -> (b, kvh, g, n, n)
        attn_scores = queries @ keys.transpose(3, 4)
        mask_bool = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask_bool, - torch.inf)

        attn_weights = torch.softmax(attn_scores / self.group_dim ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (b, kvh, g, n, n) @ (b, kvh, 1, n, dim) -> (b, kvh, g, n, dim) -> (b, qh, n, dim) -> (b, n, qh, dim)
        context_vec = (attn_weights @ values).flatten(start_dim=1, end_dim=2).transpose(1, 2)

        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.dim_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


class FeedForward(nn.Module):
    """
    Qwen的FeedForward和GPT的不同，有门控（SwiGLU）的思想在里面
    """
    def __init__(self, dim_in: int, dim_hidden):
        """
        前馈神经网络
        :param dim_in: 输入维度
        :param dim_hidden: 隐藏维度
        """
        super().__init__()
        self.W_gate = nn.Linear(dim_in, dim_hidden, bias=False)
        self.W_up = nn.Linear(dim_in, dim_hidden, bias=False)
        self.W_down = nn.Linear(dim_hidden, dim_in, bias=False)
        self.silu = nn.SiLU()


    def forward(self, x: torch.Tensor):
        gate = self.W_gate(x)
        up = self.W_up(x)
        activated = self.silu(gate) * up
        return self.W_down(activated)


class TransformerBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            num_query_heads: int,
            num_key_value_heads: int,
            hidden_dim: int,
            rope: RoPE,
            drop_rate: float,
            qkv_bias: bool = False
    ):
        """
        transformer单元
        :param embedding_dim: 嵌入维度
        :param num_query_heads: 查询头数
        :param num_key_value_heads: 键值头数
        :param hidden_dim: FeedForward层的隐藏层维度
        :param rope: RoPE位置编码
        :param drop_rate: dropout率
        :param qkv_bias: 是否启用qkv偏置
        """
        super().__init__()
        self.attention = GroupedQueryAttention(
            dim_in=embedding_dim,
            dim_out=embedding_dim,
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            rope=rope,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias
        )
        self.feedforward = FeedForward(embedding_dim, hidden_dim)
        self.norm1 = nn.RMSNorm(embedding_dim)
        self.norm2 = nn.RMSNorm(embedding_dim)
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


class QwenNetwork(nn.Module):
    def __init__(self, config: QwenNetworkConfig):
        """
        Qwen神经网络
        :param config:
        """
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.rope = RoPE(config.embedding_dim // config.num_query_heads, config.context_length, config.rope_theta)
        self.drop_emb = nn.Dropout(config.drop_rate)
        self.transformers = nn.Sequential(
            *[TransformerBlock(
                config.embedding_dim,
                config.num_query_heads,
                config.num_key_value_heads,
                config.hidden_dim,
                self.rope,
                config.drop_rate,
                config.qkv_bias
            ) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.RMSNorm(config.embedding_dim)

    def forward(self, x: torch.Tensor):
        tok_embs = self.tok_emb(x)
        x = self.drop_emb(tok_embs)
        x = self.transformers(x)
        x = self.final_norm(x)
        logits = x @ self.tok_emb.weight.T  # Qwen共用输入/输出权重
        return logits
