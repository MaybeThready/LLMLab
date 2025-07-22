# -*- coding: UTF-8 -*-

# 在这里写入一些你觉得十分实用的小函数

import random
from typing import Any

import torch

NUM_TYPE = int | float


def random_by_prob(prob_dict: dict[Any, float]) -> Any:
    """
    根据概率列表生成数据
    :param prob_dict:
    :return:
    """
    rate = random.random()
    current = 0
    for k, v in prob_dict.items():
        current += v
        if rate < current:
            return k
    raise ValueError("The sum of the probabilities is not 1")


def calculate_network_size(network, unit="mb") -> float:
    """
    计算网络大小
    :param network:
    :param unit: 单位
    :return:
    """
    total_params = sum(p.numel() for p in network.parameters())
    total_size_bytes = total_params * 4
    k = 1
    match unit.lower():
        case "b":
            k = 1
        case "kb":
            k = 1024
        case "mb":
            k = 1024 * 1024
        case "gb":
            k = 1024 * 1024 * 1024
        case "billion":
            return total_params / 10_000_000_000
    return total_size_bytes / k


def assign(left, right):
    """
    比对两个张量的形状，如果不同则抛出异常，相同就返回右值
    :param left:
    :param right:
    :return:
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))
