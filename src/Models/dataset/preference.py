# -*- coding: UTF-8 -*-

import json

import torch
from torch.utils.data import Dataset, DataLoader

from ..models.tokenizer import Tokenizer
from ..utils import create_prompt


class PreferenceDataset(Dataset):
    """
    偏好微调数据集构建
    """
    def __init__(self, path: str, tokenizer: Tokenizer, template: str="default"):
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        self.length = len(data)
        self.encoded_texts = []
        for entry in data:
            prompt = create_prompt(instruction=entry["input"], template=template)
            chosen = create_prompt(output_text=entry["chosen"], template=template)
            rejected = create_prompt(output_text=entry["rejected"], template=template)

            self.encoded_texts.append(
                {
                    "prompt": tokenizer.encode(prompt),
                    "chosen": tokenizer.encode(chosen),
                    "rejected": tokenizer.encode(rejected)
                }
            )

    def __getitem__(self, item):
        return self.encoded_texts[item]

    def __len__(self):
        return self.length


class CollateFunction:
    """
    预处理函数。这是一个可回调的类
    """
    def __init__(self, pad_token_id: int, max_length: int=None, mask_prompt_tokens: bool=True, device: torch.device=torch.device("cpu")):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.mask_prompt_tokens = mask_prompt_tokens
        self.device = device

    def __call__(self, batch):
        batch_data = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
            "chosen_mask": [],
            "rejected_mask": [],
        }

        max_common_length = 0
        if batch:
            for key in ["chosen", "rejected"]:
                current_max = max(len(item[key]) + 1 for item in batch)
                max_common_length = max(max_common_length, current_max)

        for item in batch:
            prompt = torch.tensor(item["prompt"], dtype=torch.long, device=self.device)
            batch_data["prompt"].append(prompt)

            for key in ["chosen", "rejected"]:
                seq = item[key]
                padded = seq + [self.pad_token_id] * (max_common_length - len(seq))

                mask = torch.ones(len(padded), device=self.device).bool()
                mask[len(seq):] = False

                if self.mask_prompt_tokens:
                    mask[:prompt.shape[0] + 1] = False

                batch_data[key].append(torch.tensor(padded, dtype=torch.long, device=self.device))
                batch_data[f"{key}_mask"].append(mask)

        for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
            tensor_stack = torch.stack(batch_data[key])

            if self.max_length is not None:
                tensor_stack = tensor_stack[:, :self.max_length]

            batch_data[key] = tensor_stack.to(self.device)

        return batch_data


def create_dataloader(path: str,
                      tokenizer: Tokenizer,
                      batch_size: int,
                      pad_token_id: int,
                      template: str = "default",
                      max_length: int=None,
                      mask_prompt_tokens: bool=True,
                      device: torch.device=torch.device("cpu"),
                      shuffle: bool=True,
                      drop_last: bool=False):
    return DataLoader(
        PreferenceDataset(path, tokenizer, template),
        batch_size=batch_size,
        collate_fn=CollateFunction(pad_token_id, max_length, mask_prompt_tokens, device),
        shuffle=shuffle,
        drop_last=drop_last
    )

            

