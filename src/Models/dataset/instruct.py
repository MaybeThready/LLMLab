# -*- coding: UTF-8 -*-


from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from ..models.tokenizer import Tokenizer
from ..utils import create_prompt
import json
import torch


class InstructDataset(Dataset):
    def __init__(self, path: str, tokenizer: Tokenizer, template: str="alpaca"):
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        self.encoded_texts = []
        for entry in data:
            prompt = create_prompt(
                instruction=entry["instruction"],
                input_text=entry["input"],
                output_text=entry["output"],
                template=template
            )
            self.encoded_texts.append(tokenizer.encode(prompt))

    def __getitem__(self, item):
        return self.encoded_texts[item]

    def __len__(self):
        return len(self.encoded_texts)


class CollateFuntion:
    def __init__(self, pad_token_id: int, ignore_index=-100, max_length: int=None, device: torch.device=torch.device("cpu")):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.max_length = max_length
        self.device = device

    def __call__(self, batch):
        batch_max_length = max(len(item) + 1 for item in batch)
        inputs_lst, targets_lst = [], []

        for item in batch:
            new_item = item.copy()
            new_item += [self.pad_token_id]

            padded = new_item + [self.pad_token_id] * (batch_max_length - len(new_item))

            inputs = torch.tensor(padded[:-1])
            targets = torch.tensor(padded[1:])

            mask = targets == self.pad_token_id
            indices = torch.nonzero(mask).squeeze()

            if indices.numel() > 1:
                targets[indices[1:]] = self.ignore_index

            if self.max_length is not None:
                inputs = inputs[:self.max_length]
                targets = targets[:self.max_length]

            inputs_lst.append(inputs)
            targets_lst.append(targets)

        inputs_tensor = torch.stack(inputs_lst).to(self.device)
        targets_tensor = torch.stack(targets_lst).to(self.device)
        return inputs_tensor, targets_tensor


def create_dataloader(path: str,
                      train_batch_size: int,
                      val_batch_size: int,
                      tokenizer: Tokenizer,
                      template: str="alpaca",
                      pad_token_id: int=None,
                      ignore_index=-100,
                      max_length: int=None,
                      device: torch.device=torch.device("cpu"),
                      train_rate: float = 0.8
                      ):
    dataset = InstructDataset(path, tokenizer, template)
    size = len(dataset)
    indices = list(range(size))
    split_index = int(train_rate * size)
    train_indices, val_indices = indices[:split_index], indices[split_index:]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    pad_token_id = tokenizer.eos_id if pad_token_id is None else pad_token_id
    collate_fn = CollateFuntion(pad_token_id, ignore_index, max_length, device)

    train_loader = DataLoader(dataset,
                              train_batch_size,
                              sampler=train_sampler,
                              collate_fn=collate_fn
                              )
    val_loader = DataLoader(dataset,
                            val_batch_size,
                            sampler=val_sampler,
                            collate_fn=collate_fn,
                            drop_last=True)

    return train_loader, val_loader


