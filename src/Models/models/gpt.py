# -*- coding: UTF-8 -*-

# GPT预训练模型类

import torch

from .tokenizer import Tokenizer
from ..networks import GPTNetwork
from ..config import GPTNetworkConfig


class GPTModel:
    def __init__(
            self,
            tokenizer: Tokenizer,
            network_config: GPTNetworkConfig,
            device: torch.device = torch.device("cpu")
    ):
        self.tokenizer = tokenizer
        self.network_config = network_config

        if self.network_config.vocab_size != -1:
            assert self.network_config.vocab_size == len(tokenizer),\
                (f"Vocab size of tokenizer (got {len(tokenizer)})"
                 f" must equal to the config (got {self.network_config.vocab_size})")
        else:
            self.network_config.vocab_size = len(tokenizer)

        self.network = GPTNetwork(network_config).to(device)
        self.device = device

    def generate_from_tensor(
            self,
            batch_input_ids: torch.Tensor,
            max_text_length: int,
            temperature: float = 0.,
            top_k: int = None,
    ) -> torch.Tensor:
        """
        通过张量生成文本ID
        :param batch_input_ids: 批量文本ID
        :param max_text_length: 生成文本的最大长度
        :param temperature: 温度。越小越自信，越大越多样
        :param top_k: top_k采样
        :return:
        """
        self.network.eval()
        for _ in range(max_text_length):
            batch_ids = batch_input_ids[:, -self.network_config.context_length:]
            with torch.no_grad():
                logits = self.network(batch_ids)
            logits: torch.Tensor = logits[:, -1, :]

            if top_k:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val,
                    torch.tensor(float("-inf")).to(logits.device),
                    logits
                )

            if temperature > 0.:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                id_next = torch.multinomial(probs, num_samples=1)
            else:
                id_next = torch.argmax(logits, dim=-1, keepdim=True)

            if id_next == self.tokenizer.eos_id:
                break

            batch_input_ids = torch.cat((batch_input_ids, id_next), dim=1)

        return batch_input_ids

    def generate_from_text(
            self,
            text: str,
            max_text_length: int,
            temperature: float = 0.,
            top_k: int = None,
    ) -> str:
        """
        通过文本生成文本
        :param text:
        :param max_text_length:
        :param temperature:
        :param top_k:
        :return:
        """
        ids = self.tokenizer.encode(text)
        batch_input_ids = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        output_ids = self.generate_from_tensor(batch_input_ids, max_text_length, temperature, top_k).squeeze(0)
        return self.tokenizer.decode(output_ids.tolist())

    def save(self, fpath: str):
        """
        保存模型
        :param fpath:
        :return:
        """
        torch.save(self.network.state_dict(), fpath)

    def load(self, fpath: str):
        """
        加载模型
        :param fpath:
        :return:
        """
        state_dict = torch.load(fpath, map_location=self.device)
        self.network.load_state_dict(state_dict)
