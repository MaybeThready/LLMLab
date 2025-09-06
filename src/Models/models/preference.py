# -*- coding: UTF-8 -*-

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from ..models.tokenizer import Tokenizer
from ..config import PreferenceTrainerConfig
from torch.utils.data import DataLoader
from copy import deepcopy
from rich import print
from rich.rule import Rule


class PreferenceModel:
    def __init__(self, network, context_length: int, tokenizer: Tokenizer, device: torch.device=torch.device("cpu")):
        self.context_length = context_length
        self.network = network.to(device)
        self.tokenizer = tokenizer
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
            batch_ids = batch_input_ids[:, -self.context_length:]
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


class PreferenceTrainer:
    def __init__(self, model: PreferenceModel, config: PreferenceTrainerConfig, train_dataloader: DataLoader, val_dataloader: DataLoader):
        self.model = model
        self.policy_network = model.network
        self.reference_network = deepcopy(model.network)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.policy_network.train()
        self.reference_network.eval()

        self.config = config
        self.optimizer = torch.optim.AdamW(self.policy_network.parameters(), lr=config.lr)

    @staticmethod
    def compute_dpo_loss(
            policy_chosen_logprobs,
            policy_rejected_logprobs,
            reference_chosen_logprobs,
            reference_rejected_logprobs,
            beta=0.1
    ):
        """
        beta -> [0.1, 0.5], 越大dpo影响越小
        """
        policy_logratios = policy_chosen_logprobs - policy_rejected_logprobs
        reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
        logits = policy_logratios - reference_logratios

        losses = -F.logsigmoid(beta * logits)
        return losses.mean()

    @staticmethod
    def compute_logprobs(logits, labels, selection_mask=None):
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = torch.gather(
            input=log_probs,
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)
        if selection_mask is not None:
            mask = selection_mask[:, 1:].clone()
            selected_log_probs = selected_log_probs * mask
            avg_log_prob = selected_log_probs.sum(-1) / mask.sum(-1)
            return avg_log_prob
        else:
            return selected_log_probs.mean(-1)

    def compute_dpo_loss_batch(self, batch, policy_network, reference_network, beta):
        policy_chosen_log_probas = self.compute_logprobs(
            logits=policy_network(batch["chosen"]),
            labels=batch["chosen"],
            selection_mask=batch["chosen_mask"]
        )
        policy_rejected_log_probas = self.compute_logprobs(
            logits=policy_network(batch["rejected"]),
            labels=batch["rejected"],
            selection_mask=batch["rejected_mask"]
        )

        with torch.no_grad():
            ref_chosen_log_probas = self.compute_logprobs(
                logits=reference_network(batch["chosen"]),
                labels=batch["chosen"],
                selection_mask=batch["chosen_mask"]
            )
            ref_rejected_log_probas = self.compute_logprobs(
                logits=reference_network(batch["rejected"]),
                labels=batch["rejected"],
                selection_mask=batch["rejected_mask"]
            )
        loss = self.compute_dpo_loss(
            policy_chosen_logprobs=policy_chosen_log_probas,
            policy_rejected_logprobs=policy_rejected_log_probas,
            reference_chosen_logprobs=ref_chosen_log_probas,
            reference_rejected_logprobs=ref_rejected_log_probas,
            beta=beta
        )
        return loss

    def train(self, test_context="", max_output_length=50):
        print(Rule("Start Training", characters='='))
        tokens_seen, step = 0, -1
        with SummaryWriter(self.config.log_dir) as writer:
            for epoch in range(self.config.epochs):
                self.policy_network.train()

                for batch in self.train_dataloader:
                    self.optimizer.zero_grad()

                    loss = self.compute_dpo_loss_batch(
                        batch=batch,
                        policy_network=self.policy_network,
                        reference_network=self.reference_network,
                        beta=self.config.beta
                    )
                    loss.backward()
                    self.optimizer.step()
                    step += 1
                    tokens_seen += batch["chosen"].numel()

                    writer.add_scalar("Loss/Train", loss.item(), step)
                    if step % self.config.eval_freq == 0:
                        loss = self.eval()
                        writer.add_scalar("Loss/Val", loss, step)
                        print(f"Epoch {epoch+1}, Step {step}, Val Loss is {loss:.3f}")

                self.model.generate_from_text(test_context, max_text_length=max_output_length)

        self.model.save(self.config.save_fpath)
        return tokens_seen

    def eval(self):
        self.policy_network.eval()
        total_loss = 0.
        for batch in self.val_dataloader:
            loss = self.compute_dpo_loss_batch(
                batch=batch,
                policy_network=self.policy_network,
                reference_network=self.reference_network,
                beta=self.config.beta
            )
            total_loss += loss.item()
        return total_loss / len(self.val_dataloader)


