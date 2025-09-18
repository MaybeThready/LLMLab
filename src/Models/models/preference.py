# -*- coding: UTF-8 -*-
import os

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from ..models import InstructModel
from ..config import PreferenceTrainerConfig
from torch.utils.data import DataLoader
from copy import deepcopy
from rich import print
from rich.rule import Rule


class PreferenceModel:
    def __init__(self, instruct_model: InstructModel):
        self.model = instruct_model
        self.network = self.model.network
        self.device = self.model.device

    def generate_from_tensor(
            self,
            batch_input_ids: torch.Tensor,
            max_text_length: int,
            temperature: float = 0.,
            top_k: int = None,
    ) -> torch.Tensor:
        return self.model.generate_from_tensor(
            batch_input_ids,
            max_text_length,
            temperature,
            top_k
        )

    def generate_from_text(
            self,
            text: str,
            max_text_length: int,
            temperature: float = 0.,
            top_k: int = None,
    ) -> str:
        return self.model.generate_from_text(
            text,
            max_text_length,
            temperature,
            top_k
        )

    def chat(
            self,
            text: str,
            max_text_length: int,
            temperature: float = 0.,
            top_k: int = None,
            input_text: str = ""
    ) -> str:
        return self.model.chat(
            text,
            max_text_length,
            temperature,
            top_k,
            input_text
        )

    def save(self, fpath: str):
        self.model.save(fpath)

    def load(self, fpath: str):
        self.model.load(fpath)


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

        # DS修改代码
        # logits = logits - logits.max(dim=-1, keepdim=True).values

        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = torch.gather(
            input=log_probs,
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)
        if selection_mask is not None:
            mask = selection_mask[:, 1:].clone()
            selected_log_probs = selected_log_probs * mask
            avg_log_prob = selected_log_probs.sum(-1) / (mask.sum(-1) + 1e-8)
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

    def train(self, test_text="", max_output_tokens=50):
        print(Rule("Start Training"))
        total_steps = self.config.epochs * len(self.train_dataloader)
        tokens_seen, step = 0, -1
        log_list = os.listdir(self.config.log_dir)
        for name in log_list:
            fp = os.path.join(self.config.log_dir, name)
            if os.path.isfile(fp):
                os.remove(fp)

        ave_loss_list = []

        with SummaryWriter(self.config.log_dir) as writer:
            for epoch in range(self.config.epochs):
                for batch in self.train_dataloader:
                    self.policy_network.train()
                    self.optimizer.zero_grad()

                    loss = self.compute_dpo_loss_batch(
                        batch=batch,
                        policy_network=self.policy_network,
                        reference_network=self.reference_network,
                        beta=self.config.beta
                    )
                    ave_loss_list.append(loss.item())
                    loss.backward()

                    # DS修改代码
                    torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)

                    self.optimizer.step()
                    step += 1
                    tokens_seen += batch["chosen"].numel()

                    if step % self.config.eval_freq == 0:
                        val_loss = self.eval()
                        train_loss = sum(ave_loss_list) / len(ave_loss_list)
                        writer.add_scalars("Loss", {
                            "Train": train_loss,
                            "Val": val_loss
                        }, step)
                        ave_loss_list = []
                        print(f"Epoch {epoch+1} / {self.config.epochs}, Step {step} / {total_steps}, train_loss is {train_loss:.3f}, val_loss is {val_loss:.3f}")

                print(f"Epoch {epoch + 1} end, generate text:")
                print(self.model.generate_from_text(test_text, max_output_tokens))

            self.model.save(
                os.path.join(self.config.model_save_path, self.config.model_save_name + f"-epoch{epoch + 1}.pth")
            )
            print("Successfully save model!")

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


