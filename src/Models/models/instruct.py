# -*- coding: UTF-8 -*-

from ..models.pretrained import PretrainedModel
from ..networks.lora import replace_linear_with_lora
from ..config import InstructTrainerConfig
from ..utils import create_prompt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from rich import print
from rich.rule import Rule


class InstructModel:
    def __init__(self, pretrained_model: PretrainedModel, rank: int, alpha: float, template: str="alpaca", module_names=None):
        self.model = pretrained_model
        self.network = self.model.network
        self.device = self.model.device
        self.template = template
        for param in self.network.parameters():
            param.requires_grad = False
        replace_linear_with_lora(self.network, rank=rank, alpha=alpha, device=self.device, module_names=module_names)

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
        text = create_prompt(text, template=self.template)
        return self.model.generate_from_text(
            text,
            max_text_length,
            temperature,
            top_k
        )

    def save(self, fpath: str):
        self.model.save(fpath)

    def load(self, fpath: str):
        self.model.load(fpath)


class InstructTrainer:
    def __init__(self, model: InstructModel, config: InstructTrainerConfig, train_loader: DataLoader, val_loader: DataLoader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.AdamW(
            self.model.network.parameters(),
            lr=config.max_learning_rate,
        )
        total_steps = self.config.epochs * len(self.train_loader)
        warmup_steps = int(self.config.warmup_rate * total_steps)
        cosine_steps = total_steps - warmup_steps
        # print(total_steps, warmup_steps, cosine_steps)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=self.config.min_learning_rate / self.config.max_learning_rate,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=cosine_steps,
            eta_min=self.config.min_learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

    @staticmethod
    def loss_from_logits(logits_batch: torch.Tensor, target_batch: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(
            logits_batch.flatten(0, 1),
            target_batch.flatten()
        )

    def loss_from_batch(self, input_batch: torch.Tensor, target_batch: torch.Tensor) -> torch.Tensor:
        input_batch = input_batch.to(self.model.device)
        target_batch = target_batch.to(self.model.device)
        logits_batch = self.model.network(input_batch)
        return self.loss_from_logits(logits_batch, target_batch)

    def eval(self):
        self.model.network.eval()
        total_loss = 0.
        for input_batch, target_batch in self.val_loader:
            total_loss += self.loss_from_batch(input_batch, target_batch).item()
        return total_loss / len(self.val_loader)

    def train(self, test_text: str="", max_output_tokens: int=50):
        print(Rule("Start Training"))
        step = -1

        with SummaryWriter(self.config.log_dir) as writer:
            for epoch in range(self.config.epochs):
                for input_batch, target_batch in self.train_loader:
                    self.model.network.train()
                    self.optimizer.zero_grad()
                    step += 1
                    loss = self.loss_from_batch(input_batch, target_batch)
                    writer.add_scalar("Loss/Train", loss.item(), step)
                    loss.backward()
                    self.optimizer.step()

                    writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], step)
                    self.scheduler.step()

                    if step % self.config.eval_freq == 0:
                        loss = self.eval()
                        writer.add_scalar("Loss/Val", loss, step)
                        print(f"Epoch {epoch+1}, Step {step}, val_loss is {loss:.3f}")

                print(f"Epoch {epoch+1} end, generate text:")
                print(self.model.generate_from_text(test_text, max_output_tokens))

        self.model.save(self.config.model_save_fpath)


