"""Local client updates for FedAvg-style training."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class LocalClient:
    """Single-process FL client holding a local model copy and dataloaders."""

    def __init__(
        self,
        idx: int,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        lr: float = 5e-4,
    ):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.local_model = model
        self.criterion = nn.CrossEntropyLoss()
        self.num_samples = len(train_loader.dataset)

    def update_local_model(self, global_state: Dict[str, torch.Tensor]) -> None:
        self.local_model.load_state_dict(global_state)

    def local_training(self, local_epoch: int) -> Tuple[Dict[str, torch.Tensor], float, float]:
        model = self.local_model
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        losses = []

        for _ in range(max(local_epoch, 1)):
            for images, labels in self.train_loader:
                if len(labels) == 0:
                    continue
                images = images.to(self.device)
                labels = labels.to(self.device)
                model.zero_grad()
                output = model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        if not losses:
            losses = [0.0, 0.0]
        return model.state_dict(), losses[0], losses[-1]

    @torch.no_grad()
    def compute_cpm(self, aux_loader: DataLoader, num_classes: int) -> torch.Tensor:
        """Class Probability Matrix on server auxiliary data (paper Eq. 1)."""
        model = self.local_model
        model.eval()
        sums = [[] for _ in range(num_classes)]
        for images, labels in aux_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            probs = F.softmax(model(images), dim=1)
            for c in range(num_classes):
                idx = torch.where(labels == c)[0]
                if len(idx) > 0:
                    sums[c].append(probs[idx])

        rows = []
        for c in range(num_classes):
            if sums[c]:
                rows.append(torch.cat(sums[c], dim=0).mean(dim=0))
            else:
                rows.append(torch.zeros(num_classes, device=self.device))
        return torch.stack(rows, dim=0)  # [C, C]
