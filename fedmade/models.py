"""Base models used in FedMADE experiments (CNN and FCNN)."""

import torch
import torch.nn as nn


class CNN(nn.Module):
    """Two Conv1d layers + dense classifier (paper CNN / CNN2_new)."""

    def __init__(self, input_dim: int = 46, num_classes: int = 7):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
        )
        self.classifier = nn.Linear(input_dim * 128, num_classes)
        self.classifier_weight_keys = ["classifier.weight", "classifier.bias"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, dim = x.shape
        x = x.view(batch_size, 1, dim)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class FCNN(nn.Module):
    """Three-layer MLP matching paper FCNN / DNN3 (50 → 25 → classes)."""

    def __init__(self, input_dim: int = 46, num_classes: int = 7):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, num_classes)
        self.relu = nn.ReLU()
        self.classifier_weight_keys = ["fc3.weight", "fc3.bias"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class FCNNDropout(nn.Module):
    """FCNN with dropout (DNN3_dropout variant used in some runs)."""

    def __init__(self, input_dim: int = 46, num_classes: int = 7):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 50)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(50, num_classes)
        self.relu = nn.ReLU()
        self.classifier_weight_keys = ["fc3.weight", "fc3.bias"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)


def build_model(name: str, input_dim: int = 46, num_classes: int = 7) -> nn.Module:
    name = name.lower()
    if name in ("cnn", "cnn2", "cnn2_new"):
        return CNN(input_dim=input_dim, num_classes=num_classes)
    if name in ("fcnn", "dnn", "dnn3"):
        return FCNN(input_dim=input_dim, num_classes=num_classes)
    if name in ("fcnn_dropout", "dnn3_dropout"):
        return FCNNDropout(input_dim=input_dim, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")
