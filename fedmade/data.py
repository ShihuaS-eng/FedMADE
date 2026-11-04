"""Load CICIoT2023 pickles, preprocess, partition to 63 clients, build aux set."""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

from fedmade.labels import X_columns, devices_ID, dict_34_classes, dict_7_classes


class IndexSubset(Dataset):
    """Index into a TensorDataset without remapping labels."""

    def __init__(self, dataset: TensorDataset, indices: List[int]):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        x, y = self.dataset[self.indices[item]]
        return x.to(dtype=torch.float32), y.long()


def _require_pickles(data_dir: str, features_pkl: str, labels_pkl: str) -> Tuple[str, str]:
    feat_path = os.path.join(data_dir, features_pkl)
    label_path = os.path.join(data_dir, labels_pkl)
    missing = [p for p in (feat_path, label_path) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required pickle(s):\n  "
            + "\n  ".join(missing)
            + "\nSee README.md Data section."
        )
    return feat_path, label_path


def load_raw_arrays(
    data_dir: str,
    features_pkl: str = "ciciot2023_features.pkl",
    labels_pkl: str = "ciciot2023_labels.pkl",
) -> Tuple[np.ndarray, np.ndarray]:
    feat_path, label_path = _require_pickles(data_dir, features_pkl, labels_pkl)
    df = pd.read_pickle(feat_path)
    y_df = pd.read_pickle(label_path)
    features = df[X_columns].to_numpy()
    labels = y_df.map(dict_34_classes).map(dict_7_classes).to_numpy().astype(np.int64)
    if len(features) != len(labels):
        raise ValueError(
            f"Feature/label length mismatch: {len(features)} vs {len(labels)}"
        )
    return features, labels


def apply_smote(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Paper-style SMOTE: web/brute x4, other minorities x2; skip DDoS/DoS/Mirai."""
    counts = {c: int(np.sum(labels == c)) for c in range(7)}
    strategy = {}
    for c in (0, 3, 4):  # Benign, Recon, Spoofing
        if counts[c] > 0:
            strategy[c] = counts[c] * 2
    for c in (5, 6):  # Web, Brute
        if counts[c] > 0:
            strategy[c] = counts[c] * 4
    if not strategy:
        return features, labels
    smote = SMOTE(sampling_strategy=strategy, random_state=42)
    return smote.fit_resample(features, labels)


def partition_ciciot_real(
    dataset: TensorDataset,
    num_users: int = 63,
    num_classes: int = 7,
    seed: int = 10,
) -> Dict[int, List[int]]:
    """Assign samples to clients by victim-device lists (7-class)."""
    rng = random.Random(seed)
    class_indices: Dict[int, List[int]] = {i: [] for i in range(num_classes)}
    labels = dataset.tensors[1]
    for c in range(num_classes):
        idx = torch.where(labels == c)[0].tolist()
        rng.shuffle(idx)
        class_indices[c] = idx

    class_per_user = {}
    for c in range(num_classes):
        n_victims = len(devices_ID[c])
        class_per_user[c] = int(len(class_indices[c]) / max(n_victims, 1))

    dict_users: Dict[int, List[int]] = {i: [] for i in range(num_users)}
    track = {c: 0 for c in range(num_classes)}
    for user in range(num_users):
        for c in range(num_classes):
            if user in devices_ID[c]:
                t = track[c]
                start = t * class_per_user[c]
                end = (t + 1) * class_per_user[c]
                dict_users[user].extend(class_indices[c][start:end])
                track[c] += 1
    return dict_users


def build_aux_loader(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 7,
    per_class: int = 10,
    batch_size: int = 64,
    seed: int = 42,
) -> DataLoader:
    """Server auxiliary set: `per_class` samples from each class (paper: 10)."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        take = min(per_class, len(idx))
        chosen = rng.choice(idx, size=take, replace=False)
        xs.append(features[chosen])
        ys.append(labels[chosen])
    x = torch.tensor(np.concatenate(xs, axis=0), dtype=torch.float32)
    y = torch.tensor(np.concatenate(ys, axis=0), dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def prepare_federated_data(
    data_dir: str,
    features_pkl: str = "ciciot2023_features.pkl",
    labels_pkl: str = "ciciot2023_labels.pkl",
    train_size: float = 0.1,
    num_users: int = 63,
    num_classes: int = 7,
    local_bs: int = 128,
    smote: bool = True,
    aux_per_class: int = 10,
    seed: int = 42,
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader, DataLoader]:
    """
    Returns:
      train_loaders, test_loaders (per client), global_test_loader, aux_loader
    """
    features, labels = load_raw_arrays(data_dir, features_pkl, labels_pkl)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=(1.0 - train_size),
        random_state=seed,
        stratify=labels,
    )

    if smote:
        print("Applying SMOTE on training split...")
        x_train, y_train = apply_smote(x_train, y_train)

    # Hold out a small slice of the leftover pool for a lighter global test if train_size is small
    # Otherwise use full held-out test. For train_size=0.1, further subsample test for speed.
    if train_size < 0.5:
        _, x_test, _, y_test = train_test_split(
            x_test, y_test, test_size=0.1, random_state=seed, stratify=y_test
        )

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    user_train = partition_ciciot_real(train_ds, num_users, num_classes, seed=10)
    user_test = partition_ciciot_real(test_ds, num_users, num_classes, seed=10)

    train_loaders, test_loaders = [], []
    for uid in range(num_users):
        train_loaders.append(
            DataLoader(
                IndexSubset(train_ds, user_train[uid]),
                batch_size=local_bs,
                shuffle=True,
            )
        )
        test_loaders.append(
            DataLoader(
                IndexSubset(test_ds, user_test[uid]),
                batch_size=local_bs,
                shuffle=False,
            )
        )

    global_test_loader = DataLoader(test_ds, batch_size=local_bs, shuffle=False)
    aux_loader = build_aux_loader(
        x_train, np.asarray(y_train), num_classes, aux_per_class, local_bs, seed
    )
    print(
        f"Clients={num_users}, train={len(train_ds)}, test={len(test_ds)}, "
        f"aux={len(aux_loader.dataset)}"
    )
    return train_loaders, test_loaders, global_test_loader, aux_loader
