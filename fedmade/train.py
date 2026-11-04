"""Training loop, evaluation, and experiment runner."""

from __future__ import annotations

import copy
import json
import os
from argparse import Namespace
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fedmade.aggregation import run_round_fedmade, run_round_size_weighted
from fedmade.client import LocalClient
from fedmade.data import prepare_federated_data
from fedmade.labels import CLASS_NAMES_7
from fedmade.models import build_model


@torch.no_grad()
def evaluate_overall(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def evaluate_per_class(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Dict[int, Optional[float]]:
    model.eval()
    correct = {c: 0 for c in range(num_classes)}
    total = {c: 0 for c in range(num_classes)}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        for c in range(num_classes):
            mask = y == c
            n = int(mask.sum().item())
            if n == 0:
                continue
            total[c] += n
            correct[c] += int((pred[mask] == y[mask]).sum().item())
    out: Dict[int, Optional[float]] = {}
    for c in range(num_classes):
        out[c] = (100.0 * correct[c] / total[c]) if total[c] > 0 else None
    return out


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(args: Namespace) -> Dict:
    _set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(f"device={device} method={args.method} model={args.model}")

    os.makedirs(args.output_dir, exist_ok=True)
    run_name = f"{args.method}_{args.model}_frac{args.frac}_seed{args.seed}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    train_loaders, test_loaders, global_test, aux_loader = prepare_federated_data(
        data_dir=args.data_dir,
        features_pkl=args.features_pkl,
        labels_pkl=args.labels_pkl,
        train_size=args.train_size,
        num_users=args.num_users,
        num_classes=args.num_classes,
        local_bs=args.local_bs,
        smote=args.smote,
        aux_per_class=args.aux_per_class,
        seed=args.seed,
    )

    global_model = build_model(
        args.model, input_dim=args.input_dim, num_classes=args.num_classes
    ).to(device)

    clients: List[LocalClient] = []
    for i in range(args.num_users):
        clients.append(
            LocalClient(
                idx=i,
                model=copy.deepcopy(global_model),
                train_loader=train_loaders[i],
                test_loader=test_loaders[i],
                device=device,
                lr=args.lr,
            )
        )

    history = {
        "overall_acc": [],
        "per_class_acc": [],
        "loss_start": [],
        "loss_end": [],
        "best_round": -1,
        "best_overall_acc": -1.0,
    }
    best_state = copy.deepcopy(global_model.state_dict())

    for rnd in range(args.epochs):
        if args.method == "fedmade":
            loss0, loss1 = run_round_fedmade(
                args, global_model, clients, aux_loader, rnd
            )
        else:
            loss0, loss1 = run_round_size_weighted(args, global_model, clients, rnd)

        overall = evaluate_overall(global_model, global_test, device)
        per_class = evaluate_per_class(
            global_model, global_test, device, args.num_classes
        )
        history["overall_acc"].append(overall)
        history["per_class_acc"].append(per_class)
        history["loss_start"].append(loss0)
        history["loss_end"].append(loss1)

        named = {
            CLASS_NAMES_7[c] if c < len(CLASS_NAMES_7) else str(c): per_class[c]
            for c in range(args.num_classes)
        }
        print(f"Round {rnd + 1}: overall={overall:.2f}% per_class={named}")

        torch.save(global_model.state_dict(), os.path.join(run_dir, f"round_{rnd:03d}.pt"))
        if overall > history["best_overall_acc"]:
            history["best_overall_acc"] = overall
            history["best_round"] = rnd
            best_state = copy.deepcopy(global_model.state_dict())
            torch.save(best_state, os.path.join(run_dir, "best.pt"))

    global_model.load_state_dict(best_state)
    final_overall = evaluate_overall(global_model, global_test, device)
    final_per_class = evaluate_per_class(
        global_model, global_test, device, args.num_classes
    )
    history["final_overall_acc"] = final_overall
    history["final_per_class_acc"] = final_per_class
    history["final_per_class_named"] = {
        CLASS_NAMES_7[c] if c < len(CLASS_NAMES_7) else str(c): final_per_class[c]
        for c in range(args.num_classes)
    }

    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(
        f"Done. best_round={history['best_round'] + 1} "
        f"best_acc={history['best_overall_acc']:.2f}% "
        f"final_best_eval={final_overall:.2f}% -> {metrics_path}"
    )
    return history
