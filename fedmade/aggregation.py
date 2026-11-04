"""Server aggregation: FedAvg (size-weighted) and FedMADE."""

from __future__ import annotations

import copy
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

from fedmade.client import LocalClient


def average_weights_weighted(
    states: List[Dict[str, torch.Tensor]],
    weights: Sequence[float],
) -> Dict[str, torch.Tensor]:
    """Weighted average of state dicts; weights are normalized to sum to 1."""
    if len(states) == 0:
        raise ValueError("No local states to aggregate")
    w = torch.tensor([float(x) for x in weights], dtype=torch.float32)
    if torch.all(w == 0):
        w = torch.ones_like(w)
    w = w / w.sum()
    agg = copy.deepcopy(states[0])
    for key in agg:
        agg[key] = torch.zeros_like(agg[key], dtype=torch.float32)
        for i, state in enumerate(states):
            agg[key] += w[i] * state[key].float()
        # preserve original dtype if possible
        agg[key] = agg[key].to(states[0][key].dtype)
    return agg


def optimize_cluster_coefficients(
    cluster_cpms: List[np.ndarray],
    num_classes: int,
    steps: int = 1000,
    lr: float = 0.01,
) -> List[float]:
    """Solve min ||sum alpha_k P_k - I||_2 with alpha_k >= 0 (clamp)."""
    matrices = [torch.tensor(p, dtype=torch.float32) for p in cluster_cpms]
    identity = torch.eye(num_classes)
    coeffs = [Variable(torch.tensor(0.0), requires_grad=True) for _ in matrices]
    optimizer = Adam(coeffs, lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        constrained = [torch.clamp(c, min=0.0) for c in coeffs]
        combo = sum(c * m for c, m in zip(constrained, matrices))
        loss = torch.nn.functional.mse_loss(combo, identity)
        loss.backward()
        optimizer.step()
    return [float(torch.clamp(c, min=0.0).item()) for c in coeffs]


def cluster_cpms(
    cpms: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 1,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    cpms: [N, C, C] -> flatten to [N, C*C], DBSCAN, return mean CPM per cluster + labels.
    """
    n_classes = cpms.shape[-1]
    flat = cpms.reshape((-1, n_classes * n_classes))
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(flat)
    # DBSCAN may mark noise as -1; treat each noise point as its own cluster index
    unique = sorted(set(labels.tolist()))
    remapped = np.zeros_like(labels)
    cluster_means = []
    new_id = 0
    for lab in unique:
        idx = np.where(labels == lab)[0]
        remapped[idx] = new_id
        cluster_means.append(np.mean(cpms[idx], axis=0))
        new_id += 1
    return cluster_means, remapped


def run_round_size_weighted(
    args,
    global_model: nn.Module,
    clients: List[LocalClient],
    round_idx: int,
) -> Tuple[float, float]:
    """FedAvg server round with sample-size weights."""
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    idx_users = sorted(np.random.choice(range(num_users), m, replace=False).tolist())
    print(f"\n---- Round {round_idx + 1} | clients={idx_users} ----")

    global_state = global_model.state_dict()
    local_states, weights, losses_start, losses_end = [], [], [], []
    for idx in idx_users:
        client = clients[idx]
        client.update_local_model(global_state)
        state, loss0, loss1 = client.local_training(args.local_epoch)
        local_states.append(copy.deepcopy(state))
        weights.append(float(client.num_samples))
        losses_start.append(loss0)
        losses_end.append(loss1)

    new_state = average_weights_weighted(local_states, weights)
    global_model.load_state_dict(new_state)
    return float(np.mean(losses_start)), float(np.mean(losses_end))


def run_round_fedmade(
    args,
    global_model: nn.Module,
    clients: List[LocalClient],
    aux_loader: DataLoader,
    round_idx: int,
) -> Tuple[float, float]:
    """FedMADE: local train → CPM on aux → DBSCAN → optimize α → weighted agg."""
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    idx_users = sorted(np.random.choice(range(num_users), m, replace=False).tolist())
    print(f"\n---- FedMADE Round {round_idx + 1} | clients={idx_users} ----")

    global_state = global_model.state_dict()
    local_states, losses_start, losses_end = [], [], []
    cpm_list = []

    for idx in idx_users:
        client = clients[idx]
        client.update_local_model(global_state)
        state, loss0, loss1 = client.local_training(args.local_epoch)
        local_states.append(copy.deepcopy(state))
        losses_start.append(loss0)
        losses_end.append(loss1)
        # Reload trained weights into client model for CPM (already there)
        cpm = client.compute_cpm(aux_loader, args.num_classes)
        cpm_np = cpm.detach().cpu().numpy()
        cpm_np = np.nan_to_num(cpm_np, nan=0.0)
        cpm_list.append(cpm_np)

    cpms = np.stack(cpm_list, axis=0)
    cluster_means, clabels = cluster_cpms(cpms)
    alphas = optimize_cluster_coefficients(cluster_means, args.num_classes)
    print(f"clusters={len(cluster_means)} alphas={alphas} labels={clabels.tolist()}")

    # beta_i = alpha_k / |C_k| for client i in cluster k; then normalize
    betas = np.zeros(len(idx_users), dtype=np.float64)
    for k, alpha in enumerate(alphas):
        members = np.where(clabels == k)[0]
        if len(members) == 0:
            continue
        share = alpha / len(members)
        for mi in members:
            betas[mi] = share
    if betas.sum() <= 0:
        betas = np.ones_like(betas)
    betas = betas / betas.sum()
    print(f"betas={betas.tolist()}")

    new_state = average_weights_weighted(local_states, betas.tolist())
    global_model.load_state_dict(new_state)
    return float(np.mean(losses_start)), float(np.mean(losses_end))
