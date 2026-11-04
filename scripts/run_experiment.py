#!/usr/bin/env python3
"""CLI entry for FedMADE / FedAvg experiments."""

from __future__ import annotations

import argparse
import os
import sys

# Limit BLAS/OpenMP threads before importing numpy/sklearn (many-core hosts).
# Shell exports still win via setdefault.
for _k, _v in (
    ("OPENBLAS_NUM_THREADS", "8"),
    ("OMP_NUM_THREADS", "8"),
    ("MKL_NUM_THREADS", "8"),
    ("NUMEXPR_NUM_THREADS", "8"),
):
    os.environ.setdefault(_k, _v)

import yaml

# Allow running without installing the package
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fedmade.train import run_experiment  # noqa: E402


def load_yaml(path: str) -> dict:
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run FedMADE / FedAvg on CICIoT2023")
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    p.add_argument("--data-dir", default=None)
    p.add_argument("--features-pkl", default=None)
    p.add_argument("--labels-pkl", default=None)
    p.add_argument("--method", choices=["fedmade", "fedavg"], default=None)
    p.add_argument("--model", choices=["cnn", "fcnn", "fcnn_dropout"], default=None)
    p.add_argument("--frac", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--local-epoch", type=int, default=None)
    p.add_argument("--local-bs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--train-size", type=float, default=None)
    p.add_argument("--num-users", type=int, default=None)
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--input-dim", type=int, default=None)
    p.add_argument("--aux-per-class", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--smote", dest="smote", action="store_true")
    p.add_argument("--no-smote", dest="smote", action="store_false")
    p.set_defaults(smote=None)
    return p


def main() -> None:
    parser = build_parser()
    cli = parser.parse_args()
    cfg = load_yaml(cli.config)

    def pick(cli_val, key, default=None):
        if cli_val is not None:
            return cli_val
        return cfg.get(key, default)

    args = argparse.Namespace(
        data_dir=pick(cli.data_dir, "data_dir"),
        features_pkl=pick(cli.features_pkl, "features_pkl", "ciciot2023_features.pkl"),
        labels_pkl=pick(cli.labels_pkl, "labels_pkl", "ciciot2023_labels.pkl"),
        method=pick(cli.method, "method", "fedmade"),
        model=pick(cli.model, "model", "cnn"),
        frac=pick(cli.frac, "frac", 1.0),
        epochs=pick(cli.epochs, "epochs", 30),
        local_epoch=pick(cli.local_epoch, "local_epoch", 2),
        local_bs=pick(cli.local_bs, "local_bs", 128),
        lr=pick(cli.lr, "lr", 5e-4),
        train_size=pick(cli.train_size, "train_size", 0.1),
        num_users=pick(cli.num_users, "num_users", 63),
        num_classes=pick(cli.num_classes, "num_classes", 7),
        input_dim=pick(cli.input_dim, "input_dim", 46),
        aux_per_class=pick(cli.aux_per_class, "aux_per_class", 10),
        seed=pick(cli.seed, "seed", 520),
        output_dir=pick(cli.output_dir, "output_dir", os.path.join(ROOT, "outputs")),
        smote=pick(cli.smote, "smote", True),
    )

    if not args.data_dir:
        raise SystemExit("--data-dir is required (see README Data section)")

    run_experiment(args)


if __name__ == "__main__":
    main()
