# FedMADE

Implementation of **FedMADE** (arXiv:[2408.07152](https://arxiv.org/abs/2408.07152)) for federated IoT intrusion detection on **CICIoT2023**.

## Install

```bash
cd FedMADE
pip install -r requirements.txt
```

## Data

The raw dataset is CICIoT2023: [https://www.unb.ca/cic/datasets/iotdataset-2023.html](https://www.unb.ca/cic/datasets/iotdataset-2023.html).

The `.pkl` files are preprocessed from the raw CICIoT2023 CSV shards: the shards are aggregated into `ciciot2023_features.pkl` (features) and `ciciot2023_labels.pkl` (labels). 

Download the preprocessed `.pkl` files from Google Drive and place them under `./data`:  
[https://drive.google.com/drive/folders/1CxGHx28n1SuPE6GFLfBNiM4TkBVd5xLW?usp=drive_link](https://drive.google.com/drive/folders/1CxGHx28n1SuPE6GFLfBNiM4TkBVd5xLW?usp=drive_link)

Then place them under `./data` (or another directory and pass it with `--data-dir`):

```bash
mkdir -p ./data
# move/copy the downloaded .pkl files into ./data/
```

The main experiment loads these pickles via `--data-dir`.

### Required files (main experiment)

Prepare the following under `--data-dir` (default `./data`):

| File                      | Size (approx.) | Description                                                                                                                                                                                                                                           |
| ------------------------- | -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ciciot2023_features.pkl` | ~13GB          | Preprocessed CICIoT2023 **feature table** (pandas DataFrame). The code reads `X_columns` (46 traffic feature columns) as model input.                                                                                                                 |
| `ciciot2023_labels.pkl`   | ~175MB         | Attack labels aligned one-to-one with the feature table (strings such as `BenignTraffic`, `DDoS-SYN_Flood`). Mapped at load time: `dict_34_classes` → `dict_7_classes` (7 classes: Benign / DDoS+DoS / Mirai / Recon / Spoofing / Web / Brute Force). |

## Run

```bash
# FedMADE (paper method)
python scripts/run_experiment.py --method fedmade --model cnn --frac 1.0 \
  --data-dir ./data

# Baseline
python scripts/run_experiment.py --method fedavg --model cnn --frac 1.0 \
  --data-dir ./data
```

Common flags:

| Flag | Default | Meaning |
|---|---|---|
| `--method` | `fedmade` | `fedmade` / `fedavg` |
| `--model` | `cnn` | `cnn` or `fcnn` |
| `--frac` | `1.0` | client sampling rate γ |
| `--epochs` | `30` | communication rounds |
| `--local-epoch` | `2` | local epochs per round |
| `--lr` | `5e-4` | Adam learning rate |
| `--train-size` | `0.1` | fraction of data used (latency-style subsample; use `0.8` for paper 80/20) |
| `--smote` | on | minority oversampling as in paper |
| `--output-dir` | `outputs` | checkpoints + metrics |

Checkpoints and per-round metrics are written under `outputs/`.

## Design notes

- **63 clients** partitioned by victim-device style lists (`devices_ID`), 7-class taxonomy (DoS merged with DDoS).
- FedMADE computes Class Probability Matrices on a **server auxiliary set** (10 samples/class), clusters with DBSCAN, then solves non-negative weights minimizing distance of aggregated CCPMs to the identity matrix.
- Best global model by validation accuracy is kept for deployment-style evaluation.

## Layout

```text
FedMADE/
  configs/default.yaml
  fedmade/
    labels.py
    data.py
    models.py
    client.py
    aggregation.py
    train.py
  scripts/run_experiment.py
  outputs/
```

## Citation

```
@article{sun2024fedmade,
  title={FedMADE: Robust Federated Learning for Intrusion Detection in IoT Networks Using a Dynamic Aggregation Method},
  author={Sun, Shihua and Sharma, Pragya and Nwodo, Kenechukwu and Stavrou, Angelos and Wang, Haining},
  journal={arXiv preprint arXiv:2408.07152},
  year={2024}
}
```
