# run_selected_embeddings_cnn_mhsa_5fold.py
# -*- coding: utf-8 -*-

import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, matthews_corrcoef,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)

# =========================
# User config: select features here
# =========================
FEATURE_DIR = "/home/wsu/iPro_SW/Features/embedding_features"

# 只跑你指定的几个：key 是你想要的名字 f1/f2/f3，value 是 FEATURE_DIR 下的 csv 文件名
FEATURES = {
    "f1": "hyenadna-tiny-1k.csv",
    "f2": "hyenadna-tiny-16k.csv",
    "f3": "hyenadna-large-1m.csv",
    "f4": "hyenadna-medium-160k.csv",
    "f5": "hyenadna-medium-450k.csv",
    "f6": "hyenadna-small-32k.csv",
    "f7": "nt-2.5b-ms.csv",
    "f8": "nt-500m-1000g.csv",
    "f9": "ntv2-50m-ms.csv",
    "f10": "ntv2-50m-ms-3kmer.csv",
    "f11": "ntv2-100m-ms.csv",
    "f12": "ntv2-250m-ms.csv",
    "f13": "ntv2-500m-ms.csv",
    "f14": "ntv3-8m-pre.csv",
    "f15": "ntv3-100m-post.csv",
    "f16": "ntv3-100m-pre.csv",
    "f17": "ntv3-650m-post.csv",
    "f18": "ntv3-650m-pre.csv",
    "f19": "evo-1.5-8k-base.csv",
    "f20": "evo-1-8k-base.csv",
    "f21": "evo-1-131k-base.csv",
    "f22": "evo2_1b_base.csv",
    "f23": "evo2_7b.csv",
    "f24": "evo2_7b_base.csv",
    "f25": "DNABERT-6.csv",
    "f26": "DNABERT-S.csv",
}

# =========================
# Output paths
# =========================
RESULT_DIR = "./Results"
DETAIL_DIR = os.path.join(RESULT_DIR, "5fold_details")      # 每个模型 5-fold 指标明细
LOSS_DIR = os.path.join(RESULT_DIR, "loss_curves")          # 每个模型训练损失曲线数据
CURVE_DIR = os.path.join(RESULT_DIR, "curve_data")          # ROC/PRC 曲线数据
RESULT_OUT = os.path.join(RESULT_DIR, "Results.csv")

THR_STAGE1 = 0.5
THR_STAGE2 = 0.5

# =========================
# Dataset fixed rule for y2
# =========================
N_TOTAL = 6764
N_PROM = 3382
N_STRONG = 1591
N_WEAK = 1791  # 1591 + 1791 = 3382


# =========================
# Reproducibility
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Dataset
# =========================
class XYDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class XOnlyDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.from_numpy(X).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


# =========================
# Modules: ConvFeatureExtractor + PositionalEncoding
# =========================
class ConvFeatureExtractor(nn.Module):
    """
    Simple 1D CNN feature extractor for embedding vectors.
    Input:  (N,1,D)
    Output: (N,C2,L')
    """
    def __init__(self, conv_channels=(64, 128), kernels=(7, 5), dropout=0.5, in_channels=1):
        super().__init__()
        c1, c2 = conv_channels
        k1, k2 = kernels
        self.conv1 = nn.Conv1d(in_channels, c1, kernel_size=k1, padding=k1 // 2)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=k2, padding=k2 // 2)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.drop(x)
        x = self.pool(self.act(self.conv2(x)))
        x = self.drop(x)
        return x


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for (N,L,D) batch_first.
    Automatically expands if L exceeds current max_len.
    """
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("pe", self._build_pe(max_len, d_model), persistent=False)

    @staticmethod
    def _build_pe(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def _ensure_len(self, L: int):
        if L <= self.pe.size(1):
            return
        new_len = max(L, int(self.pe.size(1) * 1.5))
        self.pe = self._build_pe(new_len, self.d_model).to(self.pe.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,L,D)
        L = x.size(1)
        self._ensure_len(L)
        return x + self.pe[:, :L, :]


# =========================
# Model: CNN + MHSA (your framework)
# =========================
class CNNMHSA(nn.Module):
    """
    CNN -> (N,L',C) -> linear proj -> +pos -> MHSA -> FFN -> mean pool -> FC(s) -> out
    """
    def __init__(
        self,
        conv_channels=(16, 32),
        kernels=(7, 5),
        d_model=128,
        num_heads=4,
        attn_layers=1,
        ff_dim=256,
        fc_dims=(128, 64),
        dropout=0.5
    ):
        super().__init__()
        self.feat = ConvFeatureExtractor(conv_channels, kernels, dropout, in_channels=1)
        c2 = conv_channels[1]

        self.proj = nn.Linear(c2, d_model)
        # max_len 先给 1024，若 L' 更长会自动扩展
        self.pos = PositionalEncoding(d_model=d_model, max_len=1024)

        enc_layers = []
        for _ in range(attn_layers):
            enc_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=ff_dim,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True
                )
            )
        self.encoder = nn.Sequential(*enc_layers)

        layers: List[nn.Module] = []
        in_dim = d_model
        for d in fc_dims:
            layers += [nn.Linear(in_dim, d), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = d
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.out = nn.Linear(in_dim, 1)

    def forward(self, x):
        # x: (N,1,D)
        x = self.feat(x)               # (N,C,L')
        x = x.transpose(1, 2)          # (N,L',C)
        x = self.proj(x)               # (N,L',d_model)
        x = self.pos(x)                # + positional encoding
        x = self.encoder(x)            # (N,L',d_model)
        x = x.mean(dim=1)              # (N,d_model)
        x = self.mlp(x)
        logit = self.out(x).squeeze(-1)
        return logit


# =========================
# Metrics
# =========================
def compute_metrics_binary(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)
    y_score = y_score.astype(np.float64)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sn = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    sp = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = np.nan

    auc = np.nan
    auprc = np.nan
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)

    return {
        "Sn": float(sn) if not np.isnan(sn) else np.nan,
        "Sp": float(sp) if not np.isnan(sp) else np.nan,
        "Acc": float(acc),
        "MCC": float(mcc) if not np.isnan(mcc) else np.nan,
        "AUC": float(auc) if not np.isnan(auc) else np.nan,
        "AUPRC": float(auprc) if not np.isnan(auprc) else np.nan,
        "F1": float(f1),
    }


# =========================
# Train utils
# =========================
@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.5
    patience: int = 5
    grad_clip: float = 1.0
    num_workers: int = 0


def build_pos_weight(y: np.ndarray) -> Optional[torch.Tensor]:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos <= 0:
        return None
    return torch.tensor([neg / pos], dtype=torch.float32)


def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # X: (N,1,D)
    x = X[:, 0, :]
    mu = x.mean(axis=0, keepdims=True)
    sigma = x.std(axis=0, keepdims=True) + 1e-8
    mu = mu.reshape(1, 1, -1).astype(np.float32)
    sigma = sigma.reshape(1, 1, -1).astype(np.float32)
    return mu, sigma


def standardize_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu) / sigma


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns y_true, y_prob, avg_loss
    """
    model.eval()
    ys, ps = [], []
    losses = []
    criterion = nn.BCEWithLogitsLoss()
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        logits = model(X)
        loss = criterion(logits, y).detach().cpu().item()
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        ys.append(y.detach().cpu().numpy())
        ps.append(prob)
        losses.append(loss)
    y_true = np.concatenate(ys, axis=0)
    y_prob = np.concatenate(ps, axis=0)
    avg_loss = float(np.mean(losses)) if losses else np.nan
    return y_true, y_prob, avg_loss


@torch.no_grad()
def infer_probs_on_X(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    ds = XOnlyDataset(X)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    out = []
    for Xb in dl:
        Xb = Xb.to(device)
        prob = torch.sigmoid(model(Xb)).detach().cpu().numpy()
        out.append(prob)
    return np.concatenate(out, axis=0)


def train_one_fold(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    cfg: TrainConfig,
    model_kwargs: dict,
    device: torch.device,
    thr: float = 0.5
) -> Tuple[nn.Module, Dict[str, float], Dict[str, List[float]], np.ndarray, np.ndarray]:
    """
    Returns:
      model(best),
      final_metrics(on val),
      history: {"train_loss": [...], "val_loss": [...]},
      y_val_true (np),
      y_val_prob (np)
    """
    train_ds = XYDataset(X_train, y_train)
    val_ds = XYDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = CNNMHSA(dropout=cfg.dropout, **model_kwargs).to(device)

    pos_weight = build_pos_weight(y_train)
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_score = -1.0
    best_state = None
    bad = 0

    history = {"train_loss": [], "val_loss": []}

    for _epoch in range(cfg.epochs):
        model.train()
        train_losses = []
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            train_losses.append(loss.detach().cpu().item())

        # val
        y_true, y_prob, val_loss = predict_probs(model, val_loader, device)
        y_pred = (y_prob >= thr).astype(np.int32)
        metrics = compute_metrics_binary(y_true, y_prob, y_pred)

        history["train_loss"].append(float(np.mean(train_losses)) if train_losses else np.nan)
        history["val_loss"].append(val_loss)

        score = metrics["AUC"] if not np.isnan(metrics["AUC"]) else metrics["Acc"]
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final val outputs
    y_true, y_prob, _val_loss = predict_probs(model, val_loader, device)
    y_pred = (y_prob >= thr).astype(np.int32)
    final_metrics = compute_metrics_binary(y_true, y_prob, y_pred)

    return model, final_metrics, history, y_true, y_prob


# =========================
# Load one feature CSV and build (X, y1, y2)
# =========================
def load_feature_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    df = pd.read_csv(csv_path)

    if "y1" not in df.columns:
        raise ValueError(f"{csv_path} 缺少 y1 列")
    if len(df) != N_TOTAL:
        raise ValueError(f"{csv_path} 行数不为 {N_TOTAL}，实际为 {len(df)}")

    y1 = pd.to_numeric(df["y1"], errors="coerce").fillna(0).astype(int).values

    feat_cols = [c for c in df.columns if c not in ("y1", "y2")]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)  # (N,D)
    X = X[:, None, :]  # (N,1,D)

    # build y2 (promoters only)
    y2 = np.full((len(df),), -1, dtype=np.int32)
    prom_idx = np.where(y1 == 1)[0]
    if len(prom_idx) < N_PROM:
        raise ValueError(f"{csv_path} 启动子样本(y1=1)不足 {N_PROM}，实际 {len(prom_idx)}")

    # 按 y1==1 的出现顺序赋值
    y2[prom_idx[:N_STRONG]] = 1
    y2[prom_idx[N_STRONG:N_PROM]] = 0

    feat_dim = X.shape[2]
    return X, y1.astype(np.int32), y2.astype(np.int32), feat_dim


# =========================
# Save curve data
# =========================
def save_roc_prc(feature_id: str, stage: str, fold: int, y_true: np.ndarray, y_prob: np.ndarray):
    # ROC
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    df_roc = pd.DataFrame({"fold": fold, "fpr": fpr, "tpr": tpr, "threshold": thr})
    roc_path = os.path.join(CURVE_DIR, f"{feature_id}_{stage}_ROC.csv")
    df_roc.to_csv(roc_path, mode="a", header=not os.path.exists(roc_path), index=False, encoding="utf-8-sig")

    # PRC
    precision, recall, thr2 = precision_recall_curve(y_true, y_prob)
    # precision/recall length = len(thr2)+1，最后一个点没有 threshold
    thr_full = np.concatenate([thr2, [np.nan]])
    df_prc = pd.DataFrame({"fold": fold, "precision": precision, "recall": recall, "threshold": thr_full})
    prc_path = os.path.join(CURVE_DIR, f"{feature_id}_{stage}_PRC.csv")
    df_prc.to_csv(prc_path, mode="a", header=not os.path.exists(prc_path), index=False, encoding="utf-8-sig")


def save_loss_history(feature_id: str, stage: str, fold: int, history: Dict[str, List[float]]):
    rows = []
    for i, (tr, va) in enumerate(zip(history["train_loss"], history["val_loss"]), start=1):
        rows.append({"fold": fold, "epoch": i, "train_loss": tr, "val_loss": va})
    df = pd.DataFrame(rows)
    out_path = os.path.join(LOSS_DIR, f"{feature_id}_{stage}_loss.csv")
    df.to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False, encoding="utf-8-sig")


# =========================
# Evaluate one feature file with 5-fold two-stage
# =========================
def eval_one_feature_file(
    feature_id: str,
    csv_path: str,
    cfg: TrainConfig,
    model_kwargs: dict,
    device: torch.device
) -> Tuple[Dict[str, object], pd.DataFrame]:
    X_all, y1_all, y2_all, feat_dim = load_feature_csv(csv_path)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    metric_cols = ["Sn", "Sp", "Acc", "MCC", "AUC", "AUPRC", "F1"]
    details_rows: List[Dict[str, object]] = []

    stage1_fold_metrics = []
    stage2_fold_metrics = []

    # 清理旧的曲线/损失文件（避免重复追加）
    for suffix in ["Stage1_ROC.csv", "Stage1_PRC.csv", "Stage2_ROC.csv", "Stage2_PRC.csv"]:
        p = os.path.join(CURVE_DIR, f"{feature_id}_{suffix}")
        if os.path.exists(p):
            os.remove(p)
    for suffix in ["Stage1_loss.csv", "Stage2_loss.csv"]:
        p = os.path.join(LOSS_DIR, f"{feature_id}_{suffix}")
        if os.path.exists(p):
            os.remove(p)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y1_all), start=1):
        X_tr, X_va = X_all[tr_idx], X_all[va_idx]
        y1_tr, y1_va = y1_all[tr_idx], y1_all[va_idx]
        y2_tr, y2_va = y2_all[tr_idx], y2_all[va_idx]

        # ---------- Stage-1 standardize ----------
        mu1, sd1 = standardize_fit(X_tr)
        X_tr1 = standardize_apply(X_tr, mu1, sd1)
        X_va1 = standardize_apply(X_va, mu1, sd1)

        # ---------- Stage-1 ----------
        stage1_model, stage1_metrics, hist1, y1_true_va, y1_prob_va = train_one_fold(
            X_tr1, y1_tr.astype(np.float32),
            X_va1, y1_va.astype(np.float32),
            cfg, model_kwargs, device, thr=THR_STAGE1
        )
        stage1_fold_metrics.append(stage1_metrics)

        # 保存 Stage1 loss 和 ROC/PRC 数据
        save_loss_history(feature_id, "Stage1", fold, hist1)
        if len(np.unique(y1_true_va.astype(int))) == 2:
            save_roc_prc(feature_id, "Stage1", fold, y1_true_va.astype(int), y1_prob_va)

        # ---------- Stage-2 promoters only ----------
        prom_tr_mask = (y1_tr == 1) & (y2_tr >= 0)
        prom_va_mask = (y1_va == 1) & (y2_va >= 0)

        X2_tr = X_tr[prom_tr_mask]
        y2_tr_bin = y2_tr[prom_tr_mask].astype(np.float32)

        X2_va = X_va[prom_va_mask]
        y2_va_bin = y2_va[prom_va_mask].astype(np.int32)

        if len(X2_tr) == 0 or len(X2_va) == 0:
            raise RuntimeError(f"[{feature_id}] Fold {fold}: Stage2 promoters set empty, please check data.")

        # Stage2 standardize on promoter-train
        mu2, sd2 = standardize_fit(X2_tr)
        X2_tr2 = standardize_apply(X2_tr, mu2, sd2)
        X2_va2 = standardize_apply(X2_va, mu2, sd2)

        stage2_model, stage2_metrics, hist2, y2_true_va, y2_prob_va = train_one_fold(
            X2_tr2, y2_tr_bin,
            X2_va2, y2_va_bin.astype(np.float32),
            cfg, model_kwargs, device, thr=THR_STAGE2
        )
        stage2_fold_metrics.append(stage2_metrics)

        # 保存 Stage2 loss 和 ROC/PRC 数据
        save_loss_history(feature_id, "Stage2", fold, hist2)
        if len(np.unique(y2_true_va.astype(int))) == 2:
            save_roc_prc(feature_id, "Stage2", fold, y2_true_va.astype(int), y2_prob_va)

        # fold detail
        detail = {"feature": feature_id, "fold": fold}
        detail.update({f"Stage1_{k}": stage1_metrics[k] for k in metric_cols})
        detail.update({f"Stage2_{k}": stage2_metrics[k] for k in metric_cols})
        details_rows.append(detail)

        print(
            f"[{feature_id} | Fold {fold}] "
            f"Stage1 Acc={stage1_metrics['Acc']:.4f} AUC={stage1_metrics['AUC']:.4f} | "
            f"Stage2 Acc={stage2_metrics['Acc']:.4f} AUC={stage2_metrics['AUC']:.4f}"
        )

    # avg
    def avg_metrics(ms: List[Dict[str, float]]) -> Dict[str, float]:
        out = {}
        for c in metric_cols:
            out[c] = float(np.nanmean([m[c] for m in ms]))
        return out

    s1_avg = avg_metrics(stage1_fold_metrics)
    s2_avg = avg_metrics(stage2_fold_metrics)

    df_details = pd.DataFrame(details_rows)
    avg_row = {"feature": feature_id, "fold": "avg"}
    avg_row.update({f"Stage1_{k}": s1_avg[k] for k in metric_cols})
    avg_row.update({f"Stage2_{k}": s2_avg[k] for k in metric_cols})
    df_details = pd.concat([df_details, pd.DataFrame([avg_row])], ignore_index=True)

    summary_row = {
        "feature": feature_id,
        "csv_path": csv_path,
        "feat_dim": int(feat_dim),
        **{f"Stage1_{k}": v for k, v in s1_avg.items()},
        **{f"Stage2_{k}": v for k, v in s2_avg.items()},
    }
    return summary_row, df_details


# =========================
# Main
# =========================
def main():
    set_seed(42)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(DETAIL_DIR, exist_ok=True)
    os.makedirs(LOSS_DIR, exist_ok=True)
    os.makedirs(CURVE_DIR, exist_ok=True)

    if not FEATURES:
        raise ValueError(
            "你还没有在 FEATURES={} 里指定要跑的特征文件。"
            "请把 f1/f2/f3 和对应 csv 文件名填进去。"
        )

    # check files
    selected = []
    for fid, fn in FEATURES.items():
        path = os.path.join(FEATURE_DIR, fn)
        if not os.path.exists(path):
            raise FileNotFoundError(f"[{fid}] not found: {path}")
        selected.append((fid, path))

    cfg = TrainConfig(
        epochs=30,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-4,
        dropout=0.5,
        patience=5,
        grad_clip=1.0,
        num_workers=0,
    )

    # CNN+MHSA hyperparams（可按需改）
    model_kwargs = dict(
        conv_channels=(64, 128),
        kernels=(7, 5),
        d_model=128,
        num_heads=4,
        attn_layers=1,
        ff_dim=256,
        fc_dims=(128, 64),
    )

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Selected features:")
    for fid, p in selected:
        print(f"  - {fid}: {p}")

    results = []
    for fid, csv_path in selected:
        print(f"\n===== Evaluating {fid}: {csv_path} =====")
        summary_row, df_details = eval_one_feature_file(fid, csv_path, cfg, model_kwargs, device)
        results.append(summary_row)

        # save per-feature fold details
        detail_out = os.path.join(DETAIL_DIR, f"{fid}.csv")
        df_details.to_csv(detail_out, index=False, encoding="utf-8-sig")
        print(f"[DETAIL] 5-fold metrics saved: {detail_out}")

        # loss + curve files saved automatically under LOSS_DIR / CURVE_DIR
        print(f"[LOSS]   {LOSS_DIR}/{fid}_Stage1_loss.csv , {LOSS_DIR}/{fid}_Stage2_loss.csv")
        print(f"[CURVE]  {CURVE_DIR}/{fid}_Stage1_ROC.csv  , {CURVE_DIR}/{fid}_Stage1_PRC.csv")
        print(f"[CURVE]  {CURVE_DIR}/{fid}_Stage2_ROC.csv  , {CURVE_DIR}/{fid}_Stage2_PRC.csv")

    # summary
    df_out = pd.DataFrame(results)
    df_out.to_csv(RESULT_OUT, index=False, encoding="utf-8-sig")
    print(f"\n[FINISH] Saved summary to: {RESULT_OUT}")
    print(df_out.sort_values(by="Stage1_AUC", ascending=False))


if __name__ == "__main__":
    main()