"""Preprocessing helpers for the fraud-detection datasets.

Works with both the bundled synthetic fraud data and (if you download the
Kaggle 'creditcardfraud' CSV yourself) the real ULB dataset. The real file
has columns Time, V1..V28, Amount, Class.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score


def load_fraud(path: str | None = None) -> pd.DataFrame:
    """Load fraud data. If path is None, generate the synthetic version."""
    if path:
        return pd.read_csv(path)
    from academy.datasets.manager import load_dataset
    return load_dataset("synthetic_fraud", seed=42)


def preprocess(df: pd.DataFrame, test_size: float = 0.2,
               random_state: int = 42):
    """Scale features; stratified split; returns X_train/X_test/y_train/y_test/scaler."""
    feature_cols = [c for c in df.columns if c not in ("Class", "Time")]
    X = df[feature_cols].values
    y = df["Class"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    scaler = StandardScaler().fit(X_train)  # fit on train only (no leakage)
    return (scaler.transform(X_train), scaler.transform(X_test),
            y_train, y_test, scaler)


def fraud_metrics(y_true, y_prob) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "fraud_rate": float(np.mean(y_true)),
    }


if __name__ == "__main__":
    df = load_fraud()
    Xtr, Xte, ytr, yte, _ = preprocess(df)
    print(f"train={Xtr.shape} test={Xte.shape} fraud_rate={ytr.mean():.4f}")
