"""
train.py
--------
End-to-end training pipeline:

    raw log --> log_parser --> feature_engineering --> Isolation Forest --> joblib model

Usage:
    python src/train.py --log-file data/raw/access.log \
                         --model-out models/isolation_forest.joblib \
                         --contamination 0.02

Notes on `contamination`:
    This is an *assumption*, not a measured ground truth, since we have no
    attack labels. 0.02 (2%) is a conservative default meaning "assume
    roughly the most unusual 2% of requests are worth a human look" -- it
    controls the anomaly score threshold, not the accuracy of the model.
    Tune it based on how many alerts your team can realistically triage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import FEATURE_COLUMNS, build_features  # noqa: E402
from log_parser import parse_log_file  # noqa: E402


def train(
    log_file: str,
    model_out: str,
    contamination: float = 0.02,
    n_estimators: int = 200,
    random_state: int = 42,
    processed_out: str | None = None,
):
    print(f"[train] parsing {log_file} ...")
    t0 = time.time()
    df, stats = parse_log_file(log_file)
    print(f"[train] parsed {len(df):,} rows in {time.time() - t0:.1f}s ({stats.summary()})")

    print("[train] building features ...")
    t0 = time.time()
    features = build_features(df)
    print(f"[train] built {features.shape[1]} features for {len(features):,} rows in {time.time() - t0:.1f}s")

    if processed_out:
        os.makedirs(os.path.dirname(processed_out), exist_ok=True)
        features.to_csv(processed_out, index=False)
        print(f"[train] saved processed feature matrix to {processed_out}")

    scaler = StandardScaler()
    X = scaler.fit_transform(features[FEATURE_COLUMNS].to_numpy())

    print(f"[train] fitting IsolationForest (n_estimators={n_estimators}, contamination={contamination}) ...")
    t0 = time.time()
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)
    print(f"[train] fit complete in {time.time() - t0:.1f}s")

    # anomaly_score: higher = more anomalous (we flip sklearn's convention,
    # where decision_function is higher for normal points, so downstream
    # code and reports read more intuitively).
    raw_scores = model.decision_function(X)
    anomaly_scores = -raw_scores
    predictions = model.predict(X)  # -1 = anomaly, 1 = normal

    n_anomalies = int((predictions == -1).sum())
    print(f"[train] baseline flags {n_anomalies:,} / {len(df):,} requests as anomalous "
          f"({100 * n_anomalies / len(df):.2f}%)")

    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "contamination": contamination,
        "trained_on": log_file,
        "n_training_rows": len(df),
    }
    joblib.dump(bundle, model_out)
    print(f"[train] saved model bundle to {model_out}")

    metadata = {
        "trained_on": log_file,
        "n_rows": len(df),
        "n_features": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "contamination": contamination,
        "n_estimators": n_estimators,
        "n_anomalies_in_training_baseline": n_anomalies,
        "anomaly_rate_in_training_baseline": n_anomalies / len(df),
        "parse_stats": {
            "total_lines": stats.total_lines,
            "parsed_strict": stats.parsed_strict,
            "parsed_loose": stats.parsed_loose,
            "dropped": stats.dropped,
        },
    }
    meta_path = os.path.splitext(model_out)[0] + "_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[train] saved training metadata to {meta_path}")

    return model, scaler, features, anomaly_scores, predictions, df


def main():
    parser = argparse.ArgumentParser(description="Train the Overwatch Isolation Forest baseline.")
    parser.add_argument("--log-file", default="data/raw/access.log")
    parser.add_argument("--model-out", default="models/isolation_forest.joblib")
    parser.add_argument("--contamination", type=float, default=0.02)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--processed-out", default="data/processed/features.csv")
    args = parser.parse_args()

    train(
        log_file=args.log_file,
        model_out=args.model_out,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        processed_out=args.processed_out,
    )


if __name__ == "__main__":
    main()
