"""
detect.py
---------
Command-line inference: score a (new) log file against a previously
trained Isolation Forest baseline and produce a Threat Intelligence Report.

Usage:
    python src/detect.py --log-file sample/suspicious_logs.log \
                          --model models/isolation_forest.joblib \
                          --dataset-label synthetic_test \
                          --json-out reports/sample_report.json \
                          --text-out reports/sample_report.txt
"""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import build_features  # noqa: E402
from log_parser import parse_log_file, parse_log_text  # noqa: E402
from report_generator import (  # noqa: E402
    build_json_report,
    build_text_report,
    write_json_report,
    write_text_report,
)


def score_dataframe(df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """Run the trained Isolation Forest on an already-parsed log DataFrame.

    Same feature engineering, scaler, and model as detect() — no ML changes.
    """
    if len(df) == 0:
        raise ValueError("No rows could be parsed from the supplied log file.")

    print(f"[detect] loading model bundle from {model_path} ...")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_columns = bundle["feature_columns"]

    print("[detect] building features (same pipeline used in training) ...")
    features = build_features(df)

    X = scaler.transform(features[feature_columns].to_numpy())
    raw_scores = model.decision_function(X)
    predictions = model.predict(X)

    # `features` duplicates a couple of column names already present in df
    # (e.g. "size" is both the raw parsed field and a feature); keep the
    # raw df version and drop the duplicate before concatenating.
    dup_cols = [c for c in features.columns if c in df.columns]
    features_for_concat = features.drop(columns=dup_cols)
    scored = pd.concat([df.reset_index(drop=True), features_for_concat.reset_index(drop=True)], axis=1)
    scored["anomaly_score"] = -raw_scores
    scored["is_anomaly"] = predictions == -1

    n_anom = int(scored["is_anomaly"].sum())
    print(f"[detect] flagged {n_anom:,} / {len(scored):,} requests as anomalous "
          f"({100 * n_anom / len(scored):.2f}%)")

    return scored


def detect(
    log_file: str,
    model_path: str,
    dataset_label: str = "real",
) -> pd.DataFrame:
    print(f"[detect] parsing {log_file} ...")
    df, stats = parse_log_file(log_file)
    print(f"[detect] parsed {len(df):,} rows ({stats.summary()})")
    return score_dataframe(df, model_path)


def detect_from_text(
    log_text: str,
    model_path: str,
) -> tuple[pd.DataFrame, object]:
    """Parse in-memory log text and score it with the trained model.

    Returns (scored_df, parse_stats). Used by the Streamlit frontend.
    """
    print("[detect] parsing in-memory log text ...")
    df, stats = parse_log_text(log_text, verbose=True)
    print(f"[detect] parsed {len(df):,} rows ({stats.summary()})")
    scored = score_dataframe(df, model_path)
    return scored, stats


def main():
    parser = argparse.ArgumentParser(description="Score a log file against the trained Overwatch baseline.")
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--model", default="models/isolation_forest.joblib")
    parser.add_argument("--dataset-label", default="real", choices=["real", "synthetic_test"])
    parser.add_argument("--json-out", default="reports/report.json")
    parser.add_argument("--text-out", default="reports/report.txt")
    parser.add_argument("--top-n", type=int, default=100, help="Max flagged entries to include in the JSON report")
    args = parser.parse_args()

    scored = detect(args.log_file, args.model, dataset_label=args.dataset_label)

    report = build_json_report(
        scored, source_file=args.log_file, dataset_label=args.dataset_label, top_n=args.top_n
    )

    os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.text_out) or ".", exist_ok=True)
    write_json_report(report, args.json_out)
    text = build_text_report(report)
    write_text_report(text, args.text_out)

    print(f"[detect] JSON report written to {args.json_out}")
    print(f"[detect] Text report written to {args.text_out}")
    print()
    print(text)


if __name__ == "__main__":
    main()
