"""
report_generator.py
--------------------
Builds a structured Threat Intelligence Report (JSON) and a human-readable
text summary from a scored, parsed log DataFrame.

This module makes NO claims about "confirmed attacks." It reports
statistical anomalies relative to the trained baseline and lets a human
analyst decide what's actually malicious. Language is deliberately
hedged ("unusual", "flagged", "worth review") rather than accusatory.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd


def _entry_record(row: pd.Series) -> dict:
    return {
        "ip": row["ip"],
        "detection_time_utc": row["timestamp"].tz_convert("UTC").isoformat()
        if row["timestamp"].tzinfo is not None
        else row["timestamp"].isoformat(),
        "endpoint": row["path"],
        "http_method": row["method"],
        "status_code": int(row["status"]),
        "response_size_bytes": int(row["size"]),
        "anomaly_score": round(float(row["anomaly_score"]), 6),
        "indicators": {
            "hour": int(row["hour"]),
            "is_error": bool(row["is_error"]),
            "suspicious_pattern": int(row["suspicious_pattern"]),
            "malformed_request": bool(row.get("malformed_request", False)),
            "ip_requests_so_far": int(row["ip_requests_so_far"]),
            "ip_error_rate_so_far": round(float(row["ip_error_rate_so_far"]), 4),
            "ip_unique_paths_so_far": int(row["ip_unique_paths_so_far"]),
            "ip_requests_last_60s": int(row["ip_requests_last_60s"]),
        },
        "raw_log_line": row["raw_line"],
    }


def build_json_report(
    scored_df: pd.DataFrame,
    source_file: str,
    dataset_label: str = "real",
    top_n: int = 100,
) -> dict:
    """scored_df must already contain: anomaly_score, is_anomaly (bool),
    plus all parsed + feature columns, and raw_line.
    """
    anomalies = scored_df[scored_df["is_anomaly"]].copy()
    anomalies = anomalies.sort_values("anomaly_score", ascending=False)

    top_ips = (
        anomalies.groupby("ip")
        .size()
        .sort_values(ascending=False)
        .head(20)
        .to_dict()
    )

    status_dist = scored_df["status"].value_counts().to_dict()
    anomaly_status_dist = anomalies["status"].value_counts().to_dict()

    report = {
        "report_generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": source_file,
        "dataset_label": dataset_label,  # "real" or "synthetic_test"
        "summary": {
            "total_requests_analyzed": int(len(scored_df)),
            "anomalous_requests_flagged": int(len(anomalies)),
            "anomaly_rate": round(len(anomalies) / len(scored_df), 4) if len(scored_df) else 0.0,
            "anomaly_score_min": round(float(scored_df["anomaly_score"].min()), 6),
            "anomaly_score_max": round(float(scored_df["anomaly_score"].max()), 6),
            "anomaly_score_mean": round(float(scored_df["anomaly_score"].mean()), 6),
            "anomaly_score_std": round(float(scored_df["anomaly_score"].std()), 6),
        },
        "top_suspicious_ips": [{"ip": ip, "flagged_requests": int(c)} for ip, c in top_ips.items()],
        "status_code_distribution_all_traffic": {str(k): int(v) for k, v in status_dist.items()},
        "status_code_distribution_anomalies": {str(k): int(v) for k, v in anomaly_status_dist.items()},
        "flagged_entries": [_entry_record(row) for _, row in anomalies.head(top_n).iterrows()],
        "disclaimer": (
            "This report identifies statistical outliers relative to a "
            "learned baseline of normal traffic using unsupervised "
            "Isolation Forest. It does NOT confirm malicious intent. Every "
            "flagged entry should be reviewed by a human analyst before "
            "any action is taken."
        ),
    }
    return report


def write_json_report(report: dict, out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)


def build_text_report(report: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("OVERWATCH — THREAT INTELLIGENCE REPORT (unsupervised anomaly scan)")
    lines.append("=" * 70)
    lines.append(f"Generated (UTC):      {report['report_generated_utc']}")
    lines.append(f"Source file:          {report['source_file']}")
    lines.append(f"Dataset label:        {report['dataset_label']}")
    lines.append("")
    s = report["summary"]
    lines.append(f"Total requests analyzed:   {s['total_requests_analyzed']:,}")
    lines.append(f"Flagged as anomalous:      {s['anomalous_requests_flagged']:,} "
                 f"({s['anomaly_rate'] * 100:.2f}%)")
    lines.append(f"Anomaly score range:       [{s['anomaly_score_min']}, {s['anomaly_score_max']}]  "
                 f"mean={s['anomaly_score_mean']} std={s['anomaly_score_std']}")
    lines.append("")
    lines.append("Top suspicious IPs (by number of flagged requests):")
    for entry in report["top_suspicious_ips"][:10]:
        lines.append(f"  - {entry['ip']:<40} {entry['flagged_requests']} flagged requests")
    lines.append("")
    lines.append("Status code distribution (all traffic):")
    for k, v in sorted(report["status_code_distribution_all_traffic"].items(), key=lambda x: -x[1]):
        lines.append(f"  {k}: {v:,}")
    lines.append("")
    lines.append("Status code distribution (flagged anomalies only):")
    for k, v in sorted(report["status_code_distribution_anomalies"].items(), key=lambda x: -x[1]):
        lines.append(f"  {k}: {v:,}")
    lines.append("")
    lines.append(f"Top {min(10, len(report['flagged_entries']))} highest-scoring flagged entries:")
    for e in report["flagged_entries"][:10]:
        lines.append(
            f"  score={e['anomaly_score']:.4f}  ip={e['ip']}  "
            f"[{e['detection_time_utc']}]  {e['http_method']} {e['endpoint']} -> {e['status_code']}"
        )
        lines.append(f"      raw: {e['raw_log_line']}")
    lines.append("")
    lines.append("-" * 70)
    lines.append(report["disclaimer"])
    lines.append("=" * 70)
    return "\n".join(lines)


def write_text_report(text: str, out_path: str) -> None:
    with open(out_path, "w") as f:
        f.write(text)
