"""
Overwatch — Streamlit frontend for the existing Isolation Forest pipeline.

This UI does not reimplement anomaly detection. It calls the same parser,
feature engineering, trained model, and report generator used by
`python src/detect.py`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
MODEL_PATH = ROOT / "models" / "isolation_forest.joblib"
METADATA_PATH = ROOT / "models" / "isolation_forest_metadata.json"
DEMO_LOG_PATH = ROOT / "sample" / "suspicious_logs.log"

# Keep uploads practical for free-tier Render memory limits.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MB
MAX_LINES = 50_000

sys.path.insert(0, str(SRC))
from detect import detect_from_text  # noqa: E402
from report_generator import build_json_report, build_text_report  # noqa: E402


def _load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _validate_extension(filename: str | None) -> None:
    if not filename:
        raise ValueError("No filename provided.")
    lower = filename.lower()
    if not (lower.endswith(".log") or lower.endswith(".txt")):
        raise ValueError(
            "Unsupported file type. Please upload an Apache/Nginx access log "
            "with a .log or .txt extension."
        )


def _decode_upload(raw: bytes) -> str:
    if raw is None or len(raw) == 0:
        raise ValueError("The uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"File is too large ({len(raw):,} bytes). "
            f"For the live demo, please keep uploads under "
            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB / {MAX_LINES:,} lines."
        )
    text = raw.decode("utf-8", errors="replace")
    line_count = text.count("\n") + (0 if text.endswith("\n") or not text else 1)
    if line_count > MAX_LINES:
        raise ValueError(
            f"Log has about {line_count:,} lines. "
            f"For the live demo, please keep uploads under {MAX_LINES:,} lines "
            f"(offline CLI supports the full NASA dataset)."
        )
    if not text.strip():
        raise ValueError("The uploaded file contains no usable text.")
    return text


def _run_pipeline(log_text: str, source_label: str, dataset_label: str):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH.as_posix()}. "
            "Ensure models/isolation_forest.joblib is present in the repository."
        )
    scored, stats = detect_from_text(log_text, str(MODEL_PATH))
    report = build_json_report(
        scored,
        source_file=source_label,
        dataset_label=dataset_label,
        top_n=100,
    )
    text_report = build_text_report(report)
    return scored, stats, report, text_report


def _anomaly_display_frame(anomalies: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "ip",
        "timestamp",
        "method",
        "status",
        "path",
        "size",
        "anomaly_score",
        "raw_line",
    ]
    present = [c for c in cols if c in anomalies.columns]
    out = anomalies[present].sort_values("anomaly_score", ascending=False).copy()
    if "timestamp" in out.columns:
        out["timestamp"] = out["timestamp"].astype(str)
    if "anomaly_score" in out.columns:
        out["anomaly_score"] = out["anomaly_score"].round(6)
    return out.reset_index(drop=True)


def _render_charts(anomalies: pd.DataFrame) -> None:
    if anomalies.empty:
        st.info("No anomalous requests to chart for this run.")
        return

    st.subheader("Anomalies by HTTP Status")
    status_counts = (
        anomalies["status"]
        .value_counts()
        .sort_index()
        .rename_axis("status")
        .reset_index(name="count")
        .set_index("status")
    )
    st.bar_chart(status_counts)

    st.subheader("Anomalies by IP")
    ip_counts = (
        anomalies["ip"]
        .value_counts()
        .head(15)
        .rename_axis("ip")
        .reset_index(name="count")
        .set_index("ip")
    )
    st.bar_chart(ip_counts)

    if "timestamp" in anomalies.columns and anomalies["timestamp"].notna().any():
        st.subheader("Timeline")
        ts = pd.to_datetime(anomalies["timestamp"], errors="coerce").dropna()
        if not ts.empty:
            # Minute buckets keep the demo readable for small sample logs.
            timeline = (
                ts.dt.floor("min")
                .value_counts()
                .sort_index()
                .rename_axis("minute")
                .reset_index(name="anomalies")
                .set_index("minute")
            )
            st.line_chart(timeline)


def main() -> None:
    st.set_page_config(
        page_title="Overwatch — Server Log Anomaly Detection",
        layout="wide",
    )

    st.title("Overwatch — Server Log Anomaly Detection")
    st.subheader(
        "Unsupervised ML-based anomaly detection for Apache/Nginx server logs"
    )
    st.write(
        "Upload a server access log and Overwatch parses the requests, "
        "engineers behavioral features, and uses Isolation Forest to identify "
        "statistically anomalous requests."
    )
    st.caption(
        f"Demo size limit: {MAX_UPLOAD_BYTES // (1024 * 1024)} MB / "
        f"{MAX_LINES:,} lines. Larger offline runs use `python src/detect.py`."
    )

    metadata = _load_metadata()

    if "log_text" not in st.session_state:
        st.session_state.log_text = None
        st.session_state.source_label = None
        st.session_state.dataset_label = None

    uploaded = st.file_uploader(
        "Upload Apache/Nginx access log (.log or .txt)",
        type=["log", "txt"],
        help="Common Log Format (CLF) access logs are supported.",
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        use_demo = st.button("Use Demo Log", width="stretch")
    with col_b:
        run_detection = st.button(
            "Run Anomaly Detection",
            type="primary",
            width="stretch",
        )

    if use_demo:
        if not DEMO_LOG_PATH.exists():
            st.error(
                f"Demo log not found at {DEMO_LOG_PATH.as_posix()}. "
                "Expected sample/suspicious_logs.log in the repository."
            )
        else:
            st.session_state.log_text = DEMO_LOG_PATH.read_text(
                encoding="utf-8", errors="replace"
            )
            st.session_state.source_label = "sample/suspicious_logs.log"
            st.session_state.dataset_label = "synthetic_test"
            st.success(
                "Loaded demo log: sample/suspicious_logs.log "
                "(synthetic, clearly labeled — see sample/README_SYNTHETIC.md)."
            )
    elif uploaded is not None:
        try:
            _validate_extension(uploaded.name)
            raw = uploaded.getvalue()
            st.session_state.log_text = _decode_upload(raw)
            st.session_state.source_label = uploaded.name
            st.session_state.dataset_label = "uploaded"
            st.info(f"Ready: {uploaded.name} ({len(raw):,} bytes).")
        except ValueError as exc:
            st.error(str(exc))

    if run_detection:
        if not st.session_state.log_text:
            st.error("Please upload a log file or click **Use Demo Log** first.")
        else:
            with st.spinner("Running Overwatch pipeline…"):
                try:
                    scored, stats, report, text_report = _run_pipeline(
                        st.session_state.log_text,
                        st.session_state.source_label or "upload",
                        st.session_state.dataset_label or "uploaded",
                    )
                except ValueError as exc:
                    st.error(str(exc))
                    return
                except FileNotFoundError as exc:
                    st.error(str(exc))
                    return
                except Exception as exc:  # noqa: BLE001 — show clear UI errors
                    st.error(f"Inference failed: {exc}")
                    return

            anomalies = scored[scored["is_anomaly"]].copy()
            total = int(len(scored))
            n_anom = int(len(anomalies))
            rate = (n_anom / total) if total else 0.0
            unique_ips = int(anomalies["ip"].nunique()) if n_anom else 0

            st.markdown("---")
            st.header("Results")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Requests", f"{total:,}")
            m2.metric("Anomalies Detected", f"{n_anom:,}")
            m3.metric("Anomaly Rate", f"{rate * 100:.2f}%")
            m4.metric("Unique Anomalous IPs", f"{unique_ips:,}")

            st.caption(
                f"Parse stats: {stats.summary()} — anomalies are statistical "
                "outliers relative to the trained baseline, not confirmed attacks."
            )

            st.subheader("Anomalous Requests")
            if anomalies.empty:
                st.info("No requests were flagged as anomalous for this log.")
            else:
                st.dataframe(
                    _anomaly_display_frame(anomalies),
                    width="stretch",
                    height=420,
                )

            _render_charts(anomalies)

            st.subheader("Threat Intelligence Report")
            st.write(
                "Structured report from the existing Overwatch report generator. "
                "Language is deliberately hedged: flagged entries are statistical "
                "outliers worth human review."
            )
            flagged = report.get("flagged_entries", [])
            if flagged:
                preview_rows = []
                for entry in flagged[:50]:
                    preview_rows.append(
                        {
                            "ip": entry.get("ip"),
                            "timestamp": entry.get("detection_time_utc"),
                            "http_method": entry.get("http_method"),
                            "status": entry.get("status_code"),
                            "request": entry.get("endpoint"),
                            "anomaly_score": entry.get("anomaly_score"),
                            "raw_log_line": entry.get("raw_log_line"),
                        }
                    )
                st.dataframe(
                    pd.DataFrame(preview_rows),
                    width="stretch",
                    height=320,
                )
            else:
                st.info("No flagged entries in the threat intelligence report.")

            st.download_button(
                "Download Threat Intelligence Report",
                data=json.dumps(report, indent=2, default=str),
                file_name="overwatch_threat_intelligence_report.json",
                mime="application/json",
            )
            st.download_button(
                "Download Text Report",
                data=text_report,
                file_name="overwatch_threat_intelligence_report.txt",
                mime="text/plain",
            )
            with st.expander("Preview text report"):
                st.code(text_report, language="text")

    with st.expander("About the Model"):
        n_features = metadata.get("n_features", len(metadata.get("feature_columns", [])))
        contamination = metadata.get("contamination", 0.02)
        n_estimators = metadata.get("n_estimators", 200)
        feature_columns = metadata.get("feature_columns", [])
        st.markdown(
            f"""
- **Algorithm:** Isolation Forest (`sklearn.ensemble.IsolationForest`)
- **Learning type:** Unsupervised anomaly detection
- **Purpose:** Identify statistically unusual server-log behavior relative to a
  learned baseline of normal traffic
- **Input:** Parsed HTTP/server-log behavioral features
  ({n_features} features{': `' + '`, `'.join(feature_columns) + '`' if feature_columns else ''})
- **Configuration (from training metadata):**
  `{n_estimators}` trees, `contamination={contamination}`
  (a stated assumption that ~{contamination * 100:.0f}% of traffic is worth human review —
  not a measured attack rate)
- **Model file:** `models/isolation_forest.joblib` (scaler + model bundle)
            """.strip()
        )

    with st.expander("Limitations"):
        st.markdown(
            """
Overwatch identifies **statistically anomalous** server-log behavior. An anomaly
is not automatically a confirmed cyberattack. Detected events should be
investigated using additional security context and evidence.

Additional project-specific limitations:
- Real 1995 NASA traffic used for training contains no confirmed attacks, so
  real-data results validate *outlier detection*, not attack-detection accuracy.
- `contamination` is a tunable assumption; changing it changes how many
  requests get flagged, not the model's underlying scoring.
- No `referrer` / `user-agent` fields in Common Log Format limits available
  features compared to modern combined-format logs.
- A static, one-time-trained model does not adapt to gradual traffic drift.
- The live demo caps upload size for hosting memory limits; the offline CLI
  can process the full NASA dataset.
            """.strip()
        )


if __name__ == "__main__":
    # Render / local: prefer PORT from the environment when present.
    # `streamlit run app.py --server.port $PORT ...` is the supported start command.
    _ = os.environ.get("PORT")
    main()
