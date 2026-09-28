# Overwatch — Server Log Anomaly Detection

Unsupervised anomaly detection for Apache/NGINX-style server access logs, using
Isolation Forest. Built and validated against a real 1.57-million-row NASA-HTTP
access log (August 1995, standard Apache Common Log Format).

## Live Demo

Interactive Streamlit UI (upload a log or use the built-in demo sample):

`https://YOUR-RENDER-URL.onrender.com`

The live demo lets visitors upload a small Apache/Nginx Common Log Format
access log (or load `sample/suspicious_logs.log`), run the existing Overwatch
parser → feature engineering → Isolation Forest pipeline, view summary metrics
and charts, and download the Threat Intelligence Report. Anomalies are
statistical outliers relative to the trained baseline — not confirmed attacks.
Uploads are capped for hosting memory limits; full-scale offline runs still use
the CLI below.

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Deploy on [Render](https://render.com) as a **Web Service**:

**Build command:**

```bash
pip install -r requirements.txt
```

**Start command:**

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

Do not hard-code a port. Ensure `models/isolation_forest.joblib` is present in
the repository (paths are repository-relative and Linux-compatible).

## Objective

Detect unusual, potentially suspicious HTTP requests in a server access log
*without* labeled attack data, by learning a baseline of "normal" traffic
behavior and flagging statistical outliers for human review. This is an
internship-scale project: no supervised attack labels, no toy classifier, one
well-understood unsupervised model, applied honestly to a real dataset.

## Architecture / Pipeline

```
raw access.log
      │
      ▼
src/log_parser.py            Regex-based CLF parser (strict + loose fallback)
      │
      ▼
src/feature_engineering.py   Per-request + causal per-IP behavioral features
      │
      ▼
src/train.py                 StandardScaler + IsolationForest, saved with joblib
      │
      ▼
models/isolation_forest.joblib
      │
      ▼
src/detect.py  ──(new log file)──►  score requests ──► src/report_generator.py
                                                              │
                                                              ▼
                                              reports/*.json + reports/*.txt
```

## Dataset

`data/raw/access.log` — the real **NASA-HTTP** access log, August 1995,
~1.57 million requests, standard Apache Common Log Format:

```
host - - [dd/Mon/yyyy:HH:MM:SS -zzzz] "METHOD path HTTP/version" status size
```

Key facts discovered during inspection (see the notebook, section 1–3):
- 99.9994% of lines parse cleanly with the strict CLF regex; a small number
  (3,591) need a looser fallback due to unescaped quotes/HTML embedded in
  the request field (some of these are themselves interesting — a legacy
  client sending an unescaped `<IMG SRC=...>` string, for instance);
  10 lines are unparseable binary noise and are dropped. Rows recovered via
  the loose fallback are flagged with `malformed_request=True`.
- **Parser fix:** the loose fallback previously collapsed everything after
  the first space in a malformed request to a single truncated token,
  silently discarding embedded quotes/HTML/payload characters even though
  `raw_line` kept them. It now preserves the **complete** malformed request
  content in `path` (stripping only a trailing `HTTP/x.y` token if present),
  so nothing is lost before feature engineering sees it. `raw_line` was
  never altered either way.
- `size` is `-` (no body) on ~14k rows — normalized to `0`.
- Status codes are overwhelmingly benign: ~89% `200`, ~8.5% `304`, ~1.7%
  `302`, ~0.6% `404`. 4xx/5xx server errors total under 200 requests out of
  1.57M. **This dataset contains no confirmed attacks** — it is ordinary
  1995 public web traffic, and this README does not claim otherwise.
- No `referrer` or `user-agent` fields exist (this predates the "combined"
  log format), so feature engineering works from method/path/status/size/time
  only.

Because the real data has no attack-like traffic to validate detection
against, a small **synthetic** log is included separately, purely to prove
the pipeline actually fires on attack-shaped behavior:

`sample/suspicious_logs.log` — **fabricated, clearly labeled** (see
`sample/README_SYNTHETIC.md`). Contains a fast scan of admin/CGI paths, a
login-hammering burst from one IP, and a few SQLi/path-traversal/XSS-style
payloads, interleaved with normal-looking background traffic. This is never
mixed into the real training baseline — it is only used at inference time to
demonstrate detection.

## Features

Isolation Forest needs numeric, behaviorally meaningful inputs, not raw IP or
path strings. Two groups of features are engineered (full rationale and code
in `src/feature_engineering.py`):

**Per-request:** `hour`, `method_code`, `status_class`, `is_error`, `size`,
`path_depth`, `path_length`, `has_query`, `is_static_asset`, `suspicious_pattern`.

**Per-IP behavioral (causal / leakage-safe):** `ip_requests_so_far`,
`ip_error_rate_so_far`, `ip_unique_paths_so_far`, `ip_requests_last_60s`.

These are computed using **only each IP's requests strictly before the
current one** (an expanding/trailing-window calculation over time-sorted
data), so no future information about an IP leaks into the features used to
judge a request happening earlier in time — the same information a real-time
detector would actually have available.

**`suspicious_pattern` (new):** an integer count (0–7) of how many distinct
attack-*shaped* syntax categories appear in a request's text — path
traversal (`../`, `%2e%2e`), SQL-injection-shaped syntax (`UNION SELECT`,
`OR 1=1`, `--`, `SLEEP(`), script-tag/XSS-shaped syntax (`<script`,
`javascript:`, `onerror=`), command-injection-shaped syntax (`; cat`,
`` `...` ``, `$(`), suspicious percent-encoded bytes (`%00`, `%3c`, `%27`),
raw quote/angle-bracket characters, and a `+1` if `log_parser` had to use
its loose fallback on that line at all. **This is a behavioral signal for
the model to weigh alongside everything else — a nonzero value means "this
request's text has attack-shaped syntax," not a confirmed verdict that the
request is malicious.** Legitimate URLs can occasionally trip a category
(see Results below for a real example), which is exactly why the JSON
report always includes the raw log line for human review.

## Model

`sklearn.ensemble.IsolationForest` (200 trees, `contamination=0.02`), fit on
`StandardScaler`-normalized features. `contamination` is a **stated
assumption** (assume ~2% of traffic is worth a human look), not a measured
error rate — there is no ground truth to measure against. The model and
scaler are bundled together and saved with `joblib`.

## Training

```bash
pip install -r requirements.txt

python src/train.py \
  --log-file data/raw/access.log \
  --model-out models/isolation_forest.joblib \
  --contamination 0.02 \
  --processed-out data/processed/features.csv
```

On the full 1.57M-row real log this takes well under a minute end-to-end
(parsing ~13s, feature engineering ~25s, model fit ~12s on a standard CPU).
Training metadata (row counts, parse stats, anomaly rate at training time) is
written alongside the model as `models/isolation_forest_metadata.json`.

## Inference

```bash
python src/detect.py \
  --log-file sample/suspicious_logs.log \
  --model models/isolation_forest.joblib \
  --dataset-label synthetic_test \
  --json-out reports/sample_report.json \
  --text-out reports/sample_report.txt
```

`detect.py` parses the supplied log with the same parser, builds the same
features used in training, loads the saved model bundle, scores every
request, and writes both a structured JSON report and a human-readable text
report. Every flagged entry keeps the **original raw log line** so a human
analyst can verify exactly what triggered the flag.

## Results

*(Regenerated after the three fixes below — retrained model, re-run
inference, both reports and the notebook are current as of this version.)*

**Real NASA data** (`reports/real_nasa_report.txt` / `.json`): 31,398 / 1,569,888
requests flagged (2.00%, matching the configured `contamination`). With
`suspicious_pattern` now implemented, the **top two** highest-scoring
entries are a request with a stray embedded quote in the path
(`.../KSC-95EC-0946.gif"`, score 0.1484) — a 1995 client-rendering quirk that
now correctly registers as attack-*shaped* syntax even though it is clearly
benign. The rest of the top-10 is unchanged in character: large ISP/proxy
servers (Prodigy, AOL, `beta.xerox.com`) making unusually high request
volumes on behalf of many real users, and broken/typo'd links generating
repeated 404s. Top flagged IPs: `163.206.89.4` (1,560), `piweba3y.prodigy.com`
(1,280), `piweba4y.prodigy.com` (1,174), `piweba5y.prodigy.com` (1,165),
`www-d1.proxy.aol.com` (1,065). These remain genuine statistical outliers,
**not evidence of attacks** — reported honestly rather than dressed up, and
this real example is exactly why the feature is framed as behavioral, not
accusatory.

**Synthetic test data** (`reports/sample_report.txt` / `.json`): 99 / 149
requests flagged (66.4%). With `suspicious_pattern` implemented, the
**top three** highest-scoring entries are now the path-traversal
(`../../../../etc/passwd`, score 0.1502), SQL-injection-shaped
(`id=1' OR '1'='1`, score 0.1489), and command-injection-shaped
(`cmd=;cat%20/etc/shadow`, score 0.1469) payloads — previously these ranked
behind the login-hammering burst; the new feature correctly pulls the
injection-style requests to the top. The login-hammering IP
(`198.51.100.23`, 79 flagged requests) and the scanning burst are still
almost entirely flagged, confirming the pipeline works end-to-end on
attack-shaped behavior.

See `reports/anomaly_analysis_overview.png` for the anomaly-score
distribution, status-code comparison, hour-of-day comparison, and top
suspicious IPs, and `notebooks/01_training_and_evaluation.ipynb` for the full
walkthrough with all cells executed.

## Evaluation approach

Because this is unsupervised anomaly detection with no ground-truth attack
labels, this project does **not** report fabricated accuracy/precision/recall.
Instead it reports: anomaly count and rate, anomaly-score distribution,
normal-vs-anomalous visualizations, top suspicious IPs, status-code
distribution (overall and among anomalies), and concrete examples of
detected anomalies with their raw log lines — all in the notebook and the
generated reports.

## Limitations

- Real 1995 NASA traffic contains no confirmed attacks, so real-data results
  above validate *outlier detection*, not attack-detection accuracy —
  meaningful validation of true/false positive rates requires either labeled
  incidents or ongoing analyst review of live alerts.
- `contamination` is a tunable assumption; changing it changes how many
  requests get flagged, not the model's underlying scoring.
- No `referrer`/`user-agent` in this log format limits available features
  compared to modern combined-format logs.
- A static, one-time-trained model doesn't adapt to gradual traffic drift;
  a production deployment would retrain on a rolling window.

## How to run everything

```bash
pip install -r requirements.txt

# 1. Train on the real dataset (retrains the model + scaler bundle)
python src/train.py \
  --log-file data/raw/access.log \
  --model-out models/isolation_forest.joblib \
  --contamination 0.02

# 2. Score the synthetic demo log
python src/detect.py \
  --log-file sample/suspicious_logs.log \
  --model models/isolation_forest.joblib \
  --dataset-label synthetic_test \
  --json-out reports/sample_report.json \
  --text-out reports/sample_report.txt

# 3. (Optional) re-score the real log itself for the "real" report
python src/detect.py \
  --log-file data/raw/access.log \
  --model models/isolation_forest.joblib \
  --dataset-label real \
  --json-out reports/real_nasa_report.json \
  --text-out reports/real_nasa_report.txt

# 4. (Optional) reproduce the full notebook walkthrough
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/01_training_and_evaluation.ipynb
```

Verified from a clean state: `rm -rf src/__pycache__ models/*.joblib models/*.json`,
then steps 1–4 above reproduce identical row counts, an identical 2.00%
training-baseline anomaly rate, and the same top-ranked entries described
in Results (Isolation Forest's `random_state=42` is fixed, so results are
deterministic given the same input file).

## Project structure

```
Overwatch/
├── app.py                             # Streamlit frontend (calls existing pipeline)
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/access.log                 # real NASA-HTTP dataset
│   └── processed/                     # generated feature matrix (gitignored)
├── notebooks/
│   └── 01_training_and_evaluation.ipynb
├── src/
│   ├── log_parser.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── detect.py
│   └── report_generator.py
├── models/
│   ├── isolation_forest.joblib
│   └── isolation_forest_metadata.json
├── reports/
│   ├── real_nasa_report.json / .txt
│   ├── sample_report.json / .txt
│   └── anomaly_analysis_overview.png
└── sample/
    ├── suspicious_logs.log            # synthetic, clearly labeled
    └── README_SYNTHETIC.md
```
