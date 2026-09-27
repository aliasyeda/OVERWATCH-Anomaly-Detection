"""
feature_engineering.py
-----------------------
Turns parsed log rows into a numeric feature matrix suitable for
Isolation Forest.

Why these features (and not raw IP strings)?
Isolation Forest isolates points via random splits on numeric axes.
Feeding it a raw IP address or path string gives it nothing meaningful to
split on. Instead we describe the *behavior* around each request:

Per-request features
    hour                  - time-of-day context; scanning/attacks often
                            cluster at unusual hours.
    method_code           - GET/POST/HEAD/OTHER encoded as an int; unusual
                            methods (PUT/DELETE/TRACE) are rarer and often
                            probing.
    status_class          - 2/3/4/5 (first digit of HTTP status); error-
                            heavy behavior is a core signal.
    is_error              - 1 if status >= 400, else 0.
    size                  - response size in bytes; 0-byte or unusually
                            large responses can indicate probing or
                            data exfiltration attempts.
    path_depth            - number of "/" segments; deep/unusual paths can
                            indicate directory traversal attempts.
    path_length            - length of the requested path string; very long
                            paths often carry injection payloads.
    has_query             - whether the path contains "?"; more common in
                            dynamic/attack requests than static assets.
    is_static_asset       - request for gif/jpg/css/js/etc (normal browsing
                            noise) vs. dynamic/script paths.
    suspicious_pattern     - count (0-N) of distinct suspicious URL/request
                            pattern categories matched in the path (and, for
                            malformed lines, in the full malformed request
                            content preserved by log_parser): encoded/plain
                            path traversal, SQL-injection-shaped syntax,
                            script-tag/XSS-shaped syntax, command-injection-
                            shaped syntax, suspicious percent-encoded bytes,
                            and raw quote/angle-bracket characters or a
                            malformed request line itself. This is a
                            BEHAVIORAL signal for the anomaly model to weigh
                            alongside everything else -- a nonzero value
                            means "this request's text has attack-shaped
                            syntax", NOT a claim that the request is
                            confirmed malicious (plenty of legitimate 1995
                            NASA URLs contain none of this and some
                            benign-but-odd URLs may still trip a category).

Per-IP behavioral features (CAUSAL / LEAKAGE-SAFE)
    All of the following are computed as of the moment of each request,
    using ONLY that IP's requests strictly before the current one (an
    expanding or trailing window "as of now"). This avoids leaking future
    behavior of an IP into features used to judge a request as normal or
    anomalous -- a request is judged by what was known about that IP up to
    that point in time, exactly as a real-time detector would see it.

    ip_requests_so_far     - running count of this IP's prior requests.
    ip_error_rate_so_far   - running fraction of prior requests that errored.
    ip_unique_paths_so_far - running count of distinct endpoints hit so far.
    ip_requests_last_60s   - burst signal: how many requests this IP made
                            in the trailing 60-second window (excludes the
                            current request itself).
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

STATIC_EXTENSIONS = {
    "gif", "jpg", "jpeg", "png", "bmp", "ico", "css", "js",
    "wav", "mpg", "mp3", "xbm", "txt",
}

METHOD_CODES = {"GET": 0, "HEAD": 1, "POST": 2}
DEFAULT_METHOD_CODE = 3  # anything else (PUT, DELETE, TRACE, malformed...)

FEATURE_COLUMNS = [
    "hour",
    "method_code",
    "status_class",
    "is_error",
    "size",
    "path_depth",
    "path_length",
    "has_query",
    "is_static_asset",
    "suspicious_pattern",
    "ip_requests_so_far",
    "ip_error_rate_so_far",
    "ip_unique_paths_so_far",
    "ip_requests_last_60s",
]

# Named categories of attack-*shaped* syntax. Each is deliberately broad
# (favoring recall over precision) since this only feeds a behavioral
# anomaly signal -- false positives just mean "this request's text looked
# a bit unusual," not a false accusation of an attack. Matching is done
# case-insensitively against the (lowercased) path/request text.
_SUSPICIOUS_PATTERNS = [
    ("path_traversal", re.compile(r"(?:\.\./|\.\.\\|%2e%2e|\.\.%2f|%2e%2e%2f)")),
    ("sql_injection", re.compile(
        r"(?:\bunion\s+select\b|\bselect\b.*\bfrom\b|\bdrop\s+table\b|"
        r"\bor\b\s*['\"]?\s*\d+\s*['\"]?\s*=\s*['\"]?\s*\d+|--\s|;--|"
        r"\bsleep\s*\(|\bbenchmark\s*\()"
    )),
    ("script_tag", re.compile(r"(?:<script|javascript:|onerror\s*=|onload\s*=|<img[^>]*src)")),
    ("command_injection", re.compile(
        r"(?:;\s*(?:cat|ls|rm|wget|curl|id|whoami)\b|\|\s*(?:cat|ls|id|whoami)\b|`[^`]*`|\$\()"
    )),
    ("encoded_special_bytes", re.compile(r"(?:%00|%0a|%0d|%3c|%3e|%27|%22)")),
    ("raw_quote_or_bracket", re.compile(r"[\"'<>]")),
]


def _suspicious_pattern_score(text_lower: pd.Series) -> pd.Series:
    """Count how many distinct suspicious-pattern categories match each row.

    Vectorized (regex applied once per category across the whole column)
    rather than looping per-row, since this runs over 1.5M+ rows.
    """
    score = pd.Series(0, index=text_lower.index, dtype=int)
    for _name, pattern in _SUSPICIOUS_PATTERNS:
        score = score + text_lower.str.contains(pattern, regex=True, na=False).astype(int)
    return score


def _static_asset_flag(path: str) -> int:
    clean = path.split("?", 1)[0]
    if "." not in clean:
        return 0
    ext = clean.rsplit(".", 1)[-1].lower()
    return int(ext in STATIC_EXTENSIONS)


def _per_request_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["hour"] = df["timestamp"].dt.hour
    out["method_code"] = df["method"].map(METHOD_CODES).fillna(DEFAULT_METHOD_CODE).astype(int)
    out["status_class"] = (df["status"] // 100).clip(lower=1, upper=5)
    out["is_error"] = (df["status"] >= 400).astype(int)
    out["size"] = df["size"].clip(lower=0)
    out["path_depth"] = df["path"].str.split("?").str[0].str.count("/")
    out["path_length"] = df["path"].str.len()
    out["has_query"] = df["path"].str.contains(r"\?", regex=True).astype(int)
    out["is_static_asset"] = df["path"].apply(_static_asset_flag)

    # suspicious_pattern: count of attack-shaped syntax categories found in
    # the path text, plus +1 if log_parser flagged this row's request line
    # as malformed (embedded quotes/HTML broke the strict CLF pattern --
    # itself a mild behavioral signal worth folding in here).
    path_lower = df["path"].astype(str).str.lower()
    suspicious = _suspicious_pattern_score(path_lower)
    if "malformed_request" in df.columns:
        suspicious = suspicious + df["malformed_request"].fillna(False).astype(int)
    out["suspicious_pattern"] = suspicious
    return out


def _causal_ip_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute leakage-safe, time-ordered per-IP behavioral features.

    Assumes df is already sorted by timestamp ascending (log_parser does
    this). Uses groupby + shift/expanding so that the feature for row i
    only reflects information available strictly before row i.
    """
    n = len(df)
    ip_requests_so_far = np.zeros(n, dtype=np.int64)
    ip_error_rate_so_far = np.zeros(n, dtype=np.float64)
    ip_unique_paths_so_far = np.zeros(n, dtype=np.int64)
    ip_requests_last_60s = np.zeros(n, dtype=np.int64)

    is_error = (df["status"].to_numpy() >= 400).astype(np.int64)
    timestamps = df["timestamp"].astype("int64").to_numpy()  # ns since epoch
    window_ns = 60 * 1_000_000_000

    for ip, idx in df.groupby("ip").groups.items():
        idx = np.asarray(idx)
        cnt = len(idx)
        errs = is_error[idx]
        paths = df.loc[idx, "path"].to_numpy()
        ts = timestamps[idx]

        # running count / error rate: value at position k = stats over
        # positions [0, k) only (strictly before current request).
        running_count = np.arange(cnt)
        running_errors = np.concatenate(([0], np.cumsum(errs)[:-1])) if cnt > 0 else np.array([])

        # running unique-path count strictly before current row
        seen = set()
        unique_so_far = np.zeros(cnt, dtype=np.int64)
        for k in range(cnt):
            unique_so_far[k] = len(seen)
            seen.add(paths[k])

        # trailing 60s request count, excluding current request, via two-pointer
        last_60s = np.zeros(cnt, dtype=np.int64)
        left = 0
        for k in range(cnt):
            while ts[k] - ts[left] > window_ns:
                left += 1
            last_60s[k] = k - left  # requests strictly before k within window

        ip_requests_so_far[idx] = running_count
        ip_error_rate_so_far[idx] = np.divide(
            running_errors, running_count,
            out=np.zeros(cnt, dtype=np.float64),
            where=running_count > 0,
        )
        ip_unique_paths_so_far[idx] = unique_so_far
        ip_requests_last_60s[idx] = last_60s

    return pd.DataFrame(
        {
            "ip_requests_so_far": ip_requests_so_far,
            "ip_error_rate_so_far": ip_error_rate_so_far,
            "ip_unique_paths_so_far": ip_unique_paths_so_far,
            "ip_requests_last_60s": ip_requests_last_60s,
        },
        index=df.index,
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full feature matrix from a parsed log DataFrame.

    df must contain: ip, timestamp, method, path, status, size
    and must be sorted by timestamp ascending.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    request_feats = _per_request_features(df)
    ip_feats = _causal_ip_features(df)
    features = pd.concat([request_feats, ip_feats], axis=1)
    features = features[FEATURE_COLUMNS]
    return features


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from log_parser import parse_log_file

    log_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/access.log"
    df, stats = parse_log_file(log_path)
    feats = build_features(df)
    print(feats.describe())
