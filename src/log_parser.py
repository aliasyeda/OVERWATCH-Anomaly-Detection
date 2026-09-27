"""
log_parser.py
-------------
Parses Apache/NCSA "Common Log Format" (CLF) access logs into a pandas
DataFrame. Built and tested against the real NASA-HTTP August 1995 access
log, but works for any standard CLF file.

Log line shape:
    host ident authuser [timestamp] "METHOD path HTTP/version" status size

Example:
    in24.inetnebr.com - - [01/Aug/1995:00:00:01 -0400] \
        "GET /shuttle/missions/sts-68/news/sts-68-mcc-05.txt HTTP/1.0" 200 1839

Design notes
------------
- The real dataset contains a handful of lines that break the strict CLF
  pattern (e.g. an unescaped double-quote inside a query string that was
  clearly a client attempting an XSS-style injection, or a couple of lines
  of raw binary noise). Rather than silently dropping the interesting rows,
  we use a strict primary regex first and fall back to a looser regex
  before giving up on a line.
- `size` is "-" when Apache logged no response body (common on 304/302);
  we normalize that to 0.
- We keep the raw original line for every parsed row so any detected
  anomaly can be traced back to the exact triggering log entry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

# Strict Common Log Format pattern.
_STRICT_PATTERN = re.compile(
    r'^(?P<host>\S+) (?P<ident>\S+) (?P<authuser>\S+) '
    r'\[(?P<timestamp>[^\]]+)\] '
    r'"(?P<method>[A-Z]+) (?P<path>\S+) (?P<protocol>HTTP/[\d.]+)" '
    r'(?P<status>\d{3}) (?P<size>\S+)$'
)

# Looser fallback: tolerates malformed/unescaped request strings by matching
# the method at the start and status/size at the end, treating everything
# in between as the "path" (this recovers a small number of rows that would
# otherwise be discarded, e.g. injection-attempt payloads worth keeping for
# a security-focused dataset). `.*` is greedy and the pattern is anchored
# with `$`, so `rest` naturally captures every embedded quote/HTML/payload
# character between the method and the final `" status size` -- nothing is
# truncated here at the regex level.
_LOOSE_PATTERN = re.compile(
    r'^(?P<host>\S+) (?P<ident>\S+) (?P<authuser>\S+) '
    r'\[(?P<timestamp>[^\]]+)\] '
    r'"(?P<method>[A-Z]+) (?P<rest>.*)" '
    r'(?P<status>\d{3}) (?P<size>\S+)$'
)

# Used only to strip a trailing " HTTP/x.y" protocol token off the tail of
# a loose-matched `rest` string, if one is present, so that `path` reflects
# the request target while keeping every other malformed character intact.
_TRAILING_PROTOCOL = re.compile(r'\s(HTTP/\d\.\d)$')

_TIMESTAMP_FORMAT = "%d/%b/%Y:%H:%M:%S %z"


@dataclass
class ParseStats:
    total_lines: int = 0
    parsed_strict: int = 0
    parsed_loose: int = 0
    dropped: int = 0

    def summary(self) -> str:
        return (
            f"total_lines={self.total_lines} "
            f"parsed_strict={self.parsed_strict} "
            f"parsed_loose={self.parsed_loose} "
            f"dropped={self.dropped}"
        )


def _parse_size(raw: str) -> int:
    if raw == "-" or raw is None:
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


def parse_log_file(path: str, verbose: bool = True) -> tuple[pd.DataFrame, ParseStats]:
    """Parse a CLF access log file into a DataFrame.

    Returns
    -------
    (df, stats) : the parsed DataFrame and a ParseStats summary.

    Columns produced
    ----------------
    ip, timestamp (tz-aware datetime), method, path, protocol,
    status (int), size (int, bytes), raw_line (original untouched line)
    """
    stats = ParseStats()
    rows = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stats.total_lines += 1
            raw_line = line.rstrip("\n")

            m = _STRICT_PATTERN.match(raw_line)
            if m:
                stats.parsed_strict += 1
                d = m.groupdict()
                path_val = d["path"]
                method_val = d["method"]
                is_malformed = False
            else:
                m = _LOOSE_PATTERN.match(raw_line)
                if not m:
                    stats.dropped += 1
                    continue
                stats.parsed_loose += 1
                d = m.groupdict()
                # In the loose match, "rest" holds the entire malformed
                # request content (path plus any embedded quotes/HTML/
                # injection-style payload, and usually a trailing protocol
                # token). Previously this was collapsed to just its first
                # whitespace-delimited token, which silently threw away
                # everything after the first space (e.g. injected
                # <IMG SRC=...> markup or extra quote characters) even
                # though `raw_line` still had it. We now keep the FULL
                # content: strip a trailing " HTTP/x.y" if present, and use
                # everything else, unmodified, as `path`. This preserves
                # the complete request/payload for malformed lines while
                # `raw_line` remains the untouched original.
                method_val = d["method"]
                rest = d["rest"] or ""
                proto_match = _TRAILING_PROTOCOL.search(rest)
                path_val = rest[: proto_match.start()] if proto_match else rest
                if not path_val:
                    path_val = "-"
                is_malformed = True

            rows.append(
                {
                    "ip": d["host"],
                    "timestamp_raw": d["timestamp"],
                    "method": method_val,
                    "path": path_val,
                    "status": int(d["status"]) if d["status"].isdigit() else -1,
                    "size": _parse_size(d["size"]),
                    "malformed_request": is_malformed,
                    "raw_line": raw_line,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(
            df["timestamp_raw"], format=_TIMESTAMP_FORMAT, errors="coerce"
        )
        # Drop the tiny number of rows where even the timestamp itself
        # was unparseable (should be ~0 for well-formed CLF).
        bad_ts = df["timestamp"].isna().sum()
        if bad_ts:
            stats.dropped += int(bad_ts)
            df = df[df["timestamp"].notna()].copy()
        df = df.drop(columns=["timestamp_raw"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    if verbose:
        print(f"[log_parser] {stats.summary()}")

    return df, stats


if __name__ == "__main__":
    import sys

    log_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/access.log"
    df, stats = parse_log_file(log_path)
    print(df.head())
    print(df.dtypes)
    print(stats.summary())
