"""
verify_data.py  ·  MLB Swing Intelligence — Data Diagnostics
=============================================================
Run LOCALLY before pushing to Streamlit Cloud to catch data
issues early.

    python verify_data.py
    python verify_data.py --dir /path/to/data --seasons 2024 2025

Checks performed:
  1. File existence (CSV and/or Parquet)
  2. Required columns present
  3. Column dtype sanity
  4. Numeric range plausibility (tilt 0–90, xwoba 0–1, etc.)
  5. Duplicate batter × zone × pitch_group rows (should be unique)
  6. Pitcher contamination remaining after filter
  7. Missing value rates per column
  8. Cross-season batter overlap (how many batters appear in multiple seasons)

Prints a colour-coded report and exits with code 1 if any ERROR found.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Colour codes (ANSI) ──────────────────────────────────────────────
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET}  {msg}")
def err(msg):  print(f"  {RED}✗{RESET}  {msg}")
def hdr(msg):  print(f"\n{BOLD}{CYAN}{msg}{RESET}")

# ── Expected schema ──────────────────────────────────────────────────
REQUIRED_COLS = ["batter_name", "zone", "pitch_group", "avg_tilt", "swings"]
OPTIONAL_COLS = ["pitch_type", "avg_aa", "avg_bat_speed", "avg_swing_len",
                 "batting_avg", "xwoba", "avg_exit_velocity", "avg_launch_angle"]

PLAUSIBLE_RANGES = {
    "avg_tilt":          (0,   90),
    "avg_aa":            (-45, 45),
    "avg_bat_speed":     (40,  120),
    "avg_swing_len":     (2,   15),
    "batting_avg":       (0,   1),
    "xwoba":             (0,   1.2),
    "avg_exit_velocity": (40,  130),
    "avg_launch_angle":  (-90, 90),
    "swings":            (1,   5000),
    "zone":              (1,   14),
}


def check_season(data_dir: Path, season: int) -> tuple[bool, int]:
    """Returns (has_error, warning_count)."""
    has_error = False
    warn_count = 0

    hdr(f"Season {season}")

    # 1. File existence
    csv_path = data_dir / f"detail_zone_pitchgroup_{season}.csv"
    pq_path  = data_dir / f"detail_zone_pitchgroup_{season}.parquet"

    if pq_path.exists():
        ok(f"Parquet found: {pq_path.name}  ({pq_path.stat().st_size/1024:.0f} KB)")
        path = pq_path
    elif csv_path.exists():
        ok(f"CSV found:     {csv_path.name}  ({csv_path.stat().st_size/1024:.0f} KB)")
        path = csv_path
    else:
        warn(f"No detail file for {season} — skipping season")
        warn_count += 1
        return has_error, warn_count

    # 2. Load
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    ok(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

    # 3. Duplicate columns
    dupes = df.columns[df.columns.duplicated()].tolist()
    if dupes:
        err(f"Duplicate column names: {dupes}  — will break narwhals/plotly")
        has_error = True
    else:
        ok("No duplicate column names")

    # 4. Required columns
    missing_req = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_req:
        err(f"Missing required columns: {missing_req}")
        has_error = True
    else:
        ok(f"All {len(REQUIRED_COLS)} required columns present")

    present_opt = [c for c in OPTIONAL_COLS if c in df.columns]
    absent_opt  = [c for c in OPTIONAL_COLS if c not in df.columns]
    ok(f"Optional columns present: {present_opt}")
    if absent_opt:
        warn(f"Optional columns absent:  {absent_opt}")
        warn_count += len(absent_opt)

    # 5. Numeric coercion check
    print()
    print(f"  {'Column':<24} {'dtype':<12} {'nulls':>6}  {'%null':>6}  range")
    print(f"  {'─'*24} {'─'*12} {'─'*6}  {'─'*6}  {'─'*20}")
    for col in REQUIRED_COLS + [c for c in OPTIONAL_COLS if c in df.columns]:
        if col == "batter_name": continue
        series = pd.to_numeric(df[col], errors="coerce")
        n_null = series.isna().sum()
        pct    = 100 * n_null / max(len(df), 1)
        lo, hi = series.min(), series.max()
        range_str = f"{lo:.1f} – {hi:.1f}" if not np.isnan(lo) else "all null"
        flag = ""
        if pct > 20:  flag = f"  {YELLOW}⚠ high nulls{RESET}"
        if pct == 100: flag = f"  {RED}✗ all null!{RESET}"; has_error = True
        print(f"  {col:<24} {str(series.dtype):<12} {n_null:>6}  {pct:>5.1f}%  {range_str}{flag}")

    # 6. Plausibility checks
    print()
    for col, (lo_ok, hi_ok) in PLAUSIBLE_RANGES.items():
        if col not in df.columns: continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        out = ((series < lo_ok) | (series > hi_ok)).sum()
        if out > 0:
            warn(f"{col}: {out} values outside plausible range [{lo_ok}, {hi_ok}]")
            warn_count += 1
        else:
            ok(f"{col}: all values in plausible range [{lo_ok}, {hi_ok}]")

    # 7. Duplicate key rows
    key_cols = [c for c in ["batter_name", "zone", "pitch_group", "pitch_type"] if c in df.columns]
    dup_rows = df.duplicated(subset=key_cols).sum()
    if dup_rows > 0:
        warn(f"{dup_rows} duplicate (batter × zone × pitch_group) rows")
        warn_count += 1
    else:
        ok("No duplicate (batter × zone × pitch_group) rows")

    # 8. Pitcher contamination
    import re
    pitcher_re = re.compile(r" pitcher| P$", re.IGNORECASE)
    pitchers = df["batter_name"].str.contains(pitcher_re, na=False).sum()
    if pitchers > 0:
        warn(f"{pitchers} rows with pitcher names still present (will be filtered at runtime)")
        warn_count += 1
    else:
        ok("No pitcher contamination detected")

    # 9. Batter count
    n_batters = df["batter_name"].nunique()
    ok(f"{n_batters} unique batters")
    if n_batters < 50:
        warn(f"Very few batters ({n_batters}) — is this a full-season file?")
        warn_count += 1

    return has_error, warn_count


def check_cross_season_overlap(data_dir: Path, seasons: list[int]):
    """Report how many batters appear across multiple seasons."""
    hdr("Cross-Season Batter Overlap")
    season_batters: dict[int, set] = {}

    for s in seasons:
        path = data_dir / f"detail_zone_pitchgroup_{s}.csv"
        pq   = data_dir / f"detail_zone_pitchgroup_{s}.parquet"
        f = pq if pq.exists() else (path if path.exists() else None)
        if f is None:
            continue
        df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
        season_batters[s] = set(df["batter_name"].dropna().unique())

    s_list = sorted(season_batters.keys())
    for i in range(len(s_list)):
        for j in range(i+1, len(s_list)):
            a, b = s_list[i], s_list[j]
            overlap = len(season_batters[a] & season_batters[b])
            only_a  = len(season_batters[a] - season_batters[b])
            only_b  = len(season_batters[b] - season_batters[a])
            ok(f"{a} ∩ {b}: {overlap} shared batters  "
               f"(only in {a}: {only_a},  only in {b}: {only_b})")


def main():
    parser = argparse.ArgumentParser(description="Verify MLB dashboard data files")
    parser.add_argument("--dir",     default=".", help="Data directory path")
    parser.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    args = parser.parse_args()

    data_dir = Path(args.dir)
    print(f"\n{BOLD}MLB Swing Intelligence — Data Verification{RESET}")
    print(f"Directory: {data_dir.resolve()}")
    print("=" * 55)

    total_errors   = 0
    total_warnings = 0

    for season in args.seasons:
        has_err, n_warn = check_season(data_dir, season)
        if has_err: total_errors += 1
        total_warnings += n_warn

    check_cross_season_overlap(data_dir, args.seasons)

    # Final report
    print(f"\n{'='*55}")
    if total_errors == 0:
        print(f"{GREEN}{BOLD}✓ All checks passed{RESET}  "
              f"(warnings: {total_warnings})")
    else:
        print(f"{RED}{BOLD}✗ {total_errors} season(s) have errors{RESET}  "
              f"(warnings: {total_warnings})")
        print(f"{RED}Fix errors before deploying to Streamlit Cloud.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
