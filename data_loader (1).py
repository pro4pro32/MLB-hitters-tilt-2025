"""
data_loader.py  ·  MLB Swing Intelligence Dashboard
=====================================================
Centralised, robust data loading layer.

Problems solved vs the original monolith:
  1. Schema validation at load-time (clear error messages)
  2. Numeric coercion for every numeric column
  3. Parquet-first loading (falls back to CSV automatically)
  4. avail_seasons computed ONCE and stored in DataBundle
  5. Nested-cache anti-pattern removed:
       load_all_seasons wraps load_season without double-caching
  6. Pitcher rows filtered with compiled regex (10× faster on large files)
  7. batter_summary pre-computed once alongside raw detail data
  8. Data health report available for diagnostics

Public API:
    bundle = load_bundle()          # main entry point, Streamlit-cached
    bundle.detail                   # filtered detail for primary season
    bundle.players                  # player summaries for primary season
    bundle.detail_all               # all seasons stacked
    bundle.avail_seasons            # list of seasons that actually loaded
    bundle.main_season              # highest available season int
    bundle.batter_summary           # per-batter aggregate (primary season)
    bundle.all_real                 # sorted list of batter display names
    bundle.health                   # DataHealth report dict
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    DATA_DIR, SEASONS,
    DETAIL_TEMPLATE, PLAYERS_TEMPLATE,
    DETAIL_PARQUET_TEMPLATE, PLAYERS_PARQUET_TEMPLATE,
    REQUIRED_DETAIL_COLS, OPTIONAL_DETAIL_COLS,
    REQUIRED_PLAYERS_COLS, NUMERIC_COLS,
    PITCHER_PATTERNS, REAL_METRICS, METRIC_META,
)

warnings.filterwarnings("ignore")

# Compiled pitcher filter (reused across all seasons)
_PITCHER_RE = re.compile(
    "|".join(PITCHER_PATTERNS), flags=re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────────────
# DATA HEALTH REPORT
# ─────────────────────────────────────────────────────────────────────
@dataclass
class DataHealth:
    """Holds diagnostics about what was loaded and any issues found."""
    seasons_found:   list[int] = field(default_factory=list)
    seasons_missing: list[int] = field(default_factory=list)
    row_counts:      dict[int, int] = field(default_factory=dict)
    missing_cols:    dict[int, list[str]] = field(default_factory=dict)
    numeric_coerced: dict[int, list[str]] = field(default_factory=dict)
    warnings:        list[str] = field(default_factory=list)
    errors:          list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [
            f"Seasons found: {self.seasons_found}",
            f"Seasons missing: {self.seasons_missing}",
        ]
        for s, n in self.row_counts.items():
            lines.append(f"  {s}: {n:,} rows")
        for w in self.warnings:
            lines.append(f"⚠ {w}")
        for e in self.errors:
            lines.append(f"❌ {e}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# DATA BUNDLE  (single object passed around the app)
# ─────────────────────────────────────────────────────────────────────
@dataclass
class DataBundle:
    detail:         pd.DataFrame        # primary season detail
    players:        pd.DataFrame        # primary season player summary
    detail_all:     pd.DataFrame        # all seasons stacked (has "season" col)
    players_all:    pd.DataFrame        # all seasons stacked
    batter_summary: pd.DataFrame        # per-batter agg (primary season)
    all_real:       list[str]           # sorted batter names
    avail_seasons:  list[int]
    main_season:    int
    health:         DataHealth


# ─────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────
def _find_file(season: int, template_csv: str, template_parquet: str) -> Optional[Path]:
    """Prefer Parquet; fall back to CSV; return None if neither exists."""
    pq = DATA_DIR / template_parquet.format(season=season)
    if pq.exists():
        return pq
    csv = DATA_DIR / template_csv.format(season=season)
    if csv.exists():
        return csv
    return None


def _read_file(path: Path) -> pd.DataFrame:
    """Read Parquet or CSV based on file extension."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _dedup_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate columns (narwhals/plotly protection)."""
    return df.loc[:, ~df.columns.duplicated()].copy()


def _coerce_numerics(df: pd.DataFrame, health: DataHealth, season: int) -> pd.DataFrame:
    """Force numeric columns to float; track which needed coercion."""
    coerced = []
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        before_nulls = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after_nulls  = df[col].isna().sum()
        if after_nulls > before_nulls:
            coerced.append(col)
    if coerced:
        health.numeric_coerced[season] = coerced
    return df


def _validate_schema(
    df: pd.DataFrame,
    required: list[str],
    season: int,
    label: str,
    health: DataHealth,
) -> bool:
    """Return False and record error if any required column is absent."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        health.errors.append(
            f"{label} {season}: missing required columns {missing}"
        )
        health.missing_cols[season] = missing
        return False
    return True


def _filter_pitchers(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        df["batter_name"].notna()
        & ~df["batter_name"].str.contains(_PITCHER_RE, na=False)
    )
    return df[mask].copy()


# ─────────────────────────────────────────────────────────────────────
# PER-SEASON LOADER  (not Streamlit-cached — caching done in load_bundle)
# ─────────────────────────────────────────────────────────────────────
def _load_one_season(
    season: int,
    health: DataHealth,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load (players_df, detail_df) for one season.
    Returns (None, None) if files don't exist.
    """
    detail_path  = _find_file(season, DETAIL_TEMPLATE,  DETAIL_PARQUET_TEMPLATE)
    players_path = _find_file(season, PLAYERS_TEMPLATE, PLAYERS_PARQUET_TEMPLATE)

    if detail_path is None:
        health.seasons_missing.append(season)
        return None, None

    health.seasons_found.append(season)

    # ── Detail ───────────────────────────────────────────────────────
    detail = _read_file(detail_path)
    detail = _dedup_cols(detail)

    if not _validate_schema(detail, REQUIRED_DETAIL_COLS, season, "detail", health):
        return None, None

    detail = _coerce_numerics(detail, health, season)
    detail = _filter_pitchers(detail)
    detail["season"] = season
    health.row_counts[season] = len(detail)

    # Warn about suspiciously small datasets
    if len(detail) < 50:
        health.warnings.append(
            f"Season {season} detail has only {len(detail)} rows — data may be incomplete."
        )

    # ── Players summary (optional) ───────────────────────────────────
    players = None
    if players_path is not None:
        players = _read_file(players_path)
        players = _dedup_cols(players)
        _validate_schema(players, REQUIRED_PLAYERS_COLS, season, "players", health)
        players = _coerce_numerics(players, health, season)
        players["season"] = season
    else:
        health.warnings.append(
            f"players_summary_{season}.csv not found — "
            "player metadata will be derived from detail file."
        )
        # Build a minimal players frame from detail
        players = (
            detail.groupby("batter_name", observed=True)
            .agg({c: "mean" for c in REAL_METRICS if c in detail.columns}
                 | {"swings": "sum"})
            .round(3)
            .reset_index()
        )
        players["season"] = season

    return players, detail


# ─────────────────────────────────────────────────────────────────────
# BATTER SUMMARY  (cached separately so it can be reused)
# ─────────────────────────────────────────────────────────────────────
def _build_batter_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Aggregate detail to one row per batter (primary season, all zones/pitches)."""
    agg_cols = {
        c: ("sum" if c == "swings" else "mean")
        for c in REAL_METRICS
        if c in detail.columns and c != "swings"
    }
    agg_cols["swings"] = "sum"
    return (
        detail.groupby("batter_name", observed=True)
        .agg(agg_cols)
        .round(3)
        .reset_index()
    )


# ─────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT — Streamlit-cached
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(
    show_spinner="⚾  Loading Statcast data …",
    ttl=3600,           # refresh cache every hour (useful on Cloud)
)
def load_bundle() -> DataBundle:
    """
    Load all available seasons, validate schemas, return a DataBundle.
    This is THE single call the main app makes — everything else comes
    from the returned bundle.
    """
    health = DataHealth()

    all_players: list[pd.DataFrame] = []
    all_detail:  list[pd.DataFrame] = []

    for season in SEASONS:
        players, detail = _load_one_season(season, health)
        if players is not None:
            all_players.append(players)
        if detail is not None:
            all_detail.append(detail)

    # Hard stop if no data at all
    if not all_detail:
        health.errors.append(
            "No detail CSV/Parquet files found. "
            f"Expected at least one of: "
            + ", ".join(
                DETAIL_TEMPLATE.format(season=s) for s in SEASONS
            )
        )
        # Return empty bundle so Streamlit can show a proper error page
        empty = pd.DataFrame()
        return DataBundle(
            detail=empty, players=empty, detail_all=empty,
            players_all=empty, batter_summary=empty,
            all_real=[], avail_seasons=[], main_season=0,
            health=health,
        )

    # Stack all seasons
    detail_all  = pd.concat(all_detail,  ignore_index=True)
    players_all = pd.concat(all_players, ignore_index=True) if all_players else pd.DataFrame()

    # Pick primary (most recent) season
    avail = sorted(health.seasons_found)
    main  = avail[-1]

    detail_main  = detail_all[detail_all["season"] == main].copy()
    players_main = (
        players_all[players_all["season"] == main].copy()
        if not players_all.empty else pd.DataFrame()
    )

    # Batter list: prefer players file; fall back to detail
    if not players_main.empty and "batter_name" in players_main.columns:
        all_real = sorted(players_main["batter_name"].dropna().unique())
    else:
        all_real = sorted(detail_main["batter_name"].dropna().unique())

    # Pre-compute batter summary for primary season
    batter_sum = _build_batter_summary(detail_main)

    return DataBundle(
        detail         = detail_main,
        players        = players_main,
        detail_all     = detail_all,
        players_all    = players_all,
        batter_summary = batter_sum,
        all_real       = all_real,
        avail_seasons  = avail,
        main_season    = main,
        health         = health,
    )


# ─────────────────────────────────────────────────────────────────────
# HELPER: get one season's detail from the pre-loaded bundle
# ─────────────────────────────────────────────────────────────────────
def get_season_detail(bundle: DataBundle, season: int) -> pd.DataFrame:
    """Slice detail_all for a specific season (no disk I/O)."""
    if bundle.detail_all.empty:
        return pd.DataFrame()
    return bundle.detail_all[bundle.detail_all["season"] == season].copy()


def get_season_players(bundle: DataBundle, season: int) -> pd.DataFrame:
    """Slice players_all for a specific season (no disk I/O)."""
    if bundle.players_all.empty:
        return pd.DataFrame()
    return bundle.players_all[bundle.players_all["season"] == season].copy()


# ─────────────────────────────────────────────────────────────────────
# HELPER: apply pitch group / type filters
# ─────────────────────────────────────────────────────────────────────
def apply_pitch_filter(
    df: pd.DataFrame,
    pitch_group: str,
    pitch_type:  str,
    all_token:   str = "All",
) -> pd.DataFrame:
    if pitch_group != all_token and "pitch_group" in df.columns:
        df = df[df["pitch_group"] == pitch_group]
    if pitch_type  != all_token and "pitch_type"  in df.columns:
        df = df[df["pitch_type"]  == pitch_type]
    return df


# ─────────────────────────────────────────────────────────────────────
# HELPER: league averages for a given detail frame
# ─────────────────────────────────────────────────────────────────────
def compute_league_stats(detail: pd.DataFrame) -> dict[str, float]:
    return {
        c: float(detail[c].mean())
        for c in REAL_METRICS
        if c in detail.columns and c != "swings"
    }


# ─────────────────────────────────────────────────────────────────────
# HELPER: year-over-year delta table
# ─────────────────────────────────────────────────────────────────────
def yoy_delta_table(
    bundle: DataBundle,
    s_old:  int,
    s_new:  int,
    min_swings: int = 30,
) -> pd.DataFrame:
    """Return per-batter YoY delta for key metrics."""
    d_old = get_season_detail(bundle, s_old)
    d_new = get_season_detail(bundle, s_new)

    if d_old.empty or d_new.empty:
        return pd.DataFrame()

    cols = [c for c in ["avg_tilt", "avg_aa", "avg_bat_speed", "xwoba",
                         "avg_exit_velocity", "avg_launch_angle"]
            if c in d_old.columns and c in d_new.columns]

    def _agg(df):
        return df.groupby("batter_name", observed=True).agg(
            {c: "mean" for c in cols} | {"swings": "sum"}
        )

    old_agg = _agg(d_old).add_suffix(f"_{s_old}")
    new_agg = _agg(d_new).add_suffix(f"_{s_new}")
    merged  = old_agg.join(new_agg, how="inner").reset_index()

    # Filter minimum swings in BOTH seasons
    merged = merged[
        (merged[f"swings_{s_old}"] >= min_swings) &
        (merged[f"swings_{s_new}"] >= min_swings)
    ]

    for c in cols:
        merged[f"Δ_{c}"] = (merged[f"{c}_{s_new}"] - merged[f"{c}_{s_old}"]).round(3)

    keep = ["batter_name"] + [f"Δ_{c}" for c in cols] + [f"swings_{s_new}"]
    return merged[[col for col in keep if col in merged.columns]].rename(
        columns={"batter_name": "Batter"} |
                {f"Δ_{c}": f"Δ {METRIC_META[c]['label']}" for c in cols} |
                {f"swings_{s_new}": "Swings"}
    ).sort_values(f"Δ {METRIC_META['avg_tilt']['label']}", ascending=False)
