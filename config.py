"""
config.py  ·  MLB Swing Intelligence Dashboard
================================================
Single source of truth for every constant, metric definition,
data schema, and colour used across the app.

Import with:
    from config import SEASONS, METRIC_META, REQUIRED_DETAIL_COLS, ...
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# FILE SYSTEM
# ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(".")           # Streamlit Cloud: repo root

# Season list — add future seasons here; the loader skips missing files
SEASONS: list[int] = [2024, 2025, 2026]

# File-name templates  (formatted with season int)
DETAIL_TEMPLATE  = "detail_zone_pitchgroup_{season}.csv"
PLAYERS_TEMPLATE = "players_summary_{season}.csv"

# Also try Parquet variants (faster; loader prefers them over CSV)
DETAIL_PARQUET_TEMPLATE  = "detail_zone_pitchgroup_{season}.parquet"
PLAYERS_PARQUET_TEMPLATE = "players_summary_{season}.parquet"

# ─────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────
SHRINKAGE_K   = 50      # Swings needed for 50 % weight on player estimate
TILT_MIN      = 8.0
TILT_MAX      = 62.0
TILT_GRID_N   = 80
MIN_SWINGS_MODEL = 5    # Rows with fewer swings are excluded from model training

FEATURE_COLS: list[str] = [
    "avg_tilt", "avg_aa", "avg_bat_speed", "avg_swing_len",
    "zone_enc", "group_enc", "tilt_x_aa", "tilt_x_group",
]

# ─────────────────────────────────────────────────────────────────────
# DATA SCHEMA  (minimum required columns)
# ─────────────────────────────────────────────────────────────────────
# Loader validates these exist; missing optional cols are silently skipped.

REQUIRED_DETAIL_COLS: list[str] = [
    "batter_name",
    "zone",
    "pitch_group",
    "avg_tilt",
    "swings",
]

OPTIONAL_DETAIL_COLS: list[str] = [
    "pitch_type",
    "avg_aa",
    "avg_bat_speed",
    "avg_swing_len",
    "batting_avg",
    "xwoba",
    "avg_exit_velocity",
    "avg_launch_angle",
    "batter_id",          # Only present in some exports
]

REQUIRED_PLAYERS_COLS: list[str] = [
    "batter_name",
]

# Columns that should be numeric (coerced with pd.to_numeric)
NUMERIC_COLS: list[str] = [
    "avg_tilt", "avg_aa", "avg_bat_speed", "avg_swing_len",
    "swings", "batting_avg", "xwoba",
    "avg_exit_velocity", "avg_launch_angle",
    "zone",
]

# Pitcher filter patterns applied to batter_name
PITCHER_PATTERNS: list[str] = [r" pitcher", r" P$"]

# ─────────────────────────────────────────────────────────────────────
# METRIC METADATA
# Drives: labels, colour coding, heatmap scale, formatting
# ─────────────────────────────────────────────────────────────────────
METRIC_META: dict[str, dict] = {
    "avg_tilt": {
        "label":   "Tilt (°)",
        "good":    "high",
        "elite":   (42, 999),   # value range that = elite (green)
        "warn":    (0,  20),    # value range that = concern (red)
        "fmt":     ".1f",
        "hm_range":(8, 60),
        "cmap":    "YlOrRd",
    },
    "avg_aa": {
        "label":   "Attack Angle (°)",
        "good":    "mid",
        "elite":   (8, 15),
        "warn":    (-99, 0),
        "fmt":     ".1f",
        "hm_range":(-35, 35),
        "cmap":    "RdYlGn",
    },
    "avg_bat_speed": {
        "label":   "Bat Speed (mph)",
        "good":    "high",
        "elite":   (76, 999),
        "warn":    (0, 68),
        "fmt":     ".1f",
        "hm_range":(55, 88),
        "cmap":    "YlOrRd",
    },
    "avg_swing_len": {
        "label":   "Swing Length (ft)",
        "good":    "mid",
        "elite":   (6.5, 8.0),
        "warn":    (0, 5.5),
        "fmt":     ".2f",
        "hm_range":(4.5, 9.5),
        "cmap":    "YlOrRd",
    },
    "batting_avg": {
        "label":   "Batting Average",
        "good":    "high",
        "elite":   (0.290, 9),
        "warn":    (0, 0.215),
        "fmt":     ".3f",
        "hm_range":(0.150, 0.400),
        "cmap":    "RdYlGn",
    },
    "xwoba": {
        "label":   "xwOBA",
        "good":    "high",
        "elite":   (0.380, 9),
        "warn":    (0, 0.275),
        "fmt":     ".3f",
        "hm_range":(0.200, 0.600),
        "cmap":    "RdYlGn",
    },
    "avg_exit_velocity": {
        "label":   "Exit Velocity (mph)",
        "good":    "high",
        "elite":   (92, 999),
        "warn":    (0, 86),
        "fmt":     ".1f",
        "hm_range":(75, 105),
        "cmap":    "YlOrRd",
    },
    "avg_launch_angle": {
        "label":   "Launch Angle (°)",
        "good":    "mid",
        "elite":   (10, 18),
        "warn":    (-99, -1),
        "fmt":     ".1f",
        "hm_range":(-15, 45),
        "cmap":    "RdYlGn",
    },
    "swings": {
        "label":   "Swings",
        "good":    "high",
        "elite":   (300, 9999),
        "warn":    (0, 30),
        "fmt":     ".0f",
        "hm_range":(0, 400),
        "cmap":    "Blues",
    },
    # Derived / heatmap-only metrics
    "tilt_std": {
        "label":   "Tilt StdDev",
        "good":    "low",
        "elite":   (0, 5),
        "warn":    (15, 999),
        "fmt":     ".1f",
        "hm_range":(0, 20),
        "cmap":    "YlOrRd",
    },
    "delta_tilt": {
        "label":   "Δ Tilt vs League",
        "good":    "high",
        "elite":   (8, 999),
        "warn":    (-999, -8),
        "fmt":     ".1f",
        "hm_range":(-20, 20),
        "cmap":    "RdBu_r",
    },
    "aa_std": {
        "label":   "AA StdDev",
        "good":    "low",
        "elite":   (0, 4),
        "warn":    (12, 999),
        "fmt":     ".1f",
        "hm_range":(0, 20),
        "cmap":    "YlOrRd",
    },
}

# Quick-access list: metrics that exist as real columns (not derived)
REAL_METRICS: list[str] = [
    m for m in METRIC_META
    if m not in ("tilt_std", "delta_tilt", "aa_std")
]

# ─────────────────────────────────────────────────────────────────────
# COLOURS  (Stadium Night palette)
# ─────────────────────────────────────────────────────────────────────
CLR = {
    "bg_deep":    "#0d1117",
    "bg_card":    "#161b22",
    "bg_hover":   "#1c2333",
    "border":     "#21262d",
    "border_soft":"#30363d",
    "amber":      "#f0a500",
    "blue":       "#388bfd",
    "green":      "#3fb950",
    "red":        "#f85149",
    "purple":     "#bc8cff",
    "text":       "#e6edf3",
    "text_muted": "#8b949e",
    "text_body":  "#c9d1d9",
}

PLOTLY_COLORS = [
    CLR["amber"], CLR["blue"], CLR["green"],
    CLR["red"],   CLR["purple"], "#58a6ff", "#56d364",
]

# ─────────────────────────────────────────────────────────────────────
# TILT PROFILE LABELS  (used in Player Card and Glossary)
# ─────────────────────────────────────────────────────────────────────
def tilt_profile_label(tilt: float) -> tuple[str, str]:
    """Returns (label, colour_hex) for a given tilt value."""
    if tilt >= 45: return "Elite Uppercut",  CLR["red"]
    if tilt >= 38: return "Power Tilt",      CLR["amber"]
    if tilt >= 28: return "Balanced",        CLR["green"]
    if tilt >= 18: return "Line Drive",      CLR["blue"]
    return               "Flat / GB",        CLR["text_muted"]

def aa_profile_label(aa: float) -> str:
    if aa >= 20: return "High — pop-up risk"
    if aa >= 8:  return "Optimal — lift zone"
    if aa >= 0:  return "Neutral — slight GB"
    return              "Negative — ground ball"
