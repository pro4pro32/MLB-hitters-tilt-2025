"""
MLB 2025 – Swing Path Tilt & Attack Angle Dashboard  v2.0
==========================================================
Professional-grade baseball analytics tool for scouting
and player development.

Key upgrades over v1:
  • GAM (pyGAM) or Gradient Boosting model layer
  • Optimal Tilt Simulator tab with PDP curves + CI bands
  • 2-D interaction heatmap (tilt × attack angle)
  • Player Tilt Rankings / optimization table
  • Bayesian shrinkage for small samples
  • Percentile-rank & shrunk-estimate heatmap modes
  • Improved caching, modularity, and UX
"""

# ─────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.stats import percentileofscore

try:
    from pygam import LinearGAM, s, f as gam_f
    HAS_PYGAM = True
except ImportError:
    HAS_PYGAM = False

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLB Bat Tracking 2025",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────
SHRINKAGE_K  = 50          # Equivalent swings for 50 % weight on league mean
TILT_MIN     = 8.0
TILT_MAX     = 62.0
TILT_GRID_N  = 80          # Points on the tilt sweep grid
DATA_DIR     = "."

FEATURE_COLS = [
    "avg_tilt", "avg_aa", "avg_bat_speed", "avg_swing_len",
    "zone_enc", "group_enc", "tilt_x_aa", "tilt_x_group",
]

METRIC_LABELS = {
    "avg_tilt":          "Swing Path Tilt (°)",
    "avg_aa":            "Attack Angle (°)",
    "avg_bat_speed":     "Bat Speed (mph)",
    "avg_swing_len":     "Swing Length (ft)",
    "batting_avg":       "Batting Average",
    "xwoba":             "xwOBA",
    "avg_exit_velocity": "Exit Velocity (mph)",
    "avg_launch_angle":  "Launch Angle (°)",
    "swings":            "Swings",
}

HEATMAP_RANGES = {
    "avg_tilt":          (8,   60),
    "tilt_std":          (0,   20),
    "delta_tilt":        (-20, 20),
    "avg_aa":            (-35, 35),
    "aa_std":            (0,   20),
    "avg_bat_speed":     (55,  88),
    "avg_swing_len":     (4.5, 9.5),
    "swings":            (0,   400),
    "batting_avg":       (0.150, 0.400),
    "xwoba":             (0.200, 0.600),
    "avg_exit_velocity": (75,  105),
    "avg_launch_angle":  (-15, 45),
}

# ─────────────────────────────────────────────────────────────────────
# TRANSLATIONS
# ─────────────────────────────────────────────────────────────────────
TEXTS = {
    "en": {
        "title":              "MLB 2025 – Swing Path Tilt & Attack Angle Dashboard",
        "subtitle":           "Scouting & Player Development Analytics Platform",
        "sidebar_players":    "Select Batters",
        "sidebar_min_swings": "Min. swings (tables)",
        "sidebar_pitch_group":"Pitch group",
        "sidebar_pitch_type": "Pitch type",
        "all":                "All",
        "league_avg":         "League Average",
        "tab_summary":        "📊 Summary",
        "tab_groups":         "📦 Group Comp.",
        "tab_heatmaps":       "🔥 Heatmaps",
        "tab_side_by_side":   "↔ Side-by-Side",
        "tab_player_compare": "🎯 Batter Metrics",
        "tab_tilt_sim":       "🔬 Tilt Optimizer",
        "tab_tilt_rankings":  "🏆 Tilt Rankings",
        "selected_players":   "**Selected Batters**",
        "all_players":        "**All Batters (post-filter)**",
        "no_player":          "Select at least one real batter",
        "no_data":            "No data after filters",
        "metric":             "Metric",
        "compare_left":       "Left Batter",
        "compare_right":      "Right Batter",
        "no_data_for":        "No data for",
        "detailed_table":     "Detailed Table",
        "select_player":      "Select Batter",
        "left_metric":        "Left Metric",
        "right_metric":       "Right Metric",
        "optimal_tilt":       "Optimal Tilt (shrunk)",
        "current_tilt":       "Current Avg Tilt",
        "tilt_delta":         "Δ Tilt  (Current − Optimal)",
        "pred_xwoba":         "Pred. xwOBA @ Optimal",
        "conf_weight":        "Confidence (0–1)",
        "heatmap_view":       "Heatmap view mode",
        "show_ci":            "Show 80 % confidence bands",
        "model_info":         "Model",
    },
    "pl": {
        "title":              "MLB 2025 – Swing Path Tilt & Attack Angle Dashboard",
        "subtitle":           "Platforma analityczna dla scoutów i development graczy",
        "sidebar_players":    "Wybierz batterów",
        "sidebar_min_swings": "Min. swingów (tabele)",
        "sidebar_pitch_group":"Grupa rzutów",
        "sidebar_pitch_type": "Typ rzutu",
        "all":                "Wszystkie",
        "league_avg":         "Średnia ligi",
        "tab_summary":        "📊 Podsumowanie",
        "tab_groups":         "📦 Grupy",
        "tab_heatmaps":       "🔥 Heatmapy",
        "tab_side_by_side":   "↔ Porównanie",
        "tab_player_compare": "🎯 Metryki",
        "tab_tilt_sim":       "🔬 Symulator",
        "tab_tilt_rankings":  "🏆 Ranking",
        "selected_players":   "**Wybrani batterzy**",
        "all_players":        "**Wszyscy batterzy (po filtrach)**",
        "no_player":          "Wybierz co najmniej jednego battera",
        "no_data":            "Brak danych po filtrach",
        "metric":             "Metryka",
        "compare_left":       "Batter lewy",
        "compare_right":      "Batter prawy",
        "no_data_for":        "Brak danych dla",
        "detailed_table":     "Tabela szczegółowa",
        "select_player":      "Wybierz battera",
        "left_metric":        "Metryka lewa",
        "right_metric":       "Metryka prawa",
        "optimal_tilt":       "Optymalny tilt (shrunk)",
        "current_tilt":       "Aktualny śr. tilt",
        "tilt_delta":         "Δ Tilt (aktualny − optymalny)",
        "pred_xwoba":         "Prognozowane xwOBA @ optimum",
        "conf_weight":        "Pewność (0–1)",
        "heatmap_view":       "Tryb widoku heatmapy",
        "show_ci":            "Pokaż 80 % przedziały ufności",
        "model_info":         "Model",
    },
}

# ─────────────────────────────────────────────────────────────────────
# SESSION STATE & LANGUAGE
# ─────────────────────────────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "en"

with st.sidebar:
    lang_sel = st.selectbox("Language / Język", ["English", "Polski"], index=0)
    new_lang = {"English": "en", "Polski": "pl"}[lang_sel]
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

t = TEXTS[st.session_state.lang]

# ─────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⚾ Loading Statcast data …")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    players = pd.read_csv(Path(DATA_DIR) / "players_summary_2025.csv")
    detail  = pd.read_csv(Path(DATA_DIR) / "detail_zone_pitchgroup_2025.csv")
    mask = (
        detail["batter_name"].notna()
        & ~detail["batter_name"].str.contains(
            r" pitcher| P$", case=False, na=False, regex=True
        )
    )
    return players, detail[mask].copy()

players_raw, detail_full = load_data()

# ─────────────────────────────────────────────────────────────────────
# PLAYER ID / DISPLAY-NAME RESOLUTION
# ─────────────────────────────────────────────────────────────────────
use_id        = False
id_col        = None
player_info   = None
display_to_id: dict = {}
id_to_display: dict = {}

for _cand in ("batter_id", "batter", "mlb_id", "player_id", "id"):
    if _cand in players_raw.columns and _cand in detail_full.columns:
        id_col = _cand
        use_id = True
        break

if use_id:
    player_info  = players_raw[[id_col, "batter_name"]].drop_duplicates()
    _name_counts = player_info["batter_name"].value_counts()
    _dupes       = _name_counts[_name_counts > 1].index.tolist()
    player_info["display_name"] = player_info.apply(
        lambda r: (
            f"{r['batter_name']} (ID:{int(r[id_col])})"
            if r["batter_name"] in _dupes else r["batter_name"]
        ), axis=1
    )
    display_to_id = dict(zip(player_info["display_name"], player_info[id_col]))
    id_to_display = dict(zip(player_info[id_col],         player_info["display_name"]))
    all_real      = sorted(player_info["display_name"])
else:
    all_real = sorted(players_raw["batter_name"].dropna().unique())

# ─────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (run once at startup)
# ─────────────────────────────────────────────────────────────────────
def _engineer_features(df: pd.DataFrame):
    """
    Add encoded categoricals and interaction terms.
    Returns (enriched_df, le_zone, le_group).
    """
    out = df.copy()
    le_zone  = LabelEncoder()
    le_group = LabelEncoder()
    out["zone_enc"]    = le_zone.fit_transform(out["zone"].astype(str))
    out["group_enc"]   = le_group.fit_transform(
        out["pitch_group"].fillna("Unknown").astype(str)
    )
    out["tilt_x_aa"]   = out["avg_tilt"] * out["avg_aa"]
    out["tilt_x_group"]= out["avg_tilt"] * out["group_enc"]
    out["sample_weight"]= np.sqrt(out["swings"].clip(lower=1))
    return out, le_zone, le_group

detail_fe, le_zone_g, le_group_g = _engineer_features(detail_full)

# ─────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Training swing model …")
def train_model():
    """
    Fit a GAM (preferred) or Gradient Boosting model.
    Target  : xwOBA
    Features: avg_tilt, avg_aa, avg_bat_speed, avg_swing_len,
              zone_enc, group_enc, tilt×aa, tilt×group
    Sample weights ~ sqrt(swings) to downplay small-sample cells.
    """
    df = detail_fe.dropna(subset=["xwoba"]).copy()
    df = df[df["swings"] >= 5]

    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median()).values
    y = df["xwoba"].values
    w = df["sample_weight"].values

    if HAS_PYGAM and len(X) >= 40:
        try:
            gam = LinearGAM(
                s(0, n_splines=12, constraints="none")   # tilt  – no monotone constraint
                + s(1, n_splines=10)                     # attack angle
                + s(2, n_splines=8)                      # bat speed
                + s(3, n_splines=6)                      # swing length
                + gam_f(4)                               # zone  (factor)
                + gam_f(5)                               # pitch group (factor)
                + s(6, n_splines=6)                      # tilt × aa
                + s(7, n_splines=6),                     # tilt × group
                fit_intercept=True,
            )
            gam.gridsearch(X, y, weights=w, progress=False)
            return gam, "GAM (pyGAM)"
        except Exception:
            pass   # fall through to GBM

    gbm = GradientBoostingRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.75,
        min_samples_leaf=4,
        max_features=0.8,
        random_state=42,
    )
    gbm.fit(X, y, sample_weight=w)
    return gbm, "Gradient Boosting (sklearn)"

model, model_type = train_model()

# ─────────────────────────────────────────────────────────────────────
# SHRINKAGE UTILITIES
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _league_stats(df: pd.DataFrame) -> dict:
    return {c: float(df[c].mean()) for c in
            ["avg_tilt", "avg_aa", "avg_bat_speed", "avg_swing_len", "xwoba"]}

league_stats = _league_stats(detail_full)


def shrink(player_val: float, league_val: float,
           n: int, k: float = SHRINKAGE_K) -> tuple[float, float]:
    """
    Bayesian (James-Stein-style) shrinkage toward league mean.
      shrunk = w * player + (1-w) * league,  w = n / (n + k)
    Returns (shrunk_value, confidence_weight).
    """
    w = n / (n + k)
    return w * player_val + (1 - w) * league_val, round(w, 3)


# ─────────────────────────────────────────────────────────────────────
# ENCODE HELPERS
# ─────────────────────────────────────────────────────────────────────
def _enc_zone(zone) -> float:
    s = str(zone)
    return float(le_zone_g.transform([s])[0]) if s in le_zone_g.classes_ else 0.0


def _enc_group(group: str) -> float:
    return float(le_group_g.transform([group])[0]) if group in le_group_g.classes_ else 0.0


# ─────────────────────────────────────────────────────────────────────
# TILT CURVE PREDICTION
# ─────────────────────────────────────────────────────────────────────
def predict_tilt_curve(
    avg_aa: float, avg_speed: float, avg_len: float,
    zone_enc: float, group_enc: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Partial-dependence style: sweep tilt, hold everything else fixed.
    Returns (tilt_grid, smoothed_xwoba_predictions).
    """
    tg = np.linspace(TILT_MIN, TILT_MAX, TILT_GRID_N)
    X  = np.column_stack([
        tg,
        np.full(TILT_GRID_N, avg_aa),
        np.full(TILT_GRID_N, avg_speed),
        np.full(TILT_GRID_N, avg_len),
        np.full(TILT_GRID_N, zone_enc),
        np.full(TILT_GRID_N, group_enc),
        tg * avg_aa,      # tilt_x_aa
        tg * group_enc,   # tilt_x_group
    ])
    raw = model.predict(X)
    return tg, gaussian_filter1d(raw, sigma=1.8)


def find_optimal_tilt(tg: np.ndarray, preds: np.ndarray) -> tuple[float, float]:
    idx = int(np.argmax(preds))
    return float(tg[idx]), float(preds[idx])


def approximate_ci(
    avg_aa: float, avg_speed: float, avg_len: float,
    zone_enc: float, group_enc: float,
    n_swings: int, n_boot: int = 150,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo CI: add feature noise inversely proportional to sample size.
    Returns (p10, p90) arrays over TILT_GRID_N.
    """
    scale = 1.0 / np.sqrt(max(n_swings, 1) / SHRINKAGE_K)
    rng   = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        _, p = predict_tilt_curve(
            avg_aa    + rng.normal(0, 8  * scale),
            avg_speed + rng.normal(0, 3  * scale),
            avg_len   + rng.normal(0, 0.3* scale),
            zone_enc, group_enc,
        )
        boots.append(p)
    arr = np.vstack(boots)
    return np.percentile(arr, 10, axis=0), np.percentile(arr, 90, axis=0)


# ─────────────────────────────────────────────────────────────────────
# FILTER HELPERS
# ─────────────────────────────────────────────────────────────────────
def apply_filters(df: pd.DataFrame, pitch_group: str, pitch_type: str) -> pd.DataFrame:
    if pitch_group != t["all"]:
        df = df[df["pitch_group"] == pitch_group]
    if pitch_type  != t["all"]:
        df = df[df["pitch_type"]  == pitch_type]
    return df


def get_player_zone_df(name: str, df_ctx: pd.DataFrame, league_z: pd.DataFrame) -> pd.DataFrame:
    if name == t["league_avg"]:
        return league_z.copy()
    agg_cols = {c: "mean" for c in METRIC_LABELS if c != "swings" and c in df_ctx.columns}
    agg_cols["swings"] = "sum"
    if use_id:
        pid = display_to_id.get(name)
        sub = df_ctx[df_ctx[id_col] == pid]
    else:
        sub = df_ctx[df_ctx["batter_name"] == name]
    if sub.empty:
        return pd.DataFrame()
    return sub.groupby("zone", as_index=False).agg(agg_cols).round(3)


def player_display_name(val) -> str:
    return id_to_display.get(val, str(val)) if use_id else str(val)


# ─────────────────────────────────────────────────────────────────────
# HEATMAP — zone layout helpers
# ─────────────────────────────────────────────────────────────────────
_ZONE_GEOM = {
    "border": 0.85, "main_size": 3.3, "split_y": 2.5,
}

def _zone_color(val, vmin, vmax, cmap, view):
    if pd.isna(val):
        return (0.91, 0.91, 0.91)
    normed = float(np.clip((val - vmin) / max(vmax - vmin, 1e-9), 0, 1))
    return cmap(normed)

def _text_color(bg):
    r, g, b = bg[:3]
    return "black" if 0.299*r + 0.587*g + 0.114*b > 0.45 else "white"

def _fmt(val, metric, view):
    if pd.isna(val):
        return "—"
    if view == "Percentile":
        return f"{int(round(val))}th"
    if metric == "swings":
        return f"{int(round(val))}"
    if metric in ("batting_avg", "xwoba"):
        return f"{val:.3f}"
    return f"{val:.1f}"


def _get_pivot(df_p: pd.DataFrame, metric: str, view: str,
               league_df: pd.DataFrame | None, df_ctx: pd.DataFrame) -> pd.Series:
    """Compute per-zone scalar values respecting the view mode."""

    if metric == "tilt_std":
        return df_p.groupby("zone")["avg_tilt"].std(ddof=1).round(1)
    if metric == "aa_std":
        return df_p.groupby("zone")["avg_aa"].std(ddof=1).round(1)
    if metric == "delta_tilt":
        pm = df_p.groupby("zone")["avg_tilt"].mean()
        lm = league_df.set_index("zone")["avg_tilt"] if league_df is not None else pd.Series()
        return (pm - lm.reindex(pm.index, fill_value=np.nan)).round(2)

    if metric == "swings":
        raw = df_p.groupby("zone")["swings"].sum()
    else:
        raw = df_p.groupby("zone")[metric].mean()

    if view == "Raw":
        return raw.round(3)

    if view == "Percentile":
        result = {}
        for zone, val in raw.items():
            zone_vals = df_ctx[df_ctx["zone"] == zone][metric].dropna()
            result[zone] = percentileofscore(zone_vals, val, kind="rank") if len(zone_vals) else np.nan
        return pd.Series(result)

    if view == "Shrunk":
        n_sw = df_p.groupby("zone")["swings"].sum()
        lg   = df_ctx.groupby("zone")[metric].mean() if metric != "swings" else df_ctx.groupby("zone")["swings"].sum()
        result = {}
        for zone, val in raw.items():
            n   = int(n_sw.get(zone, 0))
            l_v = float(lg.get(zone, df_ctx[metric].mean())) if zone in lg.index else float(df_ctx[metric].mean())
            result[zone] = round(shrink(val, l_v, n)[0], 3)
        return pd.Series(result)

    return raw.round(3)


# ─────────────────────────────────────────────────────────────────────
# HEATMAP RENDERER
# ─────────────────────────────────────────────────────────────────────
def make_heatmap(
    df_p: pd.DataFrame,
    metric: str,
    title: str,
    league_df: pd.DataFrame | None = None,
    view: str = "Raw",
    df_ctx: pd.DataFrame | None = None,
):
    if df_p is None or df_p.empty:
        st.warning(f"{t['no_data_for']} {title}")
        return

    if df_ctx is None:
        df_ctx = detail_full

    pivot = _get_pivot(df_p, metric, view, league_df, df_ctx)

    # Colour range
    if view == "Percentile" and metric not in ("tilt_std", "aa_std", "delta_tilt"):
        vmin, vmax, cmap_name = 0, 100, "RdYlGn"
    elif metric == "delta_tilt":
        vmin, vmax = HEATMAP_RANGES.get(metric, (-20, 20))
        cmap_name  = "RdBu_r"
    else:
        vmin, vmax = HEATMAP_RANGES.get(metric, (0, 100))
        cmap_name  = "YlOrRd"

    cmap = sns.color_palette(cmap_name, as_cmap=True)
    n_sw = df_p.groupby("zone")["swings"].sum()

    g  = _ZONE_GEOM
    b  = g["border"]; ms = g["main_size"]; sy = g["split_y"]
    mx, my = b, b
    tx, ty = mx + ms, my + ms
    half   = ms / 2

    fig, ax = plt.subplots(figsize=(8.2, 8.2))

    # 3 × 3 strike-zone grid
    for i in range(3):
        for j in range(3):
            zone = i * 3 + j + 1
            val  = pivot.get(zone, np.nan)
            n    = int(n_sw.get(zone, 0))
            x    = mx + j * (ms / 3)
            y    = my + (2 - i) * (ms / 3)
            col  = _zone_color(val, vmin, vmax, cmap, view)
            ax.add_patch(plt.Rectangle(
                (x, y), ms/3, ms/3,
                facecolor=col, edgecolor="black", linewidth=2.4
            ))
            txt = f"{zone}\n{_fmt(val, metric, view)}"
            if n < 20 and view != "Percentile":
                txt += f"\n⚠ n={n}"
            ax.text(
                x + ms/6, y + ms/6, txt,
                ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=_text_color(col),
            )

    # L-shaped corner zones 11-14
    l_zones = [
        (11, [(0, sy), (b, sy), (b, ty), (mx, ty),
               (mx+half, ty), (mx+half, 5), (0, 5), (0, sy)]),
        (12, [(tx, sy), (tx, ty), (mx+half, ty),
               (mx+half, 5), (5, 5), (5, sy), (tx, sy)]),
        (13, [(0, sy), (b, sy), (b, my), (mx, my),
               (mx+half, my), (mx+half, 0), (0, 0), (0, sy)]),
        (14, [(tx, sy), (tx, my), (mx+half, my),
               (mx+half, 0), (5, 0), (5, sy), (tx, sy)]),
    ]
    off = b * 0.4
    corners = [(off, 5-off), (5-off, 5-off), (off, off), (5-off, off)]
    for (z, verts), (cx, cy) in zip(l_zones, corners):
        val = pivot.get(z, np.nan)
        n   = int(n_sw.get(z, 0))
        col = _zone_color(val, vmin, vmax, cmap, view)
        ax.add_patch(PathPatch(MPath(verts),
                               facecolor=col, edgecolor="black", linewidth=2.4))
        txt = f"{z}\n{_fmt(val, metric, view)}"
        if n < 20 and view != "Percentile":
            txt += f"\n⚠ n={n}"
        ax.text(cx, cy, txt, ha="center", va="center",
                fontsize=10.5, fontweight="bold")

    # Strike-zone border
    ax.add_patch(plt.Rectangle(
        (mx, my), ms, ms, fill=False, edgecolor="red", linewidth=3.8
    ))

    view_tag = {"Percentile": " [Percentile]", "Shrunk": " [Shrunk]"}.get(view, "")
    ax.set_title(f"{title}{view_tag}", fontsize=15, pad=18, fontweight="bold")
    ax.set_xlim(0, 5); ax.set_ylim(0, 5)
    ax.set_aspect("equal"); ax.axis("off")

    lbl_map = {
        "avg_tilt": "TILT (°)", "tilt_std": "TILT STD", "delta_tilt": "Δ TILT",
        "avg_aa": "ATTACK ANGLE (°)", "aa_std": "AA STD", "avg_bat_speed": "BAT SPEED (mph)",
        "avg_swing_len": "SWING LEN (ft)", "swings": "SWINGS",
        "batting_avg": "BATTING AVG", "xwoba": "xwOBA",
        "avg_exit_velocity": "EXIT VELO (mph)", "avg_launch_angle": "LAUNCH ANGLE (°)",
    }
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, ax=ax, shrink=0.76, pad=0.04)
    cbar.set_label(lbl_map.get(metric, metric.upper()), fontsize=11)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────────────
st.title(t["title"])
st.caption(f"🤖 {t['model_info']}: **{model_type}** · Shrinkage K = {SHRINKAGE_K} swings")

with st.sidebar:
    st.markdown("### ⚾ Batter Selection")
    player_opts = [t["league_avg"]] + ["─" * 22] + all_real
    selected_players_multi = st.multiselect(
        t["sidebar_players"],
        options=player_opts,
        default=[all_real[0]] if all_real else [],
        max_selections=8,
    )
    selected_players_multi = [p for p in selected_players_multi if "─" not in p]

    st.markdown("### 🎛 Filters")
    min_swings = st.slider(t["sidebar_min_swings"], 0, 300, 0, 10)

    pitch_groups = [t["all"]] + sorted(detail_full["pitch_group"].dropna().unique())
    selected_pitch = st.selectbox(t["sidebar_pitch_group"], pitch_groups)

    _pt_src = detail_full if selected_pitch == t["all"] \
              else detail_full[detail_full["pitch_group"] == selected_pitch]
    pitch_types = [t["all"]] + sorted(_pt_src["pitch_type"].dropna().unique())
    selected_type = st.selectbox(t["sidebar_pitch_type"], pitch_types)

    st.markdown("### 🔧 Display")
    view_mode = st.radio(
        t["heatmap_view"],
        ["Raw", "Percentile", "Shrunk"],
        index=0,
        help=(
            "**Raw** = average value · "
            "**Percentile** = rank vs all batters in that zone · "
            "**Shrunk** = Bayesian blend toward league mean"
        ),
    )
    show_ci = st.checkbox(t["show_ci"], value=True)

# Apply global filters
df_filtered = apply_filters(detail_full, selected_pitch, selected_type)
df_fe_filt  = apply_filters(detail_fe,   selected_pitch, selected_type)

# League averages per zone (used by heatmaps & model)
_lagg = {c: "mean" for c in METRIC_LABELS if c != "swings" and c in df_filtered.columns}
_lagg["swings"] = "sum"
league_per_zone = df_filtered.groupby("zone", as_index=False, observed=True).agg(_lagg).round(3)
league_per_zone["batter_name"] = t["league_avg"]

# Selected batter helpers
selected_display   = [p for p in selected_players_multi if p != t["league_avg"]]
player_filter_values = (
    [display_to_id[p] for p in selected_display if p in display_to_id]
    if use_id else selected_display
)
player_filter_col  = id_col if use_id else "batter_name"

detail_tables = df_filtered[df_filtered["swings"] >= min_swings].copy()


# ─────────────────────────────────────────────────────────────────────
# OPTIMISATION TABLE (computed once, used in Tabs 6 & 7)
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⚙️ Computing tilt optimizations …", ttl=3600)
def compute_opt_table(
    _df_fe: pd.DataFrame,
    pitch_grp: str,
    pitch_typ: str,
    group_enc_val: float,
    zone_enc_val: float,
    league_opt_tilt: float,
) -> pd.DataFrame:
    """
    For each batter compute optimal tilt (shrunk), delta vs current, predicted gain.
    All features except tilt are held at the batter's own weighted averages.
    """
    grp = id_col if use_id else "batter_name"
    agg = _df_fe.groupby(grp, observed=True).agg(
        avg_tilt    = ("avg_tilt",       "mean"),
        avg_aa      = ("avg_aa",         "mean"),
        avg_bat_speed=("avg_bat_speed",  "mean"),
        avg_swing_len=("avg_swing_len",  "mean"),
        xwoba       = ("xwoba",          "mean"),
        swings      = ("swings",         "sum"),
    ).reset_index().dropna(subset=["avg_tilt", "avg_aa", "avg_bat_speed", "avg_swing_len"])

    rows = []
    for _, r in agg.iterrows():
        n    = int(r.swings)
        if n < 5:
            continue
        tg, preds = predict_tilt_curve(
            r.avg_aa, r.avg_bat_speed, r.avg_swing_len,
            zone_enc_val, group_enc_val,
        )
        raw_opt, opt_xwoba = find_optimal_tilt(tg, preds)
        shrunk_opt, conf_w = shrink(raw_opt, league_opt_tilt, n)

        rows.append({
            "Batter":             player_display_name(r[grp]),
            "Swings":             n,
            "Current Tilt":       round(float(r.avg_tilt),  1),
            "Optimal Tilt (raw)": round(raw_opt,             1),
            "Optimal Tilt":       round(shrunk_opt,          1),
            "Δ Tilt":             round(float(r.avg_tilt) - shrunk_opt, 1),
            "Pred. xwOBA @ Opt.": round(opt_xwoba,           3),
            "Confidence":         round(conf_w,               2),
            "Current xwOBA":      round(float(r.xwoba), 3) if not np.isnan(r.xwoba) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("Swings", ascending=False)


# Pre-compute league optimal (all zones, all groups)
_tg_lg, _pr_lg  = predict_tilt_curve(
    league_stats["avg_aa"], league_stats["avg_bat_speed"], league_stats["avg_swing_len"],
    zone_enc_val=float(np.median(detail_fe["zone_enc"])),
    group_enc_val=float(np.median(detail_fe["group_enc"])),
)
LEAGUE_OPT_TILT, LEAGUE_OPT_XWOBA = find_optimal_tilt(_tg_lg, _pr_lg)

# Filtered-context group/zone encodings (median of encoded values)
_ctx_group_enc = float(np.median(df_fe_filt["group_enc"])) if not df_fe_filt.empty else 0.0
_ctx_zone_enc  = float(np.median(df_fe_filt["zone_enc"]))  if not df_fe_filt.empty else 0.0

opt_table = compute_opt_table(
    df_fe_filt,
    selected_pitch, selected_type,
    _ctx_group_enc, _ctx_zone_enc,
    LEAGUE_OPT_TILT,
)


# ─────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    t["tab_summary"], t["tab_groups"], t["tab_heatmaps"],
    t["tab_side_by_side"], t["tab_player_compare"],
    t["tab_tilt_sim"], t["tab_tilt_rankings"],
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 – SUMMARY
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader(t["tab_summary"])

    _sum_cols = ["batter_name", "avg_tilt", "avg_aa", "avg_bat_speed",
                 "avg_swing_len", "swings", "batting_avg", "xwoba",
                 "avg_exit_velocity", "avg_launch_angle"]
    _agg_fn   = {c: "mean" for c in _sum_cols[1:] if c != "swings"}
    _agg_fn["swings"] = "sum"

    # Selected batters
    if selected_players_multi:
        st.markdown(t["selected_players"])
        sel_df = (
            df_filtered[df_filtered[player_filter_col].isin(player_filter_values)]
            if player_filter_values else pd.DataFrame()
        )

        if use_id and not sel_df.empty:
            sel_sum = sel_df.groupby(id_col, observed=True).agg(_agg_fn).round(3).reset_index()
            sel_sum["batter_name"] = sel_sum[id_col].map(id_to_display)
        else:
            sel_sum = sel_df.groupby("batter_name", observed=True).agg(_agg_fn).round(3).reset_index()

        sel_sum = sel_sum[[c for c in _sum_cols if c in sel_sum.columns]]

        if t["league_avg"] in selected_players_multi and not detail_tables.empty:
            la = detail_tables.mean(numeric_only=True).round(3).to_frame().T
            la["batter_name"] = t["league_avg"]
            la["swings"]      = int(detail_tables["swings"].sum())
            sel_sum = pd.concat(
                [sel_sum, la[[c for c in _sum_cols if c in la.columns]]],
                ignore_index=True
            )

        st.dataframe(
            sel_sum.sort_values("swings", ascending=False),
            use_container_width=True, hide_index=True,
        )
        st.markdown("---")

    # All batters
    st.markdown(t["all_players"])
    if not detail_tables.empty:
        if use_id:
            all_sum = detail_tables.groupby(id_col, observed=True).agg(_agg_fn).round(3).reset_index()
            all_sum["batter_name"] = all_sum[id_col].map(id_to_display)
        else:
            all_sum = detail_tables.groupby("batter_name", observed=True).agg(_agg_fn).round(3).reset_index()
        all_sum = all_sum[[c for c in _sum_cols if c in all_sum.columns]]
        st.dataframe(
            all_sum.sort_values("swings", ascending=False),
            use_container_width=True, hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════
# TAB 2 – GROUP COMPARISON
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(t["tab_groups"])
    if not selected_display:
        st.info(t["no_player"])
    elif df_filtered.empty:
        st.info(t["no_data"])
    else:
        metric_t2 = st.selectbox(
            t["metric"],
            [k for k in METRIC_LABELS],
            format_func=lambda x: METRIC_LABELS[x],
            key="t2_metric",
        )
        tmp = df_filtered[df_filtered[player_filter_col].isin(player_filter_values)].copy()
        tmp["player_display"] = (
            tmp[id_col].map(id_to_display) if use_id else tmp["batter_name"]
        )

        c1, c2 = st.columns(2)
        with c1:
            fig_box = px.box(
                tmp, x="pitch_group", y=metric_t2, color="player_display",
                points="outliers",
                labels={"pitch_group": "Pitch Group", metric_t2: METRIC_LABELS[metric_t2]},
                title=f"Distribution – {METRIC_LABELS[metric_t2]}",
            )
            st.plotly_chart(fig_box, use_container_width=True)
        with c2:
            avg_d = tmp.groupby(["player_display", "pitch_group"], as_index=False)[metric_t2].mean()
            fig_bar = px.bar(
                avg_d, x="pitch_group", y=metric_t2, color="player_display",
                barmode="group",
                labels={"pitch_group": "Pitch Group", metric_t2: METRIC_LABELS[metric_t2]},
                title=f"Average – {METRIC_LABELS[metric_t2]}",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Scatter: tilt vs selected metric
        st.markdown("#### Tilt vs Metric (scatter)")
        tmp2 = tmp.groupby("player_display", as_index=False).agg(
            avg_tilt=(metric_t2, "mean"), metric_val=(metric_t2, "mean"),
            avg_tilt2=("avg_tilt", "mean"),
        )
        fig_sc = px.scatter(
            tmp, x="avg_tilt", y=metric_t2, color="player_display",
            trendline="lowess",
            labels={"avg_tilt": "Avg Tilt (°)", metric_t2: METRIC_LABELS[metric_t2]},
            title=f"Tilt vs {METRIC_LABELS[metric_t2]} (LOWESS trend)",
        )
        st.plotly_chart(fig_sc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 3 – HEATMAPS + DETAILED TABLES
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader(t["tab_heatmaps"])
    if not selected_players_multi:
        st.info("Select batters in the sidebar.")
    else:
        metric_t3 = st.radio(
            t["metric"],
            ["avg_tilt", "tilt_std", "delta_tilt", "avg_aa", "aa_std",
             "avg_bat_speed", "avg_swing_len", "batting_avg", "xwoba",
             "avg_exit_velocity", "avg_launch_angle", "swings"],
            format_func=lambda x: METRIC_LABELS.get(x, x),
            horizontal=True,
            key="t3_metric",
        )

        for p in selected_players_multi:
            df_p = (
                league_per_zone.copy() if p == t["league_avg"]
                else get_player_zone_df(p, df_filtered, league_per_zone)
            )

            if df_p is None or df_p.empty:
                st.caption(f"{t['no_data_for']} {p}")
                continue

            make_heatmap(df_p, metric_t3, p, league_per_zone, view_mode, df_filtered)

            # Detailed per-zone table for real batters
            if p != t["league_avg"]:
                st.markdown("---")
                st.subheader(f"{t['detailed_table']} – {p}")

                pid_filter = display_to_id.get(p) if use_id else p
                pdata = (
                    detail_tables[detail_tables[id_col] == pid_filter]
                    if use_id else
                    detail_tables[detail_tables["batter_name"] == p]
                )

                if not pdata.empty:
                    zagg = pdata.groupby("zone", as_index=False).agg({
                        "swings":            "sum",
                        "avg_tilt":          ["mean", "std"],
                        "avg_aa":            ["mean", "std"],
                        "avg_bat_speed":     "mean",
                        "avg_swing_len":     "mean",
                        "batting_avg":       "mean",
                        "xwoba":             "mean",
                        "avg_exit_velocity": "mean",
                        "avg_launch_angle":  "mean",
                    }).round(3)
                    zagg.columns = [
                        "_".join(filter(None, col)).strip("_")
                        for col in zagg.columns
                    ]
                    st.dataframe(
                        zagg.sort_values("swings_sum", ascending=False),
                        use_container_width=True, hide_index=True,
                    )

    # All-batters reference table
    st.markdown("---")
    st.markdown(t["all_players"])
    if not detail_tables.empty:
        _cols_all = ["swings", "avg_tilt", "avg_aa", "avg_bat_speed",
                     "avg_swing_len", "batting_avg", "xwoba",
                     "avg_exit_velocity", "avg_launch_angle"]
        if use_id:
            all_t3 = detail_tables.groupby(id_col, observed=True).agg(
                {c: ("sum" if c == "swings" else "mean") for c in _cols_all}
            ).round(3).reset_index()
            all_t3["batter_name"] = all_t3[id_col].map(id_to_display)
            all_t3 = all_t3[["batter_name"] + _cols_all]
        else:
            all_t3 = detail_tables.groupby("batter_name", observed=True).agg(
                {c: ("sum" if c == "swings" else "mean") for c in _cols_all}
            ).round(3).reset_index()

        st.dataframe(
            all_t3.sort_values("swings", ascending=False),
            use_container_width=True, hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════
# TAB 4 – SIDE-BY-SIDE
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader(t["tab_side_by_side"])
    _cmp_opts = [t["league_avg"]] + (
        sorted(player_info["display_name"]) if player_info is not None else all_real
    )
    cL, cR = st.columns(2)
    p_left  = cL.selectbox(t["compare_left"],  _cmp_opts, index=0, key="t4L")
    p_right = cR.selectbox(t["compare_right"], _cmp_opts,
                            index=min(1, len(_cmp_opts)-1), key="t4R")

    metric_t4 = st.radio(
        t["metric"],
        [k for k in METRIC_LABELS if k not in ("tilt_std", "aa_std")],
        format_func=lambda x: METRIC_LABELS[x],
        horizontal=True, key="t4_metric",
    )
    dfL = get_player_zone_df(p_left,  df_filtered, league_per_zone)
    dfR = get_player_zone_df(p_right, df_filtered, league_per_zone)
    cL2, cR2 = st.columns(2)
    with cL2:
        make_heatmap(dfL, metric_t4, p_left,  league_per_zone, view_mode, df_filtered) \
            if not (dfL is None or dfL.empty) else st.info(f"{t['no_data_for']} {p_left}")
    with cR2:
        make_heatmap(dfR, metric_t4, p_right, league_per_zone, view_mode, df_filtered) \
            if not (dfR is None or dfR.empty) else st.info(f"{t['no_data_for']} {p_right}")


# ══════════════════════════════════════════════════════════════════════
# TAB 5 – BATTER METRICS (two heatmaps side by side)
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader(t["tab_player_compare"])
    _p5_opts = selected_display if selected_display else all_real
    sel_p5   = st.selectbox(t["select_player"], _p5_opts, key="t5_p")
    cL5, cR5 = st.columns(2)
    left_m5  = cL5.selectbox(t["left_metric"],  list(METRIC_LABELS.keys()),
                               format_func=lambda x: METRIC_LABELS[x],
                               index=0, key="t5L")
    right_m5 = cR5.selectbox(t["right_metric"], list(METRIC_LABELS.keys()),
                               format_func=lambda x: METRIC_LABELS[x],
                               index=6, key="t5R")
    df_p5 = get_player_zone_df(sel_p5, df_filtered, league_per_zone)
    if df_p5 is None or df_p5.empty:
        st.info(f"{t['no_data_for']} {sel_p5}")
    else:
        cL5b, cR5b = st.columns(2)
        with cL5b:
            make_heatmap(df_p5, left_m5,
                         f"{sel_p5} – {METRIC_LABELS[left_m5]}",
                         league_per_zone, view_mode, df_filtered)
        with cR5b:
            make_heatmap(df_p5, right_m5,
                         f"{sel_p5} – {METRIC_LABELS[right_m5]}",
                         league_per_zone, view_mode, df_filtered)


# ══════════════════════════════════════════════════════════════════════
# TAB 6 – TILT OPTIMIZER  🔬
# ══════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader(t["tab_tilt_sim"])
    st.markdown(
        """
        The **Tilt Optimizer** fits a partial-dependence curve: all other features
        (attack angle, bat speed, swing length, zone, pitch group) are held at the
        batter's own averages while tilt is swept from 8° → 62°.
        Shrinkage blends the player-specific optimum toward the league optimum,
        weighted by sample size.
        """
    )

    # ── Controls ─────────────────────────────────────────────────────
    cc1, cc2, cc3 = st.columns([2, 1, 1])
    sel_p6 = cc1.selectbox(
        t["select_player"],
        options=all_real,
        key="t6_batter",
    )
    pg_opts6 = [t["all"]] + sorted(detail_full["pitch_group"].dropna().unique())
    sel_pg6  = cc2.selectbox("Pitch group", pg_opts6, key="t6_pg")
    zone_opts6 = ["All"] + [str(z) for z in range(1, 15)]
    sel_z6   = cc3.selectbox("Zone filter", zone_opts6, key="t6_zone")

    # ── Filter data for this batter ──────────────────────────────────
    if use_id:
        _pid6 = display_to_id.get(sel_p6)
        p6_df = detail_fe[detail_fe[id_col] == _pid6].copy()
    else:
        p6_df = detail_fe[detail_fe["batter_name"] == sel_p6].copy()

    if sel_pg6 != t["all"]:
        p6_df = p6_df[p6_df["pitch_group"] == sel_pg6]
    if sel_z6 != "All":
        p6_df = p6_df[p6_df["zone"] == int(sel_z6)]

    if p6_df.empty:
        st.warning(f"{t['no_data_for']} {sel_p6} with current filters.")
    else:
        n6       = int(p6_df["swings"].sum())
        aa6      = float(p6_df["avg_aa"].mean())
        spd6     = float(p6_df["avg_bat_speed"].mean())
        len6     = float(p6_df["avg_swing_len"].mean())
        cur_tilt = float(p6_df["avg_tilt"].mean())
        grp_enc6 = float(p6_df["group_enc"].mean())
        zon_enc6 = float(p6_df["zone_enc"].mean())

        tg6, pr6 = predict_tilt_curve(aa6, spd6, len6, zon_enc6, grp_enc6)
        raw_opt6, opt_xwoba6 = find_optimal_tilt(tg6, pr6)
        shr_opt6, conf_w6    = shrink(raw_opt6, LEAGUE_OPT_TILT, n6)

        # League curve (same context)
        tg_lg6, pr_lg6 = predict_tilt_curve(
            league_stats["avg_aa"], league_stats["avg_bat_speed"],
            league_stats["avg_swing_len"], zon_enc6, grp_enc6,
        )
        raw_lg6, _ = find_optimal_tilt(tg_lg6, pr_lg6)

        # ── KPI Row ──────────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric(t["current_tilt"],       f"{cur_tilt:.1f}°")
        k2.metric(t["optimal_tilt"],       f"{shr_opt6:.1f}°")
        k3.metric(t["tilt_delta"],         f"{cur_tilt - shr_opt6:+.1f}°",
                  delta=f"{-(cur_tilt - shr_opt6):+.1f}° to optimal",
                  delta_color="inverse")
        k4.metric(t["pred_xwoba"],         f"{opt_xwoba6:.3f}")
        k5.metric(t["conf_weight"],
                  f"{conf_w6:.0%}",
                  help=f"n={n6} swings · K={SHRINKAGE_K}")

        if n6 < 30:
            st.warning(
                f"⚠️ Small sample (n={n6} swings) — optimal tilt estimate heavily "
                f"shrunk toward league average ({LEAGUE_OPT_TILT:.1f}°). "
                "Interpret with caution."
            )

        # ── Main Tilt Curve ──────────────────────────────────────────
        fig6 = go.Figure()

        if show_ci:
            lo6, hi6 = approximate_ci(aa6, spd6, len6, zon_enc6, grp_enc6, n6)
            fig6.add_trace(go.Scatter(
                x=np.concatenate([tg6, tg6[::-1]]),
                y=np.concatenate([hi6, lo6[::-1]]),
                fill="toself", fillcolor="rgba(99,110,250,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="80% CI", hoverinfo="skip",
            ))

        fig6.add_trace(go.Scatter(
            x=tg6, y=pr6, mode="lines",
            line=dict(color="#636EFA", width=2.8),
            name=f"{sel_p6}",
        ))
        fig6.add_trace(go.Scatter(
            x=tg_lg6, y=pr_lg6, mode="lines",
            line=dict(color="#EF553B", width=1.8, dash="dot"),
            name="League avg features",
        ))

        # Vertical lines
        for x_val, color, lbl, dash in [
            (cur_tilt,   "#00CC96",  f"Current ({cur_tilt:.1f}°)",        "dash"),
            (shr_opt6,   "#AB63FA",  f"Optimal–shrunk ({shr_opt6:.1f}°)", "solid"),
            (raw_opt6,   "#FFA15A",  f"Optimal–raw ({raw_opt6:.1f}°)",    "dot"),
            (LEAGUE_OPT_TILT, "#EF553B", f"League opt. ({LEAGUE_OPT_TILT:.1f}°)", "dashdot"),
        ]:
            fig6.add_vline(
                x=x_val,
                line=dict(color=color, width=1.8, dash=dash),
                annotation_text=lbl,
                annotation_position="top",
                annotation_font_size=11,
            )

        fig6.update_layout(
            title=f"Predicted xwOBA vs Tilt — {sel_p6}"
                  f" [{sel_pg6} / Zone {sel_z6}]",
            xaxis_title="Swing Path Tilt (°)",
            yaxis_title="Predicted xwOBA",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25),
            height=460,
            hovermode="x unified",
        )
        st.plotly_chart(fig6, use_container_width=True)

        # ── 2-D Interaction Heatmap: Tilt × Attack Angle ─────────────
        st.markdown("#### Interaction Effect: Tilt × Attack Angle → Predicted xwOBA")
        st.caption("Holding bat speed, swing length, zone, and pitch group at player averages.")

        _tg2d  = np.linspace(TILT_MIN, TILT_MAX, 30)
        _aa2d  = np.linspace(-30, 30, 25)
        TT, AA = np.meshgrid(_tg2d, _aa2d)
        _n2d   = TT.size

        X2d = np.column_stack([
            TT.ravel(), AA.ravel(),
            np.full(_n2d, spd6), np.full(_n2d, len6),
            np.full(_n2d, zon_enc6), np.full(_n2d, grp_enc6),
            TT.ravel() * AA.ravel(),
            TT.ravel() * grp_enc6,
        ])
        Z2d = model.predict(X2d).reshape(TT.shape)

        fig2d = go.Figure(data=go.Heatmap(
            z=Z2d, x=_tg2d, y=_aa2d,
            colorscale="RdYlGn",
            colorbar=dict(title="Pred. xwOBA"),
            hovertemplate="Tilt: %{x:.1f}°<br>Attack Angle: %{y:.1f}°<br>xwOBA: %{z:.3f}<extra></extra>",
        ))
        # Mark current batter position
        fig2d.add_trace(go.Scatter(
            x=[cur_tilt], y=[aa6], mode="markers",
            marker=dict(color="white", size=14, symbol="star",
                        line=dict(color="black", width=2)),
            name=sel_p6,
        ))
        fig2d.update_layout(
            xaxis_title="Swing Path Tilt (°)",
            yaxis_title="Attack Angle (°)",
            height=420,
            title="xwOBA surface: Tilt × Attack Angle (⭐ = current batter position)",
        )
        st.plotly_chart(fig2d, use_container_width=True)

        # ── Tilt × Pitch-Group breakdown ──────────────────────────────
        groups_avail = sorted(detail_full["pitch_group"].dropna().unique())
        if len(groups_avail) > 1:
            st.markdown("#### Optimal Tilt by Pitch Group (for this batter)")
            rows_pg = []
            for pg in groups_avail:
                pg_enc = _enc_group(pg)
                tgp, prp = predict_tilt_curve(aa6, spd6, len6, zon_enc6, pg_enc)
                ro, ro_x  = find_optimal_tilt(tgp, prp)
                rows_pg.append({"Pitch Group": pg,
                                 "Optimal Tilt (raw)": round(ro, 1),
                                 "Pred. xwOBA": round(ro_x, 3)})
            df_pg = pd.DataFrame(rows_pg)
            figpg = px.bar(
                df_pg, x="Pitch Group", y="Optimal Tilt (raw)",
                color="Pred. xwOBA", color_continuous_scale="RdYlGn",
                text="Optimal Tilt (raw)",
                title=f"Optimal Tilt by Pitch Group — {sel_p6}",
            )
            figpg.add_hline(
                y=cur_tilt, line_dash="dash", line_color="steelblue",
                annotation_text=f"Current avg tilt ({cur_tilt:.1f}°)",
            )
            figpg.update_traces(texttemplate="%{text:.1f}°", textposition="outside")
            figpg.update_layout(height=360)
            st.plotly_chart(figpg, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 7 – TILT RANKINGS  🏆
# ══════════════════════════════════════════════════════════════════════
with tab7:
    st.subheader(t["tab_tilt_rankings"])
    st.markdown(
        f"""
        Ranks all batters by the **gap between their current tilt and model-optimal tilt**.
        Optimal is shrunk toward the league optimum ({LEAGUE_OPT_TILT:.1f}°) using
        Bayesian shrinkage (K = {SHRINKAGE_K} swings).
        Batters with low confidence should be viewed as directional signals only.
        """
    )

    if opt_table.empty:
        st.warning("No optimization data available for current filters.")
    else:
        sort_col = st.selectbox(
            "Sort by",
            ["Δ Tilt", "Swings", "Pred. xwOBA @ Opt.", "Current Tilt", "Optimal Tilt"],
            index=0, key="t7_sort",
        )
        asc7 = st.checkbox("Ascending", value=False, key="t7_asc")

        # Highlight over/under-tilted
        min_conf_filter = st.slider("Min. confidence to show", 0.0, 1.0, 0.0, 0.05)
        tbl7 = opt_table[opt_table["Confidence"] >= min_conf_filter].copy()
        tbl7 = tbl7.sort_values(sort_col, ascending=asc7, na_position="last")

        def _color_delta(val):
            if pd.isna(val):
                return ""
            if val >  5:  return "background-color: #ffe0e0"   # under-tilted
            if val < -5:  return "background-color: #e0e8ff"   # over-tilted
            return "background-color: #e0ffe8"                  # near-optimal

        st.dataframe(
            tbl7.style.applymap(_color_delta, subset=["Δ Tilt"])
                      .format({
                          "Current Tilt":       "{:.1f}°",
                          "Optimal Tilt (raw)": "{:.1f}°",
                          "Optimal Tilt":       "{:.1f}°",
                          "Δ Tilt":             "{:+.1f}°",
                          "Pred. xwOBA @ Opt.": "{:.3f}",
                          "Current xwOBA":      "{:.3f}",
                          "Confidence":         "{:.0%}",
                      }),
            use_container_width=True, hide_index=True,
        )

        # ── Under / Over adjusted bar chart ──────────────────────────
        st.markdown("#### Δ Tilt: Current minus Optimal (top 30 by |Δ|)")
        top30 = tbl7.reindex(tbl7["Δ Tilt"].abs().sort_values(ascending=False).index).head(30)

        fig7 = px.bar(
            top30,
            x="Δ Tilt", y="Batter",
            orientation="h",
            color="Δ Tilt",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            text="Δ Tilt",
            hover_data=["Current Tilt", "Optimal Tilt", "Confidence", "Swings"],
            title="Δ Tilt: Positive = too flat, Negative = too steep",
        )
        fig7.update_traces(texttemplate="%{text:+.1f}°", textposition="outside")
        fig7.add_vline(x=0, line_width=2, line_dash="solid", line_color="black")
        fig7.update_layout(
            yaxis=dict(autorange="reversed"),
            height=max(400, 20 * len(top30)),
            showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig7, use_container_width=True)

        # ── Scatter: current xwOBA vs Δ Tilt ─────────────────────────
        if "Current xwOBA" in tbl7.columns and tbl7["Current xwOBA"].notna().any():
            st.markdown("#### Current xwOBA vs Δ Tilt")
            fig7b = px.scatter(
                tbl7.dropna(subset=["Current xwOBA"]),
                x="Δ Tilt", y="Current xwOBA",
                size="Swings", color="Confidence",
                color_continuous_scale="Viridis",
                hover_name="Batter",
                hover_data=["Swings", "Optimal Tilt", "Current Tilt"],
                trendline="lowess",
                labels={
                    "Δ Tilt": "Δ Tilt (Current − Optimal, °)",
                    "Current xwOBA": "Current xwOBA",
                    "Confidence": "Conf.",
                },
                title="Do batters closer to optimal tilt perform better?",
            )
            fig7b.add_vline(x=0, line_dash="dash", line_color="grey")
            fig7b.update_layout(height=430)
            st.plotly_chart(fig7b, use_container_width=True)

        # ── Feature importance (GBM only) ────────────────────────────
        if "Gradient Boosting" in model_type:
            st.markdown("#### Model: Feature Importance")
            imp = pd.Series(model.feature_importances_, index=FEATURE_COLS) \
                    .sort_values(ascending=True)
            fig_imp = px.bar(
                imp.reset_index(), x=0, y="index",
                orientation="h",
                labels={"0": "Importance", "index": "Feature"},
                title="Gradient Boosting Feature Importance",
                color=0, color_continuous_scale="Blues",
            )
            fig_imp.update_layout(height=360, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)
