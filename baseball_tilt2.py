"""
MLB 2025 – Swing Path Tilt & Attack Angle Dashboard  v2.1
==========================================================
Reads:  players_summary_2025.csv
        detail_zone_pitchgroup_2025.csv

New in v2:
  • Gradient Boosting model (GAM optional via pygam)
  • Optimal Tilt Simulator tab – PDP curve, CI bands,
    2-D tilt × attack-angle interaction heatmap
  • Tilt Rankings tab – all batters, Bayesian shrinkage,
    delta bar chart, feature importance
  • Percentile / Shrunk heatmap modes
  • Sample-size warnings throughout
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
# PAGE CONFIG  ← must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLB Bat Tracking 2025",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────
SHRINKAGE_K = 50          # swings for 50 % weight on league mean
TILT_MIN    = 8.0
TILT_MAX    = 62.0
TILT_GRID_N = 80
DATA_DIR    = Path(".")

# Features fed to the model
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
    },
    "pl": {
        "title":              "MLB 2025 – Swing Path Tilt & Attack Angle Dashboard",
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
    },
}

# ─────────────────────────────────────────────────────────────────────
# LANGUAGE
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
# DATA LOADING  (CSV)
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⚾ Loading data …")
def load_data():
    players = pd.read_csv(DATA_DIR / "players_summary_2025.csv")
    detail  = pd.read_csv(DATA_DIR / "detail_zone_pitchgroup_2025.csv")
    # Drop pitcher rows
    mask = (
        detail["batter_name"].notna()
        & ~detail["batter_name"].str.contains(
            r" pitcher| P$", case=False, na=False, regex=True
        )
    )
    return players, detail[mask].copy()

players_raw, detail_full = load_data()

# ─────────────────────────────────────────────────────────────────────
# PLAYER ID RESOLUTION  (supports optional batter_id column)
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
    player_info = players_raw[[id_col, "batter_name"]].drop_duplicates()
    _dupes = player_info["batter_name"].value_counts()
    _dupes = _dupes[_dupes > 1].index.tolist()
    player_info["display_name"] = player_info.apply(
        lambda r: (
            f"{r['batter_name']} (ID:{int(r[id_col])})"
            if r["batter_name"] in _dupes else r["batter_name"]
        ), axis=1,
    )
    display_to_id = dict(zip(player_info["display_name"], player_info[id_col]))
    id_to_display = dict(zip(player_info[id_col], player_info["display_name"]))
    all_real = sorted(player_info["display_name"])
else:
    all_real = sorted(players_raw["batter_name"].dropna().unique())

def _display_name(val) -> str:
    return id_to_display.get(val, str(val)) if use_id else str(val)

def _filter_by_player(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if use_id:
        pid = display_to_id.get(name)
        return df[df[id_col] == pid] if pid is not None else pd.DataFrame()
    return df[df["batter_name"] == name]

# ─────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (run once at startup)
# ─────────────────────────────────────────────────────────────────────
def _engineer(df: pd.DataFrame):
    out = df.copy()
    le_zone  = LabelEncoder()
    le_group = LabelEncoder()
    out["zone_enc"]     = le_zone.fit_transform(out["zone"].astype(str))
    out["group_enc"]    = le_group.fit_transform(
        out["pitch_group"].fillna("Unknown").astype(str)
    )
    out["tilt_x_aa"]    = out["avg_tilt"] * out["avg_aa"]
    out["tilt_x_group"] = out["avg_tilt"] * out["group_enc"]
    out["sample_weight"]= np.sqrt(out["swings"].clip(lower=1))
    return out, le_zone, le_group

detail_fe, le_zone_g, le_group_g = _engineer(detail_full)

def _enc_group(name: str) -> float:
    return float(le_group_g.transform([name])[0]) if name in le_group_g.classes_ else 0.0

# ─────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Training swing model …")
def train_model():
    df = detail_fe.dropna(subset=["xwoba"]).query("swings >= 5").copy()
    if df.empty:
        return None, "No data"

    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median()).values
    y = df["xwoba"].values
    w = df["sample_weight"].values

    if HAS_PYGAM and len(X) >= 40:
        try:
            gam = LinearGAM(
                s(0, n_splines=12, constraints="none")
                + s(1, n_splines=10)
                + s(2, n_splines=8)
                + s(3, n_splines=6)
                + gam_f(4)
                + gam_f(5)
                + s(6, n_splines=6)
                + s(7, n_splines=6),
                fit_intercept=True,
            )
            gam.gridsearch(X, y, weights=w, progress=False)
            return gam, "GAM (pyGAM)"
        except Exception:
            pass

    gbm = GradientBoostingRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.035,
        subsample=0.75, min_samples_leaf=4, max_features=0.8,
        random_state=42,
    )
    gbm.fit(X, y, sample_weight=w)
    return gbm, "Gradient Boosting"

model, model_type = train_model()

# ─────────────────────────────────────────────────────────────────────
# SHRINKAGE
# ─────────────────────────────────────────────────────────────────────
def shrink(player_val: float, league_val: float,
           n: int, k: float = SHRINKAGE_K) -> tuple[float, float]:
    """Blend player estimate toward league mean. Returns (shrunk, weight)."""
    w = n / (n + k)
    return w * player_val + (1 - w) * league_val, round(w, 3)

# ─────────────────────────────────────────────────────────────────────
# TILT CURVE  (partial-dependence style)
# ─────────────────────────────────────────────────────────────────────
def predict_tilt_curve(
    avg_aa: float,
    avg_speed: float,
    avg_len: float,
    zone_enc: float,
    group_enc: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep tilt from TILT_MIN → TILT_MAX, hold everything else fixed."""
    if model is None:
        tg = np.linspace(TILT_MIN, TILT_MAX, TILT_GRID_N)
        return tg, np.full(TILT_GRID_N, np.nan)

    tg = np.linspace(TILT_MIN, TILT_MAX, TILT_GRID_N)
    X  = np.column_stack([
        tg,
        np.full(TILT_GRID_N, avg_aa),
        np.full(TILT_GRID_N, avg_speed),
        np.full(TILT_GRID_N, avg_len),
        np.full(TILT_GRID_N, zone_enc),
        np.full(TILT_GRID_N, group_enc),
        tg * avg_aa,
        tg * group_enc,
    ])
    raw = model.predict(X)
    return tg, gaussian_filter1d(raw, sigma=1.8)


def find_optimal(tg: np.ndarray, preds: np.ndarray) -> tuple[float, float]:
    if np.all(np.isnan(preds)):
        return float(np.mean(tg)), float("nan")
    idx = int(np.nanargmax(preds))
    return float(tg[idx]), float(preds[idx])


def approx_ci(
    avg_aa: float, avg_speed: float, avg_len: float,
    zone_enc: float, group_enc: float,
    n_swings: int, n_boot: int = 150,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte-Carlo CI bands (noise ∝ 1/√n)."""
    scale = 1.0 / np.sqrt(max(n_swings, 1) / SHRINKAGE_K)
    rng   = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        _, p = predict_tilt_curve(
            avg_aa    + rng.normal(0, 8   * scale),
            avg_speed + rng.normal(0, 3   * scale),
            avg_len   + rng.normal(0, 0.3 * scale),
            zone_enc, group_enc,
        )
        boots.append(p)
    arr = np.vstack(boots)
    return np.percentile(arr, 10, axis=0), np.percentile(arr, 90, axis=0)

# ─────────────────────────────────────────────────────────────────────
# COMPUTE GLOBAL LEAGUE OPTIMAL TILT  (safe, always runs)
# ─────────────────────────────────────────────────────────────────────
def _safe_mean(series) -> float:
    v = series.dropna()
    return float(v.mean()) if len(v) else 0.0

_lg_aa    = _safe_mean(detail_full["avg_aa"])
_lg_speed = _safe_mean(detail_full["avg_bat_speed"])
_lg_len   = _safe_mean(detail_full["avg_swing_len"])
_lg_zone  = float(np.median(detail_fe["zone_enc"]))  if not detail_fe.empty else 0.0
_lg_group = float(np.median(detail_fe["group_enc"])) if not detail_fe.empty else 0.0

_tg_lg, _pr_lg          = predict_tilt_curve(_lg_aa, _lg_speed, _lg_len, _lg_zone, _lg_group)
LEAGUE_OPT_TILT, _       = find_optimal(_tg_lg, _pr_lg)
LEAGUE_XWOBA             = _safe_mean(detail_full["xwoba"])

# ─────────────────────────────────────────────────────────────────────
# FILTERING HELPERS
# ─────────────────────────────────────────────────────────────────────
def apply_pitch_filter(df: pd.DataFrame, pitch_grp: str, pitch_typ: str) -> pd.DataFrame:
    if pitch_grp != t["all"]:
        df = df[df["pitch_group"] == pitch_grp]
    if pitch_typ  != t["all"]:
        df = df[df["pitch_type"]  == pitch_typ]
    return df


def get_player_zone_df(name: str, df_ctx: pd.DataFrame,
                        league_z: pd.DataFrame) -> pd.DataFrame:
    if name == t["league_avg"]:
        return league_z.copy()
    sub = _filter_by_player(df_ctx, name)
    if sub.empty:
        return pd.DataFrame()
    agg = {c: "mean" for c in METRIC_LABELS if c != "swings" and c in sub.columns}
    agg["swings"] = "sum"
    return sub.groupby("zone", as_index=False).agg(agg).round(3)

# ─────────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────────────
st.title(t["title"])
st.caption(
    f"🤖 Model: **{model_type}** · "
    f"League opt. tilt: **{LEAGUE_OPT_TILT:.1f}°** · "
    f"Shrinkage K = {SHRINKAGE_K} swings"
)

with st.sidebar:
    st.markdown("### ⚾ Batter Selection")
    _sep = "─" * 22
    player_opts = [t["league_avg"], _sep] + all_real
    selected_players_multi = st.multiselect(
        t["sidebar_players"],
        options=player_opts,
        default=[all_real[0]] if all_real else [],
        max_selections=8,
    )
    selected_players_multi = [p for p in selected_players_multi if p != _sep]

    st.markdown("### 🎛 Filters")
    min_swings = st.slider(t["sidebar_min_swings"], 0, 300, 0, 10)

    pitch_groups_list = [t["all"]] + sorted(detail_full["pitch_group"].dropna().unique())
    selected_pitch = st.selectbox(t["sidebar_pitch_group"], pitch_groups_list)

    _pt_src = (
        detail_full if selected_pitch == t["all"]
        else detail_full[detail_full["pitch_group"] == selected_pitch]
    )
    pitch_types_list = [t["all"]] + sorted(_pt_src["pitch_type"].dropna().unique())
    selected_type = st.selectbox(t["sidebar_pitch_type"], pitch_types_list)

    st.markdown("### 🔧 Display")
    view_mode = st.radio(
        t["heatmap_view"], ["Raw", "Percentile", "Shrunk"], index=0,
        help="Raw=average · Percentile=rank vs zone peers · Shrunk=Bayesian blend",
    )
    show_ci = st.checkbox(t["show_ci"], value=True)

# ── Apply global filters ──────────────────────────────────────────────
df_filtered = apply_pitch_filter(detail_full.copy(), selected_pitch, selected_type)
df_fe_filt  = apply_pitch_filter(detail_fe.copy(),   selected_pitch, selected_type)

_LAGG = {c: "mean" for c in METRIC_LABELS if c != "swings" and c in df_filtered.columns}
_LAGG["swings"] = "sum"
league_per_zone = (
    df_filtered.groupby("zone", as_index=False, observed=True)
               .agg(_LAGG).round(3)
)
league_per_zone["batter_name"] = t["league_avg"]

selected_display = [p for p in selected_players_multi if p != t["league_avg"]]

if use_id:
    player_filter_col    = id_col
    player_filter_values = [display_to_id[p] for p in selected_display if p in display_to_id]
else:
    player_filter_col    = "batter_name"
    player_filter_values = selected_display

detail_tables = df_filtered[df_filtered["swings"] >= min_swings].copy()

# ─────────────────────────────────────────────────────────────────────
# OPTIMISATION TABLE
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⚙️ Computing optimizations …", ttl=3600)
def build_opt_table(
    _df_fe:       pd.DataFrame,
    pitch_grp:    str,
    pitch_typ:    str,
    lg_opt_tilt:  float,
    ctx_zone_enc: float,
    ctx_group_enc:float,
) -> pd.DataFrame:
    grp_col = id_col if use_id else "batter_name"
    agg = _df_fe.groupby(grp_col, observed=True).agg(
        avg_tilt     =("avg_tilt",      "mean"),
        avg_aa       =("avg_aa",        "mean"),
        avg_bat_speed=("avg_bat_speed", "mean"),
        avg_swing_len=("avg_swing_len", "mean"),
        xwoba        =("xwoba",         "mean"),
        swings       =("swings",        "sum"),
    ).reset_index().dropna(subset=["avg_tilt", "avg_aa", "avg_bat_speed", "avg_swing_len"])

    rows = []
    for _, r in agg.iterrows():
        n = int(r.swings)
        if n < 5:
            continue
        tg, preds = predict_tilt_curve(
            float(r.avg_aa), float(r.avg_bat_speed), float(r.avg_swing_len),
            ctx_zone_enc, ctx_group_enc,
        )
        raw_opt, opt_xwoba = find_optimal(tg, preds)
        shrunk_opt, conf_w = shrink(raw_opt, lg_opt_tilt, n)
        rows.append({
            "Batter":             _display_name(r[grp_col]),
            "Swings":             n,
            "Current Tilt":       round(float(r.avg_tilt),  1),
            "Optimal Tilt (raw)": round(raw_opt,             1),
            "Optimal Tilt":       round(shrunk_opt,          1),
            "Δ Tilt":             round(float(r.avg_tilt) - shrunk_opt, 1),
            "Pred. xwOBA @ Opt.": round(opt_xwoba,           3),
            "Confidence":         round(conf_w,               2),
            "Current xwOBA":      round(float(r.xwoba), 3) if not np.isnan(r.xwoba) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("Swings", ascending=False).reset_index(drop=True)

# Context encodings for the selected filters
_ctx_zone_enc  = float(np.median(df_fe_filt["zone_enc"])) if not df_fe_filt.empty else _lg_zone
_ctx_group_enc = float(np.median(df_fe_filt["group_enc"])) if not df_fe_filt.empty else _lg_group

opt_table = build_opt_table(
    df_fe_filt, selected_pitch, selected_type,
    LEAGUE_OPT_TILT, _ctx_zone_enc, _ctx_group_enc,
)

# ─────────────────────────────────────────────────────────────────────
# HEATMAP HELPERS
# ─────────────────────────────────────────────────────────────────────
def _zone_color(val, vmin, vmax, cmap):
    if pd.isna(val):
        return (0.91, 0.91, 0.91)
    return cmap(float(np.clip((val - vmin) / max(vmax - vmin, 1e-9), 0, 1)))

def _text_col(bg):
    r, g, b = bg[:3]
    return "black" if 0.299*r + 0.587*g + 0.114*b > 0.45 else "white"

def _fmt(val, metric, view):
    if pd.isna(val): return "—"
    if view == "Percentile": return f"{int(round(val))}th"
    if metric == "swings":   return f"{int(round(val))}"
    if metric in ("batting_avg", "xwoba"): return f"{val:.3f}"
    return f"{val:.1f}"

def _get_pivot(df_p, metric, view, league_df, df_ctx) -> pd.Series:
    if metric == "tilt_std":
        return df_p.groupby("zone")["avg_tilt"].std(ddof=1).round(1)
    if metric == "aa_std":
        return df_p.groupby("zone")["avg_aa"].std(ddof=1).round(1)
    if metric == "delta_tilt":
        pm = df_p.groupby("zone")["avg_tilt"].mean()
        lm = league_df.set_index("zone")["avg_tilt"] if league_df is not None else pd.Series()
        return (pm - lm.reindex(pm.index, fill_value=np.nan)).round(2)

    raw = (df_p.groupby("zone")["swings"].sum()
           if metric == "swings"
           else df_p.groupby("zone")[metric].mean())

    if view == "Raw":
        return raw.round(3)

    if view == "Percentile":
        result = {}
        for zone, val in raw.items():
            vals = df_ctx[df_ctx["zone"] == zone][metric].dropna()
            result[zone] = percentileofscore(vals, val, kind="rank") if len(vals) else np.nan
        return pd.Series(result)

    if view == "Shrunk":
        n_sw = df_p.groupby("zone")["swings"].sum()
        lg   = (df_ctx.groupby("zone")[metric].sum()
                if metric == "swings"
                else df_ctx.groupby("zone")[metric].mean())
        result = {}
        for zone, val in raw.items():
            n   = int(n_sw.get(zone, 0))
            lgv = float(lg.get(zone, df_ctx[metric].mean())) if zone in lg.index else float(df_ctx[metric].mean())
            result[zone] = round(shrink(val, lgv, n)[0], 3)
        return pd.Series(result)
    return raw.round(3)

# ─────────────────────────────────────────────────────────────────────
# HEATMAP RENDERER
# ─────────────────────────────────────────────────────────────────────
def make_heatmap(df_p, metric, title,
                 league_df=None, view="Raw", df_ctx=None):
    if df_p is None or df_p.empty:
        st.warning(f"{t['no_data_for']} {title}")
        return
    if df_ctx is None:
        df_ctx = detail_full

    pivot  = _get_pivot(df_p, metric, view, league_df, df_ctx)
    n_sw   = df_p.groupby("zone")["swings"].sum() if "swings" in df_p.columns else pd.Series()

    if view == "Percentile" and metric not in ("tilt_std", "aa_std", "delta_tilt"):
        vmin, vmax, cmap_name = 0, 100, "RdYlGn"
    elif metric == "delta_tilt":
        vmin, vmax = HEATMAP_RANGES.get(metric, (-20, 20))
        cmap_name  = "RdBu_r"
    else:
        vmin, vmax = HEATMAP_RANGES.get(metric, (0, 100))
        cmap_name  = "YlOrRd"

    cmap = sns.color_palette(cmap_name, as_cmap=True)

    # Zone geometry
    b, ms, sy = 0.85, 3.3, 2.5
    mx, my = b, b
    tx, ty = mx + ms, my + ms
    half   = ms / 2

    fig, ax = plt.subplots(figsize=(8, 8))

    # 3×3 main zones
    for i in range(3):
        for j in range(3):
            zone = i * 3 + j + 1
            val  = pivot.get(zone, np.nan)
            n    = int(n_sw.get(zone, 0))
            x    = mx + j * (ms / 3)
            y    = my + (2 - i) * (ms / 3)
            col  = _zone_color(val, vmin, vmax, cmap)
            ax.add_patch(plt.Rectangle(
                (x, y), ms/3, ms/3, facecolor=col, edgecolor="black", linewidth=2.4
            ))
            txt = f"{zone}\n{_fmt(val, metric, view)}"
            if n < 20 and view != "Percentile":
                txt += f"\n⚠n={n}"
            ax.text(x + ms/6, y + ms/6, txt, ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color=_text_col(col))

    # L-shaped zones 11-14
    l_data = [
        (11, [(0, sy),(b, sy),(b, ty),(mx, ty),(mx+half, ty),(mx+half, 5),(0, 5),(0, sy)],
              (b*0.4, 5 - b*0.4)),
        (12, [(tx, sy),(tx, ty),(mx+half, ty),(mx+half, 5),(5, 5),(5, sy),(tx, sy)],
              (5 - b*0.4, 5 - b*0.4)),
        (13, [(0, sy),(b, sy),(b, my),(mx, my),(mx+half, my),(mx+half, 0),(0, 0),(0, sy)],
              (b*0.4, b*0.4)),
        (14, [(tx, sy),(tx, my),(mx+half, my),(mx+half, 0),(5, 0),(5, sy),(tx, sy)],
              (5 - b*0.4, b*0.4)),
    ]
    for z, verts, (cx, cy) in l_data:
        val = pivot.get(z, np.nan)
        n   = int(n_sw.get(z, 0))
        col = _zone_color(val, vmin, vmax, cmap)
        ax.add_patch(PathPatch(MPath(verts), facecolor=col, edgecolor="black", linewidth=2.4))
        txt = f"{z}\n{_fmt(val, metric, view)}"
        if n < 20 and view != "Percentile":
            txt += f"\n⚠n={n}"
        ax.text(cx, cy, txt, ha="center", va="center", fontsize=10.5, fontweight="bold")

    ax.add_patch(plt.Rectangle(
        (mx, my), ms, ms, fill=False, edgecolor="red", linewidth=3.8
    ))

    vsuffix = {"Percentile": " [Percentile]", "Shrunk": " [Shrunk]"}.get(view, "")
    ax.set_title(f"{title}{vsuffix}", fontsize=14, pad=16, fontweight="bold")
    ax.set_xlim(0, 5); ax.set_ylim(0, 5)
    ax.set_aspect("equal"); ax.axis("off")

    lbl_map = {
        "avg_tilt": "TILT (°)", "tilt_std": "TILT STD", "delta_tilt": "Δ TILT",
        "avg_aa": "ATTACK ANGLE (°)", "aa_std": "AA STD",
        "avg_bat_speed": "BAT SPEED (mph)", "avg_swing_len": "SWING LEN (ft)",
        "swings": "SWINGS", "batting_avg": "BATTING AVG", "xwoba": "xwOBA",
        "avg_exit_velocity": "EXIT VELO (mph)", "avg_launch_angle": "LAUNCH ANGLE (°)",
    }
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, ax=ax, shrink=0.76, pad=0.04)
    cbar.set_label(lbl_map.get(metric, metric.upper()), fontsize=11)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

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
    _COLS = ["batter_name", "avg_tilt", "avg_aa", "avg_bat_speed",
             "avg_swing_len", "swings", "batting_avg", "xwoba",
             "avg_exit_velocity", "avg_launch_angle"]
    _AGG  = {c: ("sum" if c == "swings" else "mean")
             for c in _COLS[1:] if c in detail_tables.columns}

    if selected_players_multi:
        st.markdown(t["selected_players"])
        sel_sub = (
            detail_tables[detail_tables[player_filter_col].isin(player_filter_values)]
            if player_filter_values else pd.DataFrame()
        )
        if not sel_sub.empty:
            if use_id:
                sel_sum = sel_sub.groupby(id_col, observed=True).agg(_AGG).round(3).reset_index()
                sel_sum["batter_name"] = sel_sum[id_col].map(id_to_display)
            else:
                sel_sum = sel_sub.groupby("batter_name", observed=True).agg(_AGG).round(3).reset_index()
            sel_sum = sel_sum[[c for c in _COLS if c in sel_sum.columns]]
            if t["league_avg"] in selected_players_multi:
                la = detail_tables.mean(numeric_only=True).round(3).to_frame().T
                la["batter_name"] = t["league_avg"]
                la["swings"]      = int(detail_tables["swings"].sum())
                sel_sum = pd.concat([sel_sum, la[[c for c in _COLS if c in la.columns]]], ignore_index=True)
            st.dataframe(sel_sum.sort_values("swings", ascending=False),
                         use_container_width=True, hide_index=True)
        st.markdown("---")

    st.markdown(t["all_players"])
    if not detail_tables.empty:
        if use_id:
            all_s = detail_tables.groupby(id_col, observed=True).agg(_AGG).round(3).reset_index()
            all_s["batter_name"] = all_s[id_col].map(id_to_display)
        else:
            all_s = detail_tables.groupby("batter_name", observed=True).agg(_AGG).round(3).reset_index()
        all_s = all_s[[c for c in _COLS if c in all_s.columns]]
        st.dataframe(all_s.sort_values("swings", ascending=False),
                     use_container_width=True, hide_index=True)

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
            t["metric"], list(METRIC_LABELS),
            format_func=lambda x: METRIC_LABELS[x], key="t2m",
        )
        tmp = df_filtered[df_filtered[player_filter_col].isin(player_filter_values)].copy()
        tmp["player_display"] = (
            tmp[id_col].map(id_to_display) if use_id else tmp["batter_name"]
        )

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.box(
                tmp, x="pitch_group", y=metric_t2, color="player_display",
                points="outliers",
                labels={"pitch_group": "Pitch Group", metric_t2: METRIC_LABELS[metric_t2]},
                title=f"Distribution – {METRIC_LABELS[metric_t2]}",
            ), use_container_width=True)
        with c2:
            avg_d = tmp.groupby(["player_display", "pitch_group"], as_index=False)[metric_t2].mean()
            st.plotly_chart(px.bar(
                avg_d, x="pitch_group", y=metric_t2, color="player_display",
                barmode="group",
                labels={"pitch_group": "Pitch Group", metric_t2: METRIC_LABELS[metric_t2]},
                title=f"Average – {METRIC_LABELS[metric_t2]}",
            ), use_container_width=True)

        st.markdown("#### Tilt vs Metric (scatter)")
        st.plotly_chart(px.scatter(
            tmp, x="avg_tilt", y=metric_t2, color="player_display",
            trendline="lowess",
            labels={"avg_tilt": "Avg Tilt (°)", metric_t2: METRIC_LABELS[metric_t2]},
            title=f"Tilt vs {METRIC_LABELS[metric_t2]}",
        ), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 3 – HEATMAPS + TABLES
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
            horizontal=True, key="t3m",
        )
        for p in selected_players_multi:
            df_p = get_player_zone_df(p, df_filtered, league_per_zone)
            if df_p is None or df_p.empty:
                st.caption(f"{t['no_data_for']} {p}")
                continue
            make_heatmap(df_p, metric_t3, p, league_per_zone, view_mode, df_filtered)

            if p != t["league_avg"]:
                st.markdown("---")
                st.subheader(f"{t['detailed_table']} – {p}")
                pdata = _filter_by_player(detail_tables, p)
                if not pdata.empty:
                    zagg = pdata.groupby("zone", as_index=False).agg({
                        "swings": "sum", "avg_tilt": ["mean", "std"],
                        "avg_aa": ["mean", "std"], "avg_bat_speed": "mean",
                        "avg_swing_len": "mean", "batting_avg": "mean",
                        "xwoba": "mean", "avg_exit_velocity": "mean",
                        "avg_launch_angle": "mean",
                    }).round(3)
                    zagg.columns = ["_".join(filter(None, c)) for c in zagg.columns]
                    st.dataframe(zagg.sort_values("swings_sum", ascending=False),
                                 use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(t["all_players"])
    if not detail_tables.empty:
        _c = ["swings","avg_tilt","avg_aa","avg_bat_speed","avg_swing_len",
              "batting_avg","xwoba","avg_exit_velocity","avg_launch_angle"]
        if use_id:
            at3 = detail_tables.groupby(id_col, observed=True).agg(
                {c: ("sum" if c == "swings" else "mean") for c in _c if c in detail_tables.columns}
            ).round(3).reset_index()
            at3["batter_name"] = at3[id_col].map(id_to_display)
        else:
            at3 = detail_tables.groupby("batter_name", observed=True).agg(
                {c: ("sum" if c == "swings" else "mean") for c in _c if c in detail_tables.columns}
            ).round(3).reset_index()
        st.dataframe(at3.sort_values("swings", ascending=False),
                     use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 4 – SIDE-BY-SIDE
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader(t["tab_side_by_side"])
    _cmp = [t["league_avg"]] + (
        sorted(player_info["display_name"]) if player_info is not None else all_real
    )
    cL, cR = st.columns(2)
    p_left  = cL.selectbox(t["compare_left"],  _cmp, index=0, key="t4L")
    p_right = cR.selectbox(t["compare_right"], _cmp,
                            index=min(1, len(_cmp)-1), key="t4R")
    m4 = st.radio(t["metric"],
                  [k for k in METRIC_LABELS if k not in ("tilt_std","aa_std")],
                  format_func=lambda x: METRIC_LABELS[x],
                  horizontal=True, key="t4m")
    cL2, cR2 = st.columns(2)
    with cL2:
        dfL = get_player_zone_df(p_left,  df_filtered, league_per_zone)
        make_heatmap(dfL, m4, p_left,  league_per_zone, view_mode, df_filtered) \
            if dfL is not None and not dfL.empty else st.info(f"{t['no_data_for']} {p_left}")
    with cR2:
        dfR = get_player_zone_df(p_right, df_filtered, league_per_zone)
        make_heatmap(dfR, m4, p_right, league_per_zone, view_mode, df_filtered) \
            if dfR is not None and not dfR.empty else st.info(f"{t['no_data_for']} {p_right}")

# ══════════════════════════════════════════════════════════════════════
# TAB 5 – BATTER METRICS (two heatmaps for one batter)
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader(t["tab_player_compare"])
    p5_opts  = selected_display if selected_display else all_real
    sel_p5   = st.selectbox(t["select_player"], p5_opts, key="t5p")
    cL5, cR5 = st.columns(2)
    lm5 = cL5.selectbox(t["left_metric"],  list(METRIC_LABELS),
                         format_func=lambda x: METRIC_LABELS[x], index=0, key="t5L")
    rm5 = cR5.selectbox(t["right_metric"], list(METRIC_LABELS),
                         format_func=lambda x: METRIC_LABELS[x], index=6, key="t5R")
    df5 = get_player_zone_df(sel_p5, df_filtered, league_per_zone)
    if df5 is None or df5.empty:
        st.info(f"{t['no_data_for']} {sel_p5}")
    else:
        c1b, c2b = st.columns(2)
        with c1b:
            make_heatmap(df5, lm5, f"{sel_p5} – {METRIC_LABELS[lm5]}",
                         league_per_zone, view_mode, df_filtered)
        with c2b:
            make_heatmap(df5, rm5, f"{sel_p5} – {METRIC_LABELS[rm5]}",
                         league_per_zone, view_mode, df_filtered)

# ══════════════════════════════════════════════════════════════════════
# TAB 6 – TILT OPTIMIZER 🔬
# ══════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader(t["tab_tilt_sim"])
    st.markdown(
        "Partial-dependence curve: all other features held at the batter's "
        "own averages while **tilt is swept 8° → 62°**. "
        "Optimal is shrunk toward the league optimum proportionally to sample size."
    )

    if model is None:
        st.error("Model could not be trained (insufficient data). Check your CSV files.")
    else:
        cc1, cc2, cc3 = st.columns([2, 1, 1])
        sel_p6   = cc1.selectbox(t["select_player"], all_real, key="t6p")
        pg_opts6 = [t["all"]] + sorted(detail_full["pitch_group"].dropna().unique())
        sel_pg6  = cc2.selectbox("Pitch group", pg_opts6, key="t6pg")
        z_opts6  = ["All"] + [str(z) for z in range(1, 15)]
        sel_z6   = cc3.selectbox("Zone", z_opts6, key="t6z")

        # Filter data for this batter
        p6_df = _filter_by_player(detail_fe, sel_p6)
        if sel_pg6 != t["all"]:
            p6_df = p6_df[p6_df["pitch_group"] == sel_pg6]
        if sel_z6 != "All":
            p6_df = p6_df[p6_df["zone"] == int(sel_z6)]

        if p6_df.empty:
            st.warning(f"{t['no_data_for']} {sel_p6} with current filters.")
        else:
            n6       = int(p6_df["swings"].sum())
            aa6      = _safe_mean(p6_df["avg_aa"])
            spd6     = _safe_mean(p6_df["avg_bat_speed"])
            len6     = _safe_mean(p6_df["avg_swing_len"])
            cur_tilt = _safe_mean(p6_df["avg_tilt"])
            grp_enc6 = float(p6_df["group_enc"].mean())
            zon_enc6 = float(p6_df["zone_enc"].mean())

            tg6, pr6              = predict_tilt_curve(aa6, spd6, len6, zon_enc6, grp_enc6)
            raw_opt6, opt_xwoba6  = find_optimal(tg6, pr6)
            shr_opt6, conf_w6     = shrink(raw_opt6, LEAGUE_OPT_TILT, n6)

            # League curve (same context zone/group)
            tg_lg6, pr_lg6        = predict_tilt_curve(_lg_aa, _lg_speed, _lg_len, zon_enc6, grp_enc6)

            # KPIs
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric(t["current_tilt"],  f"{cur_tilt:.1f}°")
            k2.metric(t["optimal_tilt"],  f"{shr_opt6:.1f}°")
            delta_v = cur_tilt - shr_opt6
            k3.metric(t["tilt_delta"],    f"{delta_v:+.1f}°",
                      delta=f"{-delta_v:+.1f}° to optimal", delta_color="inverse")
            k4.metric(t["pred_xwoba"],    f"{opt_xwoba6:.3f}")
            k5.metric(t["conf_weight"],   f"{conf_w6:.0%}",
                      help=f"n={n6} swings · K={SHRINKAGE_K}")

            if n6 < 30:
                st.warning(
                    f"⚠️ Small sample (n={n6} swings). Estimate heavily shrunk "
                    f"toward league optimum ({LEAGUE_OPT_TILT:.1f}°). "
                    "Treat as directional signal only."
                )

            # ── Main tilt curve ───────────────────────────────────────
            fig6 = go.Figure()

            if show_ci:
                lo6, hi6 = approx_ci(aa6, spd6, len6, zon_enc6, grp_enc6, n6)
                fig6.add_trace(go.Scatter(
                    x=np.concatenate([tg6, tg6[::-1]]),
                    y=np.concatenate([hi6, lo6[::-1]]),
                    fill="toself", fillcolor="rgba(99,110,250,0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="80% CI", hoverinfo="skip",
                ))

            fig6.add_trace(go.Scatter(
                x=tg6, y=pr6, mode="lines",
                line=dict(color="#636EFA", width=2.8),
                name=sel_p6,
            ))
            fig6.add_trace(go.Scatter(
                x=tg_lg6, y=pr_lg6, mode="lines",
                line=dict(color="#EF553B", width=1.8, dash="dot"),
                name="League avg features",
            ))

            vlines = [
                (cur_tilt,       "#00CC96", f"Current ({cur_tilt:.1f}°)",        "dash"),
                (shr_opt6,       "#AB63FA", f"Optimal-shrunk ({shr_opt6:.1f}°)", "solid"),
                (raw_opt6,       "#FFA15A", f"Optimal-raw ({raw_opt6:.1f}°)",    "dot"),
                (LEAGUE_OPT_TILT,"#EF553B", f"League opt. ({LEAGUE_OPT_TILT:.1f}°)", "dashdot"),
            ]
            for xv, col, lbl, dash in vlines:
                fig6.add_vline(x=xv, line=dict(color=col, width=1.8, dash=dash),
                               annotation_text=lbl, annotation_position="top",
                               annotation_font_size=10)

            fig6.update_layout(
                title=f"Predicted xwOBA vs Tilt – {sel_p6}  [{sel_pg6} / Zone {sel_z6}]",
                xaxis_title="Swing Path Tilt (°)",
                yaxis_title="Predicted xwOBA",
                legend=dict(orientation="h", yanchor="bottom", y=-0.28),
                height=460, hovermode="x unified",
            )
            st.plotly_chart(fig6, use_container_width=True)

            # ── 2-D interaction: Tilt × Attack Angle ─────────────────
            st.markdown("#### Interaction: Tilt × Attack Angle → Predicted xwOBA")
            _tg2d = np.linspace(TILT_MIN, TILT_MAX, 30)
            _aa2d = np.linspace(-30, 30, 25)
            TT, AA = np.meshgrid(_tg2d, _aa2d)
            n2d = TT.size
            X2d = np.column_stack([
                TT.ravel(), AA.ravel(),
                np.full(n2d, spd6), np.full(n2d, len6),
                np.full(n2d, zon_enc6), np.full(n2d, grp_enc6),
                TT.ravel() * AA.ravel(), TT.ravel() * grp_enc6,
            ])
            Z2d = model.predict(X2d).reshape(TT.shape)

            fig2d = go.Figure(data=go.Heatmap(
                z=Z2d, x=_tg2d.round(1), y=_aa2d.round(1),
                colorscale="RdYlGn",
                colorbar=dict(title="Pred. xwOBA"),
                hovertemplate=(
                    "Tilt: %{x:.1f}°<br>"
                    "Attack Angle: %{y:.1f}°<br>"
                    "xwOBA: %{z:.3f}<extra></extra>"
                ),
            ))
            fig2d.add_trace(go.Scatter(
                x=[cur_tilt], y=[aa6], mode="markers",
                marker=dict(color="white", size=14, symbol="star",
                            line=dict(color="black", width=2)),
                name=f"{sel_p6} (current)",
            ))
            fig2d.update_layout(
                xaxis_title="Swing Path Tilt (°)",
                yaxis_title="Attack Angle (°)",
                height=420,
                title="xwOBA surface: Tilt × Attack Angle  (⭐ = current batter)",
            )
            st.plotly_chart(fig2d, use_container_width=True)

            # ── Tilt × Pitch Group breakdown ─────────────────────────
            groups_avail = sorted(detail_full["pitch_group"].dropna().unique())
            if len(groups_avail) > 1:
                st.markdown("#### Optimal Tilt by Pitch Group")
                pg_rows = []
                for pg in groups_avail:
                    ge = _enc_group(pg)
                    tgp, prp = predict_tilt_curve(aa6, spd6, len6, zon_enc6, ge)
                    ro, rox  = find_optimal(tgp, prp)
                    pg_rows.append({"Pitch Group": pg,
                                    "Optimal Tilt (raw)": round(ro, 1),
                                    "Pred. xwOBA": round(rox, 3)})
                df_pg = pd.DataFrame(pg_rows)
                fig_pg = px.bar(
                    df_pg, x="Pitch Group", y="Optimal Tilt (raw)",
                    color="Pred. xwOBA", color_continuous_scale="RdYlGn",
                    text="Optimal Tilt (raw)",
                    title=f"Optimal Tilt by Pitch Group – {sel_p6}",
                )
                fig_pg.add_hline(y=cur_tilt, line_dash="dash", line_color="steelblue",
                                  annotation_text=f"Current tilt ({cur_tilt:.1f}°)")
                fig_pg.update_traces(texttemplate="%{text:.1f}°", textposition="outside")
                fig_pg.update_layout(height=360)
                st.plotly_chart(fig_pg, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 7 – TILT RANKINGS 🏆
# ══════════════════════════════════════════════════════════════════════
with tab7:
    st.subheader(t["tab_tilt_rankings"])
    st.markdown(
        f"Ranks every batter by **gap between current tilt and model-optimal tilt** "
        f"(shrunk toward league opt. = **{LEAGUE_OPT_TILT:.1f}°**, K={SHRINKAGE_K} swings). "
        "🟥 too flat · 🟦 too steep · 🟩 near-optimal (within ±5°)."
    )

    if opt_table.empty:
        st.warning("No optimization data for current filters.")
    else:
        sort_col = st.selectbox(
            "Sort by",
            ["Δ Tilt", "Swings", "Pred. xwOBA @ Opt.", "Current Tilt", "Optimal Tilt"],
            key="t7s",
        )
        asc7 = st.checkbox("Ascending", value=False, key="t7a")
        min_conf = st.slider("Min. confidence to display", 0.0, 1.0, 0.0, 0.05)

        tbl7 = (opt_table[opt_table["Confidence"] >= min_conf]
                        .sort_values(sort_col, ascending=asc7, na_position="last"))

        def _color_delta(val):
            if pd.isna(val): return ""
            if val >  5:  return "background-color:#ffe0e0"
            if val < -5:  return "background-color:#e0e8ff"
            return "background-color:#e0ffe8"

        fmt_map = {
            "Current Tilt":       "{:.1f}°",
            "Optimal Tilt (raw)": "{:.1f}°",
            "Optimal Tilt":       "{:.1f}°",
            "Δ Tilt":             "{:+.1f}°",
            "Pred. xwOBA @ Opt.": "{:.3f}",
            "Current xwOBA":      "{:.3f}",
            "Confidence":         "{:.0%}",
        }
        st.dataframe(
            tbl7.style.applymap(_color_delta, subset=["Δ Tilt"]).format(fmt_map),
            use_container_width=True, hide_index=True,
        )

        # Bar chart
        st.markdown("#### Δ Tilt – Top 30 by |Δ|")
        top30 = tbl7.assign(_abs=tbl7["Δ Tilt"].abs()).nlargest(30, "_abs")
        fig7 = px.bar(
            top30, x="Δ Tilt", y="Batter", orientation="h",
            color="Δ Tilt", color_continuous_scale="RdBu",
            color_continuous_midpoint=0, text="Δ Tilt",
            hover_data=["Current Tilt", "Optimal Tilt", "Confidence", "Swings"],
            title="Positive = too flat  |  Negative = too steep",
        )
        fig7.update_traces(texttemplate="%{text:+.1f}°", textposition="outside")
        fig7.add_vline(x=0, line_width=2, line_color="black")
        fig7.update_layout(
            yaxis=dict(autorange="reversed"),
            height=max(400, 22 * len(top30)),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig7, use_container_width=True)

        # Scatter: current xwOBA vs Δ Tilt
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
                title="Do batters nearer their optimal tilt perform better?",
            )
            fig7b.add_vline(x=0, line_dash="dash", line_color="grey")
            fig7b.update_layout(height=430)
            st.plotly_chart(fig7b, use_container_width=True)

        # Feature importance (GBM only)
        if "Gradient Boosting" in model_type and hasattr(model, "feature_importances_"):
            st.markdown("#### Model Feature Importance")
            imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
            fig_imp = px.bar(
                imp.reset_index(), x=0, y="index", orientation="h",
                labels={"0": "Importance", "index": "Feature"},
                color=0, color_continuous_scale="Blues",
                title="Gradient Boosting – Feature Importance",
            )
            fig_imp.update_layout(height=360, coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)
