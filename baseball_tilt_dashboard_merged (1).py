"""
MLB Bat Tracking 2025-2026  ·  Swing Intelligence Dashboard
============================================================
Połączona wersja: oryginalne heatmapy + działające 2 sezony 2025/2026
"""

import warnings
warnings.filterwarnings("ignore")

import os
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

try:
    from pygam import LinearGAM, s, f as gam_f
    HAS_PYGAM = True
except ImportError:
    HAS_PYGAM = False

try:
    import statsmodels
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import requests

st.set_page_config(
    page_title="MLB Swing Intelligence 2025-2026",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# CSS (oryginalny)
# ─────────────────────────────────────────────────────────────────────
DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Oswald:wght@500;700&display=swap');
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d1117 !important; color: #e6edf3; font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #21262d; }
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
.dash-header {
    background: linear-gradient(135deg, #0d1117 0%, #1a2332 50%, #0d2137 100%);
    border-bottom: 2px solid #f0a500; padding: 1.5rem 2rem; margin-bottom: 1.5rem;
    border-radius: 0 0 8px 8px;
}
.dash-title { font-family: 'Oswald', sans-serif; font-size: 2.2rem; font-weight: 700; color: #f0a500; letter-spacing: 2px; text-transform: uppercase; margin: 0; }
.dash-subtitle { font-size: 0.85rem; color: #8b949e; margin: 0.25rem 0 0; letter-spacing: 1px; }
.player-card {
    background: linear-gradient(145deg, #161b22, #1c2333); border: 1px solid #21262d;
    border-left: 4px solid #f0a500; border-radius: 8px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
}
.player-name { font-family: 'Oswald', sans-serif; font-size: 1.35rem; font-weight: 700; color: #f0a500; text-transform: uppercase; letter-spacing: 1px; }
.player-pos { font-size: 0.75rem; color: #8b949e; letter-spacing: 2px; }
.metric-pill { display: inline-block; background: #21262d; border-radius: 20px; padding: 0.3rem 0.8rem; margin: 0.2rem; font-size: 0.8rem; font-weight: 600; }
.pill-good  { border: 1px solid #3fb950; color: #3fb950; }
.pill-bad   { border: 1px solid #f85149; color: #f85149; }
.pill-avg   { border: 1px solid #8b949e; color: #8b949e; }
.pill-elite { border: 1px solid #f0a500; color: #f0a500; }
.section-hdr {
    font-family: 'Oswald', sans-serif; font-size: 1.1rem; color: #f0a500; letter-spacing: 2px;
    text-transform: uppercase; border-bottom: 1px solid #21262d; padding-bottom: 0.4rem; margin: 1.2rem 0 0.8rem;
}
.info-box { background: #161b22; border: 1px solid #21262d; border-left: 3px solid #388bfd; border-radius: 6px; padding: 1rem 1.2rem; margin: 0.5rem 0; font-size: 0.88rem; color: #c9d1d9; }
.warn-box { background: #1c1810; border-left: 3px solid #f0a500; border-radius: 6px; padding: 0.8rem 1.2rem; margin: 0.5rem 0; font-size: 0.85rem; color: #e6c77a; }
.ai-box { background: linear-gradient(135deg, #0d1f0d, #0f1f2d); border: 1px solid #3fb950; border-radius: 8px; padding: 1.2rem 1.4rem; font-size: 0.9rem; color: #c9d1d9; }
.ai-label { font-family:'Oswald',sans-serif; color: #3fb950; font-size: 0.75rem; letter-spacing: 2px; margin-bottom: 0.5rem; }
[data-testid="stTabs"] button { font-family: 'Oswald', sans-serif; letter-spacing: 1px; font-size: 0.85rem; color: #8b949e !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #f0a500 !important; border-bottom: 2px solid #f0a500 !important; }
[data-testid="stMetric"] { background: #161b22; border: 1px solid #21262d; border-radius: 6px; padding: 0.6rem 0.8rem; }
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.72rem !important; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; }
.js-plotly-plot .plotly { background: transparent !important; }
[data-testid="stDataFrame"] { border: 1px solid #21262d; border-radius: 6px; }
hr { border-color: #21262d !important; }
.gloss-term { font-family:'Oswald',sans-serif; color:#f0a500; font-size:1rem; font-weight:600; margin-top:1rem; }
.gloss-def { color:#c9d1d9; font-size:0.87rem; line-height:1.7; }
.gloss-example { color:#8b949e; font-size:0.8rem; font-style:italic; margin-top:0.25rem; }
[data-testid="stDownloadButton"] button { background: #21262d !important; color: #e6edf3 !important; border: 1px solid #30363d !important; border-radius: 6px !important; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

def styled_fig(fig, height=420):
    fig.update_layout(
        height=height, paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(color="#c9d1d9", family="Inter"),
        title_font=dict(color="#f0a500", family="Oswald", size=16),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#21262d"),
        xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
        yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────
SHRINKAGE_K = 50
TILT_MIN, TILT_MAX, TILT_GRID_N = 8.0, 62.0, 80
DATA_DIR = Path(".")
SEASONS = [2025, 2026]
CURRENT_SEASON = 2026

FEATURE_COLS = ["avg_tilt", "avg_aa", "avg_bat_speed", "avg_swing_len",
                "zone_enc", "group_enc", "tilt_x_aa", "tilt_x_group"]

METRIC_META = {
    "avg_tilt":          {"label": "Tilt (°)", "good": "high", "elite": (42, 999), "warn": (0, 25), "fmt": ".1f"},
    "avg_aa":            {"label": "Attack Angle (°)", "good": "mid", "elite": (8, 14), "warn": (-99, 2), "fmt": ".1f"},
    "avg_bat_speed":     {"label": "Bat Speed (mph)", "good": "high", "elite": (76, 999), "warn": (0, 68), "fmt": ".1f"},
    "avg_swing_len":     {"label": "Swing Len (ft)", "good": "mid", "elite": (6.5, 8), "warn": (0, 5.5), "fmt": ".2f"},
    "xwoba":             {"label": "xwOBA", "good": "high", "elite": (0.380, 9), "warn": (0, 0.280), "fmt": ".3f"},
    "batting_avg":       {"label": "BA", "good": "high", "elite": (0.285, 9), "warn": (0, 0.220), "fmt": ".3f"},
    "avg_exit_velocity": {"label": "EV (mph)", "good": "high", "elite": (92, 999), "warn": (0, 86), "fmt": ".1f"},
    "avg_launch_angle":  {"label": "LA (°)", "good": "mid", "elite": (10, 18), "warn": (-99, 0), "fmt": ".1f"},
    "swings":            {"label": "Swings", "good": "high", "elite": (300, 999), "warn": (0, 50), "fmt": ".0f"},
}

HEATMAP_RANGES = {
    "avg_tilt": (8, 60), "tilt_std": (0, 20), "delta_tilt": (-20, 20),
    "avg_aa": (-35, 35), "aa_std": (0, 20), "avg_bat_speed": (55, 88),
    "avg_swing_len": (4.5, 9.5), "swings": (0, 400), "batting_avg": (0.150, 0.400),
    "xwoba": (0.200, 0.600), "avg_exit_velocity": (75, 105), "avg_launch_angle": (-15, 45),
}

# ─────────────────────────────────────────────────────────────────────
# DATA LOADING – odporne na brak jednego sezonu
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⚾ Loading Statcast data…")
def load_season(season: int):
    pf = DATA_DIR / f"players_summary_{season}.csv"
    df_f = DATA_DIR / f"detail_zone_pitchgroup_{season}.csv"
    players = None
    detail = None
    if pf.exists():
        players = pd.read_csv(pf)
        players = players.loc[:, ~players.columns.duplicated()].copy()
        players["season"] = season
    if df_f.exists():
        detail = pd.read_csv(df_f)
        detail = detail.loc[:, ~detail.columns.duplicated()].copy()
        mask = (detail["batter_name"].notna() &
                ~detail["batter_name"].str.contains(r" pitcher| P$", case=False, na=False, regex=True))
        detail = detail[mask].copy()
        detail["season"] = season
    return players, detail

@st.cache_data(show_spinner="⚾ Building multi-season dataset…")
def load_all_seasons():
    all_players, all_detail = [], []
    for s in SEASONS:
        p, d = load_season(s)
        if p is not None:
            all_players.append(p)
        if d is not None:
            all_detail.append(d)
    players = pd.concat(all_players, ignore_index=True) if all_players else pd.DataFrame()
    detail = pd.concat(all_detail, ignore_index=True) if all_detail else pd.DataFrame()
    return players, detail

players_all, detail_all = load_all_seasons()
avail_seasons = sorted(detail_all["season"].dropna().unique().tolist()) if not detail_all.empty else []

with st.sidebar:
    st.markdown("### 📅 Season")
    st.caption(f"Wykryte sezony: **{avail_seasons}**")
    if len(avail_seasons) > 1:
        sel_season = st.selectbox("Primary season", avail_seasons, index=len(avail_seasons)-1, key="season_pick")
    elif avail_seasons:
        sel_season = avail_seasons[0]
        st.caption(f"Only **{sel_season}** available")
    else:
        sel_season = CURRENT_SEASON
        st.error("Brak plików CSV!")

MAIN_SEASON = sel_season
players_raw = players_all[players_all["season"] == MAIN_SEASON].copy() if not players_all.empty else pd.DataFrame()
detail_full = detail_all[detail_all["season"] == MAIN_SEASON].copy() if not detail_all.empty else pd.DataFrame()

if detail_full.empty:
    st.error(f"❌ Brak danych dla sezonu {MAIN_SEASON}.")
    st.stop()

_obs_tilt = detail_full["avg_tilt"].dropna()
if len(_obs_tilt) >= 20:
    TILT_MIN = float(max(5.0, np.percentile(_obs_tilt, 1) - 2.0))
    TILT_MAX = float(min(75.0, np.percentile(_obs_tilt, 99) + 2.0))

TILT_SEARCH_WINDOW = 15.0
all_real = sorted(players_raw["batter_name"].dropna().unique()) if not players_raw.empty else sorted(detail_full["batter_name"].dropna().unique())

def _fp(df, name):
    return df[df["batter_name"] == name].copy()

def _engineer(df):
    out = df.copy()
    le_z = LabelEncoder()
    le_g = LabelEncoder()
    out["zone_enc"] = le_z.fit_transform(out["zone"].astype(str))
    out["group_enc"] = le_g.fit_transform(out["pitch_group"].fillna("Unknown").astype(str))
    out["tilt_x_aa"] = out["avg_tilt"] * out["avg_aa"]
    out["tilt_x_group"] = out["avg_tilt"] * out["group_enc"]
    out["sample_weight"] = np.sqrt(out["swings"].clip(lower=1))
    return out, le_z, le_g

detail_fe, le_zone_g, le_group_g = _engineer(detail_full)

@st.cache_resource(show_spinner="🤖 Training model…")
def train_model():
    df = detail_fe.dropna(subset=["xwoba"]).query("swings >= 5").copy()
    if df.empty:
        return None, "No data", None
    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median()).values
    y = df["xwoba"].values
    w = df["sample_weight"].values
    metrics = None
    try:
        from sklearn.model_selection import KFold
        from sklearn.metrics import r2_score, mean_absolute_error
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        r2s, maes = [], []
        for tr_idx, te_idx in kf.split(X):
            gbm_cv = GradientBoostingRegressor(n_estimators=400, max_depth=4, learning_rate=0.035,
                                               subsample=0.75, min_samples_leaf=4, max_features=0.8, random_state=42)
            gbm_cv.fit(X[tr_idx], y[tr_idx], sample_weight=w[tr_idx])
            pred = gbm_cv.predict(X[te_idx])
            r2s.append(r2_score(y[te_idx], pred))
            maes.append(mean_absolute_error(y[te_idx], pred))
        metrics = {"r2": float(np.mean(r2s)), "mae": float(np.mean(maes))}
    except Exception:
        pass
    gbm = GradientBoostingRegressor(n_estimators=400, max_depth=4, learning_rate=0.035,
                                    subsample=0.75, min_samples_leaf=4, max_features=0.8, random_state=42)
    gbm.fit(X, y, sample_weight=w)
    return gbm, "Gradient Boosting", metrics

model, model_type, MODEL_METRICS = train_model()

def _sm(s):
    v = s.dropna()
    return float(v.mean()) if len(v) else np.nan

def shrink(pv, lv, n, k=SHRINKAGE_K):
    w = n / (n + k)
    return w * pv + (1 - w) * lv, round(w, 3)

def predict_tilt_curve(avg_aa, avg_speed, avg_len, zone_enc, group_enc):
    tg = np.linspace(TILT_MIN, TILT_MAX, TILT_GRID_N)
    if model is None:
        return tg, np.full(TILT_GRID_N, np.nan)
    X = np.column_stack([
        tg, np.full(TILT_GRID_N, avg_aa), np.full(TILT_GRID_N, avg_speed),
        np.full(TILT_GRID_N, avg_len), np.full(TILT_GRID_N, zone_enc),
        np.full(TILT_GRID_N, group_enc), tg * avg_aa, tg * group_enc
    ])
    return tg, gaussian_filter1d(model.predict(X), sigma=1.8)

def find_optimal(tg, preds):
    if np.all(np.isnan(preds)):
        return float(np.mean(tg)), float("nan")
    return float(tg[int(np.nanargmax(preds))]), float(np.nanmax(preds))

def find_optimal_near(tg, preds, center, window=TILT_SEARCH_WINDOW):
    if np.all(np.isnan(preds)):
        return float(np.mean(tg)), float("nan"), True
    mask = (tg >= center - window) & (tg <= center + window)
    if not mask.any():
        mask = np.ones_like(tg, dtype=bool)
    idx_all = np.where(mask)[0]
    j = int(np.nanargmax(preds[idx_all]))
    idx_best = idx_all[j]
    at_edge = (j == 0) or (j == len(idx_all) - 1)
    return float(tg[idx_best]), float(preds[idx_best]), bool(at_edge)

@st.cache_data
def league_stats_for(df):
    cols = [c for c in METRIC_META if c in df.columns]
    return {c: float(df[c].mean()) for c in cols}

LG = league_stats_for(detail_full)

_lg_ze = float(np.median(detail_fe["zone_enc"])) if not detail_fe.empty else 0.0
_lg_ge = float(np.median(detail_fe["group_enc"])) if not detail_fe.empty else 0.0
_tg_lg, _pr_lg = predict_tilt_curve(
    _sm(detail_full["avg_aa"]), _sm(detail_full["avg_bat_speed"]),
    _sm(detail_full["avg_swing_len"]), _lg_ze, _lg_ge
)
LEAGUE_OPT_TILT, _ = find_optimal(_tg_lg, _pr_lg)

@st.cache_data(show_spinner="Computing rankings…", ttl=3600)
def build_opt_table(_df, ze, ge, lo, window=TILT_SEARCH_WINDOW):
    agg = _df.groupby("batter_name", observed=True).agg(
        avg_tilt=("avg_tilt", "mean"), avg_aa=("avg_aa", "mean"),
        avg_bat_speed=("avg_bat_speed", "mean"), avg_swing_len=("avg_swing_len", "mean"),
        xwoba=("xwoba", "mean"), swings=("swings", "sum"),
    ).reset_index().dropna(subset=["avg_tilt", "avg_aa", "avg_bat_speed", "avg_swing_len"])
    agg = agg[agg["swings"] >= 5].reset_index(drop=True)
    if agg.empty or model is None:
        return pd.DataFrame()
    n = len(agg)
    tg = np.linspace(TILT_MIN, TILT_MAX, TILT_GRID_N)
    aa = agg["avg_aa"].to_numpy()[:, None]
    spd = agg["avg_bat_speed"].to_numpy()[:, None]
    ln = agg["avg_swing_len"].to_numpy()[:, None]
    cur = agg["avg_tilt"].to_numpy()
    TG = np.tile(tg, (n, 1))
    AA = np.tile(aa, (1, TILT_GRID_N))
    SPD = np.tile(spd, (1, TILT_GRID_N))
    LN = np.tile(ln, (1, TILT_GRID_N))
    ZE = np.full_like(TG, ze)
    GE = np.full_like(TG, ge)
    X = np.column_stack([TG.ravel(), AA.ravel(), SPD.ravel(), LN.ravel(),
                         ZE.ravel(), GE.ravel(), (TG * AA).ravel(), (TG * ge).ravel()])
    preds = model.predict(X).reshape(n, TILT_GRID_N)
    preds = gaussian_filter1d(preds, sigma=1.8, axis=1)
    win_mask = np.abs(TG - cur[:, None]) <= window
    has_support = win_mask.any(axis=1)
    preds_masked = np.where(win_mask, preds, -np.inf)
    preds_masked[~has_support] = preds[~has_support]
    idx_opt = np.argmax(preds_masked, axis=1)
    ro_arr = tg[idx_opt]
    ox_arr = preds_masked[np.arange(n), idx_opt]
    first_true = win_mask.argmax(axis=1)
    last_true = TILT_GRID_N - 1 - win_mask[:, ::-1].argmax(axis=1)
    extrapolated = (~has_support) | (idx_opt == first_true) | (idx_opt == last_true)
    n_arr = agg["swings"].to_numpy()
    w_arr = n_arr / (n_arr + SHRINKAGE_K)
    so_arr = w_arr * ro_arr + (1 - w_arr) * lo
    so_arr = np.where(extrapolated, cur + 0.4 * (so_arr - cur), so_arr)
    w_arr_disp = np.where(extrapolated, w_arr * 0.5, w_arr)
    delta = cur - so_arr
    direction = np.where(np.abs(delta) <= 2.5, "≈ near-optimal",
                         np.where(delta > 0, "↓ flatten " + np.round(np.abs(delta), 1).astype(str) + "°",
                                  "↑ steepen " + np.round(np.abs(delta), 1).astype(str) + "°"))
    out = pd.DataFrame({
        "Batter": agg["batter_name"], "Swings": n_arr.astype(int),
        "Current Tilt": np.round(cur, 1), "Optimal Tilt (raw)": np.round(ro_arr, 1),
        "Optimal Tilt": np.round(so_arr, 1), "Δ Tilt": np.round(delta, 1),
        "Direction": direction, "Pred. xwOBA @ Opt.": np.round(ox_arr, 3),
        "Confidence": np.round(w_arr_disp, 2), "⚠ Extrapolated": extrapolated,
        "Current xwOBA": agg["xwoba"].round(3),
    })
    return out.sort_values("Swings", ascending=False).reset_index(drop=True)

def _clean(df, *cols):
    return df[list(dict.fromkeys(c for c in cols if c in df.columns))].copy()

@st.cache_data
def batter_summary(df):
    cols = [c for c in METRIC_META if c in df.columns] + ["swings"]
    agg = {c: ("sum" if c == "swings" else "mean") for c in set(cols)}
    return df.groupby("batter_name", observed=True).agg(agg).round(3).reset_index()

summary_df = batter_summary(detail_full)

def get_batter_row(name):
    row = summary_df[summary_df["batter_name"] == name]
    return row.iloc[0] if not row.empty else None

def metric_class(metric, value):
    if pd.isna(value):
        return "pill-avg"
    m = METRIC_META.get(metric, {})
    lo_e, hi_e = m.get("elite", (999, -999))
    lo_w, hi_w = m.get("warn", (-999, 999))
    if lo_e <= value <= hi_e:
        return "pill-elite"
    if lo_w <= value <= hi_w:
        return "pill-bad"
    lg = LG.get(metric, value)
    return "pill-good" if value >= lg else "pill-avg"

def fmt_val(metric, value):
    if pd.isna(value):
        return "—"
    f = METRIC_META.get(metric, {}).get("fmt", ".1f")
    return format(value, f)

def render_player_card(name, row):
    if row is None:
        st.warning(f"No data for {name}")
        return
    pills = ""
    for m in ["avg_tilt", "avg_aa", "avg_bat_speed", "xwoba", "avg_exit_velocity"]:
        if m not in row.index or pd.isna(row[m]):
            continue
        cls = metric_class(m, row[m])
        pills += f'<span class="metric-pill {cls}">{METRIC_META[m]["label"]}: {fmt_val(m, row[m])}</span>'
    tilt = row.get("avg_tilt", np.nan)
    if pd.isna(tilt):
        profile = "Unknown"
    elif tilt >= 45:
        profile = "Elite Uppercut"
    elif tilt >= 35:
        profile = "Power Tilt"
    elif tilt >= 25:
        profile = "Balanced"
    elif tilt >= 15:
        profile = "Line Drive"
    else:
        profile = "Flat / Ground Ball"
    n = int(row.get("swings", 0))
    st.markdown(f"""
    <div class="player-card">
      <div class="player-name">{name}</div>
      <div class="player-pos">SWING PROFILE: {profile} · {n} swings</div>
      <div style="margin-top:0.8rem">{pills}</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# HEATMAP – ORYGINALNA FUNKCJA (ze strefami 1-14)
# ─────────────────────────────────────────────────────────────────────
def _zone_color(val, vmin, vmax, cmap):
    if pd.isna(val):
        return (0.12, 0.13, 0.15)
    return cmap(float(np.clip((val - vmin) / max(vmax - vmin, 1e-9), 0, 1)))

def _tc(bg):
    r, g, b = bg[:3]
    return "#e6edf3" if 0.299 * r + 0.587 * g + 0.114 * b < 0.45 else "#0d1117"

def _fv2(val, metric):
    if pd.isna(val):
        return "—"
    if metric == "swings":
        return str(int(round(val)))
    if metric in ("batting_avg", "xwoba"):
        return f"{val:.3f}"
    return f"{val:.1f}"

def _get_pivot(df_p, metric, league_df, df_ctx):
    if metric == "tilt_std":
        return df_p.groupby("zone")["avg_tilt"].std(ddof=1).round(1)
    if metric == "aa_std":
        return df_p.groupby("zone")["avg_aa"].std(ddof=1).round(1)
    if metric == "delta_tilt":
        pm = df_p.groupby("zone")["avg_tilt"].mean()
        lm = league_df.set_index("zone")["avg_tilt"] if league_df is not None else pd.Series()
        return (pm - lm.reindex(pm.index, fill_value=np.nan)).round(2)
    return (df_p.groupby("zone")["swings"].sum() if metric == "swings"
            else df_p.groupby("zone")[metric].mean()).round(3)

def make_heatmap(df_p, metric, title, league_df=None, df_ctx=None, figsize=(7, 7)):
    if df_p is None or df_p.empty:
        st.warning(f"No data for {title}")
        return
    if df_ctx is None:
        df_ctx = detail_full
    pivot = _get_pivot(df_p, metric, league_df, df_ctx)
    nsw = df_p.groupby("zone")["swings"].sum() if "swings" in df_p.columns else pd.Series()
    if metric == "delta_tilt":
        vmin, vmax = HEATMAP_RANGES.get(metric, (-20, 20))
        cn = "RdBu_r"
    else:
        vmin, vmax = HEATMAP_RANGES.get(metric, (0, 100))
        cn = "YlOrRd"
    cmap = sns.color_palette(cn, as_cmap=True)
    b, ms, sy = 0.85, 3.3, 2.5
    mx, my = b, b
    tx, ty = mx + ms, my + ms
    half = ms / 2
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    for i in range(3):
        for j in range(3):
            zone = i * 3 + j + 1
            val = pivot.get(zone, np.nan)
            n = int(nsw.get(zone, 0))
            x = mx + j * (ms / 3)
            y = my + (2 - i) * (ms / 3)
            col = _zone_color(val, vmin, vmax, cmap)
            ax.add_patch(plt.Rectangle((x, y), ms / 3, ms / 3, facecolor=col, edgecolor="#21262d", linewidth=2))
            txt = f"{zone}\n{_fv2(val, metric)}" + (f"\n⚠n={n}" if n < 20 else "")
            ax.text(x + ms / 6, y + ms / 6, txt, ha="center", va="center", fontsize=9.5,
                    fontweight="bold", color=_tc(col), fontfamily="monospace")
    ld = [
        (11, [(0, sy), (b, sy), (b, ty), (mx, ty), (mx + half, ty), (mx + half, 5), (0, 5), (0, sy)], (b * 0.4, 5 - b * 0.4)),
        (12, [(tx, sy), (tx, ty), (mx + half, ty), (mx + half, 5), (5, 5), (5, sy), (tx, sy)], (5 - b * 0.4, 5 - b * 0.4)),
        (13, [(0, sy), (b, sy), (b, my), (mx, my), (mx + half, my), (mx + half, 0), (0, 0), (0, sy)], (b * 0.4, b * 0.4)),
        (14, [(tx, sy), (tx, my), (mx + half, my), (mx + half, 0), (5, 0), (5, sy), (tx, sy)], (5 - b * 0.4, b * 0.4))
    ]
    for z, verts, (cx, cy) in ld:
        val = pivot.get(z, np.nan)
        n = int(nsw.get(z, 0))
        col = _zone_color(val, vmin, vmax, cmap)
        ax.add_patch(PathPatch(MPath(verts), facecolor=col, edgecolor="#21262d", linewidth=2))
        txt = f"{z}\n{_fv2(val, metric)}" + (f"\n⚠n={n}" if n < 20 else "")
        ax.text(cx, cy, txt, ha="center", va="center", fontsize=9.5, fontweight="bold", color=_tc(col))
    ax.add_patch(plt.Rectangle((mx, my), ms, ms, fill=False, edgecolor="#f0a500", linewidth=2.5))
    ax.set_title(title, fontsize=13, fontweight="bold", color="#f0a500", pad=14)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, ax=ax, shrink=0.72, pad=0.04)
    cbar.set_label(metric.upper(), fontsize=9, color="#8b949e")
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b949e", fontsize=8)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────
# HEADER + SIDEBAR
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <div class="dash-title">⚾ MLB Swing Intelligence</div>
  <div class="dash-subtitle">BAT TRACKING · 2025–2026</div>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚾ Batter Selection")
    spm = st.multiselect("Select Batters", all_real, default=[all_real[0]] if all_real else [], max_selections=6)
    st.markdown("### 🎛 Pitch Filters")
    pg_list = ["All"] + sorted(detail_full["pitch_group"].dropna().unique())
    sel_pitch = st.selectbox("Pitch Group", pg_list)
    pt_list = ["All"]
    if "pitch_type" in detail_full.columns:
        _pts = detail_full if sel_pitch == "All" else detail_full[detail_full["pitch_group"] == sel_pitch]
        pt_list += sorted(_pts["pitch_type"].dropna().unique())
    sel_type = st.selectbox("Pitch Type", pt_list)
    min_swings = st.slider("Min. swings filter", 0, 300, 0, 10)
    show_ci = st.checkbox("Show CI bands", value=True)
    st.caption(f"Season: **{MAIN_SEASON}** · Model: **{model_type}**")
    st.caption(f"League opt. tilt: **{LEAGUE_OPT_TILT:.1f}°**")

def apf(df, pg, pt):
    if pg != "All":
        df = df[df["pitch_group"] == pg]
    if pt != "All" and "pitch_type" in df.columns:
        df = df[df["pitch_type"] == pt]
    return df

dff = apf(detail_full.copy(), sel_pitch, sel_type)
lpz = dff.groupby("zone", as_index=False, observed=True).agg(
    {c: "mean" for c in METRIC_META if c in dff.columns} | {"swings": "sum"}
).round(3)
lpz["batter_name"] = "League Average"
dfe_f = apf(detail_fe.copy(), sel_pitch, sel_type)
_cze = float(np.median(dfe_f["zone_enc"])) if not dfe_f.empty else _lg_ze
_cge = float(np.median(dfe_f["group_enc"])) if not dfe_f.empty else _lg_ge

# ─────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────
tab_ov, tab_exp, tab_cmp, tab_tr, tab_opt, tab_rank, tab_gl = st.tabs([
    "🏠 Overview", "🔍 Player Explorer", "⚔️ Comparisons",
    "📈 Trends", "🔬 Tilt Optimizer", "🏆 Rankings", "📚 Glossary"
])

# ── OVERVIEW ─────────────────────────────────────────────────────────
with tab_ov:
    st.markdown('<div class="section-hdr">League Snapshot</div>', unsafe_allow_html=True)
    k_cols = st.columns(5)
    for col, m in zip(k_cols, ["avg_tilt", "avg_aa", "avg_bat_speed", "xwoba", "avg_exit_velocity"]):
        col.metric(METRIC_META[m]["label"], fmt_val(m, LG.get(m, np.nan)), "League Avg", delta_color="off")

# ── PLAYER EXPLORER ──────────────────────────────────────────────────
with tab_exp:
    st.markdown('<div class="section-hdr">Player Explorer</div>', unsafe_allow_html=True)
    sel_exp = st.selectbox("Select Batter", all_real, key="exp_sel")
    row_exp = get_batter_row(sel_exp)
    render_player_card(sel_exp, row_exp)

    st.markdown('<div class="section-hdr">Zone Heatmaps</div>', unsafe_allow_html=True)
    hz_metric = st.radio("Metric", ["avg_tilt", "delta_tilt", "avg_aa", "xwoba", "avg_bat_speed", "avg_exit_velocity"],
                         format_func=lambda x: METRIC_META.get(x, {}).get("label", x), horizontal=True, key="hz_m")
    df_exp = _fp(dff, sel_exp)
    make_heatmap(df_exp, hz_metric, sel_exp, lpz, dff)

    if not df_exp.empty:
        st.markdown('<div class="section-hdr">Zone Breakdown</div>', unsafe_allow_html=True)
        z_cols = [c for c in ["zone", "swings", "avg_tilt", "avg_aa", "avg_bat_speed", "xwoba"] if c in df_exp.columns]
        zt = df_exp.groupby("zone", as_index=False)[z_cols[1:]].agg(
            {c: ("sum" if c == "swings" else "mean") for c in z_cols[1:]}
        ).round(3)
        st.dataframe(zt.sort_values("swings", ascending=False), use_container_width=True, hide_index=True)

# ── COMPARISONS ──────────────────────────────────────────────────────
with tab_cmp:
    st.markdown('<div class="section-hdr">Multi-Batter Comparison</div>', unsafe_allow_html=True)
    if not spm:
        st.info("Select 2–6 batters in the sidebar.")
    else:
        for name in spm:
            render_player_card(name, get_batter_row(name))
        st.markdown('<div class="section-hdr">Side-by-Side Zone Heatmaps</div>', unsafe_allow_html=True)
        hz2 = st.radio("Heatmap metric", ["avg_tilt", "delta_tilt", "xwoba", "avg_bat_speed"],
                       format_func=lambda x: METRIC_META.get(x, {}).get("label", x), horizontal=True, key="cmp_hz")
        hm_cols = st.columns(min(len(spm), 3))
        for i, name in enumerate(spm[:3]):
            with hm_cols[i]:
                make_heatmap(_fp(dff, name), hz2, name, lpz, dff, figsize=(5.5, 5.5))

# ── TRENDS ───────────────────────────────────────────────────────────
with tab_tr:
    st.markdown('<div class="section-hdr">Trends 2025 → 2026</div>', unsafe_allow_html=True)
    if len(avail_seasons) < 2:
        st.warning("Potrzebne dane z obu sezonów.")
    else:
        tr_metric = st.selectbox("Metric", ["avg_tilt", "avg_aa", "avg_bat_speed", "xwoba"],
                                 format_func=lambda x: METRIC_META[x]["label"], key="trm")
        fig_tr = go.Figure()
        lg_vals = []
        for s in avail_seasons:
            _, d = load_season(s)
            if d is not None and tr_metric in d.columns:
                lg_vals.append({"Season": str(s), "Value": round(d[tr_metric].mean(), 3)})
        if lg_vals:
            lgt = pd.DataFrame(lg_vals)
            fig_tr.add_trace(go.Scatter(x=lgt["Season"], y=lgt["Value"], mode="lines+markers",
                                        name="League", line=dict(color="#388bfd", width=2.5), marker=dict(size=10)))
        for name in spm:
            vals = []
            for s in avail_seasons:
                _, d = load_season(s)
                if d is None:
                    continue
                sub = d[d["batter_name"] == name]
                val = sub[tr_metric].mean() if not sub.empty and tr_metric in sub.columns else np.nan
                vals.append({"Season": str(s), "Value": round(val, 3) if not pd.isna(val) else None})
            bdf = pd.DataFrame(vals).dropna()
            if not bdf.empty:
                fig_tr.add_trace(go.Scatter(x=bdf["Season"], y=bdf["Value"], mode="lines+markers",
                                            name=name, line=dict(width=2), marker=dict(size=9)))
        fig_tr = styled_fig(fig_tr, 420)
        fig_tr.update_layout(title=f"{METRIC_META[tr_metric]['label']} — 2025 vs 2026",
                             xaxis_title="Season", yaxis_title=METRIC_META[tr_metric]["label"],
                             xaxis=dict(type="category"))
        st.plotly_chart(fig_tr, use_container_width=True)

# ── OPTIMIZER ────────────────────────────────────────────────────────
with tab_opt:
    st.markdown('<div class="section-hdr">Tilt Optimizer</div>', unsafe_allow_html=True)
    if model is None:
        st.error("Model niedostępny.")
    else:
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1.2])
        sp6 = c1.selectbox("Batter", all_real, key="optp")
        pg6 = c2.selectbox("Pitch Group", ["All"] + sorted(detail_full["pitch_group"].dropna().unique()), key="optpg")
        z6 = c3.selectbox("Zone", ["All"] + [str(z) for z in range(1, 15)], key="optz")
        win6 = c4.slider("Window (°)", 5, 30, 15, 1, key="optwin")
        p6d = _fp(detail_fe, sp6)
        if pg6 != "All":
            p6d = p6d[p6d["pitch_group"] == pg6]
        if z6 != "All":
            p6d = p6d[p6d["zone"] == int(z6)]
        if p6d.empty:
            st.warning("Brak danych.")
        else:
            n6 = int(p6d["swings"].sum())
            aa6 = _sm(p6d["avg_aa"])
            spd6 = _sm(p6d["avg_bat_speed"])
            len6 = _sm(p6d["avg_swing_len"])
            ct = _sm(p6d["avg_tilt"])
            ge6 = float(p6d["group_enc"].mean())
            ze6 = float(p6d["zone_enc"].mean())
            tg6, pr6 = predict_tilt_curve(aa6, spd6, len6, ze6, ge6)
            ro6, ox6, extrap = find_optimal_near(tg6, pr6, ct, window=float(win6))
            so6, cw6 = shrink(ro6, LEAGUE_OPT_TILT, n6)
            if extrap:
                so6 = ct + 0.4 * (so6 - ct)
                cw6 *= 0.5
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Current", f"{ct:.1f}°")
            k2.metric("Optimal", f"{so6:.1f}°")
            k3.metric("Δ", f"{ct - so6:+.1f}°")
            k4.metric("Conf.", f"{cw6:.0%}")
            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(x=tg6, y=pr6, mode="lines", line=dict(color="#f0a500", width=2.5), name=sp6))
            fig6.add_vline(x=ct, line=dict(color="#3fb950", dash="dash"), annotation_text="Current")
            fig6.add_vline(x=so6, line=dict(color="#f0a500"), annotation_text="Optimal")
            fig6 = styled_fig(fig6, 420)
            fig6.update_layout(title=f"xwOBA vs Tilt — {sp6}", xaxis_title="Tilt (°)", yaxis_title="Pred. xwOBA")
            st.plotly_chart(fig6, use_container_width=True)

# ── RANKINGS ─────────────────────────────────────────────────────────
with tab_rank:
    st.markdown('<div class="section-hdr">Tilt Rankings</div>', unsafe_allow_html=True)
    opt_table = build_opt_table(dff, _cze, _cge, LEAGUE_OPT_TILT)
    if not opt_table.empty:
        st.dataframe(opt_table, use_container_width=True, hide_index=True)
        st.download_button("⬇ Download rankings CSV", opt_table.to_csv(index=False),
                           "tilt_rankings.csv", "text/csv", key="dl_rank")
    else:
        st.info("Brak danych.")

# ── GLOSSARY ─────────────────────────────────────────────────────────
with tab_gl:
    st.markdown('<div class="section-hdr">Glossary</div>', unsafe_allow_html=True)
    st.markdown("""
**Swing Path Tilt** — kąt płaszczyzny swinga względem ziemi.  
**Attack Angle** — kąt lufy w momencie kontaktu (opt. 8–15°).  
**Bat Speed** — prędkość lufy (mph).  
**xwOBA** — expected wOBA.  
**Optimal Tilt** — tilt maksymalizujący xwOBA według modelu.
    """)
