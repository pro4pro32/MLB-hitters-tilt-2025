"""
MLB Bat Tracking 2025-2026  ·  Swing Intelligence Dashboard
============================================================
Full redesign: dark stadium theme, Player Cards, AI Insights,
multi-year trends, education layer, scatter correlations.
Analyzed seasons: 2025–2026
"""

import warnings; warnings.filterwarnings("ignore")

import os, json, re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.stats import percentileofscore, pearsonr

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

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLB Swing Intelligence 2025-2026",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# THEME  — Stadium Night palette
# ─────────────────────────────────────────────────────────────────────
DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Oswald:wght@500;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d1117 !important;
    color: #e6edf3;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }

.dash-header {
    background: linear-gradient(135deg, #0d1117 0%, #1a2332 50%, #0d2137 100%);
    border-bottom: 2px solid #f0a500;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    border-radius: 0 0 8px 8px;
}
.dash-title {
    font-family: 'Oswald', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #f0a500;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 0;
}
.dash-subtitle { font-size: 0.85rem; color: #8b949e; margin: 0.25rem 0 0; letter-spacing: 1px; }

.player-card {
    background: linear-gradient(145deg, #161b22, #1c2333);
    border: 1px solid #21262d;
    border-left: 4px solid #f0a500;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.player-card::before {
    content: "";
    position: absolute;
    top: -30px; right: -30px;
    width: 100px; height: 100px;
    border: 2px solid rgba(240,165,0,0.12);
    border-radius: 50%;
}
.player-name {
    font-family: 'Oswald', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #f0a500;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.player-pos { font-size: 0.75rem; color: #8b949e; letter-spacing: 2px; }

.metric-pill {
    display: inline-block;
    background: #21262d;
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    margin: 0.2rem;
    font-size: 0.8rem;
    font-weight: 600;
}
.pill-good  { border: 1px solid #3fb950; color: #3fb950; }
.pill-bad   { border: 1px solid #f85149; color: #f85149; }
.pill-avg   { border: 1px solid #8b949e; color: #8b949e; }
.pill-elite { border: 1px solid #f0a500; color: #f0a500; }

.metric-big { font-family:'Oswald',sans-serif; font-size:2rem; font-weight:700; line-height:1; }
.metric-lbl { font-size:0.7rem; color:#8b949e; letter-spacing:1px; text-transform:uppercase; }

.section-hdr {
    font-family: 'Oswald', sans-serif;
    font-size: 1.1rem;
    color: #f0a500;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.4rem;
    margin: 1.2rem 0 0.8rem;
}

.info-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid #388bfd;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    line-height: 1.6;
    color: #c9d1d9;
}
.warn-box {
    background: #1c1810;
    border-left: 3px solid #f0a500;
    border-radius: 6px;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: #e6c77a;
}

.ai-box {
    background: linear-gradient(135deg, #0d1f0d, #0f1f2d);
    border: 1px solid #3fb950;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #c9d1d9;
    position: relative;
}
.ai-label {
    font-family:'Oswald',sans-serif;
    color: #3fb950;
    font-size: 0.75rem;
    letter-spacing: 2px;
    margin-bottom: 0.5rem;
}

[data-testid="stTabs"] button {
    font-family: 'Oswald', sans-serif;
    letter-spacing: 1px;
    font-size: 0.85rem;
    color: #8b949e !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #f0a500 !important;
    border-bottom: 2px solid #f0a500 !important;
}

[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 0.6rem 0.8rem;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.72rem !important; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; }

.js-plotly-plot .plotly { background: transparent !important; }
[data-testid="stDataFrame"] { border: 1px solid #21262d; border-radius: 6px; }

[data-testid="stSelectbox"] > div, [data-testid="stMultiSelect"] > div {
    background: #161b22 !important;
    border-color: #21262d !important;
}

hr { border-color: #21262d !important; }

.gloss-term {
    font-family:'Oswald',sans-serif;
    color:#f0a500;
    font-size:1rem;
    font-weight:600;
    margin-top:1rem;
}
.gloss-def { color:#c9d1d9; font-size:0.87rem; line-height:1.7; }
.gloss-example { color:#8b949e; font-size:0.8rem; font-style:italic; margin-top:0.25rem; }

[data-testid="stDownloadButton"] button {
    background: #21262d !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# PLOTLY TEMPLATE
# ─────────────────────────────────────────────────────────────────────
PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font_color="#c9d1d9",
    font_family="Inter",
    title_font_family="Oswald",
    title_font_color="#f0a500",
    colorway=["#f0a500","#388bfd","#3fb950","#f85149","#bc8cff","#58a6ff","#56d364"],
)

def styled_fig(fig, height=420):
    fig.update_layout(
        height=height,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
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
SEASONS = [2025, 2026]          # ← tylko 2025-2026
CURRENT_SEASON = 2026

FEATURE_COLS = ["avg_tilt","avg_aa","avg_bat_speed","avg_swing_len",
                "zone_enc","group_enc","tilt_x_aa","tilt_x_group"]

METRIC_META = {
    "avg_tilt":          {"label":"Tilt (°)",         "good":"high",  "elite":(42,999), "warn":(0,25),  "fmt":".1f"},
    "avg_aa":            {"label":"Attack Angle (°)",  "good":"mid",   "elite":(8,14),   "warn":(-99,2), "fmt":".1f"},
    "avg_bat_speed":     {"label":"Bat Speed (mph)",   "good":"high",  "elite":(76,999), "warn":(0,68),  "fmt":".1f"},
    "avg_swing_len":     {"label":"Swing Len (ft)",    "good":"mid",   "elite":(6.5,8),  "warn":(0,5.5), "fmt":".2f"},
    "xwoba":             {"label":"xwOBA",             "good":"high",  "elite":(0.380,9),  "warn":(0,0.280),"fmt":".3f"},
    "batting_avg":       {"label":"BA",                "good":"high",  "elite":(0.285,9),  "warn":(0,0.220),"fmt":".3f"},
    "avg_exit_velocity": {"label":"EV (mph)",          "good":"high",  "elite":(92,999), "warn":(0,86),  "fmt":".1f"},
    "avg_launch_angle":  {"label":"LA (°)",            "good":"mid",   "elite":(10,18),  "warn":(-99,0), "fmt":".1f"},
    "swings":            {"label":"Swings",            "good":"high",  "elite":(300,999),"warn":(0,50),  "fmt":".0f"},
}

HEATMAP_RANGES = {
    "avg_tilt":(8,60),"tilt_std":(0,20),"delta_tilt":(-20,20),
    "avg_aa":(-35,35),"aa_std":(0,20),"avg_bat_speed":(55,88),
    "avg_swing_len":(4.5,9.5),"swings":(0,400),"batting_avg":(0.150,0.400),
    "xwoba":(0.200,0.600),"avg_exit_velocity":(75,105),"avg_launch_angle":(-15,45),
}

# ─────────────────────────────────────────────────────────────────────
# DATA LOADING  (multi-season)
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⚾  Loading Statcast data …")
def load_season(season: int):
    """Load players_summary & detail for a given season. Returns (players, detail) or (None,None)."""
    pf = DATA_DIR / f"players_summary_{season}.csv"
    df_f = DATA_DIR / f"detail_zone_pitchgroup_{season}.csv"
    if not pf.exists():
        return None, None
    players = pd.read_csv(pf)
    players = players.loc[:,~players.columns.duplicated()].copy()
    players["season"] = season
    if not df_f.exists():
        return players, None
    detail = pd.read_csv(df_f)
    detail = detail.loc[:,~detail.columns.duplicated()].copy()
    mask = (detail["batter_name"].notna() &
            ~detail["batter_name"].str.contains(r" pitcher| P$",case=False,na=False,regex=True))
    detail = detail[mask].copy()
    detail["season"] = season
    return players, detail

@st.cache_data(show_spinner="⚾  Building multi-season dataset …")
def load_all_seasons():
    all_players, all_detail = [], []
    for s in SEASONS:
        p, d = load_season(s)
        if p is not None: all_players.append(p)
        if d is not None: all_detail.append(d)
    players = pd.concat(all_players, ignore_index=True) if all_players else pd.DataFrame()
    detail  = pd.concat(all_detail,  ignore_index=True) if all_detail  else pd.DataFrame()
    return players, detail

players_all, detail_all = load_all_seasons()

# Which seasons actually have data on disk
avail_seasons = sorted(detail_all["season"].dropna().unique().tolist()) if not detail_all.empty else []

with st.sidebar:
    st.markdown("### 📅 Season")
    if len(avail_seasons) > 1:
        sel_season = st.selectbox("Primary season", avail_seasons,
                                   index=len(avail_seasons) - 1, key="season_pick")
    elif avail_seasons:
        sel_season = avail_seasons[0]
        st.caption(f"Only season **{sel_season}** available.")
    else:
        sel_season = CURRENT_SEASON

MAIN_SEASON = sel_season
players_raw = players_all[players_all["season"]==MAIN_SEASON].copy() if not players_all.empty else pd.DataFrame()
detail_full = detail_all[detail_all["season"]==MAIN_SEASON].copy()  if not detail_all.empty else pd.DataFrame()

if detail_full.empty:
    st.error(f"❌  No detail CSV found for season {MAIN_SEASON}. "
             f"Make sure detail_zone_pitchgroup_{MAIN_SEASON}.csv exists.")
    st.stop()

# Rescale tilt grid to observed data
_obs_tilt = detail_full["avg_tilt"].dropna()
if len(_obs_tilt) >= 20:
    TILT_MIN = float(max(5.0,  np.percentile(_obs_tilt, 1) - 2.0))
    TILT_MAX = float(min(75.0, np.percentile(_obs_tilt, 99) + 2.0))

TILT_SEARCH_WINDOW = 15.0

all_real = sorted(players_raw["batter_name"].dropna().unique()) if not players_raw.empty \
           else sorted(detail_full["batter_name"].dropna().unique())

def _fp(df, name):
    return df[df["batter_name"]==name].copy()

# ─────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────
def _engineer(df):
    out=df.copy(); le_z=LabelEncoder(); le_g=LabelEncoder()
    out["zone_enc"]     =le_z.fit_transform(out["zone"].astype(str))
    out["group_enc"]    =le_g.fit_transform(out["pitch_group"].fillna("Unknown").astype(str))
    out["tilt_x_aa"]    =out["avg_tilt"]*out["avg_aa"]
    out["tilt_x_group"] =out["avg_tilt"]*out["group_enc"]
    out["sample_weight"]=np.sqrt(out["swings"].clip(lower=1))
    return out, le_z, le_g

detail_fe, le_zone_g, le_group_g = _engineer(detail_full)

# ─────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖  Training swing model …")
def train_model():
    df=detail_fe.dropna(subset=["xwoba"]).query("swings>=5").copy()
    if df.empty: return None,"No data",None
    X=df[FEATURE_COLS].fillna(df[FEATURE_COLS].median()).values
    y=df["xwoba"].values; w=df["sample_weight"].values

    metrics=None
    try:
        from sklearn.model_selection import KFold
        from sklearn.metrics import r2_score, mean_absolute_error
        kf=KFold(n_splits=5,shuffle=True,random_state=42)
        r2s,maes=[],[]
        for tr_idx,te_idx in kf.split(X):
            gbm_cv=GradientBoostingRegressor(n_estimators=400,max_depth=4,learning_rate=0.035,
                subsample=0.75,min_samples_leaf=4,max_features=0.8,random_state=42)
            gbm_cv.fit(X[tr_idx],y[tr_idx],sample_weight=w[tr_idx])
            pred=gbm_cv.predict(X[te_idx])
            r2s.append(r2_score(y[te_idx],pred)); maes.append(mean_absolute_error(y[te_idx],pred))
        metrics={"r2":float(np.mean(r2s)),"mae":float(np.mean(maes)),"n_folds":5}
    except Exception:
        metrics=None

    if HAS_PYGAM and len(X)>=40:
        try:
            gam=LinearGAM(s(0,n_splines=12,constraints="none")+s(1,n_splines=10)
                          +s(2,n_splines=8)+s(3,n_splines=6)+gam_f(4)+gam_f(5)
                          +s(6,n_splines=6)+s(7,n_splines=6),fit_intercept=True)
            gam.gridsearch(X,y,weights=w,progress=False); return gam,"GAM",metrics
        except: pass
    gbm=GradientBoostingRegressor(n_estimators=400,max_depth=4,learning_rate=0.035,
        subsample=0.75,min_samples_leaf=4,max_features=0.8,random_state=42)
    gbm.fit(X,y,sample_weight=w); return gbm,"Gradient Boosting",metrics

model, model_type, MODEL_METRICS = train_model()

# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
def _sm(s): v=s.dropna(); return float(v.mean()) if len(v) else np.nan

def shrink(pv,lv,n,k=SHRINKAGE_K):
    w=n/(n+k); return w*pv+(1-w)*lv, round(w,3)

def predict_tilt_curve(avg_aa,avg_speed,avg_len,zone_enc,group_enc):
    tg=np.linspace(TILT_MIN,TILT_MAX,TILT_GRID_N)
    if model is None: return tg,np.full(TILT_GRID_N,np.nan)
    X=np.column_stack([tg,np.full(TILT_GRID_N,avg_aa),np.full(TILT_GRID_N,avg_speed),
                        np.full(TILT_GRID_N,avg_len),np.full(TILT_GRID_N,zone_enc),
                        np.full(TILT_GRID_N,group_enc),tg*avg_aa,tg*group_enc])
    return tg, gaussian_filter1d(model.predict(X),sigma=1.8)

def find_optimal(tg,preds):
    if np.all(np.isnan(preds)): return float(np.mean(tg)),float("nan")
    return float(tg[int(np.nanargmax(preds))]), float(np.nanmax(preds))

def find_optimal_near(tg, preds, center, window=TILT_SEARCH_WINDOW):
    if np.all(np.isnan(preds)):
        return float(np.mean(tg)), float("nan"), True
    mask = (tg >= center - window) & (tg <= center + window)
    if not mask.any():
        mask = np.ones_like(tg, dtype=bool)
    idx_all = np.where(mask)[0]
    sub_pr = preds[idx_all]
    j = int(np.nanargmax(sub_pr))
    idx_best = idx_all[j]
    at_edge = (j == 0) or (j == len(idx_all) - 1)
    return float(tg[idx_best]), float(preds[idx_best]), bool(at_edge)

@st.cache_data
def league_stats_for(df):
    cols=[c for c in METRIC_META if c in df.columns]
    return {c: float(df[c].mean()) for c in cols}

LG = league_stats_for(detail_full)

def _enc_group(name):
    return float(le_group_g.transform([name])[0]) if name in le_group_g.classes_ else 0.0

_lg_ze = float(np.median(detail_fe["zone_enc"])) if not detail_fe.empty else 0.0
_lg_ge = float(np.median(detail_fe["group_enc"])) if not detail_fe.empty else 0.0
_tg_lg,_pr_lg = predict_tilt_curve(_sm(detail_full["avg_aa"]),_sm(detail_full["avg_bat_speed"]),
                                     _sm(detail_full["avg_swing_len"]),_lg_ze,_lg_ge)
LEAGUE_OPT_TILT,_ = find_optimal(_tg_lg,_pr_lg)

@st.cache_data(show_spinner="Computing league-wide tilt rankings ...", ttl=3600)
def build_opt_table(_df, ze, ge, lo, window=TILT_SEARCH_WINDOW):
    agg = _df.groupby("batter_name", observed=True).agg(
        avg_tilt=("avg_tilt","mean"), avg_aa=("avg_aa","mean"),
        avg_bat_speed=("avg_bat_speed","mean"), avg_swing_len=("avg_swing_len","mean"),
        xwoba=("xwoba","mean"), swings=("swings","sum"),
    ).reset_index().dropna(subset=["avg_tilt","avg_aa","avg_bat_speed","avg_swing_len"])
    agg = agg[agg["swings"] >= 5].reset_index(drop=True)
    if agg.empty or model is None:
        return pd.DataFrame()

    n = len(agg)
    tg = np.linspace(TILT_MIN, TILT_MAX, TILT_GRID_N)
    aa  = agg["avg_aa"].to_numpy()[:, None]
    spd = agg["avg_bat_speed"].to_numpy()[:, None]
    ln  = agg["avg_swing_len"].to_numpy()[:, None]
    cur = agg["avg_tilt"].to_numpy()

    TG  = np.tile(tg, (n, 1))
    AA  = np.tile(aa, (1, TILT_GRID_N))
    SPD = np.tile(spd, (1, TILT_GRID_N))
    LN  = np.tile(ln, (1, TILT_GRID_N))
    ZE  = np.full_like(TG, ze)
    GE  = np.full_like(TG, ge)

    X = np.column_stack([TG.ravel(), AA.ravel(), SPD.ravel(), LN.ravel(),
                          ZE.ravel(), GE.ravel(), (TG*AA).ravel(), (TG*ge).ravel()])
    preds = model.predict(X).reshape(n, TILT_GRID_N)
    preds = gaussian_filter1d(preds, sigma=1.8, axis=1)

    win_mask = np.abs(TG - cur[:, None]) <= window
    has_support = win_mask.any(axis=1)
    preds_masked = np.where(win_mask, preds, -np.inf)
    preds_masked[~has_support] = preds[~has_support]

    idx_opt = np.argmax(preds_masked, axis=1)
    ro_arr  = tg[idx_opt]
    ox_arr  = preds_masked[np.arange(n), idx_opt]

    first_true = win_mask.argmax(axis=1)
    last_true  = TILT_GRID_N - 1 - win_mask[:, ::-1].argmax(axis=1)
    extrapolated = (~has_support) | (idx_opt == first_true) | (idx_opt == last_true)

    n_arr  = agg["swings"].to_numpy()
    w_arr  = n_arr / (n_arr + SHRINKAGE_K)
    so_arr = w_arr*ro_arr + (1-w_arr)*lo

    so_arr = np.where(extrapolated, cur + 0.4*(so_arr-cur), so_arr)
    w_arr_disp = np.where(extrapolated, w_arr*0.5, w_arr)

    delta  = cur - so_arr
    direction = np.where(np.abs(delta) <= 2.5, "≈ near-optimal",
                 np.where(delta > 0, "↓ flatten " + np.round(np.abs(delta),1).astype(str) + "°",
                                      "↑ steepen " + np.round(np.abs(delta),1).astype(str) + "°"))

    out = pd.DataFrame({
        "Batter": agg["batter_name"],
        "Swings": n_arr.astype(int),
        "Current Tilt": np.round(cur, 1),
        "Optimal Tilt (raw)": np.round(ro_arr, 1),
        "Optimal Tilt": np.round(so_arr, 1),
        "Δ Tilt": np.round(delta, 1),
        "Direction": direction,
        "Pred. xwOBA @ Opt.": np.round(ox_arr, 3),
        "Current xwOBA": np.round(agg["xwoba"].to_numpy(), 3),
        "Confidence": np.round(w_arr_disp, 3),
        "⚠ Extrapolated": extrapolated,
    })
    return out

# ─────────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS + HEADER
# ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="dash-header">
    <p class="dash-title">⚾ MLB Swing Intelligence</p>
    <p class="dash-subtitle">Bat Tracking · Tilt Optimizer · 2025–2026 Seasons</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔍 Filters")
    spm = st.multiselect("Players (multi-select for comparison)", all_real,
                         default=all_real[:3] if len(all_real)>=3 else all_real, key="spm")
    pg_filter = st.multiselect("Pitch Group", sorted(detail_full["pitch_group"].dropna().unique()),
                               default=list(detail_full["pitch_group"].dropna().unique()), key="pgf")
    min_swings = st.slider("Min. swings", 5, 200, 30, 5, key="minsw")
    show_ci = st.checkbox("Show confidence bands on curves", value=True, key="showci")

# Apply filters
dff = detail_full.copy()
if pg_filter:
    dff = dff[dff["pitch_group"].isin(pg_filter)]
dff = dff[dff["swings"] >= min_swings] if "swings" in dff.columns else dff

# Encode for current filters
_cze = float(np.median(dff["zone_enc"])) if "zone_enc" in dff.columns and not dff.empty else _lg_ze
_cge = float(np.median(dff["group_enc"])) if "group_enc" in dff.columns and not dff.empty else _lg_ge

# ─────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────
tab_exp, tab_cmp, tab_scat, tab_tr, tab_opt, tab_rank, tab_gl = st.tabs([
    "👤 Player Explorer", "⚔️ Compare", "📊 Correlations", "📈 Trends",
    "🎯 Tilt Optimizer", "🏆 Rankings", "📖 Glossary"
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — PLAYER EXPLORER
# ══════════════════════════════════════════════════════════════════════
with tab_exp:
    st.markdown('<div class="section-hdr">Player Explorer</div>', unsafe_allow_html=True)
    if not spm:
        st.info("Select at least one player in the sidebar.")
    else:
        for name in spm:
            pdf = _fp(dff, name)
            if pdf.empty:
                st.warning(f"No data for {name} with current filters.")
                continue
            n = int(pdf["swings"].sum())
            ct = _sm(pdf["avg_tilt"])
            aa = _sm(pdf["avg_aa"])
            spd = _sm(pdf["avg_bat_speed"])
            xw = _sm(pdf["xwoba"])
            st.markdown(f"""
            <div class="player-card">
                <div class="player-name">{name}</div>
                <div class="player-pos">Season {MAIN_SEASON} · {n} swings</div>
                <div style="margin-top:0.8rem">
                    <span class="metric-pill pill-elite">Tilt {ct:.1f}°</span>
                    <span class="metric-pill pill-avg">AA {aa:.1f}°</span>
                    <span class="metric-pill pill-good">Bat Speed {spd:.1f} mph</span>
                    <span class="metric-pill pill-avg">xwOBA {xw:.3f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARE
# ══════════════════════════════════════════════════════════════════════
with tab_cmp:
    st.markdown('<div class="section-hdr">Player Comparison</div>', unsafe_allow_html=True)
    if len(spm) < 2:
        st.info("Select 2+ players in the sidebar to compare.")
    else:
        rows = []
        for name in spm:
            pdf = _fp(dff, name)
            if pdf.empty: continue
            rows.append({
                "Batter": name,
                "Swings": int(pdf["swings"].sum()),
                "Tilt": round(_sm(pdf["avg_tilt"]),1),
                "AA": round(_sm(pdf["avg_aa"]),1),
                "Bat Speed": round(_sm(pdf["avg_bat_speed"]),1),
                "Swing Len": round(_sm(pdf["avg_swing_len"]),2),
                "xwOBA": round(_sm(pdf["xwoba"]),3),
                "EV": round(_sm(pdf["avg_exit_velocity"]),1),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — CORRELATIONS
# ══════════════════════════════════════════════════════════════════════
with tab_scat:
    st.markdown('<div class="section-hdr">Scatter Correlations</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    x_m = c1.selectbox("X axis", list(METRIC_META.keys()), index=0, key="scx",
                       format_func=lambda x: METRIC_META[x]["label"])
    y_m = c2.selectbox("Y axis", list(METRIC_META.keys()), index=4, key="scy",
                       format_func=lambda x: METRIC_META[x]["label"])
    if x_m in dff.columns and y_m in dff.columns:
        fig = px.scatter(dff, x=x_m, y=y_m, color="pitch_group", size="swings",
                         hover_data=["batter_name"], opacity=0.7,
                         title=f"{METRIC_META[x_m]['label']} vs {METRIC_META[y_m]['label']}")
        fig = styled_fig(fig)
        st.plotly_chart(fig, width="stretch")

# ══════════════════════════════════════════════════════════════════════
# TAB 4 — TRENDS
# ══════════════════════════════════════════════════════════════════════
with tab_tr:
    st.markdown('<div class="section-hdr">Multi-Season Trends (2025–2026)</div>', unsafe_allow_html=True)
    if len(avail_seasons) < 2:
        st.info("Need data for both 2025 and 2026 to show trends.")
    else:
        tr_metric = st.selectbox("Trend metric",
                                 ["avg_tilt","avg_aa","avg_bat_speed","xwoba","avg_exit_velocity"],
                                 format_func=lambda x: METRIC_META[x]["label"], key="tr_m")
        lg_trend = []
        for s in avail_seasons:
            p, d = load_season(s)
            if d is not None and tr_metric in d.columns:
                v = d[tr_metric].mean()
                lg_trend.append({"Season": str(s), "Value": round(v,3), "Type": "League Avg"})
        if lg_trend:
            lgt = pd.DataFrame(lg_trend)
            fig_tr = go.Figure()
            fig_tr.add_trace(go.Scatter(x=lgt["Season"], y=lgt["Value"], mode="lines+markers",
                name="League Avg", line=dict(color="#388bfd", width=2.5),
                marker=dict(size=10, color="#388bfd")))
            for name in spm:
                btr = []
                for s in avail_seasons:
                    _, d = load_season(s)
                    if d is None: continue
                    sub = d[d["batter_name"]==name]
                    v = sub[tr_metric].mean() if not sub.empty and tr_metric in sub.columns else np.nan
                    btr.append({"Season": str(s), "Value": round(v,3)})
                if btr and any(not pd.isna(r["Value"]) for r in btr):
                    bdf = pd.DataFrame(btr)
                    fig_tr.add_trace(go.Scatter(x=bdf["Season"], y=bdf["Value"], mode="lines+markers",
                        name=name, line=dict(width=2), marker=dict(size=9)))
            fig_tr = styled_fig(fig_tr, 420)
            fig_tr.update_layout(title=f"{METRIC_META[tr_metric]['label']} — Season Trend (2025–2026)",
                xaxis_title="Season", yaxis_title=METRIC_META[tr_metric]["label"])
            st.plotly_chart(fig_tr, width="stretch")

# ══════════════════════════════════════════════════════════════════════
# TAB 5 — TILT OPTIMIZER
# ══════════════════════════════════════════════════════════════════════
with tab_opt:
    st.markdown('<div class="section-hdr">Tilt Optimizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Partial-dependence curve: all features fixed at batter averages, <strong>tilt swept across observed range</strong>. Optimal tilt is shrunk toward the league optimum (Bayesian shrinkage, K=50).</div>',
                unsafe_allow_html=True)

    if model is None:
        st.error("Model not available.")
    else:
        c1,c2,c3,c4 = st.columns([2,1,1,1.3])
        sp6 = c1.selectbox("Batter", all_real, key="t6p")
        pg6 = c2.selectbox("Pitch Group", ["All"]+sorted(detail_full["pitch_group"].dropna().unique()), key="t6pg")
        z6  = c3.selectbox("Zone", ["All"]+[str(z) for z in range(1,15)], key="t6z")
        win6= c4.slider("Search window (°)", 5, 30, int(TILT_SEARCH_WINDOW), 1, key="t6win")

        p6d = _fp(detail_fe, sp6)
        if pg6 != "All": p6d = p6d[p6d["pitch_group"]==pg6]
        if z6  != "All": p6d = p6d[p6d["zone"]==int(z6)]

        if p6d.empty:
            st.warning(f"No data for {sp6} with these filters.")
        else:
            n6  = int(p6d["swings"].sum())
            aa6 = _sm(p6d["avg_aa"])
            spd6= _sm(p6d["avg_bat_speed"])
            len6= _sm(p6d["avg_swing_len"])
            ct  = _sm(p6d["avg_tilt"])
            ge6 = float(p6d["group_enc"].mean())
            ze6 = float(p6d["zone_enc"].mean())
            tg6, pr6 = predict_tilt_curve(aa6, spd6, len6, ze6, ge6)
            ro6, ox6, extrap6 = find_optimal_near(tg6, pr6, ct, window=float(win6))
            so6, cw6 = shrink(ro6, LEAGUE_OPT_TILT, n6)
            if extrap6:
                so6 = ct + 0.4*(so6-ct)
                cw6 = cw6*0.5
            tgl6, prl6 = predict_tilt_curve(_sm(detail_full["avg_aa"]), _sm(detail_full["avg_bat_speed"]),
                                            _sm(detail_full["avg_swing_len"]), ze6, ge6)
            k1,k2,k3,k4,k5 = st.columns(5)
            k1.metric("Current Tilt", f"{ct:.1f}°")
            k2.metric("Optimal (shrunk)", f"{so6:.1f}°")
            dv = ct - so6
            dv_label = "lower it" if dv > 0 else "raise it"
            k3.metric("Δ Tilt", f"{dv:+.1f}°", f"{dv_label}", delta_color="inverse")
            k4.metric("Pred. xwOBA @ Opt.", f"{ox6:.3f}")
            k5.metric("Confidence", f"{cw6:.0%}", help=f"n={n6}·K=50")
            if n6 < 30:
                st.markdown('<div class="warn-box">⚠️ Small sample — estimate heavily shrunk toward league optimum.</div>', unsafe_allow_html=True)
            if extrap6:
                st.markdown(f'<div class="warn-box">⚠️ Optimal sits at the edge of the ±{int(win6)}° window — treat as directional only.</div>', unsafe_allow_html=True)

            fig6 = go.Figure()
            if show_ci:
                scale = 1.0 / np.sqrt(max(n6,1)/50)
                rng = np.random.default_rng(0)
                boots = []
                for _ in range(120):
                    _, p = predict_tilt_curve(aa6+rng.normal(0,8*scale), spd6+rng.normal(0,3*scale),
                                              len6+rng.normal(0,0.3*scale), ze6, ge6)
                    boots.append(p)
                arr = np.vstack(boots)
                lo = np.percentile(arr, 10, axis=0)
                hi = np.percentile(arr, 90, axis=0)
                fig6.add_trace(go.Scatter(x=np.concatenate([tg6, tg6[::-1]]),
                    y=np.concatenate([hi, lo[::-1]]), fill="toself",
                    fillcolor="rgba(240,165,0,0.1)", line=dict(color="rgba(0,0,0,0)"),
                    name="80% CI", hoverinfo="skip"))
            fig6.add_trace(go.Scatter(x=tg6, y=pr6, mode="lines",
                line=dict(color="#f0a500", width=2.8), name=sp6))
            fig6.add_trace(go.Scatter(x=tgl6, y=prl6, mode="lines",
                line=dict(color="#388bfd", width=1.5, dash="dot"), name="League avg features"))
            fig6.add_vrect(x0=max(TILT_MIN, ct-win6), x1=min(TILT_MAX, ct+win6),
                fillcolor="rgba(63,185,80,0.06)", line_width=0,
                annotation_text="trusted search window", annotation_position="top left",
                annotation_font_size=9, annotation_font_color="#3fb950")
            for xv, col, lbl, dash in [(ct, "#3fb950", f"Current {ct:.1f}°", "dash"),
                                       (so6, "#f0a500", f"Optimal {so6:.1f}°", "solid"),
                                       (LEAGUE_OPT_TILT, "#388bfd", f"League opt {LEAGUE_OPT_TILT:.1f}°", "dashdot")]:
                fig6.add_vline(x=xv, line=dict(color=col, width=1.5, dash=dash),
                               annotation_text=lbl, annotation_font_color=col, annotation_font_size=10)
            fig6 = styled_fig(fig6, 460)
            fig6.update_layout(title=f"Predicted xwOBA vs Tilt — {sp6}",
                xaxis_title="Swing Path Tilt (°)", yaxis_title="Predicted xwOBA",
                legend=dict(orientation="h", y=-0.22), hovermode="x unified")
            st.plotly_chart(fig6, width="stretch")

            # 2-D interaction
            st.markdown('<div class="section-hdr">Tilt × Attack Angle Interaction Surface</div>', unsafe_allow_html=True)
            tg2 = np.linspace(TILT_MIN, TILT_MAX, 30)
            aa2 = np.linspace(-30, 30, 25)
            TT, AA = np.meshgrid(tg2, aa2)
            n2 = TT.size
            X2 = np.column_stack([TT.ravel(), AA.ravel(), np.full(n2, spd6), np.full(n2, len6),
                                  np.full(n2, ze6), np.full(n2, ge6), TT.ravel()*AA.ravel(), TT.ravel()*ge6])
            Z2 = model.predict(X2).reshape(TT.shape)
            f2d = go.Figure(data=go.Heatmap(z=Z2, x=tg2.round(1), y=aa2.round(1),
                colorscale="RdYlGn", colorbar=dict(title="xwOBA"),
                hovertemplate="Tilt: %{x:.1f}°<br>AA: %{y:.1f}°<br>xwOBA: %{z:.3f}<extra></extra>"))
            f2d.add_trace(go.Scatter(x=[ct], y=[aa6], mode="markers",
                marker=dict(color="white", size=14, symbol="star", line=dict(color="#0d1117", width=2)),
                name=f"{sp6}"))
            f2d = styled_fig(f2d, 420)
            f2d.update_layout(title="xwOBA Surface: Tilt × Attack Angle  (⭐ = current batter)",
                xaxis_title="Tilt (°)", yaxis_title="Attack Angle (°)")
            st.plotly_chart(f2d, width="stretch")

# ══════════════════════════════════════════════════════════════════════
# TAB 6 — TILT RANKINGS
# ══════════════════════════════════════════════════════════════════════
with tab_rank:
    st.markdown('<div class="section-hdr">League-Wide Tilt Rankings</div>', unsafe_allow_html=True)

    win_c1, win_c2 = st.columns([3,1])
    with win_c1:
        st.markdown(
            f'<div class="info-box">Every batter ranked by gap to model-optimal tilt, '
            f'searched only within ± the window of their own current tilt, then shrunk toward '
            f'the league optimum <strong>{LEAGUE_OPT_TILT:.1f}°</strong> (K=50).<br>'
            f'🟥 too steep · 🟦 too flat · 🟩 near-optimal (±2.5°)</div>',
            unsafe_allow_html=True)
    with win_c2:
        rank_window = st.slider("Search window (°)", 5, 30, int(TILT_SEARCH_WINDOW), 1, key="rank_window")

    opt_table_dyn = build_opt_table(dff, _cze, _cge, LEAGUE_OPT_TILT, window=float(rank_window))

    if opt_table_dyn.empty:
        st.warning("No optimization data for current filters.")
    else:
        c1,c2,c3,c4 = st.columns([2,1,1.4,1.4])
        sort_col = c1.selectbox("Sort by",
            ["Δ Tilt","Swings","Pred. xwOBA @ Opt.","Current Tilt","Optimal Tilt"], key="rank_sort")
        ascending = c2.checkbox("Ascending", value=False, key="rank_asc")
        min_conf  = c3.slider("Min. confidence", 0.0, 1.0, 0.0, 0.05, key="rank_conf")
        hide_extrap = c4.checkbox("Hide extrapolated", value=False, key="rank_hide_extrap")

        tbl = opt_table_dyn[opt_table_dyn["Confidence"] >= min_conf]
        if hide_extrap:
            tbl = tbl[~tbl["⚠ Extrapolated"]]
        tbl = tbl.sort_values(sort_col, ascending=ascending, na_position="last")

        n_extrap = int(opt_table_dyn["⚠ Extrapolated"].sum())
        if n_extrap:
            st.caption(f"⚠️ {n_extrap} of {len(opt_table_dyn)} batters have edge-pinned recommendations.")

        def _delta_style(v):
            if pd.isna(v): return ""
            if v > 2.5:  return "background-color:#2d1414;color:#f85149"
            if v < -2.5: return "background-color:#0f1f2d;color:#58a6ff"
            return "background-color:#0d1f0d;color:#3fb950"

        fmt = {"Current Tilt":"{:.1f}°","Optimal Tilt (raw)":"{:.1f}°","Optimal Tilt":"{:.1f}°",
               "Δ Tilt":"{:+.1f}°","Pred. xwOBA @ Opt.":"{:.3f}","Current xwOBA":"{:.3f}",
               "Confidence":"{:.0%}"}
        _styler = (tbl.style.map(_delta_style, subset=["Δ Tilt"]) if hasattr(tbl.style, "map")
                   else tbl.style.applymap(_delta_style, subset=["Δ Tilt"]))
        st.dataframe(_styler.format(fmt), width="stretch", hide_index=True)

        # JEDYNY przycisk download – z unikalnym key
        csv_rank = tbl.to_csv(index=False)
        st.download_button(
            "⬇ Download rankings CSV",
            csv_rank,
            "tilt_rankings.csv",
            "text/csv",
            key="download_rankings_unique"
        )

        st.markdown('<div class="section-hdr">Δ Tilt — Top 30 by |Δ|</div>', unsafe_allow_html=True)
        top30 = tbl.assign(_a=tbl["Δ Tilt"].abs()).nlargest(30, "_a")
        fig_rank = px.bar(top30, x="Δ Tilt", y="Batter", orientation="h", color="Δ Tilt",
            color_continuous_scale="RdBu_r", color_continuous_midpoint=0, text="Δ Tilt",
            hover_data=["Current Tilt","Optimal Tilt","Confidence","Swings"],
            title="Positive = too steep · Negative = too flat")
        fig_rank.update_traces(texttemplate="%{text:+.1f}°", textposition="outside")
        fig_rank.add_vline(x=0, line_width=2, line_color="#8b949e")
        fig_rank = styled_fig(fig_rank, max(400, 22*len(top30)))
        fig_rank.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig_rank, width="stretch")

        if "Current xwOBA" in tbl.columns and tbl["Current xwOBA"].notna().any():
            st.markdown('<div class="section-hdr">Current xwOBA vs Δ Tilt</div>', unsafe_allow_html=True)
            sd = tbl.dropna(subset=["Current xwOBA"])
            fig_rx = px.scatter(sd, x="Δ Tilt", y="Current xwOBA", color="Confidence", size="Swings")
                hover_data=["Batter","Optimal Tilt","Current Tilt"],
