"""
MLB Bat Tracking 2025-2026  ·  Swing Intelligence Dashboard
============================================================
Full redesign: dark stadium theme, Player Cards, AI Insights,
multi-year trends, education layer, scatter correlations.
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
    page_title="MLB Swing Intelligence",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# THEME  — Stadium Night palette
# Deep navy + electric amber + chalk white + warning red
# Signature element: the diagonal "chalk line" accent on cards
# ─────────────────────────────────────────────────────────────────────
DARK_CSS = """
<style>
/* ── Global ── */
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

/* ── Header ── */
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

/* ── Cards ── */
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

/* ── Section headers ── */
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

/* ── Info boxes ── */
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

/* ── AI insight box ── */
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

/* ── Tabs ── */
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

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 0.6rem 0.8rem;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.72rem !important; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; }

/* ── Plotly override (background) ── */
.js-plotly-plot .plotly { background: transparent !important; }

/* ── Tables ── */
[data-testid="stDataFrame"] { border: 1px solid #21262d; border-radius: 6px; }

/* ── Selectbox / slider ── */
[data-testid="stSelectbox"] > div, [data-testid="stMultiSelect"] > div {
    background: #161b22 !important;
    border-color: #21262d !important;
}

/* ── Divider ── */
hr { border-color: #21262d !important; }

/* ── Glossary entry ── */
.gloss-term {
    font-family:'Oswald',sans-serif;
    color:#f0a500;
    font-size:1rem;
    font-weight:600;
    margin-top:1rem;
}
.gloss-def { color:#c9d1d9; font-size:0.87rem; line-height:1.7; }
.gloss-example { color:#8b949e; font-size:0.8rem; font-style:italic; margin-top:0.25rem; }

/* ── Download btn ── */
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
# PLOTLY TEMPLATE  (dark stadium)
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
SEASONS = [2024, 2025, 2026]
CURRENT_SEASON = 2025

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

# Current season as primary working frame
MAIN_SEASON = max(s for s in SEASONS
                  if (DATA_DIR / f"players_summary_{s}.csv").exists())
players_raw = players_all[players_all["season"]==MAIN_SEASON].copy() if not players_all.empty else pd.DataFrame()
detail_full = detail_all[detail_all["season"]==MAIN_SEASON].copy()  if not detail_all.empty else pd.DataFrame()

if detail_full.empty:
    st.error("❌  No detail CSV found. Make sure detail_zone_pitchgroup_2025.csv exists.")
    st.stop()

# Player list
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
    if df.empty: return None,"No data"
    X=df[FEATURE_COLS].fillna(df[FEATURE_COLS].median()).values
    y=df["xwoba"].values; w=df["sample_weight"].values
    if HAS_PYGAM and len(X)>=40:
        try:
            gam=LinearGAM(s(0,n_splines=12,constraints="none")+s(1,n_splines=10)
                          +s(2,n_splines=8)+s(3,n_splines=6)+gam_f(4)+gam_f(5)
                          +s(6,n_splines=6)+s(7,n_splines=6),fit_intercept=True)
            gam.gridsearch(X,y,weights=w,progress=False); return gam,"GAM"
        except: pass
    gbm=GradientBoostingRegressor(n_estimators=400,max_depth=4,learning_rate=0.035,
        subsample=0.75,min_samples_leaf=4,max_features=0.8,random_state=42)
    gbm.fit(X,y,sample_weight=w); return gbm,"Gradient Boosting"

model, model_type = train_model()

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

# League stats
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

def _clean(df,*cols):
    return df[list(dict.fromkeys(c for c in cols if c in df.columns))].copy()

def _safe_scatter(df,x,y,color=None,size=None,hover_data=None,trendline=None,labels=None,title=""):
    if x==y: return None
    if trendline and not HAS_STATSMODELS: trendline=None
    cols=[c for c in [x,y,color,size]+(hover_data or []) if c]
    hd=[c for c in (hover_data or []) if c not in (x,y)]
    return px.scatter(_clean(df,*cols),x=x,y=y,color=color,size=size,
                      hover_data=hd or None,trendline=trendline,
                      labels=labels or {},title=title)

# ─────────────────────────────────────────────────────────────────────
# PLAYER SUMMARY  (batter-level aggregation)
# ─────────────────────────────────────────────────────────────────────
@st.cache_data
def batter_summary(df):
    cols=[c for c in METRIC_META if c in df.columns]; cols.append("swings")
    agg={c:("sum" if c=="swings" else "mean") for c in set(cols)}
    return df.groupby("batter_name",observed=True).agg(agg).round(3).reset_index()

summary_df = batter_summary(detail_full)

def get_batter_row(name):
    row = summary_df[summary_df["batter_name"]==name]
    return row.iloc[0] if not row.empty else None

# ─────────────────────────────────────────────────────────────────────
# METRIC COLORING
# ─────────────────────────────────────────────────────────────────────
def metric_class(metric, value):
    """Return CSS class: pill-elite / pill-good / pill-avg / pill-bad."""
    if pd.isna(value): return "pill-avg"
    m = METRIC_META.get(metric, {})
    lo_e, hi_e = m.get("elite",(999,-999))
    lo_w, hi_w = m.get("warn",(-999,999))
    if lo_e <= value <= hi_e: return "pill-elite"
    if lo_w <= value <= hi_w: return "pill-bad"
    lg = LG.get(metric, value)
    return "pill-good" if value >= lg else "pill-avg"

def metric_delta_color(metric, value):
    lg = LG.get(metric, value)
    if pd.isna(value) or pd.isna(lg): return "#8b949e"
    if value > lg: return "#3fb950"
    if value < lg: return "#f85149"
    return "#8b949e"

def fmt_val(metric, value):
    if pd.isna(value): return "—"
    f = METRIC_META.get(metric,{}).get("fmt",".1f")
    return format(value, f)


# ─────────────────────────────────────────────────────────────────────
# AI INSIGHTS  (Anthropic API)
# ─────────────────────────────────────────────────────────────────────
def generate_ai_insight(batter_name, row, lg):
    """Call Claude to produce a scouting paragraph."""
    if row is None: return "No data available for this batter."
    metrics_txt = "\n".join(
        f"  {METRIC_META[m]['label']}: {fmt_val(m,row.get(m,np.nan))}  (lg avg: {fmt_val(m,lg.get(m,np.nan))})"
        for m in ["avg_tilt","avg_aa","avg_bat_speed","avg_swing_len","xwoba","avg_exit_velocity","avg_launch_angle"]
        if m in row.index
    )
    prompt = f"""You are an elite MLB hitting analyst. Analyze the following bat-tracking metrics for {batter_name} and produce a concise scouting insight (4-6 sentences). Focus on:
1. Swing path profile (tilt interpretation)
2. Attack angle implications for contact vs power
3. One key strength and one area to watch
4. A practical development recommendation

Metrics:
{metrics_txt}

Write in clear, professional scouting language. Do not repeat the raw numbers verbatim — interpret them."""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type":"application/json"},
            json={"model":"claude-sonnet-4-6","max_tokens":350,
                  "messages":[{"role":"user","content":prompt}]},
            timeout=25,
        )
        data = resp.json()
        return data["content"][0]["text"]
    except Exception as e:
        return f"AI insight unavailable: {e}"

# ─────────────────────────────────────────────────────────────────────
# PLAYER CARD HTML
# ─────────────────────────────────────────────────────────────────────
def render_player_card(name, row):
    if row is None:
        st.warning(f"No data for {name}")
        return
    pills = ""
    for m in ["avg_tilt","avg_aa","avg_bat_speed","xwoba","avg_exit_velocity"]:
        if m not in row.index or pd.isna(row[m]): continue
        cls = metric_class(m, row[m])
        lbl = METRIC_META[m]["label"]
        val = fmt_val(m, row[m])
        pills += f'<span class="metric-pill {cls}">{lbl}: {val}</span>'

    # Tilt profile label
    tilt = row.get("avg_tilt", np.nan)
    if pd.isna(tilt): profile = "Unknown"
    elif tilt >= 45:  profile = "Elite Uppercut"
    elif tilt >= 35:  profile = "Power Tilt"
    elif tilt >= 25:  profile = "Balanced"
    elif tilt >= 15:  profile = "Line Drive"
    else:             profile = "Flat / Ground Ball"

    n = int(row.get("swings",0))
    st.markdown(f"""
<div class="player-card">
  <div class="player-name">{name}</div>
  <div class="player-pos">SWING PROFILE: {profile} &nbsp;·&nbsp; {n} swings tracked</div>
  <div style="margin-top:0.8rem">{pills}</div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# HEATMAP (matplotlib, dark-themed)
# ─────────────────────────────────────────────────────────────────────
def _zone_color(val,vmin,vmax,cmap):
    if pd.isna(val): return(0.12,0.13,0.15)
    return cmap(float(np.clip((val-vmin)/max(vmax-vmin,1e-9),0,1)))
def _tc(bg): r,g,b=bg[:3]; return "#e6edf3" if 0.299*r+0.587*g+0.114*b<0.45 else "#0d1117"
def _fv2(val,metric):
    if pd.isna(val): return "—"
    if metric=="swings": return str(int(round(val)))
    if metric in("batting_avg","xwoba"): return f"{val:.3f}"
    return f"{val:.1f}"

def _get_pivot(df_p,metric,league_df,df_ctx):
    if metric=="tilt_std":
        return df_p.groupby("zone")["avg_tilt"].std(ddof=1).round(1)
    if metric=="aa_std":
        return df_p.groupby("zone")["avg_aa"].std(ddof=1).round(1)
    if metric=="delta_tilt":
        pm=df_p.groupby("zone")["avg_tilt"].mean()
        lm=league_df.set_index("zone")["avg_tilt"] if league_df is not None else pd.Series()
        return (pm-lm.reindex(pm.index,fill_value=np.nan)).round(2)
    return (df_p.groupby("zone")["swings"].sum() if metric=="swings"
            else df_p.groupby("zone")[metric].mean()).round(3)

def make_heatmap(df_p,metric,title,league_df=None,df_ctx=None,figsize=(7,7)):
    if df_p is None or df_p.empty: st.warning(f"No data for {title}"); return
    if df_ctx is None: df_ctx=detail_full
    pivot=_get_pivot(df_p,metric,league_df,df_ctx)
    nsw=df_p.groupby("zone")["swings"].sum() if "swings" in df_p.columns else pd.Series()
    if metric=="delta_tilt":
        vmin,vmax=HEATMAP_RANGES.get(metric,(-20,20)); cn="RdBu_r"
    else:
        vmin,vmax=HEATMAP_RANGES.get(metric,(0,100)); cn="YlOrRd"
    cmap=sns.color_palette(cn,as_cmap=True)
    b,ms,sy=0.85,3.3,2.5; mx,my=b,b; tx,ty=mx+ms,my+ms; half=ms/2
    fig,ax=plt.subplots(figsize=figsize,facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    for i in range(3):
        for j in range(3):
            zone=i*3+j+1; val=pivot.get(zone,np.nan); n=int(nsw.get(zone,0))
            x=mx+j*(ms/3); y=my+(2-i)*(ms/3); col=_zone_color(val,vmin,vmax,cmap)
            ax.add_patch(plt.Rectangle((x,y),ms/3,ms/3,facecolor=col,edgecolor="#21262d",linewidth=2))
            txt=f"{zone}\n{_fv2(val,metric)}"+(f"\n⚠n={n}" if n<20 else "")
            ax.text(x+ms/6,y+ms/6,txt,ha="center",va="center",fontsize=9.5,
                    fontweight="bold",color=_tc(col),fontfamily="monospace")
    ld=[(11,[(0,sy),(b,sy),(b,ty),(mx,ty),(mx+half,ty),(mx+half,5),(0,5),(0,sy)],(b*0.4,5-b*0.4)),
        (12,[(tx,sy),(tx,ty),(mx+half,ty),(mx+half,5),(5,5),(5,sy),(tx,sy)],(5-b*0.4,5-b*0.4)),
        (13,[(0,sy),(b,sy),(b,my),(mx,my),(mx+half,my),(mx+half,0),(0,0),(0,sy)],(b*0.4,b*0.4)),
        (14,[(tx,sy),(tx,my),(mx+half,my),(mx+half,0),(5,0),(5,sy),(tx,sy)],(5-b*0.4,b*0.4))]
    for z,verts,(cx,cy) in ld:
        val=pivot.get(z,np.nan); n=int(nsw.get(z,0)); col=_zone_color(val,vmin,vmax,cmap)
        ax.add_patch(PathPatch(MPath(verts),facecolor=col,edgecolor="#21262d",linewidth=2))
        txt=f"{z}\n{_fv2(val,metric)}"+(f"\n⚠n={n}" if n<20 else "")
        ax.text(cx,cy,txt,ha="center",va="center",fontsize=9.5,fontweight="bold",color=_tc(col))
    ax.add_patch(plt.Rectangle((mx,my),ms,ms,fill=False,edgecolor="#f0a500",linewidth=2.5))
    ax.set_title(title,fontsize=13,fontweight="bold",color="#f0a500",pad=14,fontfamily="sans-serif")
    ax.set_xlim(0,5); ax.set_ylim(0,5); ax.set_aspect("equal"); ax.axis("off")
    lbl_map={"avg_tilt":"TILT (°)","tilt_std":"TILT STD","delta_tilt":"Δ TILT vs LG",
             "avg_aa":"ATTACK ANGLE (°)","aa_std":"AA STD","avg_bat_speed":"BAT SPEED (mph)",
             "avg_swing_len":"SWING LEN (ft)","swings":"SWINGS","batting_avg":"BA",
             "xwoba":"xwOBA","avg_exit_velocity":"EV (mph)","avg_launch_angle":"LA (°)"}
    sm=plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin=vmin,vmax=vmax))
    cbar=plt.colorbar(sm,ax=ax,shrink=0.72,pad=0.04)
    cbar.set_label(lbl_map.get(metric,metric.upper()),fontsize=9,color="#8b949e")
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(cbar.ax.yaxis.get_ticklabels(),color="#8b949e",fontsize=8)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <div class="dash-title">⚾ MLB Swing Intelligence</div>
  <div class="dash-subtitle">BAT TRACKING · SWING PATH TILT · ATTACK ANGLE · 2024–2026</div>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚾ Batter Selection")
    _SEP="─"*20
    spm=st.multiselect("Select Batters",options=[_SEP]+all_real,
                        default=[all_real[0]] if all_real else [],max_selections=6)
    spm=[p for p in spm if p!=_SEP]

    st.markdown("### 🎛 Pitch Filters")
    pg_list=["All"]+sorted(detail_full["pitch_group"].dropna().unique())
    sel_pitch=st.selectbox("Pitch Group",pg_list)
    _pts=(detail_full if sel_pitch=="All" else detail_full[detail_full["pitch_group"]==sel_pitch])
    pt_list=["All"]+sorted(_pts["pitch_type"].dropna().unique())
    sel_type=st.selectbox("Pitch Type",pt_list)

    st.markdown("### 🔧 Display")
    min_swings=st.slider("Min. swings filter",0,300,0,10)
    show_ci=st.checkbox("Show CI bands (Tilt Optimizer)",value=True)

    # season picker (multi-year)
    avail_seasons=[s for s in SEASONS if (DATA_DIR/f"players_summary_{s}.csv").exists()]
    if len(avail_seasons)>1:
        st.markdown("### 📅 Season")
        sel_season=st.selectbox("Primary season",avail_seasons,index=len(avail_seasons)-1)
    else:
        sel_season=MAIN_SEASON

    st.markdown("---")
    st.caption(f"Model: **{model_type}**")
    st.caption(f"League opt. tilt: **{LEAGUE_OPT_TILT:.1f}°**")

# Apply pitch filter
def apf(df,pg,pt):
    if pg!="All": df=df[df["pitch_group"]==pg]
    if pt!="All": df=df[df["pitch_type"]==pt]
    return df

dff=apf(detail_full.copy(),sel_pitch,sel_type)
_LAGG={c:"mean" for c in METRIC_META if c in dff.columns}; _LAGG["swings"]="sum"
lpz=dff.groupby("zone",as_index=False,observed=True).agg(_LAGG).round(3)
lpz["batter_name"]="League Average"
dt=dff[dff["swings"]>=min_swings].copy()


# ─────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────
TAB_NAMES=["🏠 Overview","🔍 Player Explorer","⚔️ Comparisons",
           "📈 Trends","🔬 Tilt Optimizer","📚 Glossary"]
tab_ov,tab_exp,tab_cmp,tab_tr,tab_opt,tab_gl=st.tabs(TAB_NAMES)

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════
with tab_ov:
    st.markdown('<div class="section-hdr">League Snapshot</div>', unsafe_allow_html=True)

    # KPI row
    k_cols=st.columns(5)
    kpi_metrics=["avg_tilt","avg_aa","avg_bat_speed","xwoba","avg_exit_velocity"]
    for col,m in zip(k_cols,kpi_metrics):
        v=LG.get(m,np.nan)
        col.metric(METRIC_META[m]["label"],fmt_val(m,v),"League Avg",delta_color="off")

    st.markdown("---")
    c1,c2=st.columns([3,2])

    with c1:
        st.markdown('<div class="section-hdr">Tilt Distribution — All Batters</div>', unsafe_allow_html=True)
        tilt_data=summary_df["avg_tilt"].dropna()
        fig_dist=go.Figure()
        fig_dist.add_trace(go.Histogram(x=tilt_data,nbinsx=40,name="Tilt",
            marker_color="#f0a500",opacity=0.75,
            hovertemplate="Tilt: %{x:.1f}°<br>Count: %{y}<extra></extra>"))
        fig_dist.add_vline(x=float(tilt_data.mean()),line_dash="dash",line_color="#388bfd",
            annotation_text=f"Avg {tilt_data.mean():.1f}°",annotation_font_color="#388bfd")
        fig_dist=styled_fig(fig_dist,380)
        fig_dist.update_layout(xaxis_title="Swing Path Tilt (°)",yaxis_title="# Batters",
            bargap=0.05,showlegend=False)
        st.plotly_chart(fig_dist,use_container_width=True)

    with c2:
        st.markdown('<div class="section-hdr">Tilt vs xwOBA</div>', unsafe_allow_html=True)
        sc=summary_df.dropna(subset=["avg_tilt","xwoba","swings"]).copy()
        sc=sc[sc["swings"]>=30]
        fig_sc=px.scatter(sc,x="avg_tilt",y="xwoba",size="swings",
            hover_name="batter_name",
            size_max=28,color="avg_bat_speed",color_continuous_scale="plasma",
            labels={"avg_tilt":"Tilt (°)","xwoba":"xwOBA","avg_bat_speed":"Bat Speed (mph)"},
        )
        if HAS_STATSMODELS:
            fig_sc2=px.scatter(sc,x="avg_tilt",y="xwoba",trendline="lowess")
            for tr in fig_sc2.data:
                if tr.mode=="lines": fig_sc.add_trace(tr)
        fig_sc=styled_fig(fig_sc,380)
        fig_sc.update_traces(marker_line_color="#0d1117",marker_line_width=0.5)
        fig_sc.update_layout(coloraxis_colorbar=dict(title="Bat Speed"))
        st.plotly_chart(fig_sc,use_container_width=True)

    # Top / Bottom tables
    st.markdown('<div class="section-hdr">League Leaders</div>', unsafe_allow_html=True)
    c3,c4,c5=st.columns(3)
    for col,(metric,label,asc) in zip([c3,c4,c5],[
        ("avg_tilt","Highest Tilt",False),
        ("xwoba","Best xwOBA",False),
        ("avg_bat_speed","Hardest Swingers",False),
    ]):
        with col:
            st.markdown(f"**{label}**")
            sub=summary_df[summary_df["swings"]>=50].dropna(subset=[metric]) \
                         .sort_values(metric,ascending=asc).head(8)
            sub_show=sub[["batter_name",metric,"swings"]].rename(columns={
                "batter_name":"Batter",metric:METRIC_META[metric]["label"],"swings":"Swings"})
            st.dataframe(sub_show.reset_index(drop=True),use_container_width=True,hide_index=True,height=240)

    # Correlation matrix
    st.markdown('<div class="section-hdr">Metric Correlations (batters with ≥50 swings)</div>', unsafe_allow_html=True)
    corr_cols=["avg_tilt","avg_aa","avg_bat_speed","avg_swing_len","xwoba","avg_exit_velocity","avg_launch_angle"]
    corr_df=summary_df[summary_df["swings"]>=50][corr_cols].dropna()
    if len(corr_df)>5:
        corr_mat=corr_df.corr().round(2)
        fig_corr=px.imshow(corr_mat,text_auto=True,aspect="auto",
            color_continuous_scale="RdBu",zmin=-1,zmax=1,
            labels=dict(color="r"),
            x=[METRIC_META[c]["label"] for c in corr_cols],
            y=[METRIC_META[c]["label"] for c in corr_cols])
        fig_corr=styled_fig(fig_corr,380)
        fig_corr.update_layout(title="Pearson Correlation Matrix")
        st.plotly_chart(fig_corr,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — PLAYER EXPLORER
# ══════════════════════════════════════════════════════════════════════
with tab_exp:
    st.markdown('<div class="section-hdr">Player Explorer</div>', unsafe_allow_html=True)
    sel_exp=st.selectbox("Select Batter",all_real,key="exp_sel")
    row_exp=get_batter_row(sel_exp)
    render_player_card(sel_exp,row_exp)

    # Metrics vs league
    st.markdown('<div class="section-hdr">Metrics vs League Average</div>', unsafe_allow_html=True)
    m_cols=st.columns(4)
    for i,(m,meta) in enumerate(METRIC_META.items()):
        if m=="swings" or m not in (row_exp.index if row_exp is not None else []): continue
        val=row_exp.get(m,np.nan) if row_exp is not None else np.nan
        lg_v=LG.get(m,np.nan)
        delta=round(val-lg_v,3) if not pd.isna(val) and not pd.isna(lg_v) else None
        delta_str=f"{delta:+.3f}" if delta is not None else None
        with m_cols[i%4]:
            st.metric(meta["label"],fmt_val(m,val),delta_str)

    # Zone heatmaps
    st.markdown('<div class="section-hdr">Zone Heatmaps</div>', unsafe_allow_html=True)
    hz_metric=st.radio("Metric",["avg_tilt","delta_tilt","avg_aa","xwoba","avg_bat_speed","avg_exit_velocity"],
                        format_func=lambda x:METRIC_META.get(x,{}).get("label",x),
                        horizontal=True,key="hz_m")
    df_exp=_fp(dff,sel_exp)
    make_heatmap(df_exp,hz_metric,sel_exp,lpz,dff)

    # Zone detail table
    if not df_exp.empty:
        st.markdown('<div class="section-hdr">Zone Breakdown Table</div>', unsafe_allow_html=True)
        z_cols=[c for c in ["zone","swings","avg_tilt","avg_aa","avg_bat_speed","xwoba","avg_exit_velocity"] if c in df_exp.columns]
        zt=df_exp.groupby("zone",as_index=False)[z_cols[1:]].agg(
            {c:("sum" if c=="swings" else "mean") for c in z_cols[1:]}).round(3)
        zt.insert(0,"zone",zt.pop("zone") if "zone" in zt.columns else range(len(zt)))
        st.dataframe(zt.sort_values("swings",ascending=False),use_container_width=True,hide_index=True)

    # Download
    if not df_exp.empty:
        csv=df_exp.to_csv(index=False)
        st.download_button(f"⬇ Download {sel_exp} data",csv,f"{sel_exp.replace(' ','_')}_data.csv","text/csv")

    # AI Insight
    st.markdown('<div class="section-hdr">AI Scouting Insight</div>', unsafe_allow_html=True)
    if st.button("🤖 Generate AI Insight",key="ai_btn"):
        with st.spinner("Analyzing swing profile …"):
            insight=generate_ai_insight(sel_exp,row_exp,LG)
        st.markdown(f'<div class="ai-box"><div class="ai-label">⚡ AI SCOUTING REPORT — {sel_exp.upper()}</div>{insight}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">Click the button above to generate a personalized scouting insight powered by Claude AI.</div>',
                    unsafe_allow_html=True)

    # Pitch-group breakdown
    st.markdown('<div class="section-hdr">Pitch Group Breakdown</div>', unsafe_allow_html=True)
    df_exp2=_fp(detail_full,sel_exp)
    if not df_exp2.empty and "pitch_group" in df_exp2.columns:
        pg_agg=df_exp2.groupby("pitch_group",observed=True).agg(
            avg_tilt=("avg_tilt","mean"),avg_aa=("avg_aa","mean"),
            xwoba=("xwoba","mean"),swings=("swings","sum")).round(3).reset_index()
        fig_pg=px.bar(pg_agg,x="pitch_group",y="avg_tilt",color="xwoba",
            color_continuous_scale="RdYlGn",text="avg_tilt",
            labels={"pitch_group":"Pitch Group","avg_tilt":"Avg Tilt (°)"},
            title=f"{sel_exp} — Tilt by Pitch Group")
        fig_pg.update_traces(texttemplate="%{text:.1f}°",textposition="outside")
        fig_pg.add_hline(y=LG.get("avg_tilt",30),line_dash="dot",line_color="#8b949e",
            annotation_text="League avg",annotation_font_color="#8b949e")
        fig_pg=styled_fig(fig_pg,360)
        st.plotly_chart(fig_pg,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — COMPARISONS
# ══════════════════════════════════════════════════════════════════════
with tab_cmp:
    st.markdown('<div class="section-hdr">Multi-Batter Comparison</div>', unsafe_allow_html=True)
    if not spm:
        st.markdown('<div class="info-box">Select 2–6 batters in the sidebar to compare.</div>',
                    unsafe_allow_html=True)
    else:
        # Player cards row
        card_cols=st.columns(min(len(spm),3))
        for i,name in enumerate(spm):
            with card_cols[i%3]:
                render_player_card(name,get_batter_row(name))

        # Radar chart
        st.markdown('<div class="section-hdr">Swing Profile Radar</div>', unsafe_allow_html=True)
        radar_metrics=["avg_tilt","avg_aa","avg_bat_speed","xwoba","avg_exit_velocity"]
        radar_labels=[METRIC_META[m]["label"] for m in radar_metrics]
        fig_radar=go.Figure()
        for name in spm:
            row=get_batter_row(name)
            if row is None: continue
            # normalise 0-1 vs league
            vals=[]
            for m in radar_metrics:
                v=row.get(m,np.nan); lo,hi=HEATMAP_RANGES.get(m,(0,1))
                vals.append(float(np.clip((v-lo)/(hi-lo),0,1)) if not pd.isna(v) else 0)
            vals.append(vals[0])
            fig_radar.add_trace(go.Scatterpolar(r=vals,
                theta=radar_labels+[radar_labels[0]],fill="toself",name=name,opacity=0.7))
        fig_radar.update_layout(polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(visible=True,range=[0,1],color="#8b949e",gridcolor="#21262d"),
            angularaxis=dict(color="#c9d1d9",gridcolor="#21262d")),
            paper_bgcolor="#0d1117",font_color="#c9d1d9",
            title_text="Normalised Swing Profile",title_font_color="#f0a500",
            legend=dict(bgcolor="rgba(0,0,0,0)"),height=450)
        st.plotly_chart(fig_radar,use_container_width=True)

        # Side-by-side metric bar chart
        st.markdown('<div class="section-hdr">Metric by Metric</div>', unsafe_allow_html=True)
        cmp_metric=st.selectbox("Metric",list(METRIC_META.keys()),
                                format_func=lambda x:METRIC_META[x]["label"],key="cmp_m")
        cmp_data=[]
        for name in spm:
            row=get_batter_row(name)
            if row is None: continue
            cmp_data.append({"Batter":name,"Value":row.get(cmp_metric,np.nan),"Type":"Player"})
        cmp_data.append({"Batter":"League Avg","Value":LG.get(cmp_metric,np.nan),"Type":"League"})
        cdf=pd.DataFrame(cmp_data).dropna()
        if not cdf.empty:
            fig_cmp=px.bar(cdf,x="Batter",y="Value",color="Type",
                color_discrete_map={"Player":"#f0a500","League":"#388bfd"},
                labels={"Value":METRIC_META[cmp_metric]["label"]},
                title=METRIC_META[cmp_metric]["label"])
            fig_cmp=styled_fig(fig_cmp,360)
            fig_cmp.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_cmp,use_container_width=True)

        # Scatter: tilt vs selected output metric
        st.markdown('<div class="section-hdr">Scatter: Tilt vs Performance</div>', unsafe_allow_html=True)
        sc_y=st.selectbox("Y axis",["xwoba","batting_avg","avg_exit_velocity","avg_launch_angle"],
                           format_func=lambda x:METRIC_META[x]["label"],key="sc_y")
        sc_df=summary_df.dropna(subset=["avg_tilt",sc_y,"swings"]).copy()
        sc_df=sc_df[sc_df["swings"]>=30]
        fig_big_sc=px.scatter(sc_df,x="avg_tilt",y=sc_y,size="swings",
            hover_name="batter_name",size_max=24,
            color=sc_df["batter_name"].apply(lambda n:"Selected" if n in spm else "League"),
            color_discrete_map={"Selected":"#f0a500","League":"#388bfd"},
            opacity=0.7,
            labels={"avg_tilt":"Tilt (°)",sc_y:METRIC_META[sc_y]["label"]},
        )
        if HAS_STATSMODELS:
            fig_tr=px.scatter(sc_df,x="avg_tilt",y=sc_y,trendline="lowess")
            for tr in fig_tr.data:
                if hasattr(tr,"mode") and tr.mode=="lines":
                    tr.line.color="#8b949e"; fig_big_sc.add_trace(tr)
        fig_big_sc=styled_fig(fig_big_sc,450)
        st.plotly_chart(fig_big_sc,use_container_width=True)

        # Head-to-head zone heatmaps
        st.markdown('<div class="section-hdr">Side-by-Side Zone Heatmaps</div>', unsafe_allow_html=True)
        hz2=st.radio("Heatmap metric",["avg_tilt","delta_tilt","xwoba","avg_bat_speed"],
                      format_func=lambda x:METRIC_META.get(x,{}).get("label",x),
                      horizontal=True,key="cmp_hz")
        n_show=min(len(spm),4)
        hm_cols=st.columns(n_show)
        for i,name in enumerate(spm[:n_show]):
            with hm_cols[i]:
                make_heatmap(_fp(dff,name),hz2,name,lpz,dff,figsize=(5.5,5.5))

        # Download comparison CSV
        rows_dl=[]
        for name in spm:
            r=get_batter_row(name)
            if r is not None:
                d={"Batter":name}; d.update(r.to_dict()); rows_dl.append(d)
        if rows_dl:
            cmp_csv=pd.DataFrame(rows_dl).to_csv(index=False)
            st.download_button("⬇ Download comparison data",cmp_csv,"comparison.csv","text/csv")

# ══════════════════════════════════════════════════════════════════════
# TAB 4 — TRENDS  (multi-season)
# ══════════════════════════════════════════════════════════════════════
with tab_tr:
    st.markdown('<div class="section-hdr">Season-over-Season Trends</div>', unsafe_allow_html=True)
    if len(avail_seasons)<2:
        st.markdown(f"""
<div class="warn-box">
Only season <strong>{avail_seasons[0] if avail_seasons else '?'}</strong> data found.<br>
To enable trend analysis add CSV files for other seasons:
<code>players_summary_2024.csv</code>, <code>players_summary_2026.csv</code>
</div>""", unsafe_allow_html=True)
    else:
        tr_metric=st.selectbox("Trend metric",["avg_tilt","avg_aa","avg_bat_speed","xwoba","avg_exit_velocity"],
                                format_func=lambda x:METRIC_META[x]["label"],key="tr_m")
        # League trend
        lg_trend=[]
        for s in avail_seasons:
            p,d=load_season(s)
            if d is not None:
                v=d[tr_metric].mean() if tr_metric in d.columns else np.nan
                lg_trend.append({"Season":str(s),"Value":round(v,3),"Type":"League Avg"})
        if lg_trend:
            lgt=pd.DataFrame(lg_trend)
            fig_tr=go.Figure()
            fig_tr.add_trace(go.Scatter(x=lgt["Season"],y=lgt["Value"],mode="lines+markers",
                name="League Avg",line=dict(color="#388bfd",width=2.5),
                marker=dict(size=10,color="#388bfd")))
            # Selected batters
            for name in spm:
                btr=[]
                for s in avail_seasons:
                    _,d=load_season(s)
                    if d is None: continue
                    sub=d[d["batter_name"]==name]
                    v=sub[tr_metric].mean() if not sub.empty and tr_metric in sub.columns else np.nan
                    btr.append({"Season":str(s),"Value":round(v,3)})
                if btr and any(not pd.isna(r["Value"]) for r in btr):
                    bdf=pd.DataFrame(btr)
                    fig_tr.add_trace(go.Scatter(x=bdf["Season"],y=bdf["Value"],mode="lines+markers",
                        name=name,line=dict(width=2),marker=dict(size=9)))
            fig_tr=styled_fig(fig_tr,420)
            fig_tr.update_layout(title=f"{METRIC_META[tr_metric]['label']} — Season Trend",
                xaxis_title="Season",yaxis_title=METRIC_META[tr_metric]["label"])
            st.plotly_chart(fig_tr,use_container_width=True)

        # YoY delta table
        if len(avail_seasons)>=2:
            st.markdown('<div class="section-hdr">Year-over-Year Changes</div>', unsafe_allow_html=True)
            s_old,s_new=avail_seasons[-2],avail_seasons[-1]
            _,d_old=load_season(s_old); _,d_new=load_season(s_new)
            if d_old is not None and d_new is not None:
                cols=[c for c in ["avg_tilt","avg_aa","avg_bat_speed","xwoba"] if c in d_old.columns and c in d_new.columns]
                old_agg=d_old.groupby("batter_name")[cols+["swings"]].agg(
                    {c:"mean" for c in cols}|{"swings":"sum"}).add_suffix(f"_{s_old}")
                new_agg=d_new.groupby("batter_name")[cols+["swings"]].agg(
                    {c:"mean" for c in cols}|{"swings":"sum"}).add_suffix(f"_{s_new}")
                merged=old_agg.join(new_agg,how="inner").reset_index()
                for c in cols:
                    merged[f"Δ_{c}"]=round(merged[f"{c}_{s_new}"]-merged[f"{c}_{s_old}"],3)
                merged=merged[merged[f"swings_{s_old}"]>=30][merged[f"swings_{s_new}"]>=30]
                show_cols=["batter_name"]+[f"Δ_{c}" for c in cols]
                show_cols=[c for c in show_cols if c in merged.columns]
                merged_show=merged[show_cols].rename(columns={"batter_name":"Batter"} | {f"Δ_{c}":f"Δ {METRIC_META[c]['label']}" for c in cols})
                sort_col=f"Δ {METRIC_META['avg_tilt']['label']}"
                if sort_col in merged_show.columns:
                    merged_show=merged_show.sort_values(sort_col,ascending=False)
                st.dataframe(merged_show.head(30).reset_index(drop=True),use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 5 — TILT OPTIMIZER
# ══════════════════════════════════════════════════════════════════════
with tab_opt:
    st.markdown('<div class="section-hdr">Tilt Optimizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Partial-dependence curve: all features fixed at batter averages, <strong>tilt swept 8° → 62°</strong>. Optimal tilt is shrunk toward the league optimum proportionally to sample size (Bayesian shrinkage, K=50).</div>',
                unsafe_allow_html=True)

    if model is None:
        st.error("Model not available.")
    else:
        c1,c2,c3=st.columns([2,1,1])
        sp6=c1.selectbox("Batter",all_real,key="t6p")
        pg6=c2.selectbox("Pitch Group",["All"]+sorted(detail_full["pitch_group"].dropna().unique()),key="t6pg")
        z6=c3.selectbox("Zone",["All"]+[str(z) for z in range(1,15)],key="t6z")

        p6d=_fp(detail_fe,sp6)
        if pg6!="All": p6d=p6d[p6d["pitch_group"]==pg6]
        if z6!="All": p6d=p6d[p6d["zone"]==int(z6)]

        if p6d.empty:
            st.warning(f"No data for {sp6} with these filters.")
        else:
            n6=int(p6d["swings"].sum()); aa6=_sm(p6d["avg_aa"]); spd6=_sm(p6d["avg_bat_speed"])
            len6=_sm(p6d["avg_swing_len"]); ct=_sm(p6d["avg_tilt"])
            ge6=float(p6d["group_enc"].mean()); ze6=float(p6d["zone_enc"].mean())
            tg6,pr6=predict_tilt_curve(aa6,spd6,len6,ze6,ge6)
            ro6,ox6=find_optimal(tg6,pr6)
            so6,cw6=shrink(ro6,LEAGUE_OPT_TILT,n6)
            tgl6,prl6=predict_tilt_curve(_sm(detail_full["avg_aa"]),_sm(detail_full["avg_bat_speed"]),
                                          _sm(detail_full["avg_swing_len"]),ze6,ge6)
            k1,k2,k3,k4,k5=st.columns(5)
            k1.metric("Current Tilt",f"{ct:.1f}°")
            k2.metric("Optimal (shrunk)",f"{so6:.1f}°")
            dv=ct-so6
            k3.metric("Δ Tilt",f"{dv:+.1f}°",f"{-dv:+.1f}° gap",delta_color="inverse")
            k4.metric("Pred. xwOBA @ Opt.",f"{ox6:.3f}")
            k5.metric("Confidence",f"{cw6:.0%}",help=f"n={n6}·K=50")
            if n6<30:
                st.markdown('<div class="warn-box">⚠️ Small sample — estimate heavily shrunk toward league optimum. Treat as directional signal only.</div>',unsafe_allow_html=True)
            # Curve
            fig6=go.Figure()
            if show_ci:
                from scipy.ndimage import gaussian_filter1d as gf
                scale=1.0/np.sqrt(max(n6,1)/50); rng=np.random.default_rng(0); boots=[]
                for _ in range(120):
                    _,p=predict_tilt_curve(aa6+rng.normal(0,8*scale),spd6+rng.normal(0,3*scale),
                                            len6+rng.normal(0,0.3*scale),ze6,ge6)
                    boots.append(p)
                arr=np.vstack(boots); lo=np.percentile(arr,10,axis=0); hi=np.percentile(arr,90,axis=0)
                fig6.add_trace(go.Scatter(x=np.concatenate([tg6,tg6[::-1]]),
                    y=np.concatenate([hi,lo[::-1]]),fill="toself",
                    fillcolor="rgba(240,165,0,0.1)",line=dict(color="rgba(0,0,0,0)"),
                    name="80% CI",hoverinfo="skip"))
            fig6.add_trace(go.Scatter(x=tg6,y=pr6,mode="lines",
                line=dict(color="#f0a500",width=2.8),name=sp6))
            fig6.add_trace(go.Scatter(x=tgl6,y=prl6,mode="lines",
                line=dict(color="#388bfd",width=1.5,dash="dot"),name="League avg features"))
            for xv,col,lbl,dash in[(ct,"#3fb950",f"Current {ct:.1f}°","dash"),
                                    (so6,"#f0a500",f"Optimal {so6:.1f}°","solid"),
                                    (LEAGUE_OPT_TILT,"#388bfd",f"League opt {LEAGUE_OPT_TILT:.1f}°","dashdot")]:
                fig6.add_vline(x=xv,line=dict(color=col,width=1.5,dash=dash),
                               annotation_text=lbl,annotation_font_color=col,annotation_font_size=10)
            fig6=styled_fig(fig6,460)
            fig6.update_layout(title=f"Predicted xwOBA vs Tilt — {sp6}",
                xaxis_title="Swing Path Tilt (°)",yaxis_title="Predicted xwOBA",
                legend=dict(orientation="h",y=-0.22),hovermode="x unified")
            st.plotly_chart(fig6,use_container_width=True)

            # 2-D interaction
            st.markdown('<div class="section-hdr">Tilt × Attack Angle Interaction Surface</div>',unsafe_allow_html=True)
            tg2=np.linspace(TILT_MIN,TILT_MAX,30); aa2=np.linspace(-30,30,25)
            TT,AA=np.meshgrid(tg2,aa2); n2=TT.size
            X2=np.column_stack([TT.ravel(),AA.ravel(),np.full(n2,spd6),np.full(n2,len6),
                                 np.full(n2,ze6),np.full(n2,ge6),TT.ravel()*AA.ravel(),TT.ravel()*ge6])
            Z2=model.predict(X2).reshape(TT.shape)
            f2d=go.Figure(data=go.Heatmap(z=Z2,x=tg2.round(1),y=aa2.round(1),
                colorscale="RdYlGn",colorbar=dict(title="xwOBA"),
                hovertemplate="Tilt: %{x:.1f}°<br>AA: %{y:.1f}°<br>xwOBA: %{z:.3f}<extra></extra>"))
            f2d.add_trace(go.Scatter(x=[ct],y=[aa6],mode="markers",
                marker=dict(color="white",size=14,symbol="star",line=dict(color="#0d1117",width=2)),
                name=f"{sp6}"))
            f2d=styled_fig(f2d,420)
            f2d.update_layout(title="xwOBA Surface: Tilt × Attack Angle  (⭐ = current batter)",
                xaxis_title="Tilt (°)",yaxis_title="Attack Angle (°)")
            st.plotly_chart(f2d,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 6 — GLOSSARY & EDUCATION
# ══════════════════════════════════════════════════════════════════════
with tab_gl:
    st.markdown('<div class="section-hdr">Bat Tracking Glossary</div>', unsafe_allow_html=True)

    glossary=[
        ("⚾ Swing Path Tilt",
         "The angle (in degrees) between the bat's swing path plane and the horizontal ground plane. "
         "A higher tilt means a more upward, angled swing; a lower tilt is flatter and more level. "
         "Think of tilt as 'how steep the bat travels through the zone.' "
         "The MLB average is roughly 25–32°. Elite power hitters often sit 35–50°.",
         "Aaron Judge: ~47° (extreme uppercut). Freddie Freeman: ~40° (power tilt). "
         "Luis Arraez: ~15° (flat, contact-first)."),

        ("📐 Attack Angle",
         "The angle at which the bat head meets the ball at the moment of contact. Positive = upward swing. "
         "Optimal attack angle for hard contact is typically 8–15°, which matches a typical pitcher's "
         "downward plane and maximizes the window where bat-meets-ball. "
         "Attack angle is different from tilt — tilt is the full swing path plane; attack angle is the specific moment of contact.",
         "8–14° → sweet spot for launch angle / line drives / fly balls. "
         "< 0° → grounds out often. > 20° → pop-up risk increases."),

        ("🏏 Steep vs Flat Swing",
         "A 'steep' swing (high tilt + high attack angle) is designed to match pitcher plane and create lift. "
         "It can sacrifice contact rate but increases HR potential. "
         "A 'flat' swing (low tilt) is horizontal, great for contact but generates more grounders. "
         "Neither is universally superior — the optimal profile depends on bat speed, pitch mix faced, and ballpark.",
         "Steep (tilt > 40°): Judge, Stanton, Bobby Dalbec — high HR/PA, high strikeout. "
         "Flat (tilt < 20°): Arraez, Amed Rosario — high BA, low power. "
         "Balanced (20–35°): Correa, Trea Turner — versatile, contact + gap power."),

        ("🚀 Bat Speed",
         "The peak speed of the bat barrel during the swing, measured in mph. "
         "Higher bat speed extends the batter's reaction time window and generates more raw power. "
         "MLB average: ~70–73 mph. Elite: 77+ mph. "
         "Bat speed alone doesn't guarantee success — efficiency (swing length) and timing matter too.",
         "75+ mph → above-average power potential. < 67 mph → contact-first profile advised."),

        ("📏 Swing Length",
         "The total distance (in feet) the barrel travels from swing initiation to contact point. "
         "Shorter swings allow later contact decisions and are generally harder to fool on breaking balls. "
         "Longer swings can generate more power but create more vulnerability to inside pitches and off-speed. "
         "Optimal range: 6.0–7.5 ft for most MLB hitters.",
         "6.0–7.0 ft → efficient, disciplined swing. "
         "< 5.5 ft → short but may sacrifice reach. > 8.5 ft → long swing, exploitable."),

        ("📊 xwOBA (Expected Weighted On-Base Average)",
         "A statistic that measures the expected run value of each batted ball based on launch angle and exit velocity, "
         "plus strikeouts and walks. It removes luck (park factors, defensive positioning, BABIP variance) "
         "and gives a purer measure of a hitter's quality of contact. "
         "Scale: < .280 = below average · .310–.340 = average · .360+ = above average · .400+ = elite.",
         "Mike Trout career: ~.420 xwOBA. League avg: ~.315. Arraez: ~.330 (high contact keeps it solid despite no power)."),

        ("🎯 Optimal Tilt",
         "The swing path tilt angle that the model predicts would maximize xwOBA for a given batter, "
         "holding all other factors (bat speed, attack angle, pitch type, zone) constant. "
         "Because sample sizes per batter are limited, the model applies Bayesian shrinkage — "
         "blending the player-specific optimum toward the league-wide optimum. "
         "The more swings tracked, the more the estimate reflects that specific player.",
         "League optimal tilt (2025): ~30–38° depending on context. "
         "Players with fast bat speed often have higher optimal tilts (can afford the steeper path)."),
    ]

    for term,defn,example in glossary:
        st.markdown(f'<div class="gloss-term">{term}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="gloss-def">{defn}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="gloss-example">📌 {example}</div>', unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

    # Interactive "what does my tilt mean?" calculator
    st.markdown('<div class="section-hdr">What Does My Tilt Mean?</div>', unsafe_allow_html=True)
    tilt_input=st.slider("Enter a swing path tilt value:",5,65,30,1,key="gl_tilt")
    aa_input=st.slider("Enter attack angle:",- 20,30,10,1,key="gl_aa")
    if tilt_input>=45:    tilt_profile="🔴 Extreme Uppercut — maximum lift potential, contact sacrifice"
    elif tilt_input>=35:  tilt_profile="🟠 Power Tilt — solid HR profile, slight strikeout risk"
    elif tilt_input>=25:  tilt_profile="🟢 Balanced — versatile, good mix of contact and lift"
    elif tilt_input>=15:  tilt_profile="🔵 Line Drive — contact-first, limited home run power"
    else:                 tilt_profile="⚪ Flat/Ground Ball — extreme contact, minimal lift"
    if aa_input>=15:      aa_profile="High AA — pop-up risk above 20°, excellent lift 8–15°"
    elif aa_input>=6:     aa_profile="Optimal AA range — matches typical pitcher downward plane"
    elif aa_input>=0:     aa_profile="Neutral — some ground ball tendency"
    else:                 aa_profile="Negative AA — ground ball swing, contact-oriented"

    col1,col2=st.columns(2)
    col1.info(f"**Tilt {tilt_input}°:** {tilt_profile}")
    col2.info(f"**Attack Angle {aa_input}°:** {aa_profile}")

