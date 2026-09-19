"""
MLB Bat Tracking 2025-2026  ·  Swing Intelligence Dashboard
============================================================
Przywrócone heatmapy, tabele, filtry trendów, szybsze ładowanie.
Sezony: tylko 2025 i 2026
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="MLB Swing Intelligence 2025-2026",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# CSS (skrócony, ale zachowany styl)
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Oswald:wght@500;700&display=swap');
html, body, [data-testid="stAppViewContainer"] { background-color: #0d1117 !important; color: #e6edf3; font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #21262d; }
.dash-header { background: linear-gradient(135deg, #0d1117 0%, #1a2332 100%); border-bottom: 2px solid #f0a500; padding: 1.2rem 1.5rem; margin-bottom: 1rem; border-radius: 0 0 8px 8px; }
.dash-title { font-family: 'Oswald', sans-serif; font-size: 1.9rem; font-weight: 700; color: #f0a500; margin: 0; letter-spacing: 1px; }
.dash-subtitle { font-size: 0.8rem; color: #8b949e; margin: 0.2rem 0 0; }
.section-hdr { font-family: 'Oswald', sans-serif; font-size: 1.05rem; color: #f0a500; letter-spacing: 1.5px; text-transform: uppercase; border-bottom: 1px solid #21262d; padding-bottom: 0.3rem; margin: 1rem 0 0.6rem; }
.info-box { background: #161b22; border-left: 3px solid #388bfd; border-radius: 6px; padding: 0.8rem 1rem; margin: 0.4rem 0; font-size: 0.85rem; color: #c9d1d9; }
.warn-box { background: #1c1810; border-left: 3px solid #f0a500; border-radius: 6px; padding: 0.7rem 1rem; margin: 0.4rem 0; font-size: 0.82rem; color: #e6c77a; }
.player-card { background: linear-gradient(145deg, #161b22, #1c2333); border: 1px solid #21262d; border-left: 4px solid #f0a500; border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 0.8rem; }
.player-name { font-family: 'Oswald', sans-serif; font-size: 1.25rem; font-weight: 700; color: #f0a500; }
.metric-pill { display: inline-block; background: #21262d; border-radius: 16px; padding: 0.25rem 0.7rem; margin: 0.15rem; font-size: 0.78rem; font-weight: 600; }
.pill-elite { border: 1px solid #f0a500; color: #f0a500; }
.pill-good  { border: 1px solid #3fb950; color: #3fb950; }
.pill-avg   { border: 1px solid #8b949e; color: #8b949e; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #f0a500 !important; border-bottom: 2px solid #f0a500 !important; }
</style>
""", unsafe_allow_html=True)

def styled_fig(fig, height=400):
    fig.update_layout(
        height=height,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(color="#c9d1d9", family="Inter"),
        title_font=dict(color="#f0a500", family="Oswald", size=15),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────
SHRINKAGE_K = 50
TILT_MIN, TILT_MAX, TILT_GRID_N = 8.0, 62.0, 60
DATA_DIR = Path(".")
SEASONS = [2025, 2026]
CURRENT_SEASON = 2026
TILT_SEARCH_WINDOW = 15.0

FEATURE_COLS = ["avg_tilt", "avg_aa", "avg_bat_speed", "avg_swing_len",
                "zone_enc", "group_enc", "tilt_x_aa", "tilt_x_group"]

METRIC_META = {
    "avg_tilt":          {"label": "Tilt (°)", "fmt": ".1f"},
    "avg_aa":            {"label": "Attack Angle (°)", "fmt": ".1f"},
    "avg_bat_speed":     {"label": "Bat Speed (mph)", "fmt": ".1f"},
    "avg_swing_len":     {"label": "Swing Len (ft)", "fmt": ".2f"},
    "xwoba":             {"label": "xwOBA", "fmt": ".3f"},
    "avg_exit_velocity": {"label": "EV (mph)", "fmt": ".1f"},
    "swings":            {"label": "Swings", "fmt": ".0f"},
}

# ─────────────────────────────────────────────────────────────────────
# DATA LOADING (szybsze)
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⚾ Loading data…", ttl=3600)
def load_season(season: int):
    pf = DATA_DIR / f"players_summary_{season}.csv"
    df_f = DATA_DIR / f"detail_zone_pitchgroup_{season}.csv"
    if not pf.exists():
        return None, None
    players = pd.read_csv(pf)
    players = players.loc[:, ~players.columns.duplicated()].copy()
    players["season"] = season
    if not df_f.exists():
        return players, None
    detail = pd.read_csv(df_f)
    detail = detail.loc[:, ~detail.columns.duplicated()].copy()
    mask = detail["batter_name"].notna() & ~detail["batter_name"].str.contains(r" pitcher| P$", case=False, na=False, regex=True)
    detail = detail[mask].copy()
    detail["season"] = season
    return players, detail

@st.cache_data(show_spinner=False)
def load_all():
    all_p, all_d = [], []
    for s in SEASONS:
        p, d = load_season(s)
        if p is not None: all_p.append(p)
        if d is not None: all_d.append(d)
    players = pd.concat(all_p, ignore_index=True) if all_p else pd.DataFrame()
    detail = pd.concat(all_d, ignore_index=True) if all_d else pd.DataFrame()
    return players, detail

players_all, detail_all = load_all()
avail_seasons = sorted(detail_all["season"].dropna().unique().tolist()) if not detail_all.empty else []

with st.sidebar:
    st.markdown("### 📅 Season")
    if len(avail_seasons) > 1:
        sel_season = st.selectbox("Primary season", avail_seasons, index=len(avail_seasons)-1, key="season_pick")
    elif avail_seasons:
        sel_season = avail_seasons[0]
        st.caption(f"Only **{sel_season}** available")
    else:
        sel_season = CURRENT_SEASON

MAIN_SEASON = sel_season
players_raw = players_all[players_all["season"] == MAIN_SEASON].copy() if not players_all.empty else pd.DataFrame()
detail_full = detail_all[detail_all["season"] == MAIN_SEASON].copy() if not detail_all.empty else pd.DataFrame()

if detail_full.empty:
    st.error(f"No data for season {MAIN_SEASON}. Check CSV files.")
    st.stop()

# Rescale tilt
_obs = detail_full["avg_tilt"].dropna()
if len(_obs) >= 20:
    TILT_MIN = float(max(5.0, np.percentile(_obs, 1) - 2))
    TILT_MAX = float(min(75.0, np.percentile(_obs, 99) + 2))

all_real = sorted(players_raw["batter_name"].dropna().unique()) if not players_raw.empty else sorted(detail_full["batter_name"].dropna().unique())

def _fp(df, name):
    return df[df["batter_name"] == name].copy()

def _sm(s):
    v = s.dropna()
    return float(v.mean()) if len(v) else np.nan

# Feature engineering
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

# Model (lekki)
@st.cache_resource(show_spinner="Training model…")
def train_model():
    df = detail_fe.dropna(subset=["xwoba"]).query("swings >= 5").copy()
    if df.empty:
        return None, "No data"
    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median()).values
    y = df["xwoba"].values
    w = df["sample_weight"].values
    gbm = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=42
    )
    gbm.fit(X, y, sample_weight=w)
    return gbm, "Gradient Boosting"

model, model_type = train_model()

def predict_tilt_curve(avg_aa, avg_speed, avg_len, zone_enc, group_enc):
    tg = np.linspace(TILT_MIN, TILT_MAX, TILT_GRID_N)
    if model is None:
        return tg, np.full(TILT_GRID_N, np.nan)
    X = np.column_stack([
        tg, np.full(TILT_GRID_N, avg_aa), np.full(TILT_GRID_N, avg_speed),
        np.full(TILT_GRID_N, avg_len), np.full(TILT_GRID_N, zone_enc),
        np.full(TILT_GRID_N, group_enc), tg * avg_aa, tg * group_enc
    ])
    return tg, gaussian_filter1d(model.predict(X), sigma=1.5)

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

def shrink(pv, lv, n, k=SHRINKAGE_K):
    w = n / (n + k)
    return w * pv + (1 - w) * lv, round(w, 3)

_lg_ze = float(np.median(detail_fe["zone_enc"])) if not detail_fe.empty else 0.0
_lg_ge = float(np.median(detail_fe["group_enc"])) if not detail_fe.empty else 0.0
_tg_lg, _pr_lg = predict_tilt_curve(
    _sm(detail_full["avg_aa"]), _sm(detail_full["avg_bat_speed"]),
    _sm(detail_full["avg_swing_len"]), _lg_ze, _lg_ge
)
LEAGUE_OPT_TILT, _ = find_optimal_near(_tg_lg, _pr_lg, 32.0, 40)

# ─────────────────────────────────────────────────────────────────────
# HEADER + FILTERS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
    <p class="dash-title">⚾ MLB Swing Intelligence</p>
    <p class="dash-subtitle">Bat Tracking · Tilt Optimizer · 2025–2026</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔍 Filters")
    spm = st.multiselect("Players", all_real, default=all_real[:2] if len(all_real) >= 2 else all_real, key="spm")
    pg_filter = st.multiselect("Pitch Group", sorted(detail_full["pitch_group"].dropna().unique()),
                               default=list(detail_full["pitch_group"].dropna().unique()), key="pgf")
    min_swings = st.slider("Min. swings", 5, 150, 20, 5, key="minsw")
    show_ci = st.checkbox("Show CI bands", value=False, key="showci")

dff = detail_full.copy()
if pg_filter:
    dff = dff[dff["pitch_group"].isin(pg_filter)]
if "swings" in dff.columns:
    dff = dff[dff["swings"] >= min_swings]

_cze = float(np.median(dff["zone_enc"])) if "zone_enc" in dff.columns and not dff.empty else _lg_ze
_cge = float(np.median(dff["group_enc"])) if "group_enc" in dff.columns and not dff.empty else _lg_ge

# ─────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────
tab_exp, tab_heat, tab_cmp, tab_tr, tab_opt, tab_rank, tab_gl = st.tabs([
    "👤 Explorer", "🔥 Heatmaps", "⚔️ Compare", "📈 Trends",
    "🎯 Optimizer", "🏆 Rankings", "📖 Glossary"
])

# ── TAB 1: Explorer ──────────────────────────────────────────────────
with tab_exp:
    st.markdown('<div class="section-hdr">Player Explorer</div>', unsafe_allow_html=True)
    if not spm:
        st.info("Wybierz graczy w sidebarze.")
    else:
        for name in spm:
            pdf = _fp(dff, name)
            if pdf.empty:
                continue
            n = int(pdf["swings"].sum())
            ct = _sm(pdf["avg_tilt"])
            aa = _sm(pdf["avg_aa"])
            spd = _sm(pdf["avg_bat_speed"])
            xw = _sm(pdf["xwoba"])
            st.markdown(f"""
            <div class="player-card">
                <div class="player-name">{name}</div>
                <div style="color:#8b949e;font-size:0.75rem;">Season {MAIN_SEASON} · {n} swings</div>
                <div style="margin-top:0.6rem">
                    <span class="metric-pill pill-elite">Tilt {ct:.1f}°</span>
                    <span class="metric-pill pill-avg">AA {aa:.1f}°</span>
                    <span class="metric-pill pill-good">Speed {spd:.1f}</span>
                    <span class="metric-pill pill-avg">xwOBA {xw:.3f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── TAB 2: Heatmaps (przywrócone) ────────────────────────────────────
with tab_heat:
    st.markdown('<div class="section-hdr">Zone Heatmaps</div>', unsafe_allow_html=True)
    heat_metric = st.selectbox("Metric", ["avg_tilt", "avg_aa", "avg_bat_speed", "xwoba", "swings"],
                               format_func=lambda x: METRIC_META.get(x, {"label": x})["label"], key="heatm")
    heat_player = st.selectbox("Player (or League)", ["League"] + all_real, key="heatp")

    if heat_player == "League":
        hdf = dff.copy()
    else:
        hdf = _fp(dff, heat_player)

    if not hdf.empty and "zone" in hdf.columns:
        heat = hdf.groupby("zone")[heat_metric].mean().reset_index()
        # Prosta tabela + bar
        st.dataframe(heat.sort_values("zone"), width="stretch", hide_index=True)
        fig_h = px.bar(heat, x="zone", y=heat_metric, title=f"{METRIC_META.get(heat_metric, {}).get('label', heat_metric)} by Zone")
        fig_h = styled_fig(fig_h, 380)
        st.plotly_chart(fig_h, width="stretch")
    else:
        st.info("Brak danych zone dla wybranego filtra.")

# ── TAB 3: Compare ───────────────────────────────────────────────────
with tab_cmp:
    st.markdown('<div class="section-hdr">Comparison Table</div>', unsafe_allow_html=True)
    if len(spm) < 2:
        st.info("Wybierz minimum 2 graczy.")
    else:
        rows = []
        for name in spm:
            pdf = _fp(dff, name)
            if pdf.empty:
                continue
            rows.append({
                "Batter": name,
                "Swings": int(pdf["swings"].sum()),
                "Tilt": round(_sm(pdf["avg_tilt"]), 1),
                "AA": round(_sm(pdf["avg_aa"]), 1),
                "Bat Speed": round(_sm(pdf["avg_bat_speed"]), 1),
                "xwOBA": round(_sm(pdf["xwoba"]), 3),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# ── TAB 4: Trends (tylko 2025 / 2026 + filtry) ───────────────────────
with tab_tr:
    st.markdown('<div class="section-hdr">Trends 2025 → 2026</div>', unsafe_allow_html=True)

    if len(avail_seasons) < 2:
        st.info("Potrzebne dane z obu sezonów 2025 i 2026.")
    else:
        c1, c2, c3 = st.columns(3)
        tr_metric = c1.selectbox("Metric", ["avg_tilt", "avg_aa", "avg_bat_speed", "xwoba"],
                                 format_func=lambda x: METRIC_META[x]["label"], key="trm")
        tr_pg = c2.selectbox("Pitch Group", ["All"] + sorted(detail_all["pitch_group"].dropna().unique()), key="trpg")
        tr_zone = c3.selectbox("Zone", ["All"] + [str(z) for z in range(1, 15)], key="trz")

        fig_tr = go.Figure()
        # League
        lg_vals = []
        for s in [2025, 2026]:
            _, d = load_season(s)
            if d is None:
                continue
            mask = pd.Series(True, index=d.index)
            if tr_pg != "All":
                mask &= d["pitch_group"] == tr_pg
            if tr_zone != "All":
                mask &= d["zone"] == int(tr_zone)
            sub = d[mask]
            val = sub[tr_metric].mean() if not sub.empty and tr_metric in sub.columns else np.nan
            lg_vals.append({"Season": str(s), "Value": round(val, 3)})
        if lg_vals:
            lgt = pd.DataFrame(lg_vals)
            fig_tr.add_trace(go.Scatter(
                x=lgt["Season"], y=lgt["Value"], mode="lines+markers",
                name="League", line=dict(color="#388bfd", width=2.5), marker=dict(size=10)
            ))

        # Selected players
        for name in spm:
            vals = []
            for s in [2025, 2026]:
                _, d = load_season(s)
                if d is None:
                    continue
                mask = d["batter_name"] == name
                if tr_pg != "All":
                    mask &= d["pitch_group"] == tr_pg
                if tr_zone != "All":
                    mask &= d["zone"] == int(tr_zone)
                sub = d[mask]
                val = sub[tr_metric].mean() if not sub.empty and tr_metric in sub.columns else np.nan
                vals.append({"Season": str(s), "Value": round(val, 3)})
            if vals and any(not pd.isna(v["Value"]) for v in vals):
                bdf = pd.DataFrame(vals)
                fig_tr.add_trace(go.Scatter(
                    x=bdf["Season"], y=bdf["Value"], mode="lines+markers",
                    name=name, line=dict(width=2), marker=dict(size=9)
                ))

        fig_tr = styled_fig(fig_tr, 420)
        fig_tr.update_layout(
            title=f"{METRIC_META[tr_metric]['label']} — 2025 vs 2026",
            xaxis_title="Season",
            yaxis_title=METRIC_META[tr_metric]["label"],
            xaxis=dict(type="category")  # czyste etykiety 2025 / 2026
        )
        st.plotly_chart(fig_tr, width="stretch")

# ── TAB 5: Optimizer ─────────────────────────────────────────────────
with tab_opt:
    st.markdown('<div class="section-hdr">Tilt Optimizer</div>', unsafe_allow_html=True)
    if model is None:
        st.error("Model niedostępny.")
    else:
        c1, c2, c3 = st.columns([2, 1, 1])
        sp6 = c1.selectbox("Batter", all_real, key="optp")
        pg6 = c2.selectbox("Pitch Group", ["All"] + sorted(detail_full["pitch_group"].dropna().unique()), key="optpg")
        win6 = c3.slider("Window (°)", 5, 25, 15, 1, key="optwin")

        p6d = _fp(detail_fe, sp6)
        if pg6 != "All":
            p6d = p6d[p6d["pitch_group"] == pg6]

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
            k3.metric("Δ", f"{ct-so6:+.1f}°")
            k4.metric("Conf.", f"{cw6:.0%}")

            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(x=tg6, y=pr6, mode="lines", line=dict(color="#f0a500", width=2.5), name=sp6))
            fig6.add_vline(x=ct, line=dict(color="#3fb950", dash="dash"), annotation_text="Current")
            fig6.add_vline(x=so6, line=dict(color="#f0a500"), annotation_text="Optimal")
            fig6 = styled_fig(fig6, 420)
            fig6.update_layout(title=f"xwOBA vs Tilt — {sp6}", xaxis_title="Tilt (°)", yaxis_title="Pred. xwOBA")
            st.plotly_chart(fig6, width="stretch")

# ── TAB 6: Rankings (przywrócone tabele) ─────────────────────────────
with tab_rank:
    st.markdown('<div class="section-hdr">Tilt Rankings</div>', unsafe_allow_html=True)

    @st.cache_data(ttl=1800)
    def build_simple_rank(_df):
        agg = _df.groupby("batter_name").agg(
            swings=("swings", "sum"),
            avg_tilt=("avg_tilt", "mean"),
            avg_aa=("avg_aa", "mean"),
            xwoba=("xwoba", "mean"),
            avg_bat_speed=("avg_bat_speed", "mean")
        ).reset_index()
        agg = agg[agg["swings"] >= 20].sort_values("avg_tilt", ascending=False)
        return agg

    rank_df = build_simple_rank(dff)
    if not rank_df.empty:
        st.dataframe(
            rank_df.round({"avg_tilt": 1, "avg_aa": 1, "xwoba": 3, "avg_bat_speed": 1}),
            width="stretch", hide_index=True
        )
        csv = rank_df.to_csv(index=False)
        st.download_button("⬇ Download CSV", csv, "tilt_rankings.csv", "text/csv", key="dl_rank")
    else:
        st.info("Brak danych do rankingu.")

# ── TAB 7: Glossary ──────────────────────────────────────────────────
with tab_gl:
    st.markdown('<div class="section-hdr">Glossary</div>', unsafe_allow_html=True)
    st.markdown("""
    **Swing Path Tilt** – kąt płaszczyzny swinga względem ziemi. Wyższy = bardziej uppercut.  
    **Attack Angle** – kąt lufy w momencie kontaktu (optymalnie 8–15°).  
    **Bat Speed** – prędkość lufy (mph).  
    **xwOBA** – expected wOBA na podstawie EV + LA.  
    **Optimal Tilt** – tilt maksymalizujący xwOBA według modelu (z shrinkage).
    """)
