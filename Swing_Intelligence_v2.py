"""
MLB 2025 – Swing Path Tilt & Attack Angle Dashboard  v2.2
Fixes vs v2.1:
  * load_data() strips duplicate CSV columns (root cause of narwhals DuplicateError)
  * _clean() helper passes only the needed columns to every plotly call
  * _safe_scatter() skips chart when x == y (e.g. metric_t2 == "avg_tilt")
"""

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

st.set_page_config(page_title="MLB Bat Tracking 2025", layout="wide",
                   initial_sidebar_state="expanded")

SHRINKAGE_K = 50
TILT_MIN, TILT_MAX, TILT_GRID_N = 8.0, 62.0, 80
DATA_DIR = Path(".")

FEATURE_COLS = ["avg_tilt","avg_aa","avg_bat_speed","avg_swing_len",
                "zone_enc","group_enc","tilt_x_aa","tilt_x_group"]

METRIC_LABELS = {
    "avg_tilt":"Swing Path Tilt (°)","avg_aa":"Attack Angle (°)",
    "avg_bat_speed":"Bat Speed (mph)","avg_swing_len":"Swing Length (ft)",
    "batting_avg":"Batting Average","xwoba":"xwOBA",
    "avg_exit_velocity":"Exit Velocity (mph)","avg_launch_angle":"Launch Angle (°)",
    "swings":"Swings",
}

HEATMAP_RANGES = {
    "avg_tilt":(8,60),"tilt_std":(0,20),"delta_tilt":(-20,20),
    "avg_aa":(-35,35),"aa_std":(0,20),"avg_bat_speed":(55,88),
    "avg_swing_len":(4.5,9.5),"swings":(0,400),"batting_avg":(0.150,0.400),
    "xwoba":(0.200,0.600),"avg_exit_velocity":(75,105),"avg_launch_angle":(-15,45),
}

TEXTS = {
    "en":{
        "title":"Swing Intelligence",
         "tab_glossary":"📖 Glossary",
        "sidebar_players":"Select Batters","sidebar_min_swings":"Min. swings (tables)",
        "sidebar_pitch_group":"Pitch group","sidebar_pitch_type":"Pitch type",
        "all":"All","league_avg":"League Average",
        "tab_summary":"📊 Summary","tab_groups":"📦 Group Comp.",
        "tab_heatmaps":"🔥 Heatmaps","tab_side_by_side":"↔ Side-by-Side",
        "tab_player_compare":"🎯 Batter Metrics","tab_tilt_sim":"🔬 Tilt Optimizer",
        "tab_tilt_rankings":"🏆 Tilt Rankings",
        "selected_players":"**Selected Batters**","all_players":"**All Batters (post-filter)**",
        "no_player":"Select at least one real batter","no_data":"No data after filters",
        "metric":"Metric","compare_left":"Left Batter","compare_right":"Right Batter",
        "no_data_for":"No data for","detailed_table":"Detailed Table",
        "select_player":"Select Batter","left_metric":"Left Metric","right_metric":"Right Metric",
        "optimal_tilt":"Optimal Tilt (shrunk)","current_tilt":"Current Avg Tilt",
        "tilt_delta":"Δ Tilt  (Current − Optimal)","pred_xwoba":"Pred. xwOBA @ Optimal",
        "conf_weight":"Confidence (0–1)","heatmap_view":"Heatmap view mode",
        "show_ci":"Show 80 % confidence bands",
    },
    "pl":{
        "title":"Swing Intelligence",
         "tab_glossary":"📖 Słownik",
        "sidebar_players":"Wybierz batterów","sidebar_min_swings":"Min. swingów (tabele)",
        "sidebar_pitch_group":"Grupa rzutów","sidebar_pitch_type":"Typ rzutu",
        "all":"Wszystkie","league_avg":"Średnia ligi",
        "tab_summary":"📊 Podsumowanie","tab_groups":"📦 Grupy",
        "tab_heatmaps":"🔥 Heatmapy","tab_side_by_side":"↔ Porównanie",
        "tab_player_compare":"🎯 Metryki","tab_tilt_sim":"🔬 Symulator",
        "tab_tilt_rankings":"🏆 Ranking",
        "selected_players":"**Wybrani batterzy**","all_players":"**Wszyscy batterzy (po filtrach)**",
        "no_player":"Wybierz co najmniej jednego battera","no_data":"Brak danych po filtrach",
        "metric":"Metryka","compare_left":"Batter lewy","compare_right":"Batter prawy",
        "no_data_for":"Brak danych dla","detailed_table":"Tabela szczegółowa",
        "select_player":"Wybierz battera","left_metric":"Metryka lewa","right_metric":"Metryka prawa",
        "optimal_tilt":"Optymalny tilt (shrunk)","current_tilt":"Aktualny śr. tilt",
        "tilt_delta":"Δ Tilt (aktualny − optymalny)","pred_xwoba":"Prognozowane xwOBA @ optimum",
        "conf_weight":"Pewność (0–1)","heatmap_view":"Tryb widoku heatmapy",
        "show_ci":"Pokaż 80 % przedziały ufności",
    },
}

if "lang" not in st.session_state:
    st.session_state.lang = "en"
with st.sidebar:
    lang_sel = st.selectbox("Language / Język", ["English","Polski"], index=0)
    new_lang = {"English":"en","Polski":"pl"}[lang_sel]
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()
t = TEXTS[st.session_state.lang]

# ── Plotly safety helpers ────────────────────────────────────────────
def _clean(df, *cols):
    """Unique-column subset — prevents narwhals DuplicateError."""
    existing = list(dict.fromkeys(c for c in cols if c in df.columns))
    return df[existing].copy()

try:
    import statsmodels  # noqa
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

def _safe_scatter(df, x, y, color=None, size=None, hover_data=None,
                  trendline=None, labels=None, title=""):
    """Returns None when x==y; drops trendline if statsmodels missing."""
    if x == y:
        return None
    # trendline="lowess"/"ols" both require statsmodels
    if trendline and not HAS_STATSMODELS:
        trendline = None
    cols = [c for c in [x,y,color,size]+(hover_data or []) if c]
    hd   = [c for c in (hover_data or []) if c not in (x,y)]
    return px.scatter(_clean(df,*cols), x=x, y=y, color=color, size=size,
                      hover_data=hd or None, trendline=trendline,
                      labels=labels or {}, title=title)

# ── Data loading ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="⚾ Loading data …")
def load_data():
    players = pd.read_csv(DATA_DIR/"players_summary_2025.csv")
    detail  = pd.read_csv(DATA_DIR/"detail_zone_pitchgroup_2025.csv")
    # FIX: remove duplicate columns before narwhals ever sees the frame
    players = players.loc[:,~players.columns.duplicated()].copy()
    detail  = detail.loc[:,~detail.columns.duplicated()].copy()
    mask = (detail["batter_name"].notna() &
            ~detail["batter_name"].str.contains(r" pitcher| P$",case=False,na=False,regex=True))
    return players, detail[mask].copy()

players_raw, detail_full = load_data()

# ── Player ID resolution ─────────────────────────────────────────────
use_id=False; id_col=None; player_info=None
display_to_id={}; id_to_display={}
for _c in ("batter_id","batter","mlb_id","player_id","id"):
    if _c in players_raw.columns and _c in detail_full.columns:
        id_col=_c; use_id=True; break

if use_id:
    player_info = players_raw[[id_col,"batter_name"]].drop_duplicates()
    _d = player_info["batter_name"].value_counts()
    _d = _d[_d>1].index.tolist()
    player_info["display_name"] = player_info.apply(
        lambda r: f"{r['batter_name']} (ID:{int(r[id_col])})"
        if r["batter_name"] in _d else r["batter_name"], axis=1)
    display_to_id = dict(zip(player_info["display_name"],player_info[id_col]))
    id_to_display = dict(zip(player_info[id_col],player_info["display_name"]))
    all_real = sorted(player_info["display_name"])
else:
    all_real = sorted(players_raw["batter_name"].dropna().unique())

def _dn(val): return id_to_display.get(val,str(val)) if use_id else str(val)
def _fp(df,name):
    if use_id:
        pid=display_to_id.get(name)
        return df[df[id_col]==pid].copy() if pid is not None else pd.DataFrame()
    return df[df["batter_name"]==name].copy()

# ── Feature engineering ──────────────────────────────────────────────
def _engineer(df):
    out=df.copy(); le_z=LabelEncoder(); le_g=LabelEncoder()
    out["zone_enc"]     =le_z.fit_transform(out["zone"].astype(str))
    out["group_enc"]    =le_g.fit_transform(out["pitch_group"].fillna("Unknown").astype(str))
    out["tilt_x_aa"]    =out["avg_tilt"]*out["avg_aa"]
    out["tilt_x_group"] =out["avg_tilt"]*out["group_enc"]
    out["sample_weight"]=np.sqrt(out["swings"].clip(lower=1))
    return out,le_z,le_g

detail_fe,le_zone_g,le_group_g = _engineer(detail_full)
def _enc_g(name): return float(le_group_g.transform([name])[0]) if name in le_group_g.classes_ else 0.0

# ── Model training ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Training swing model …")
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
            gam.gridsearch(X,y,weights=w,progress=False)
            return gam,"GAM (pyGAM)"
        except Exception: pass
    gbm=GradientBoostingRegressor(n_estimators=400,max_depth=4,learning_rate=0.035,
                                   subsample=0.75,min_samples_leaf=4,max_features=0.8,random_state=42)
    gbm.fit(X,y,sample_weight=w)
    return gbm,"Gradient Boosting"

model,model_type=train_model()

# ── Shrinkage ────────────────────────────────────────────────────────
def shrink(pv,lv,n,k=SHRINKAGE_K):
    w=n/(n+k); return w*pv+(1-w)*lv,round(w,3)

# ── Tilt curve ───────────────────────────────────────────────────────
def predict_tilt_curve(avg_aa,avg_speed,avg_len,zone_enc,group_enc):
    tg=np.linspace(TILT_MIN,TILT_MAX,TILT_GRID_N)
    if model is None: return tg,np.full(TILT_GRID_N,np.nan)
    X=np.column_stack([tg,np.full(TILT_GRID_N,avg_aa),np.full(TILT_GRID_N,avg_speed),
                        np.full(TILT_GRID_N,avg_len),np.full(TILT_GRID_N,zone_enc),
                        np.full(TILT_GRID_N,group_enc),tg*avg_aa,tg*group_enc])
    return tg,gaussian_filter1d(model.predict(X),sigma=1.8)

def find_optimal(tg,preds):
    if np.all(np.isnan(preds)): return float(np.mean(tg)),float("nan")
    idx=int(np.nanargmax(preds)); return float(tg[idx]),float(preds[idx])

def approx_ci(avg_aa,avg_speed,avg_len,zone_enc,group_enc,n_swings,n_boot=150):
    scale=1.0/np.sqrt(max(n_swings,1)/SHRINKAGE_K); rng=np.random.default_rng(0); boots=[]
    for _ in range(n_boot):
        _,p=predict_tilt_curve(avg_aa+rng.normal(0,8*scale),avg_speed+rng.normal(0,3*scale),
                                avg_len+rng.normal(0,0.3*scale),zone_enc,group_enc)
        boots.append(p)
    arr=np.vstack(boots); return np.percentile(arr,10,axis=0),np.percentile(arr,90,axis=0)

# ── League optimal ───────────────────────────────────────────────────
def _sm(s): v=s.dropna(); return float(v.mean()) if len(v) else 0.0
_lg_aa=_sm(detail_full["avg_aa"]); _lg_spd=_sm(detail_full["avg_bat_speed"])
_lg_len=_sm(detail_full["avg_swing_len"])
_lg_ze=float(np.median(detail_fe["zone_enc"])) if not detail_fe.empty else 0.0
_lg_ge=float(np.median(detail_fe["group_enc"])) if not detail_fe.empty else 0.0
_tg_lg,_pr_lg=predict_tilt_curve(_lg_aa,_lg_spd,_lg_len,_lg_ze,_lg_ge)
LEAGUE_OPT_TILT,_=find_optimal(_tg_lg,_pr_lg)

# ── Filters ──────────────────────────────────────────────────────────
def apf(df,pg,pt):
    if pg!=t["all"]: df=df[df["pitch_group"]==pg]
    if pt!=t["all"]: df=df[df["pitch_type"]==pt]
    return df

def gpzd(name,df_ctx,lz):
    if name==t["league_avg"]: return lz.copy()
    sub=_fp(df_ctx,name)
    if sub.empty: return pd.DataFrame()
    agg={c:"mean" for c in METRIC_LABELS if c!="swings" and c in sub.columns}; agg["swings"]="sum"
    return sub.groupby("zone",as_index=False).agg(agg).round(3)

# ── Sidebar ───────────────────────────────────────────────────────────
st.title(t["title"])
st.caption(f"🤖 Model: **{model_type}** · League opt. tilt: **{LEAGUE_OPT_TILT:.1f}°** · K={SHRINKAGE_K}")
with st.sidebar:
    st.markdown("### ⚾ Batter Selection")
    _SEP="─"*22
    spm=st.multiselect(t["sidebar_players"],options=[t["league_avg"],_SEP]+all_real,
                        default=[all_real[0]] if all_real else [],max_selections=8)
    spm=[p for p in spm if p!=_SEP]
    st.markdown("### 🎛 Filters")
    min_swings=st.slider(t["sidebar_min_swings"],0,300,0,10)
    pgl=[t["all"]]+sorted(detail_full["pitch_group"].dropna().unique())
    sel_pitch=st.selectbox(t["sidebar_pitch_group"],pgl)
    _pts=detail_full if sel_pitch==t["all"] else detail_full[detail_full["pitch_group"]==sel_pitch]
    ptl=[t["all"]]+sorted(_pts["pitch_type"].dropna().unique())
    sel_type=st.selectbox(t["sidebar_pitch_type"],ptl)
    st.markdown("### 🔧 Display")
    view_mode=st.radio(t["heatmap_view"],["Raw","Percentile","Shrunk"],index=0)
    show_ci=st.checkbox(t["show_ci"],value=True)

dff=apf(detail_full.copy(),sel_pitch,sel_type)
dfe_f=apf(detail_fe.copy(),sel_pitch,sel_type)
_LA={c:"mean" for c in METRIC_LABELS if c!="swings" and c in dff.columns}; _LA["swings"]="sum"
lpz=dff.groupby("zone",as_index=False,observed=True).agg(_LA).round(3)
lpz["batter_name"]=t["league_avg"]
sel_disp=[p for p in spm if p!=t["league_avg"]]
if use_id: pfc=id_col; pfv=[display_to_id[p] for p in sel_disp if p in display_to_id]
else:       pfc="batter_name"; pfv=sel_disp
dt=dff[dff["swings"]>=min_swings].copy()

# ── Opt table ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⚙️ Computing optimizations …",ttl=3600)
def build_opt_table(_df,pg,pt,lo,ze,ge):
    gc=id_col if use_id else "batter_name"
    agg=_df.groupby(gc,observed=True).agg(
        avg_tilt=("avg_tilt","mean"),avg_aa=("avg_aa","mean"),
        avg_bat_speed=("avg_bat_speed","mean"),avg_swing_len=("avg_swing_len","mean"),
        xwoba=("xwoba","mean"),swings=("swings","sum"),
    ).reset_index().dropna(subset=["avg_tilt","avg_aa","avg_bat_speed","avg_swing_len"])
    rows=[]
    for _,r in agg.iterrows():
        n=int(r.swings)
        if n<5: continue
        tg,preds=predict_tilt_curve(float(r.avg_aa),float(r.avg_bat_speed),float(r.avg_swing_len),ze,ge)
        ro,ox=find_optimal(tg,preds); so,cw=shrink(ro,lo,n)
        rows.append({"Batter":_dn(r[gc]),"Swings":n,"Current Tilt":round(float(r.avg_tilt),1),
                     "Optimal Tilt (raw)":round(ro,1),"Optimal Tilt":round(so,1),
                     "Δ Tilt":round(float(r.avg_tilt)-so,1),"Pred. xwOBA @ Opt.":round(ox,3),
                     "Confidence":round(cw,2),
                     "Current xwOBA":round(float(r.xwoba),3) if not np.isnan(r.xwoba) else np.nan})
    return pd.DataFrame(rows).sort_values("Swings",ascending=False).reset_index(drop=True)

_cze=float(np.median(dfe_f["zone_enc"])) if not dfe_f.empty else _lg_ze
_cge=float(np.median(dfe_f["group_enc"])) if not dfe_f.empty else _lg_ge
opt_table=build_opt_table(dfe_f,sel_pitch,sel_type,LEAGUE_OPT_TILT,_cze,_cge)

# ── Heatmap helpers ──────────────────────────────────────────────────
def _zc(val,vmin,vmax,cmap):
    if pd.isna(val): return(0.91,0.91,0.91)
    return cmap(float(np.clip((val-vmin)/max(vmax-vmin,1e-9),0,1)))
def _tc(bg): r,g,b=bg[:3]; return "black" if 0.299*r+0.587*g+0.114*b>0.45 else "white"
def _fv(val,metric,view):
    if pd.isna(val): return "—"
    if view=="Percentile": return f"{int(round(val))}th"
    if metric=="swings": return f"{int(round(val))}"
    if metric in("batting_avg","xwoba"): return f"{val:.3f}"
    return f"{val:.1f}"

def _gp(df_p,metric,view,ldf,df_ctx):
    if metric=="tilt_std": return df_p.groupby("zone")["avg_tilt"].std(ddof=1).round(1)
    if metric=="aa_std": return df_p.groupby("zone")["avg_aa"].std(ddof=1).round(1)
    if metric=="delta_tilt":
        pm=df_p.groupby("zone")["avg_tilt"].mean()
        lm=ldf.set_index("zone")["avg_tilt"] if ldf is not None else pd.Series()
        return(pm-lm.reindex(pm.index,fill_value=np.nan)).round(2)
    raw=(df_p.groupby("zone")["swings"].sum() if metric=="swings"
         else df_p.groupby("zone")[metric].mean())
    if view=="Raw": return raw.round(3)
    if view=="Percentile":
        res={}
        for z,v in raw.items():
            vs=df_ctx[df_ctx["zone"]==z][metric].dropna()
            res[z]=percentileofscore(vs,v,kind="rank") if len(vs) else np.nan
        return pd.Series(res)
    if view=="Shrunk":
        nsw=df_p.groupby("zone")["swings"].sum()
        lg=(df_ctx.groupby("zone")[metric].sum() if metric=="swings"
            else df_ctx.groupby("zone")[metric].mean())
        res={}
        for z,v in raw.items():
            n=int(nsw.get(z,0)); lgv=float(lg.get(z,df_ctx[metric].mean()))
            res[z]=round(shrink(v,lgv,n)[0],3)
        return pd.Series(res)
    return raw.round(3)

def make_heatmap(df_p,metric,title,ldf=None,view="Raw",df_ctx=None):
    if df_p is None or df_p.empty: st.warning(f"{t['no_data_for']} {title}"); return
    if df_ctx is None: df_ctx=detail_full
    pivot=_gp(df_p,metric,view,ldf,df_ctx)
    nsw=df_p.groupby("zone")["swings"].sum() if "swings" in df_p.columns else pd.Series()
    if view=="Percentile" and metric not in("tilt_std","aa_std","delta_tilt"):
        vmin,vmax,cn=0,100,"RdYlGn"
    elif metric=="delta_tilt":
        vmin,vmax=HEATMAP_RANGES.get(metric,(-20,20)); cn="RdBu_r"
    else:
        vmin,vmax=HEATMAP_RANGES.get(metric,(0,100)); cn="YlOrRd"
    cmap=sns.color_palette(cn,as_cmap=True)
    b,ms,sy=0.85,3.3,2.5; mx,my=b,b; tx,ty=mx+ms,my+ms; half=ms/2
    fig,ax=plt.subplots(figsize=(8,8))
    for i in range(3):
        for j in range(3):
            zone=i*3+j+1; val=pivot.get(zone,np.nan); n=int(nsw.get(zone,0))
            x=mx+j*(ms/3); y=my+(2-i)*(ms/3); col=_zc(val,vmin,vmax,cmap)
            ax.add_patch(plt.Rectangle((x,y),ms/3,ms/3,facecolor=col,edgecolor="black",linewidth=2.4))
            txt=f"{zone}\n{_fv(val,metric,view)}"+(f"\n⚠n={n}" if n<20 and view!="Percentile" else "")
            ax.text(x+ms/6,y+ms/6,txt,ha="center",va="center",fontsize=10.5,fontweight="bold",color=_tc(col))
    ld=[(11,[(0,sy),(b,sy),(b,ty),(mx,ty),(mx+half,ty),(mx+half,5),(0,5),(0,sy)],(b*0.4,5-b*0.4)),
        (12,[(tx,sy),(tx,ty),(mx+half,ty),(mx+half,5),(5,5),(5,sy),(tx,sy)],(5-b*0.4,5-b*0.4)),
        (13,[(0,sy),(b,sy),(b,my),(mx,my),(mx+half,my),(mx+half,0),(0,0),(0,sy)],(b*0.4,b*0.4)),
        (14,[(tx,sy),(tx,my),(mx+half,my),(mx+half,0),(5,0),(5,sy),(tx,sy)],(5-b*0.4,b*0.4))]
    for z,verts,(cx,cy) in ld:
        val=pivot.get(z,np.nan); n=int(nsw.get(z,0)); col=_zc(val,vmin,vmax,cmap)
        ax.add_patch(PathPatch(MPath(verts),facecolor=col,edgecolor="black",linewidth=2.4))
        txt=f"{z}\n{_fv(val,metric,view)}"+(f"\n⚠n={n}" if n<20 and view!="Percentile" else "")
        ax.text(cx,cy,txt,ha="center",va="center",fontsize=10.5,fontweight="bold")
    ax.add_patch(plt.Rectangle((mx,my),ms,ms,fill=False,edgecolor="red",linewidth=3.8))
    vs={"Percentile":" [Pct]","Shrunk":" [Shrunk]"}.get(view,"")
    ax.set_title(f"{title}{vs}",fontsize=14,pad=16,fontweight="bold")
    ax.set_xlim(0,5); ax.set_ylim(0,5); ax.set_aspect("equal"); ax.axis("off")
    lm={"avg_tilt":"TILT (°)","tilt_std":"TILT STD","delta_tilt":"Δ TILT","avg_aa":"ATTACK ANGLE (°)",
        "aa_std":"AA STD","avg_bat_speed":"BAT SPEED (mph)","avg_swing_len":"SWING LEN (ft)",
        "swings":"SWINGS","batting_avg":"BATTING AVG","xwoba":"xwOBA",
        "avg_exit_velocity":"EXIT VELO (mph)","avg_launch_angle":"LAUNCH ANGLE (°)"}
    sm=plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin=vmin,vmax=vmax))
    cbar=plt.colorbar(sm,ax=ax,shrink=0.76,pad=0.04); cbar.set_label(lm.get(metric,metric.upper()),fontsize=11)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

# ── Tabs ─────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8=st.tabs([
    t["tab_summary"],t["tab_groups"],t["tab_heatmaps"],t["tab_side_by_side"],
    t["tab_player_compare"],t["tab_tilt_sim"],t["tab_tilt_rankings"],
    t["tab_glossary"]])

# ── Tab 1: Summary ───────────────────────────────────────────────────
with tab1:
    st.subheader(t["tab_summary"])
    _C=["batter_name","avg_tilt","avg_aa","avg_bat_speed","avg_swing_len",
        "swings","batting_avg","xwoba","avg_exit_velocity","avg_launch_angle"]
    _A={c:("sum" if c=="swings" else "mean") for c in _C[1:] if c in dt.columns}
    if spm:
        st.markdown(t["selected_players"])
        ss=(dt[dt[pfc].isin(pfv)] if pfv else pd.DataFrame())
        if not ss.empty:
            g=id_col if use_id else "batter_name"
            sm2=ss.groupby(g,observed=True).agg(_A).round(3).reset_index()
            if use_id: sm2["batter_name"]=sm2[id_col].map(id_to_display)
            sm2=sm2[[c for c in _C if c in sm2.columns]]
            if t["league_avg"] in spm:
                la=dt.mean(numeric_only=True).round(3).to_frame().T
                la["batter_name"]=t["league_avg"]; la["swings"]=int(dt["swings"].sum())
                sm2=pd.concat([sm2,la[[c for c in _C if c in la.columns]]],ignore_index=True)
            st.dataframe(sm2.sort_values("swings",ascending=False),use_container_width=True,hide_index=True)
        st.markdown("---")
    st.markdown(t["all_players"])
    if not dt.empty:
        g=id_col if use_id else "batter_name"
        als=dt.groupby(g,observed=True).agg(_A).round(3).reset_index()
        if use_id: als["batter_name"]=als[id_col].map(id_to_display)
        als=als[[c for c in _C if c in als.columns]]
        st.dataframe(als.sort_values("swings",ascending=False),use_container_width=True,hide_index=True)

    st.markdown("---")
    st.markdown("**Swing Intelligence**")
    st.caption("Data source: Baseball Savant (MLB Statcast) · Bat-tracking data exported to CSV and processed for this dashboard.")

# ── Tab 2: Group Comparison ──────────────────────────────────────────
with tab2:
    st.subheader(t["tab_groups"])
    if not sel_disp: st.info(t["no_player"])
    elif dff.empty: st.info(t["no_data"])
    else:
        m2=st.selectbox(t["metric"],list(METRIC_LABELS),format_func=lambda x:METRIC_LABELS[x],key="t2m")
        tmp=dff[dff[pfc].isin(pfv)].copy()
        tmp["player_display"]=(tmp[id_col].map(id_to_display) if use_id else tmp["batter_name"])
        c1,c2=st.columns(2)
        with c1:
            st.plotly_chart(px.box(_clean(tmp,"pitch_group",m2,"player_display"),
                x="pitch_group",y=m2,color="player_display",points="outliers",
                labels={"pitch_group":"Pitch Group",m2:METRIC_LABELS[m2]},
                title=f"Distribution – {METRIC_LABELS[m2]}"),use_container_width=True)
        with c2:
            ad=tmp.groupby(["player_display","pitch_group"],as_index=False)[m2].mean()
            st.plotly_chart(px.bar(_clean(ad,"pitch_group",m2,"player_display"),
                x="pitch_group",y=m2,color="player_display",barmode="group",
                labels={"pitch_group":"Pitch Group",m2:METRIC_LABELS[m2]},
                title=f"Average – {METRIC_LABELS[m2]}"),use_container_width=True)
        st.markdown("#### Tilt vs Metric (scatter)")
        if m2=="avg_tilt": st.info("Select a different metric to show scatter vs tilt.")
        else:
            fs=_safe_scatter(tmp,x="avg_tilt",y=m2,color="player_display",trendline="lowess",
                              labels={"avg_tilt":"Avg Tilt (°)",m2:METRIC_LABELS[m2]},
                              title=f"Tilt vs {METRIC_LABELS[m2]}")
            if fs: st.plotly_chart(fs,use_container_width=True)

# ── Tab 3: Heatmaps ──────────────────────────────────────────────────
with tab3:
    st.subheader(t["tab_heatmaps"])
    if not spm: st.info("Select batters in the sidebar.")
    else:
        m3=st.radio(t["metric"],
            ["avg_tilt","tilt_std","delta_tilt","avg_aa","aa_std","avg_bat_speed","avg_swing_len",
             "batting_avg","xwoba","avg_exit_velocity","avg_launch_angle","swings"],
            format_func=lambda x:METRIC_LABELS.get(x,x),horizontal=True,key="t3m")
        for p in spm:
            dfp=gpzd(p,dff,lpz)
            if dfp is None or dfp.empty: st.caption(f"{t['no_data_for']} {p}"); continue
            make_heatmap(dfp,m3,p,lpz,view_mode,dff)
            if p!=t["league_avg"]:
                st.markdown("---"); st.subheader(f"{t['detailed_table']} – {p}")
                pd2=_fp(dt,p)
                if not pd2.empty:
                    zagg=pd2.groupby("zone",as_index=False).agg({
                        "swings":"sum","avg_tilt":["mean","std"],"avg_aa":["mean","std"],
                        "avg_bat_speed":"mean","avg_swing_len":"mean","batting_avg":"mean",
                        "xwoba":"mean","avg_exit_velocity":"mean","avg_launch_angle":"mean"}).round(3)
                    zagg.columns=["_".join(filter(None,c)) for c in zagg.columns]
                    st.dataframe(zagg.sort_values("swings_sum",ascending=False),use_container_width=True,hide_index=True)
    st.markdown("---"); st.markdown(t["all_players"])
    if not dt.empty:
        _cv=["swings","avg_tilt","avg_aa","avg_bat_speed","avg_swing_len","batting_avg","xwoba","avg_exit_velocity","avg_launch_angle"]
        g=id_col if use_id else "batter_name"
        at3=dt.groupby(g,observed=True).agg({c:("sum" if c=="swings" else "mean") for c in _cv if c in dt.columns}).round(3).reset_index()
        if use_id: at3["batter_name"]=at3[id_col].map(id_to_display)
        st.dataframe(at3.sort_values("swings",ascending=False),use_container_width=True,hide_index=True)

# ── Tab 4: Side-by-side ──────────────────────────────────────────────
with tab4:
    st.subheader(t["tab_side_by_side"])
    _co=[t["league_avg"]]+(sorted(player_info["display_name"]) if player_info is not None else all_real)
    cL,cR=st.columns(2)
    pL=cL.selectbox(t["compare_left"],_co,index=0,key="t4L")
    pR=cR.selectbox(t["compare_right"],_co,index=min(1,len(_co)-1),key="t4R")
    m4=st.radio(t["metric"],[k for k in METRIC_LABELS if k not in("tilt_std","aa_std")],
                format_func=lambda x:METRIC_LABELS[x],horizontal=True,key="t4m")
    cL2,cR2=st.columns(2)
    with cL2:
        dfL=gpzd(pL,dff,lpz)
        make_heatmap(dfL,m4,pL,lpz,view_mode,dff) if dfL is not None and not dfL.empty else st.info(f"{t['no_data_for']} {pL}")
    with cR2:
        dfR=gpzd(pR,dff,lpz)
        make_heatmap(dfR,m4,pR,lpz,view_mode,dff) if dfR is not None and not dfR.empty else st.info(f"{t['no_data_for']} {pR}")

# ── Tab 5: Batter metrics ────────────────────────────────────────────
with tab5:
    st.subheader(t["tab_player_compare"])
    p5o=sel_disp if sel_disp else all_real
    sp5=st.selectbox(t["select_player"],p5o,key="t5p")
    cL5,cR5=st.columns(2)
    lm5=cL5.selectbox(t["left_metric"],list(METRIC_LABELS),format_func=lambda x:METRIC_LABELS[x],index=0,key="t5L")
    rm5=cR5.selectbox(t["right_metric"],list(METRIC_LABELS),format_func=lambda x:METRIC_LABELS[x],index=6,key="t5R")
    df5=gpzd(sp5,dff,lpz)
    if df5 is None or df5.empty: st.info(f"{t['no_data_for']} {sp5}")
    else:
        c1b,c2b=st.columns(2)
        with c1b: make_heatmap(df5,lm5,f"{sp5} – {METRIC_LABELS[lm5]}",lpz,view_mode,dff)
        with c2b: make_heatmap(df5,rm5,f"{sp5} – {METRIC_LABELS[rm5]}",lpz,view_mode,dff)

# ── Tab 6: Tilt Optimizer ────────────────────────────────────────────
with tab6:
    st.subheader(t["tab_tilt_sim"])
    st.markdown("Partial-dependence curve: all features fixed at batter averages, **tilt swept 8°→62°**. "
                "Optimal shrunk toward league optimum proportionally to sample size.")
    if model is None: st.error("Model could not be trained. Check CSV files.")
    else:
        c1,c2,c3=st.columns([2,1,1])
        sp6=c1.selectbox(t["select_player"],all_real,key="t6p")
        pg6=c2.selectbox("Pitch group",[t["all"]]+sorted(detail_full["pitch_group"].dropna().unique()),key="t6pg")
        z6=c3.selectbox("Zone",["All"]+[str(z) for z in range(1,15)],key="t6z")
        p6d=_fp(detail_fe,sp6)
        if pg6!=t["all"]: p6d=p6d[p6d["pitch_group"]==pg6]
        if z6!="All": p6d=p6d[p6d["zone"]==int(z6)]
        if p6d.empty: st.warning(f"{t['no_data_for']} {sp6} with current filters.")
        else:
            n6=int(p6d["swings"].sum()); aa6=_sm(p6d["avg_aa"]); spd6=_sm(p6d["avg_bat_speed"])
            len6=_sm(p6d["avg_swing_len"]); ct=_sm(p6d["avg_tilt"])
            ge6=float(p6d["group_enc"].mean()); ze6=float(p6d["zone_enc"].mean())
            tg6,pr6=predict_tilt_curve(aa6,spd6,len6,ze6,ge6)
            ro6,ox6=find_optimal(tg6,pr6); so6,cw6=shrink(ro6,LEAGUE_OPT_TILT,n6)
            tgl6,prl6=predict_tilt_curve(_lg_aa,_lg_spd,_lg_len,ze6,ge6)
            k1,k2,k3,k4,k5=st.columns(5)
            k1.metric(t["current_tilt"],f"{ct:.1f}°"); k2.metric(t["optimal_tilt"],f"{so6:.1f}°")
            dv=ct-so6
            k3.metric(t["tilt_delta"],f"{dv:+.1f}°",delta=f"{-dv:+.1f}° to optimal",delta_color="inverse")
            k4.metric(t["pred_xwoba"],f"{ox6:.3f}"); k5.metric(t["conf_weight"],f"{cw6:.0%}",help=f"n={n6}·K={SHRINKAGE_K}")
            if n6<30: st.warning(f"⚠️ Small sample (n={n6}). Heavily shrunk toward {LEAGUE_OPT_TILT:.1f}°.")
            fig6=go.Figure()
            if show_ci:
                lo6,hi6=approx_ci(aa6,spd6,len6,ze6,ge6,n6)
                fig6.add_trace(go.Scatter(x=np.concatenate([tg6,tg6[::-1]]),y=np.concatenate([hi6,lo6[::-1]]),
                    fill="toself",fillcolor="rgba(99,110,250,0.15)",line=dict(color="rgba(0,0,0,0)"),name="80% CI",hoverinfo="skip"))
            fig6.add_trace(go.Scatter(x=tg6,y=pr6,mode="lines",line=dict(color="#636EFA",width=2.8),name=sp6))
            fig6.add_trace(go.Scatter(x=tgl6,y=prl6,mode="lines",line=dict(color="#EF553B",width=1.8,dash="dot"),name="League avg features"))
            for xv,col,lbl,dash in[(ct,"#00CC96",f"Current ({ct:.1f}°)","dash"),
                                    (so6,"#AB63FA",f"Opt-shrunk ({so6:.1f}°)","solid"),
                                    (ro6,"#FFA15A",f"Opt-raw ({ro6:.1f}°)","dot"),
                                    (LEAGUE_OPT_TILT,"#EF553B",f"League ({LEAGUE_OPT_TILT:.1f}°)","dashdot")]:
                fig6.add_vline(x=xv,line=dict(color=col,width=1.8,dash=dash),
                               annotation_text=lbl,annotation_position="top",annotation_font_size=10)
            fig6.update_layout(title=f"Predicted xwOBA vs Tilt – {sp6}  [{pg6}/Zone {z6}]",
                xaxis_title="Swing Path Tilt (°)",yaxis_title="Predicted xwOBA",
                legend=dict(orientation="h",yanchor="bottom",y=-0.28),height=460,hovermode="x unified")
            st.plotly_chart(fig6,use_container_width=True)
            st.markdown("#### Interaction: Tilt × Attack Angle → Predicted xwOBA")
            tg2=np.linspace(TILT_MIN,TILT_MAX,30); aa2=np.linspace(-30,30,25)
            TT,AA=np.meshgrid(tg2,aa2); n2=TT.size
            X2=np.column_stack([TT.ravel(),AA.ravel(),np.full(n2,spd6),np.full(n2,len6),
                                 np.full(n2,ze6),np.full(n2,ge6),TT.ravel()*AA.ravel(),TT.ravel()*ge6])
            Z2=model.predict(X2).reshape(TT.shape)
            f2d=go.Figure(data=go.Heatmap(z=Z2,x=tg2.round(1),y=aa2.round(1),colorscale="RdYlGn",
                colorbar=dict(title="Pred. xwOBA"),
                hovertemplate="Tilt: %{x:.1f}°<br>AA: %{y:.1f}°<br>xwOBA: %{z:.3f}<extra></extra>"))
            f2d.add_trace(go.Scatter(x=[ct],y=[aa6],mode="markers",
                marker=dict(color="white",size=14,symbol="star",line=dict(color="black",width=2)),name=f"{sp6} (current)"))
            f2d.update_layout(xaxis_title="Tilt (°)",yaxis_title="Attack Angle (°)",height=420,
                title="xwOBA surface: Tilt × Attack Angle  (⭐ = current batter)")
            st.plotly_chart(f2d,use_container_width=True)
            ga=sorted(detail_full["pitch_group"].dropna().unique())
            if len(ga)>1:
                st.markdown("#### Optimal Tilt by Pitch Group")
                pgr=[]
                for pg in ga:
                    ge=_enc_g(pg); tgp,prp=predict_tilt_curve(aa6,spd6,len6,ze6,ge)
                    ro,rox=find_optimal(tgp,prp); pgr.append({"Pitch Group":pg,"Optimal Tilt (raw)":round(ro,1),"Pred. xwOBA":round(rox,3)})
                fpg=px.bar(pd.DataFrame(pgr),x="Pitch Group",y="Optimal Tilt (raw)",color="Pred. xwOBA",
                    color_continuous_scale="RdYlGn",text="Optimal Tilt (raw)",title=f"Optimal Tilt by Pitch Group – {sp6}")
                fpg.add_hline(y=ct,line_dash="dash",line_color="steelblue",annotation_text=f"Current ({ct:.1f}°)")
                fpg.update_traces(texttemplate="%{text:.1f}°",textposition="outside"); fpg.update_layout(height=360)
                st.plotly_chart(fpg,use_container_width=True)

# ── Tab 7: Tilt Rankings ─────────────────────────────────────────────
with tab7:
    st.subheader(t["tab_tilt_rankings"])
    st.markdown(f"Every batter ranked by gap to model-optimal tilt (shrunk toward **{LEAGUE_OPT_TILT:.1f}°**, K={SHRINKAGE_K}). "
                "🟥 too flat · 🟦 too steep · 🟩 near-optimal (±5°).")
    if opt_table.empty: st.warning("No optimization data for current filters.")
    else:
        sc=st.selectbox("Sort by",["Δ Tilt","Swings","Pred. xwOBA @ Opt.","Current Tilt","Optimal Tilt"],key="t7s")
        asc=st.checkbox("Ascending",value=False,key="t7a")
        mc=st.slider("Min. confidence",0.0,1.0,0.0,0.05)
        tb=(opt_table[opt_table["Confidence"]>=mc].sort_values(sc,ascending=asc,na_position="last"))
        def _cd(v):
            if pd.isna(v): return ""
            if v>5: return "background-color:#ffe0e0"
            if v<-5: return "background-color:#e0e8ff"
            return "background-color:#e0ffe8"
        fmt={"Current Tilt":"{:.1f}°","Optimal Tilt (raw)":"{:.1f}°","Optimal Tilt":"{:.1f}°",
             "Δ Tilt":"{:+.1f}°","Pred. xwOBA @ Opt.":"{:.3f}","Current xwOBA":"{:.3f}","Confidence":"{:.0%}"}
        # pandas ≥ 2.1: Styler.applymap → Styler.map
        _styler = tb.style.map(_cd, subset=["Δ Tilt"]) if hasattr(tb.style, "map") else tb.style.applymap(_cd, subset=["Δ Tilt"])
        st.dataframe(_styler.format(fmt), use_container_width=True, hide_index=True)
        st.markdown("#### Δ Tilt – Top 30 by |Δ|")
        t30=tb.assign(_a=tb["Δ Tilt"].abs()).nlargest(30,"_a")
        f7=px.bar(_clean(t30,"Δ Tilt","Batter","Current Tilt","Optimal Tilt","Confidence","Swings"),
            x="Δ Tilt",y="Batter",orientation="h",color="Δ Tilt",color_continuous_scale="RdBu",
            color_continuous_midpoint=0,text="Δ Tilt",
            hover_data=["Current Tilt","Optimal Tilt","Confidence","Swings"],
            title="Positive = too flat  |  Negative = too steep")
        f7.update_traces(texttemplate="%{text:+.1f}°",textposition="outside")
        f7.add_vline(x=0,line_width=2,line_color="black")
        f7.update_layout(yaxis=dict(autorange="reversed"),height=max(400,22*len(t30)),coloraxis_showscale=False)
        st.plotly_chart(f7,use_container_width=True)
        if "Current xwOBA" in tb.columns and tb["Current xwOBA"].notna().any():
            st.markdown("#### Current xwOBA vs Δ Tilt")
            sd=tb.dropna(subset=["Current xwOBA"])
            f7b=_safe_scatter(sd,x="Δ Tilt",y="Current xwOBA",color="Confidence",size="Swings",
                hover_data=["Batter","Optimal Tilt","Current Tilt"],trendline="lowess",
                title="Do batters nearer their optimal tilt perform better?")
            if f7b:
                f7b.add_vline(x=0,line_dash="dash",line_color="grey"); f7b.update_layout(height=430)
                st.plotly_chart(f7b,use_container_width=True)
        if "Gradient Boosting" in model_type and hasattr(model,"feature_importances_"):
            st.markdown("#### Model Feature Importance")
            imp=pd.Series(model.feature_importances_,index=FEATURE_COLS).sort_values()
            fi=px.bar(imp.reset_index(),x=0,y="index",orientation="h",
                labels={"0":"Importance","index":"Feature"},color=0,color_continuous_scale="Blues",
                title="Gradient Boosting – Feature Importance")
            fi.update_layout(height=360,coloraxis_showscale=False)
            st.plotly_chart(fi,use_container_width=True)


# ── Tab 8: Glossary ───────────────────────────────────────────────────
with tab8:
    st.subheader(t["tab_glossary"])
    st.markdown(
        "Definitions of the main metrics used throughout **Swing Intelligence**. "
        "Values are based on MLB Statcast bat-tracking and performance data."
    )

    glossary_rows = [
        ("Swing Path Tilt (Tilt)",
         "The angle of the bat's swing path at the relevant point of the swing. "
         "It describes how steep or flat the bat is moving through the hitting zone. "
         "Tilt is the primary swing-path metric optimized by the model in this dashboard."),
        ("Attack Angle (AA)",
         "The vertical angle of the bat's path through the hitting zone. "
         "Positive values indicate an upward path; negative values indicate a downward path."),
        ("Bat Speed",
         "The speed of the bat at the relevant point of the swing, measured in mph. "
         "It describes how quickly the bat is moving through the hitting zone."),
        ("Swing Length",
         "The length of the bat's path during the swing, measured in feet. "
         "It provides context for how much distance the bat travels during the swing."),
        ("Swings",
         "The number of swings represented by the observations after the selected filters. "
         "Larger samples generally provide more stable estimates."),
        ("Batting Average (AVG)",
         "Hits divided by official at-bats. A traditional measure of how often a batter records a hit."),
        ("xwOBA",
         "Expected weighted On-Base Average. A measure of expected offensive value based on the quality and outcomes of batted balls."),
        ("Exit Velocity",
         "The speed of the ball as it leaves the bat, measured in mph. "
         "It is commonly used as an indicator of contact quality."),
        ("Launch Angle",
         "The vertical angle of the batted ball's trajectory immediately after contact, measured in degrees."),
        ("Tilt STD",
         "Standard deviation of Swing Path Tilt within the selected zone. "
         "It describes how much swing-path tilt varies across swings."),
        ("Attack Angle STD (AA STD)",
         "Standard deviation of Attack Angle within the selected zone. "
         "It describes how much attack angle varies across swings."),
        ("Δ Tilt",
         "Current average Tilt minus the model's optimal Tilt after shrinkage. "
         "Positive values indicate the current swing path is flatter than the model optimum; negative values indicate it is steeper."),
        ("Optimal Tilt",
         "The Tilt value associated with the highest model-predicted xwOBA while the other model inputs are held at the selected batter/context values. "
         "The displayed value is shrunk toward the league optimum when the sample is limited."),
        ("Predicted xwOBA @ Optimal",
         "The model-predicted xwOBA at the identified optimal Tilt. "
         "This is a model estimate, not a guarantee of future performance."),
        ("Confidence",
         "The shrinkage weight used for the optimal Tilt estimate. "
         "Higher values mean the player's sample contributes more to the displayed optimum; lower values pull the estimate more strongly toward the league optimum."),
        ("Percentile",
         "A value's position relative to the reference distribution, expressed from 0 to 100. "
         "For example, the 90th percentile means the value is higher than approximately 90% of the reference observations."),
        ("Shrunk",
         "A stabilized estimate that combines the observed value with the league average. "
         "The amount of shrinkage depends on sample size, reducing the influence of noisy small samples."),
    ]

    glossary_df = pd.DataFrame(glossary_rows, columns=["Metric", "Definition"])
    st.dataframe(
        glossary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Metric": st.column_config.TextColumn("Metric", width="medium"),
            "Definition": st.column_config.TextColumn("Definition", width="large"),
        },
    )

    st.markdown("---")
    st.caption("Source: Baseball Savant (MLB Statcast) · Metric definitions are provided for dashboard interpretation.")

