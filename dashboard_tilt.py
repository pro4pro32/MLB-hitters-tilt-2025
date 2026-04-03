import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

# ========================== TRANSLATIONS ==========================
TEXTS = {
    "pl": {
        "title": "MLB 2025 - Swing Path Tilt & Attack Angle Dashboard",
        "sidebar_players": "Batters for heatmaps / comparisons / tables",
        "sidebar_min_swings": "Minimum swings (tables only)",
        "sidebar_pitch_group": "Pitch group (whole dashboard)",
        "all": "All",
        "league_avg": "League Average",
        "tab_summary": "Summary",
        "tab_groups": "Group Comparison",
        "tab_heatmaps": "Heatmaps + Table",
        "tab_side_by_side": "Side-by-Side Comparison",
        "tab_player_compare": "Batter Metrics Comparison",
        "selected_players": "**Selected Batters**",
        "all_players_after_filters": "**All Batters After Filters**",
        "no_real_player": "Select at least one real batter",
        "no_data_after_filters": "No data after filters",
        "metric": "Metric",
        "heatmap_metric": "Metric",
        "compare_left": "Left Batter",
        "compare_right": "Right Batter",
        "no_data": "No data",
        "no_data_for": "No data for",
        "detailed_table": "Detailed Table",
        "select_player_tab5": "Select Batter",
        "left_metric_tab5": "Left Side Metric",
        "right_metric_tab5": "Right Side Metric",
    },
    "en": {
        "title": "MLB 2025 - Swing Path Tilt & Attack Angle Dashboard",
        "sidebar_players": "Batters for heatmaps / comparisons / tables",
        "sidebar_min_swings": "Minimum swings (tables only)",
        "sidebar_pitch_group": "Pitch group (whole dashboard)",
        "all": "All",
        "league_avg": "League Average",
        "tab_summary": "Summary",
        "tab_groups": "Group Comparison",
        "tab_heatmaps": "Heatmaps + Table",
        "tab_side_by_side": "Side-by-Side Comparison",
        "tab_player_compare": "Batter Metrics Comparison",
        "selected_players": "**Selected Batters**",
        "all_players_after_filters": "**All Batters After Filters**",
        "no_real_player": "Select at least one real batter",
        "no_data_after_filters": "No data after filters",
        "metric": "Metric",
        "heatmap_metric": "Metric",
        "compare_left": "Left Batter",
        "compare_right": "Right Batter",
        "no_data": "No data",
        "no_data_for": "No data for",
        "detailed_table": "Detailed Table",
        "select_player_tab5": "Select Batter",
        "left_metric_tab5": "Left Side Metric",
        "right_metric_tab5": "Right Side Metric",
    }
}

if "lang" not in st.session_state:
    st.session_state.lang = "en"

# ========================== DATA LOADING ==========================
@st.cache_data(ttl=3600)
def load_data():
    players = pd.read_parquet("players_summary_2025.parquet")
    detail = pd.read_parquet("detail_zone_pitchgroup_2025.parquet")
    detail = detail[detail["batter_name"].notna() & ~detail["batter_name"].str.contains(r" pitcher| P$", case=False, na=False)]
    return players, detail

players, detail_full = load_data()

# ========================== PLAYER HANDLING ==========================
all_real = sorted(detail_full["batter_name"].dropna().unique())

# ========================== CONFIG ==========================
st.set_page_config(page_title="MLB 2025 Bat Tracking", layout="wide", initial_sidebar_state="collapsed")
st.title(TEXTS[st.session_state.lang]["title"])

with st.sidebar:
    lang_display = st.selectbox("Language", options=["Polski", "English"], index=1)
    LANG_MAP = {"Polski": "pl", "English": "en"}
    if LANG_MAP[lang_display] != st.session_state.lang:
        st.session_state.lang = LANG_MAP[lang_display]
        st.rerun()

t = TEXTS[st.session_state.lang]

selected_players_multi = st.multiselect(
    t["sidebar_players"], 
    options=[t["league_avg"]] + ["─"*25] + all_real,
    default=[all_real[0]] if all_real else [], 
    max_selections=8
)
selected_players_multi = [p for p in selected_players_multi if "─" not in p]

min_swings = st.slider(t["sidebar_min_swings"], 0, 300, 0, 10)

pitch_groups = [t["all"]] + sorted(detail_full["pitch_group"].dropna().unique().tolist())
selected_pitch = st.selectbox(t["sidebar_pitch_group"], pitch_groups)

# ========================== FILTERED DATA ==========================
df_filtered = detail_full.copy()
if selected_pitch != t["all"]:
    df_filtered = df_filtered[df_filtered["pitch_group"] == selected_pitch]

league_per_zone = df_filtered.groupby("zone", as_index=False).agg({
    "avg_tilt": "mean", "std_tilt": "mean", "avg_aa": "mean",
    "avg_bat_speed": "mean", "avg_swing_len": "mean", "swings": "sum"
}).round(3)
league_per_zone["batter_name"] = t["league_avg"]

detail_tables = df_filtered[df_filtered["swings"] >= min_swings].copy()

# ========================== HEATMAP ==========================
HEATMAP_RANGES = {
    "avg_tilt": (8, 60), "std_tilt": (0, 20),
    "avg_aa": (-35, 35), "avg_bat_speed": (55, 88),
    "avg_swing_len": (4.5, 9.5), "swings": (0, 400)
}

def make_heatmap(df_p, metric, title):
    if df_p.empty:
        st.warning(t["no_data"])
        return

    if metric == "swings":
        pivot = df_p.groupby('zone')['swings'].sum().round(0)
    elif metric == "std_tilt":
        pivot = df_p.groupby('zone')['avg_tilt'].std(ddof=1).round(1)
    else:
        pivot = df_p.groupby('zone')[metric].mean().round(1)

    vmin, vmax = HEATMAP_RANGES.get(metric, (0, 100))
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    fig, ax = plt.subplots(figsize=(8.6, 8.6))
    border = 0.85
    main_size = 3.3
    main_x = border
    main_y = border
    top_y = main_y + main_size
    right_x = main_x + main_size
    half = main_size / 2
    split_y = 2.5

    # Main 3x3 zones
    for i in range(3):
        for j in range(3):
            zone = i * 3 + j + 1
            val = pivot.get(zone, np.nan)
            x = main_x + j * (main_size / 3)
            y = main_y + (2 - i) * (main_size / 3)
            color = cmap((val - vmin) / (vmax - vmin)) if not pd.isna(val) else (0.96, 0.96, 0.96)
            ax.add_patch(plt.Rectangle((x, y), main_size / 3, main_size / 3, facecolor=color, edgecolor='black', linewidth=2.8))
            text = str(zone) if pd.isna(val) else f"{zone}\n{val:.1f}"
            ax.text(x + (main_size / 6), y + (main_size / 6), text, ha='center', va='center', fontsize=12, fontweight='bold')

    # L-shaped zones 11-14
    for z, verts in [(11, [(0, split_y), (border, split_y), (border, top_y), (main_x, top_y), (main_x + half, top_y), (main_x + half, 5), (0, 5), (0, split_y)]),
                     (12, [(right_x, split_y), (right_x, top_y), (main_x + half, top_y), (main_x + half, 5), (5, 5), (5, split_y), (right_x, split_y)]),
                     (13, [(0, split_y), (border, split_y), (border, main_y), (main_x, main_y), (main_x + half, main_y), (main_x + half, 0), (0, 0), (0, split_y)]),
                     (14, [(right_x, split_y), (right_x, main_y), (main_x + half, main_y), (main_x + half, 0), (5, 0), (5, split_y), (right_x, split_y)])]:
        val = pivot.get(z, np.nan)
        color = cmap((val - vmin) / (vmax - vmin)) if not pd.isna(val) else (0.96, 0.96, 0.96)
        ax.add_patch(PathPatch(MPath(verts), facecolor=color, edgecolor='black', linewidth=2.8))

    ax.add_patch(plt.Rectangle((main_x, main_y), main_size, main_size, fill=False, edgecolor='red', linewidth=4.2))

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=17, pad=25)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, ax=ax, shrink=0.78, pad=0.04)
    cbar.set_label(metric.replace("_", " ").upper(), fontsize=12)

    st.pyplot(fig, use_container_width=True)

# ========================== TABS ==========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    t["tab_summary"], t["tab_groups"], t["tab_heatmaps"],
    t["tab_side_by_side"], t["tab_player_compare"]
])

with tab1:
    st.subheader(t["tab_summary"])
    st.info("Summary tab - ready")

with tab3:
    st.subheader(t["tab_heatmaps"])
    metric_tab3 = st.radio(t["heatmap_metric"], 
        ["avg_tilt", "std_tilt", "avg_aa", "avg_bat_speed", "avg_swing_len", "swings"], horizontal=True)

    for p in selected_players_multi:
        if p == t["league_avg"]:
            df_p = league_per_zone.copy()
        else:
            df_p = df_filtered[df_filtered["batter_name"] == p]
        make_heatmap(df_p, metric_tab3, p)

st.caption("MLB 2025 Swing Path Dashboard")