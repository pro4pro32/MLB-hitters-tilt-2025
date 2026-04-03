import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
import os

# ========================== TRANSLATIONS ==========================
TEXTS = {
    "pl": {
        "title": "MLB 2025 - Swing Path Tilt & Attack Angle Dashboard",
        "sidebar_players": "Batters for heatmaps / comparisons / tables",
        "sidebar_min_swings": "Minimum swings (tables only)",
        "sidebar_pitch_group": "Pitch group (whole dashboard)",
        "sidebar_pitch_type": "Pitch type (whole dashboard)",
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
        "no_data_after_filters": "No data after pitch/type filters",
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
        "sidebar_pitch_type": "Pitch type (whole dashboard)",
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
        "no_data_after_filters": "No data after pitch/type filters",
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
    try:
        players = pd.read_parquet("players_summary_2025.parquet")
        detail = pd.read_parquet("detail_zone_pitchgroup_2025.parquet")
        detail = detail[detail["batter_name"].notna() & 
                       ~detail["batter_name"].str.contains(r" pitcher| P$", case=False, na=False)]
        return players, detail
    except Exception as e:
        st.error("❌ Nie można wczytać plików .parquet")
        st.error(str(e))
        st.stop()

players, detail_full = load_data()

# ========================== PLAYER HANDLING ==========================
use_id = False
id_col = None
for possible_id in ['batter_id', 'batter', 'mlb_id', 'player_id', 'id']:
    if possible_id in players.columns and possible_id in detail_full.columns:
        id_col = possible_id
        use_id = True
        break

if use_id:
    player_info = players[[id_col, 'batter_name']].drop_duplicates()
    name_counts = player_info['batter_name'].value_counts()
    dupe_names = name_counts[name_counts > 1].index.tolist()
    player_info['display_name'] = player_info.apply(
        lambda row: f"{row['batter_name']} (ID:{int(row[id_col])})" if row['batter_name'] in dupe_names else row['batter_name'],
        axis=1
    )
    display_to_id = dict(zip(player_info['display_name'], player_info[id_col]))
    id_to_display = dict(zip(player_info[id_col], player_info['display_name']))
    all_real = sorted(player_info['display_name'])
else:
    all_real = sorted(players["batter_name"].dropna().unique())

# ========================== CONFIG ==========================
st.set_page_config(page_title="MLB 2025 Bat Tracking", layout="wide", initial_sidebar_state="collapsed")

with st.sidebar:
    lang_display = st.selectbox("Language", options=["Polski", "English"], index=1)
    LANG_MAP = {"Polski": "pl", "English": "en"}
    if LANG_MAP[lang_display] != st.session_state.lang:
        st.session_state.lang = LANG_MAP[lang_display]
        st.rerun()

t = TEXTS[st.session_state.lang]

player_options = [t["league_avg"]] + ["─"*25] + all_real
selected_players_multi = st.multiselect(
    t["sidebar_players"], options=player_options,
    default=[all_real[0]] if all_real else [], max_selections=8
)
selected_players_multi = [p for p in selected_players_multi if "─" not in p]

min_swings = st.slider(t["sidebar_min_swings"], 0, 300, 0, 10)

pitch_groups = [t["all"]] + sorted(detail_full["pitch_group"].dropna().unique().tolist())
selected_pitch = st.selectbox(t["sidebar_pitch_group"], pitch_groups)

if selected_pitch == t["all"]:
    pitch_types_avail = sorted(detail_full["pitch_type"].dropna().unique().tolist())
else:
    pitch_types_avail = sorted(detail_full[detail_full["pitch_group"] == selected_pitch]["pitch_type"].dropna().unique().tolist())

pitch_types = [t["all"]] + pitch_types_avail
selected_type = st.selectbox(t["sidebar_pitch_type"], pitch_types)

# ========================== FILTERED DATA ==========================
df_filtered = detail_full.copy()
if selected_pitch != t["all"]:
    df_filtered = df_filtered[df_filtered["pitch_group"] == selected_pitch]
if selected_type != t["all"]:
    df_filtered = df_filtered[df_filtered["pitch_type"] == selected_type]

league_agg_dict = {
    "avg_tilt": "mean", "avg_aa": "mean", "avg_bat_speed": "mean", "avg_swing_len": "mean", "swings": "sum",
    "batting_avg": "mean", "xwoba": "mean", "avg_exit_velocity": "mean", "avg_launch_angle": "mean"
}

league_per_zone = df_filtered.groupby("zone", as_index=False).agg(league_agg_dict).round(3)
league_per_zone["batter_name"] = t["league_avg"]

detail_tables = df_filtered[df_filtered["swings"] >= min_swings].copy()

selected_display = [p for p in selected_players_multi if p != t["league_avg"]]

# ========================== HEATMAP ==========================
HEATMAP_RANGES = {
    "avg_tilt": (8, 60), "tilt_std": (0, 20), "delta_tilt": (-20, 20),
    "avg_aa": (-35, 35), "aa_std": (0, 20),
    "avg_bat_speed": (55, 88), "avg_swing_len": (4.5, 9.5), "swings": (0, 400),
    "batting_avg": (0.150, 0.400), "xwoba": (0.200, 0.600),
    "avg_exit_velocity": (75, 105), "avg_launch_angle": (-15, 45),
}

def get_player_zone_df(p_sel):
    if p_sel == t["league_avg"]:
        return league_per_zone.copy()
    if use_id:
        pid = display_to_id.get(p_sel)
        return df_filtered[df_filtered[id_col] == pid].groupby("zone", as_index=False).mean(numeric_only=True).round(3)
    return df_filtered[df_filtered["batter_name"] == p_sel].groupby("zone", as_index=False).mean(numeric_only=True).round(3)

def make_heatmap(df_p, metric, title, league_df=None):
    if df_p.empty:
        st.warning(t["no_data"])
        return

    if metric == "swings":
        pivot = df_p.groupby('zone')['swings'].sum().round(0)
    elif metric == "tilt_std":
        pivot = df_p.groupby('zone')['avg_tilt'].std(ddof=1).round(1)
    elif metric == "aa_std":
        pivot = df_p.groupby('zone')['avg_aa'].std(ddof=1).round(1)
    elif metric == "delta_tilt":
        player_mean = df_p.groupby('zone')['avg_tilt'].mean().round(2)
        league_tilt = league_df.set_index('zone')['avg_tilt'] if league_df is not None else pd.Series()
        pivot = (player_mean - league_tilt.reindex(player_mean.index, fill_value=np.nan)).round(2)
    elif metric in ["batting_avg", "xwoba"]:
        pivot = df_p.groupby('zone')[metric].mean().round(3)
    else:
        pivot = df_p.groupby('zone')[metric].mean().round(1)

    vmin, vmax = HEATMAP_RANGES.get(metric, (0, 100))
    cmap = sns.color_palette("RdBu_r" if metric == "delta_tilt" else "YlOrRd", as_cmap=True)

    fig, ax = plt.subplots(figsize=(8.6, 8.6))
    border = 0.85
    main_size = 3.3
    main_x = border
    main_y = border
    top_y = main_y + main_size
    right_x = main_x + main_size
    half = main_size / 2
    split_y = 2.5

    for i in range(3):
        for j in range(3):
            zone = i * 3 + j + 1
            val = pivot.get(zone, np.nan)
            x = main_x + j * (main_size / 3)
            y = main_y + (2 - i) * (main_size / 3)
            color = cmap((val - vmin) / (vmax - vmin)) if not pd.isna(val) else (0.96, 0.96, 0.96)
            ax.add_patch(plt.Rectangle((x, y), main_size / 3, main_size / 3, facecolor=color, edgecolor='black', linewidth=2.8))
            text = str(zone) if pd.isna(val) else (f"{zone}\n{int(val)}" if metric == "swings" else f"{zone}\n{val:.3f}" if metric in ["batting_avg", "xwoba"] else f"{zone}\n{val:.1f}")
            ax.text(x + (main_size / 6), y + (main_size / 6), text, ha='center', va='center', fontsize=12, fontweight='bold')

    for z, verts in [(11, [(0, split_y), (border, split_y), (border, top_y), (main_x, top_y), (main_x + half, top_y), (main_x + half, 5), (0, 5), (0, split_y)]),
                     (12, [(right_x, split_y), (right_x, top_y), (main_x + half, top_y), (main_x + half, 5), (5, 5), (5, split_y), (right_x, split_y)]),
                     (13, [(0, split_y), (border, split_y), (border, main_y), (main_x, main_y), (main_x + half, main_y), (main_x + half, 0), (0, 0), (0, split_y)]),
                     (14, [(right_x, split_y), (right_x, main_y), (main_x + half, main_y), (main_x + half, 0), (5, 0), (5, split_y), (right_x, split_y)])]:
        val = pivot.get(z, np.nan)
        color = cmap((val - vmin) / (vmax - vmin)) if not pd.isna(val) else (0.96, 0.96, 0.96)
        ax.add_patch(PathPatch(MPath(verts), facecolor=color, edgecolor='black', linewidth=2.8))

    ax.add_patch(plt.Rectangle((main_x, main_y), main_size, main_size, fill=False, edgecolor='red', linewidth=4.2))

    offset = border * 0.38
    for pos, z in [((offset, 5 - offset), 11), ((5 - offset, 5 - offset), 12), ((offset, offset), 13), ((5 - offset, offset), 14)]:
        val = pivot.get(z, np.nan)
        txt = f"{z}\n{val:.3f}" if metric in ["batting_avg","xwoba"] and not pd.isna(val) else (f"{z}\n{val:.1f}" if not pd.isna(val) else str(z))
        ax.text(*pos, txt, ha='center', va='center', fontsize=12, fontweight='bold')

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=17, pad=25)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, ax=ax, shrink=0.78, pad=0.04)
    label_map = {
        "avg_tilt": "TILT", "tilt_std": "TILT STD", "delta_tilt": "TILT - LEAGUE AVG",
        "avg_aa": "ATTACK ANGLE", "aa_std": "AA STD", "avg_bat_speed": "BAT SPEED",
        "avg_swing_len": "SWING LENGTH", "swings": "SWINGS",
        "batting_avg": "BATTING AVG", "xwoba": "xWOBA",
        "avg_exit_velocity": "EXIT VELOCITY", "avg_launch_angle": "LAUNCH ANGLE"
    }
    cbar.set_label(label_map.get(metric, metric.upper()), fontsize=12)

    st.pyplot(fig, use_container_width=True)

# ========================== TABS ==========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    t["tab_summary"], t["tab_groups"], t["tab_heatmaps"],
    t["tab_side_by_side"], t["tab_player_compare"]
])

# TAB 1
with tab1:
    st.subheader(t["tab_summary"])
    st.info("Summary tab loaded")

# TAB 3 (Heatmaps)
with tab3:
    st.subheader(t["tab_heatmaps"])
    metric_tab3 = st.radio(t["heatmap_metric"], 
        ["avg_tilt", "tilt_std", "delta_tilt", "avg_aa", "aa_std", "avg_bat_speed", "avg_swing_len",
         "batting_avg", "xwoba", "avg_exit_velocity", "avg_launch_angle", "swings"], horizontal=True)
    
    for p in selected_players_multi:
        df_p = get_player_zone_df(p)
        make_heatmap(df_p, metric_tab3, p, league_per_zone)

# Pozostałe taby (2,4,5) możesz rozwinąć później

st.caption("MLB 2025 Dashboard • Streamlit Cloud")