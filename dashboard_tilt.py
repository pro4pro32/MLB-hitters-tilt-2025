import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

# ========================= CONFIGURATION =========================
st.set_page_config(
    page_title="MLB Bat Tracking 2025",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================= TRANSLATIONS =========================
TEXTS = {
    "en": {
        "title": "MLB 2025 – Swing Path Tilt & Attack Angle Dashboard",
        "sidebar_players": "Select Batters",
        "sidebar_min_swings": "Minimum Swings (for tables)",
        "sidebar_pitch_group": "Pitch Group",
        "sidebar_pitch_type": "Pitch Type",
        "all": "All",
        "league_avg": "League Average",
        "tab_summary": "Summary",
        "tab_groups": "Group Comparison",
        "tab_heatmaps": "Heatmaps",
        "tab_side_by_side": "Side-by-Side Comparison",
        "tab_player_compare": "Single Batter Analysis",
        "selected_players": "Selected Batters",
        "all_players": "All Batters",
        "no_data": "No data available",
        "no_data_for": "No data for",
        "metric": "Metric",
        "heatmap_metric": "Display Metric",
        "compare_left": "Left Batter",
        "compare_right": "Right Batter",
        "select_batter": "Select Batter",
        "left_metric": "Left Metric",
        "right_metric": "Right Metric",
    }
}

t = TEXTS["en"]

# ========================= OPTIMIZED DATA LOADING =========================
@st.cache_data(show_spinner="Loading data...", ttl=1800, persist="disk")
def load_data():
    players = pd.read_parquet("players_summary_2025.parquet")

    detail = pd.read_parquet(
        "detail_zone_pitchgroup_2025.parquet",
        columns=[
            "batter_name", "batter", "pitch_group", "pitch_type", "zone",
            "swings", "avg_tilt", "avg_aa", "avg_bat_speed", "avg_swing_len",
            "batting_avg", "xwoba", "avg_exit_velocity", "avg_launch_angle"
        ]
    )

    # Remove pitchers
    detail = detail[
        detail["batter_name"].notna() &
        ~detail["batter_name"].str.contains(r" pitcher| P$", case=False, na=False)
    ].copy()

    return players, detail


players, detail_full = load_data()

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("Filters")

    all_batters = sorted(detail_full["batter_name"].dropna().unique())

    selected_players = st.multiselect(
        t["sidebar_players"],
        options=[t["league_avg"]] + all_batters,
        default=[all_batters[0]] if all_batters else [],
        max_selections=8
    )

    min_swings = st.slider(t["sidebar_min_swings"], 0, 300, 10)

    pitch_groups = [t["all"]] + sorted(detail_full["pitch_group"].dropna().unique().tolist())
    selected_pitch_group = st.selectbox(t["sidebar_pitch_group"], pitch_groups)

    if selected_pitch_group == t["all"]:
        pitch_types_list = sorted(detail_full["pitch_type"].dropna().unique().tolist())
    else:
        pitch_types_list = sorted(
            detail_full[detail_full["pitch_group"] == selected_pitch_group]["pitch_type"].dropna().unique().tolist()
        )

    pitch_types = [t["all"]] + pitch_types_list
    selected_pitch_type = st.selectbox(t["sidebar_pitch_type"], pitch_types)

# ========================= DATA FILTERING =========================
df_filtered = detail_full.copy()
if selected_pitch_group != t["all"]:
    df_filtered = df_filtered[df_filtered["pitch_group"] == selected_pitch_group]
if selected_pitch_type != t["all"]:
    df_filtered = df_filtered[df_filtered["pitch_type"] == selected_pitch_type]

league_per_zone = df_filtered.groupby("zone", as_index=False).agg({
    "avg_tilt": "mean", "avg_aa": "mean", "avg_bat_speed": "mean", "avg_swing_len": "mean",
    "swings": "sum", "batting_avg": "mean", "xwoba": "mean",
    "avg_exit_velocity": "mean", "avg_launch_angle": "mean"
}).round(3)
league_per_zone["batter_name"] = t["league_avg"]

selected_real = [p for p in selected_players if p != t["league_avg"]]

# ========================= HEATMAP FUNCTION =========================
def make_heatmap(df_p, metric, title):
    if df_p is None or df_p.empty:
        st.warning("No data")
        return

    if metric == "swings":
        pivot = df_p.groupby('zone')['swings'].sum().round(0)
    elif metric == "tilt_std":
        pivot = df_p.groupby('zone')['avg_tilt'].std(ddof=1).round(1)
    elif metric == "aa_std":
        pivot = df_p.groupby('zone')['avg_aa'].std(ddof=1).round(1)
    elif metric == "delta_tilt":
        player_mean = df_p.groupby('zone')['avg_tilt'].mean().round(2)
        league_tilt = league_per_zone.set_index('zone')['avg_tilt']
        pivot = (player_mean - league_tilt.reindex(player_mean.index, fill_value=np.nan)).round(2)
    elif metric in ["batting_avg", "xwoba"]:
        pivot = df_p.groupby('zone')[metric].mean().round(3)
    else:
        pivot = df_p.groupby('zone')[metric].mean().round(1)

    vmin, vmax = {
        "avg_tilt": (8, 60), "tilt_std": (0, 20), "delta_tilt": (-20, 20),
        "avg_aa": (-35, 35), "aa_std": (0, 20),
        "avg_bat_speed": (55, 88), "avg_swing_len": (4.5, 9.5), "swings": (0, 400),
        "batting_avg": (0.150, 0.400), "xwoba": (0.200, 0.600),
        "avg_exit_velocity": (75, 105), "avg_launch_angle": (-15, 45)
    }.get(metric, (0, 100))

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

    # L-shaped zones 11-14
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
    ax.set_title(title, fontsize=16, pad=20)

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

# ========================= TABS =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    t["tab_summary"], t["tab_groups"], t["tab_heatmaps"],
    t["tab_side_by_side"], t["tab_player_compare"]
])

# TAB 1 – Summary
with tab1:
    st.subheader(t["tab_summary"])
    if selected_players:
        st.markdown(f"**{t['selected_players']}**")
        # Tutaj możesz dodać tabele wybranych pałkarzy (na razie puste – dodaj jeśli chcesz)

    st.subheader(t["all_players"])
    # Tutaj możesz dodać tabelę wszystkich pałkarzy

# TAB 5 – Single Batter Comparison (główna funkcjonalność, którą chciałeś)
with tab5:
    st.subheader(t["tab_player_compare"])

    selected_batter = st.selectbox(
        t["select_batter"],
        options=selected_real if selected_real else all_batters
    )

    metric_options = {
        "avg_tilt": "Average Tilt",
        "avg_aa": "Attack Angle",
        "avg_bat_speed": "Bat Speed",
        "avg_swing_len": "Swing Length",
        "batting_avg": "Batting Average",
        "xwoba": "xWOBA",
        "avg_exit_velocity": "Exit Velocity",
        "avg_launch_angle": "Launch Angle"
    }

    col1, col2 = st.columns(2)
    with col1:
        left_metric = st.selectbox(
            t["left_metric"],
            options=list(metric_options.keys()),
            format_func=lambda x: metric_options[x],
            index=0
        )
    with col2:
        right_metric = st.selectbox(
            t["right_metric"],
            options=list(metric_options.keys()),
            format_func=lambda x: metric_options[x],
            index=6
        )

    df_batter = df_filtered[df_filtered["batter_name"] == selected_batter].groupby("zone").mean(numeric_only=True).reset_index()

    if not df_batter.empty:
        c1, c2 = st.columns(2)
        with c1:
            make_heatmap(df_batter, left_metric, f"{selected_batter} — {metric_options[left_metric]}")
        with c2:
            make_heatmap(df_batter, right_metric, f"{selected_batter} — {metric_options[right_metric]}")
    else:
        st.info(f"{t['no_data_for']} {selected_batter}")

st.caption("MLB Bat Tracking Dashboard • 2025 Season")