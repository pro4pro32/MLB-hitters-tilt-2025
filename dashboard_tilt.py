import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

# ────────────────────────────────────────────────
# TRANSLATIONS
# ────────────────────────────────────────────────
TEXTS = {
    "pl": {
        "title": "MLB 2025 – Swing Path Tilt & Attack Angle Dashboard",
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
        "tab3_all_zones": "Zone filter (all)",
        "all_zones": "All zones",
        "no_data": "No data",
        "no_data_for": "No data for",
        "language_label": "Language",
        "detailed_table": "Detailed Table",
        "select_player_tab5": "Select Batter",
        "left_metric_tab5": "Left Side Metric",
        "right_metric_tab5": "Right Side Metric",
    },
    "en": {
        "title": "MLB 2025 – Swing Path Tilt & Attack Angle Dashboard",
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
        "tab3_all_zones": "Zone filter (all)",
        "all_zones": "All zones",
        "no_data": "No data",
        "no_data_for": "No data for",
        "language_label": "Language",
        "detailed_table": "Detailed Table",
        "select_player_tab5": "Select Batter",
        "left_metric_tab5": "Left Side Metric",
        "right_metric_tab5": "Right Side Metric",
    }
}

if "lang" not in st.session_state:
    st.session_state.lang = "en"

with st.sidebar:
    lang_display = st.selectbox("Language", options=["Polski", "English"], index=1)
    LANG_MAP = {"Polski": "pl", "English": "en"}
    new_lang = LANG_MAP[lang_display]
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

t = TEXTS[st.session_state.lang]

# =====================================================================
# DATA LOADING
# =====================================================================
DATA_DIR = "."

@st.cache_data
def load_data():
    players = pd.read_parquet(Path(DATA_DIR) / "players_summary_2025.parquet")
    detail = pd.read_parquet(Path(DATA_DIR) / "detail_zone_pitchgroup_2025.parquet")
    detail = detail[detail["batter_name"].notna() & ~detail["batter_name"].str.contains(r" pitcher| P$", case=False, na=False)]
    return players, detail

players, detail_full = load_data()

# Duplicate name handling
use_id = False
id_col = None
player_info = None
display_to_id = {}
id_to_display = {}

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

# =====================================================================
# CONFIG & FILTERS
# =====================================================================
st.set_page_config(page_title="MLB Bat Tracking 2025", layout="wide")
st.title(t["title"])

with st.sidebar:
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

df_filtered = detail_full.copy()
if selected_pitch != t["all"]:
    df_filtered = df_filtered[df_filtered["pitch_group"] == selected_pitch]
if selected_type != t["all"]:
    df_filtered = df_filtered[df_filtered["pitch_type"] == selected_type]

league_agg_dict = {
    "avg_tilt": "mean", "avg_aa": "mean", "avg_bat_speed": "mean", "avg_swing_len": "mean", "swings": "sum",
    "batting_avg": "mean", "xwoba": "mean", "avg_exit_velocity": "mean", "avg_launch_angle": "mean"
}
league_per_zone = df_filtered.groupby("zone", as_index=False, observed=True).agg(league_agg_dict).round(3)
league_per_zone["batter_name"] = t["league_avg"]

selected_display = [p for p in selected_players_multi if p != t["league_avg"]]

if use_id and selected_display:
    selected_player_ids = [display_to_id.get(p) for p in selected_display if p in display_to_id]
    player_filter_col = id_col
    player_filter_values = [pid for pid in selected_player_ids if pid is not None]
    player_name_map = id_to_display
else:
    player_filter_col = "batter_name"
    player_filter_values = selected_display
    player_name_map = {n: n for n in selected_display}

detail_heat = df_filtered[df_filtered[player_filter_col].isin(player_filter_values)].copy() if player_filter_values else pd.DataFrame()
if t["league_avg"] in selected_players_multi and not df_filtered.empty:
    detail_heat = pd.concat([detail_heat, league_per_zone], ignore_index=True)

detail_tables = df_filtered[df_filtered["swings"] >= min_swings].copy()

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
    else:
        if use_id:
            pid = display_to_id.get(p_sel)
            if pid is None: return pd.DataFrame()
            return df_filtered[df_filtered[id_col] == pid].groupby("zone", as_index=False).mean(numeric_only=True).round(3)
        else:
            return df_filtered[df_filtered["batter_name"] == p_sel].groupby("zone", as_index=False).mean(numeric_only=True).round(3)

# =====================================================================
# HEATMAP FUNCTION
# =====================================================================
def make_heatmap(df_p, metric, title, league_df=None):
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
        league_tilt = league_df.set_index('zone')['avg_tilt'] if league_df is not None and not league_df.empty else pd.Series()
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

# =====================================================================
# TABS
# =====================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    t["tab_summary"], t["tab_groups"], t["tab_heatmaps"],
    t["tab_side_by_side"], t["tab_player_compare"]
])

# TAB 1 - Summary
with tab1:
    st.subheader(t["tab_summary"])
    if selected_players_multi:
        st.markdown(t["selected_players"])
        sel_df = detail_tables[detail_tables[player_filter_col].isin(player_filter_values)].copy() if player_filter_values else pd.DataFrame()

        if use_id and not sel_df.empty:
            sel_summary = sel_df.groupby(id_col).agg({
                "avg_tilt":"mean", "avg_aa":"mean", "avg_bat_speed":"mean", "avg_swing_len":"mean", "swings":"sum",
                "batting_avg":"mean", "xwoba":"mean", "avg_exit_velocity":"mean", "avg_launch_angle":"mean"
            }).round(3).reset_index()
            sel_summary["batter_name"] = sel_summary[id_col].map(player_name_map)
            sel_summary = sel_summary[["batter_name", "avg_tilt", "avg_aa", "avg_bat_speed", "avg_swing_len", "swings",
                                       "batting_avg", "xwoba", "avg_exit_velocity", "avg_launch_angle"]]
        else:
            sel_summary = sel_df.groupby("batter_name").agg({
                "avg_tilt":"mean", "avg_aa":"mean", "avg_bat_speed":"mean", "avg_swing_len":"mean", "swings":"sum",
                "batting_avg":"mean", "xwoba":"mean", "avg_exit_velocity":"mean", "avg_launch_angle":"mean"
            }).round(3).reset_index()

        if t["league_avg"] in selected_players_multi and not detail_tables.empty:
            la = detail_tables.mean(numeric_only=True).round(3).to_frame().T
            la["batter_name"] = t["league_avg"]
            la["swings"] = detail_tables["swings"].sum()
            sel_summary = pd.concat([sel_summary, la[sel_summary.columns]], ignore_index=True)

        st.dataframe(sel_summary.sort_values("swings", ascending=False), use_container_width=True, hide_index=True)
        st.markdown("---")

    st.subheader(t["all_players_after_filters"])
    if not detail_tables.empty:
        if use_id:
            all_summary = detail_tables.groupby(id_col).agg({
                "avg_tilt":"mean", "avg_aa":"mean", "avg_bat_speed":"mean", "avg_swing_len":"mean", "swings":"sum",
                "batting_avg":"mean", "xwoba":"mean", "avg_exit_velocity":"mean", "avg_launch_angle":"mean"
            }).round(3).reset_index()
            all_summary["batter_name"] = all_summary[id_col].map(id_to_display)
            all_summary = all_summary[["batter_name", "avg_tilt", "avg_aa", "avg_bat_speed", "avg_swing_len", "swings",
                                       "batting_avg", "xwoba", "avg_exit_velocity", "avg_launch_angle"]]
        else:
            all_summary = detail_tables.groupby("batter_name").agg({
                "avg_tilt":"mean", "avg_aa":"mean", "avg_bat_speed":"mean", "avg_swing_len":"mean", "swings":"sum",
                "batting_avg":"mean", "xwoba":"mean", "avg_exit_velocity":"mean", "avg_launch_angle":"mean"
            }).round(3).reset_index()

        st.dataframe(all_summary.sort_values("swings", ascending=False), use_container_width=True, hide_index=True)

# TAB 2 - Group Comparison
with tab2:
    st.subheader(t["tab_groups"])
    if not selected_display:
        st.info(t["no_real_player"])
    elif df_filtered.empty:
        st.info(t["no_data_after_filters"])
    else:
        metric = st.selectbox(t["metric"], ["avg_tilt", "avg_aa", "avg_bat_speed", "avg_swing_len",
                                            "batting_avg", "xwoba", "avg_exit_velocity", "avg_launch_angle", "swings"])

        temp_df = df_filtered[df_filtered[player_filter_col].isin(player_filter_values)].copy() if player_filter_values else pd.DataFrame()
        if use_id and not temp_df.empty:
            temp_df["player_display"] = temp_df[id_col].map(id_to_display)
        else:
            temp_df["player_display"] = temp_df["batter_name"]

        col1, col2 = st.columns(2)
        with col1:
            fig_box = px.box(temp_df, x="pitch_group", y=metric, color="player_display", points="outliers", title="Distribution")
            st.plotly_chart(fig_box, use_container_width=True)
        with col2:
            avg_data = temp_df.groupby(["player_display", "pitch_group"], as_index=False)[metric].mean()
            fig_bar = px.bar(avg_data, x="pitch_group", y=metric, color="player_display", barmode="group", title="Average Value")
            st.plotly_chart(fig_bar, use_container_width=True)

# TAB 3 - Heatmaps + Tables (FIXED)
with tab3:
    st.subheader(t["tab_heatmaps"])
    if not selected_players_multi:
        st.info("Select batters in the sidebar")
    else:
        metric_tab3 = st.radio(t["heatmap_metric"], 
            ["avg_tilt", "tilt_std", "delta_tilt", "avg_aa", "aa_std", "avg_bat_speed", "avg_swing_len",
             "batting_avg", "xwoba", "avg_exit_velocity", "avg_launch_angle", "swings"], horizontal=True)

        for p in selected_players_multi:
            if p == t["league_avg"]:
                df_p = league_per_zone.copy()
                title_p = p
            else:
                if use_id:
                    p_id = display_to_id.get(p)
                    df_p = detail_heat[detail_heat[id_col] == p_id]
                else:
                    df_p = detail_heat[detail_heat["batter_name"] == p]
                title_p = p

            if df_p.empty:
                st.caption(f"{t['no_data_for']} {p}")
                continue

            make_heatmap(df_p, metric_tab3, title_p, league_per_zone)

            if p != t["league_avg"]:
                st.markdown("---")
                st.subheader(f"{t['detailed_table']} – {title_p}")

                if use_id:
                    player_df = detail_tables[detail_tables[id_col] == display_to_id.get(p)].copy()
                else:
                    player_df = detail_tables[detail_tables["batter_name"] == p].copy()

                if not player_df.empty:
                    zone_agg = player_df.groupby("zone", as_index=False).agg({
                        "swings": "sum",
                        "avg_tilt": ["mean", "std"],
                        "avg_aa": ["mean", "std"],
                        "avg_bat_speed": "mean",
                        "avg_swing_len": "mean",
                        "batting_avg": "mean",
                        "xwoba": "mean",
                        "avg_exit_velocity": "mean",
                        "avg_launch_angle": "mean"
                    }).round(3)

                    # Flatten multi-level columns
                    zone_agg.columns = ['Zone' if col[0] == 'zone' else f"{col[0]} {col[1]}" if isinstance(col, tuple) else col 
                                       for col in zone_agg.columns]
                    st.dataframe(zone_agg.sort_values("swings sum", ascending=False), 
                                 use_container_width=True, hide_index=True)

    # All batters table
    st.markdown("---")
    st.subheader(t["all_players_after_filters"])
    if not detail_tables.empty:
        if use_id:
            all_agg = detail_tables.groupby(id_col).agg({
                "swings": "sum", "avg_tilt":"mean", "avg_aa":"mean", "avg_bat_speed":"mean", "avg_swing_len":"mean",
                "batting_avg":"mean", "xwoba":"mean", "avg_exit_velocity":"mean", "avg_launch_angle":"mean"
            }).round(3).reset_index()
            all_agg["batter_name"] = all_agg[id_col].map(id_to_display)
            all_agg = all_agg[["batter_name", "swings", "avg_tilt", "avg_aa", "avg_bat_speed", "avg_swing_len",
                               "batting_avg", "xwoba", "avg_exit_velocity", "avg_launch_angle"]]
        else:
            all_agg = detail_tables.groupby("batter_name").agg({
                "swings": "sum", "avg_tilt":"mean", "avg_aa":"mean", "avg_bat_speed":"mean", "avg_swing_len":"mean",
                "batting_avg":"mean", "xwoba":"mean", "avg_exit_velocity":"mean", "avg_launch_angle":"mean"
            }).round(3).reset_index()

        st.dataframe(all_agg.sort_values("swings", ascending=False), use_container_width=True, hide_index=True)

# TAB 4 - Side by Side
with tab4:
    st.subheader(t["tab_side_by_side"])
    compare_options = [t["league_avg"]] + (sorted(player_info['display_name']) if player_info is not None else all_real)

    colL, colR = st.columns(2)
    with colL:
        p_left = st.selectbox(t["compare_left"], options=compare_options, index=0, key="left_tab4")
    with colR:
        p_right = st.selectbox(t["compare_right"], options=compare_options, index=1 if len(compare_options)>1 else 0, key="right_tab4")

    metric_tab4 = st.radio(t["heatmap_metric"], 
        ["avg_tilt", "avg_aa", "avg_bat_speed", "avg_swing_len", "batting_avg", "xwoba",
         "avg_exit_velocity", "avg_launch_angle", "swings"], horizontal=True, key="tab4_metric")

    df_left = get_player_zone_df(p_left)
    df_right = get_player_zone_df(p_right)

    colL, colR = st.columns(2)
    with colL:
        make_heatmap(df_left, metric_tab4, p_left, league_per_zone) if not df_left.empty else st.info(f"{t['no_data_for']} {p_left}")
    with colR:
        make_heatmap(df_right, metric_tab4, p_right, league_per_zone) if not df_right.empty else st.info(f"{t['no_data_for']} {p_right}")

# TAB 5 - Single Batter Comparison
with tab5:
    st.subheader(t["tab_player_compare"])
    player_options_tab5 = selected_display if selected_display else all_real
    selected_p = st.selectbox(t["select_player_tab5"], options=player_options_tab5, key="tab5_player")

    metric_options = {
        "avg_tilt": "Average Tilt", "avg_aa": "Attack Angle", "avg_bat_speed": "Bat Speed",
        "avg_swing_len": "Swing Length", "batting_avg": "Batting Avg", "xwoba": "xWOBA",
        "avg_exit_velocity": "Exit Velocity", "avg_launch_angle": "Launch Angle", "swings": "Swings"
    }

    colL, colR = st.columns(2)
    with colL:
        left_metric = st.selectbox(t["left_metric_tab5"], options=list(metric_options.keys()),
                                   format_func=lambda x: metric_options[x], index=0, key="tab5_left")
    with colR:
        right_metric = st.selectbox(t["right_metric_tab5"], options=list(metric_options.keys()),
                                    format_func=lambda x: metric_options[x], index=6, key="tab5_right")

    df_player = get_player_zone_df(selected_p)

    if df_player.empty:
        st.info(f"{t['no_data_for']} {selected_p}")
    else:
        col_display1, col_display2 = st.columns(2)
        with col_display1:
            make_heatmap(df_player, left_metric, f"{selected_p} — {metric_options[left_metric]}", league_per_zone)
        with col_display2:
            make_heatmap(df_player, right_metric, f"{selected_p} — {metric_options[right_metric]}", league_per_zone)
