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
    },
    "pl": {
        "title": "MLB 2025 – Dashboard Nachylenia Trajektorii Swingu i Kąta Ataku",
        "sidebar_players": "Wybierz pałkarzy",
        "sidebar_min_swings": "Minimalna liczba swingów (tabele)",
        "sidebar_pitch_group": "Grupa narzutów",
        "sidebar_pitch_type": "Typ narzutu",
        "all": "Wszystkie",
        "league_avg": "Średnia ligowa",
        "tab_summary": "Podsumowanie",
        "tab_groups": "Porównanie grup",
        "tab_heatmaps": "Heatmapy",
        "tab_side_by_side": "Porównanie bokiem",
        "tab_player_compare": "Analiza pojedynczego pałkarza",
        "selected_players": "Wybrani pałkarze",
        "all_players": "Wszyscy pałkarze",
        "no_data": "Brak danych",
        "no_data_for": "Brak danych dla",
        "metric": "Metryka",
        "heatmap_metric": "Metryka na heatmapie",
        "compare_left": "Lewy pałkarz",
        "compare_right": "Prawy pałkarz",
        "select_batter": "Wybierz pałkarza",
        "left_metric": "Metryka lewa",
        "right_metric": "Metryka prawa",
    }
}

# Language selector
if "lang" not in st.session_state:
    st.session_state.lang = "en"

with st.sidebar:
    lang_choice = st.selectbox("Language", options=["English", "Polski"], index=0)
    st.session_state.lang = "pl" if lang_choice == "Polski" else "en"

t = TEXTS[st.session_state.lang]

# ========================= DATA LOADING =========================
DATA_DIR = Path(r"C:\Users\patri\baseball_2025_data")

@st.cache_data(show_spinner="Loading data...", ttl=3600, persist="disk")
def load_data():
    players = pd.read_parquet(DATA_DIR / "players_summary_2025.parquet")
    detail = pd.read_parquet(
        DATA_DIR / "detail_zone_pitchgroup_2025.parquet",
        columns=[
            "batter_name", "batter", "pitch_group", "pitch_type", "zone",
            "swings", "avg_tilt", "avg_aa", "avg_bat_speed", "avg_swing_len",
            "batting_avg", "xwoba", "avg_exit_velocity", "avg_launch_angle"
        ]
    )
    # Remove pitchers
    detail = detail[detail["batter_name"].notna() & 
                   ~detail["batter_name"].str.contains(r" pitcher| P$", case=False, na=False)]
    return players, detail

players, detail_full = load_data()

# Handle duplicate batter names
use_id = False
id_col = None
for col in ['batter_id', 'batter', 'mlb_id', 'player_id', 'id']:
    if col in players.columns and col in detail_full.columns:
        id_col = col
        use_id = True
        break

if use_id:
    player_info = players[[id_col, 'batter_name']].drop_duplicates()
    dupe_names = player_info['batter_name'].value_counts()
    dupe_names = dupe_names[dupe_names > 1].index.tolist()
    player_info['display_name'] = player_info.apply(
        lambda row: f"{row['batter_name']} (ID:{int(row[id_col])})" 
        if row['batter_name'] in dupe_names else row['batter_name'], axis=1)
    display_to_id = dict(zip(player_info['display_name'], player_info[id_col]))
    id_to_display = dict(zip(player_info[id_col], player_info['display_name']))
    all_batters = sorted(player_info['display_name'])
else:
    all_batters = sorted(players["batter_name"].dropna().unique())

# ========================= SIDEBAR FILTERS =========================
with st.sidebar:
    st.header("Filters")
    
    player_options = [t["league_avg"]] + ["─" * 25] + all_batters
    selected_players = st.multiselect(
        t["sidebar_players"],
        options=player_options,
        default=[all_batters[0]] if all_batters else [],
        max_selections=8
    )
    selected_players = [p for p in selected_players if "─" not in p]

    min_swings = st.slider(t["sidebar_min_swings"], 0, 300, 10, 10)

    pitch_groups = [t["all"]] + sorted(detail_full["pitch_group"].dropna().unique().tolist())
    selected_pitch_group = st.selectbox(t["sidebar_pitch_group"], pitch_groups)

    if selected_pitch_group == t["all"]:
        pitch_types_list = sorted(detail_full["pitch_type"].dropna().unique().tolist())
    else:
        pitch_types_list = sorted(detail_full[detail_full["pitch_group"] == selected_pitch_group]["pitch_type"].dropna().unique().tolist())

    pitch_types = [t["all"]] + pitch_types_list
    selected_pitch_type = st.selectbox(t["sidebar_pitch_type"], pitch_types)

# ========================= DATA FILTERING =========================
df_filtered = detail_full.copy()
if selected_pitch_group != t["all"]:
    df_filtered = df_filtered[df_filtered["pitch_group"] == selected_pitch_group]
if selected_pitch_type != t["all"]:
    df_filtered = df_filtered[df_filtered["pitch_type"] == selected_pitch_type]

# League average per zone
league_per_zone = df_filtered.groupby("zone", as_index=False).agg({
    "avg_tilt": "mean", "avg_aa": "mean", "avg_bat_speed": "mean", "avg_swing_len": "mean",
    "swings": "sum", "batting_avg": "mean", "xwoba": "mean",
    "avg_exit_velocity": "mean", "avg_launch_angle": "mean"
}).round(3)
league_per_zone["batter_name"] = t["league_avg"]

selected_real = [p for p in selected_players if p != t["league_avg"]]

# Mapping for filtering
if use_id and selected_real:
    selected_ids = [display_to_id.get(p) for p in selected_real if p in display_to_id]
    filter_col = id_col
    filter_values = [pid for pid in selected_ids if pid is not None]
    name_map = id_to_display
else:
    filter_col = "batter_name"
    filter_values = selected_real
    name_map = {n: n for n in selected_real}

detail_tables = df_filtered[df_filtered[filter_col].isin(filter_values)] if filter_values else pd.DataFrame()
detail_tables = detail_tables[detail_tables["swings"] >= min_swings]

# ========================= HEATMAP FUNCTION =========================
def make_heatmap(df_p, metric, title):
    if df_p.empty:
        st.warning("No data")
        return

    # Pivot logic...
    if metric == "swings":
        pivot = df_p.groupby('zone')['swings'].sum().round(0)
    elif metric == "tilt_std":
        pivot = df_p.groupby('zone')['avg_tilt'].std(ddof=1).round(1)
    elif metric == "aa_std":
        pivot = df_p.groupby('zone')['avg_aa'].std(ddof=1).round(1)
    elif metric == "delta_tilt":
        player_mean = df_p.groupby('zone')['avg_tilt'].mean().round(2)
        league_mean = league_per_zone.set_index('zone')['avg_tilt']
        pivot = (player_mean - league_mean.reindex(player_mean.index, fill_value=np.nan)).round(2)
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

    # ... (reszta funkcji make_heatmap pozostaje taka sama jak w poprzedniej wersji)

    # (Dla skrócenia nie wklejam całej długiej funkcji rysującej strike zone – zostaw ją z poprzedniej wersji)

# ========================= TABS =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    t["tab_summary"], t["tab_groups"], t["tab_heatmaps"],
    t["tab_side_by_side"], t["tab_player_compare"]
])

# Tab 1: Summary
with tab1:
    st.subheader(t["tab_summary"])
    # ... (kod z poprzedniej wersji – z batter_name na początku)

# Tab 2, 3, 4, 5 – podobnie jak w poprzedniej profesjonalnej wersji

# Tab 5 – Single Batter Analysis (z wyborem metryk po obu stronach)
with tab5:
    st.subheader(t["tab_player_compare"])
    selected_batter = st.selectbox(t["select_batter"], options=selected_real if selected_real else all_batters)

    metric_options = {
        "avg_tilt": "Average Tilt", "avg_aa": "Attack Angle", "avg_bat_speed": "Bat Speed",
        "avg_swing_len": "Swing Length", "batting_avg": "Batting Average", "xwoba": "xWOBA",
        "avg_exit_velocity": "Exit Velocity", "avg_launch_angle": "Launch Angle"
    }

    col1, col2 = st.columns(2)
    with col1:
        left_metric = st.selectbox(t["left_metric"], options=list(metric_options.keys()),
                                   format_func=lambda x: metric_options[x], index=0)
    with col2:
        right_metric = st.selectbox(t["right_metric"], options=list(metric_options.keys()),
                                    format_func=lambda x: metric_options[x], index=6)

    df_batter = get_player_zone_df(selected_batter)
    if not df_batter.empty:
        c1, c2 = st.columns(2)
        with c1:
            make_heatmap(df_batter, left_metric, f"{selected_batter} — {metric_options[left_metric]}")
        with c2:
            make_heatmap(df_batter, right_metric, f"{selected_batter} — {metric_options[right_metric]}")