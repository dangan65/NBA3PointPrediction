"""
NBA 3-Point Props Dashboard - Streamlit Application

Purpose:
  Interactive web dashboard for analyzing NBA 3-point shooting props, tracking predictions,
  and evaluating parlay performance. Provides confidence-based pick rankings, player trend
  analysis, and historical tracking of settled bets.

Data Files:
  - ranked_value_3pt_props.csv: Value-ranked 3-point props
  - most_confident_3pt_props.csv: Confidence-ranked 3-point props
  - player_game_3pt_features.csv: Historical player game-level 3pt statistics
  - tracked_bets_3pt.csv: User-tracked bets (value approach)
  - tracked_confidence_3pt.csv: User-tracked bets (confidence approach)
  - confidence_top3_parlays.csv: Top-3 parlay combinations
  - confidence_top5_parlays.csv: Top-5 parlay combinations
  - value_top10_model_backfill.csv: Historical model backfill (value)
  - value_top10_model_backfill_summary.csv: Backfill summary (value)
  - confidence_top10_model_backfill.csv: Historical model backfill (confidence)
  - confidence_top10_model_backfill_summary.csv: Backfill summary (confidence)

Running:
  streamlit run app.py

Layout:
  - Sidebar: Controls for slate date, player selection, and filters
  - Main Area: Confidence board, player detail charts, parlay stats, tracking/backfill history
"""

from __future__ import annotations
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tracking import settle_top3_parlays, settle_top5_parlays, settle_tracked_bets_from_history


# ---------------------------------------------------------------------------
# File Paths & Constants
# ---------------------------------------------------------------------------


BASE_DIR = Path(__file__).resolve().parent
RANKED_PROPS_PATH = BASE_DIR / "data" / "ranked_value_3pt_props.csv"
CONFIDENCE_PROPS_PATH = BASE_DIR / "data" / "most_confident_3pt_props.csv"
HISTORY_PATH = BASE_DIR / "data" / "player_game_3pt_features.csv"
TRACKED_BETS_PATH = BASE_DIR / "data" / "tracked_bets_3pt.csv"
TRACKED_CONFIDENCE_PATH = BASE_DIR / "data" / "tracked_confidence_3pt.csv"
TOP3_PARLAY_PATH = BASE_DIR / "data" / "confidence_top3_parlays.csv"
TOP5_PARLAY_PATH = BASE_DIR / "data" / "confidence_top5_parlays.csv"
BACKFILL_TOP10_PATH = BASE_DIR / "data" / "value_top10_model_backfill.csv"
BACKFILL_SUMMARY_PATH = BASE_DIR / "data" / "value_top10_model_backfill_summary.csv"
BACKFILL_CONF_TOP10_PATH = BASE_DIR / "data" / "confidence_top10_model_backfill.csv"
BACKFILL_CONF_SUMMARY_PATH = BASE_DIR / "data" / "confidence_top10_model_backfill_summary.csv"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    ranked = load_csv(RANKED_PROPS_PATH)
    confident = load_csv(CONFIDENCE_PROPS_PATH)
    history = load_csv(HISTORY_PATH)
    tracked = load_csv(TRACKED_BETS_PATH)
    tracked_confidence = load_csv(TRACKED_CONFIDENCE_PATH)
    top3_parlays = load_csv(TOP3_PARLAY_PATH)
    top5_parlays = load_csv(TOP5_PARLAY_PATH)
    backfill_top10 = load_csv(BACKFILL_TOP10_PATH)
    backfill_summary = load_csv(BACKFILL_SUMMARY_PATH)
    backfill_conf_top10 = load_csv(BACKFILL_CONF_TOP10_PATH)
    backfill_conf_summary = load_csv(BACKFILL_CONF_SUMMARY_PATH)
    return (
        ranked,
        confident,
        history,
        tracked,
        tracked_confidence,
        top3_parlays,
        top5_parlays,
        backfill_top10,
        backfill_summary,
        backfill_conf_top10,
        backfill_conf_summary,
    )


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def truthy_flag(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "*"}


def player_display_value(row: pd.Series) -> str:
    if "player_display_name" in row.index and pd.notna(row.get("player_display_name")) and str(row.get("player_display_name", "")).strip():
        return str(row.get("player_display_name", "")).strip()
    if pd.isna(row.get("player_name")):
        return ""
    name = str(row.get("player_name", "")).strip()
    if not name:
        return ""
    return f"{name} *" if truthy_flag(row.get("predictability_star")) else name


def with_player_display(frame: pd.DataFrame) -> pd.DataFrame:
    displayed = frame.copy()
    if displayed.empty or "player_name" not in displayed.columns:
        return displayed
    displayed["player_display_name"] = displayed.apply(player_display_value, axis=1)
    return displayed


CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'DM Sans', sans-serif", color="#1f3147", size=13),
    margin=dict(l=48, r=16, t=28, b=48),
    xaxis=dict(
        showgrid=False,
        linecolor="#9fb0c4",
        tickfont=dict(color="#24364d", size=13),
        title_font=dict(color="#24364d", size=14),
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(145,164,188,0.28)",
        linecolor="rgba(0,0,0,0)",
        tickfont=dict(color="#24364d", size=13),
        title_font=dict(color="#24364d", size=14),
        zeroline=False,
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#c6d2df",
        borderwidth=1,
        font=dict(color="#1f3147", size=12),
    ),
)

ACCENT = "#1a6fde"
ACCENT2 = "#f5a623"
ACCENT3 = "#27ae60"

TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


# ---------------------------------------------------------------------------
# Styling & UI
# ---------------------------------------------------------------------------

def style_app() -> None:
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
        <style>
        html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], .stMarkdown, .stText, .stCaption, .stMetric, .stSelectbox label, .stSlider label, .stTabs, table, button, input {
            font-family: 'DM Sans', sans-serif !important;
        }
        .stApp {
            background: linear-gradient(160deg, #f0f4fa 0%, #f8f9fc 55%, #ffffff 100%);
        }
        header[data-testid="stHeader"] {
            background: transparent !important;
        }
        header[data-testid="stHeader"] button {
            background-color: #1a6fde !important;
            border: none !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 8px rgba(26, 111, 222, 0.22) !important;
            opacity: 1 !important;
            visibility: visible !important;
            outline: none !important;
        }
        header[data-testid="stHeader"] button:hover,
        header[data-testid="stHeader"] button:focus,
        header[data-testid="stHeader"] button:active {
            background-color: #155fc2 !important;
            opacity: 1 !important;
            outline: none !important;
        }
        header[data-testid="stHeader"] button svg {
            fill: #ffffff !important;
            stroke: #ffffff !important;
            opacity: 1 !important;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1540px;
        }
        /* Fix Streamlit dark widget backgrounds */
        .stSelectbox > div > div {
            background: white !important;
            border-radius: 10px !important;
        }
        [data-testid="stAppViewContainer"] .stSelectbox label {
            color: #1f3147 !important;
            font-weight: 600 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"],
        [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p {
            color: #1f3147 !important;
            opacity: 1 !important;
            font-weight: 600 !important;
        }
        [data-testid="stAppViewContainer"] .stSelectbox [data-baseweb="select"] {
            background: #ffffff !important;
            border: 1px solid #d2deeb !important;
            border-radius: 10px !important;
        }
        [data-testid="stAppViewContainer"] .stSelectbox [data-baseweb="select"] > div {
            color: #0d1f33 !important;
        }
        [data-testid="stAppViewContainer"] .stSelectbox svg {
            fill: #0d1f33 !important;
            color: #0d1f33 !important;
        }
        .stSlider {
            padding-top: 0.35rem;
            padding-bottom: 0.35rem;
            margin-left: 0.35rem;
            margin-right: 0.35rem;
        }
        .stDataFrame { border-radius: 12px; overflow: hidden; }
        .metric-card {
            background: #ffffff;
            border: 1px solid #e4eaf2;
            border-radius: 16px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 4px 18px rgba(26, 46, 68, 0.07);
        }
        .metric-label {
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #6b7f96;
            font-weight: 600;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #0d1f33;
            margin-top: 0.2rem;
            line-height: 1.1;
        }
        .metric-note {
            font-size: 0.85rem;
            color: #7a90a8;
            margin-top: 0.3rem;
        }
        .hero {
            background: linear-gradient(130deg, #0d2144 0%, #1a4f8c 100%);
            color: white;
            border-radius: 20px;
            padding: 1.4rem 1.8rem;
            margin-bottom: 1.4rem;
            box-shadow: 0 12px 36px rgba(13, 33, 68, 0.18);
        }
        .hero h1 {
            margin: 0;
            font-size: 1.9rem;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: -0.02em;
        }
        .hero p {
            margin: 0.4rem 0 0;
            color: rgba(255,255,255,0.72);
            font-size: 0.95rem;
        }
        .section-title {
            font-size: 1rem;
            font-weight: 600;
            color: #0d1f33;
            letter-spacing: -0.01em;
            margin-bottom: 0.6rem;
        }
        /* Fix sidebar text */
        [data-testid="stSidebar"] {
            background: #0d1f33 !important;
        }
        /* Sidebar toggle button — single clean border only on the outer wrapper */
        [data-testid="collapsedControl"] {
            background-color: #1a6fde !important;
            border-radius: 12px !important;
            box-shadow: 0 6px 16px rgba(26, 111, 222, 0.25) !important;
            opacity: 1 !important;
            visibility: visible !important;
            filter: none !important;
            border: none !important;
            outline: none !important;
        }
        /* Strip borders from all children so they don't double-up */
        [data-testid="collapsedControl"] button,
        [data-testid="collapsedControl"] [role="button"],
        [data-testid="collapsedControl"] > div,
        [data-testid="collapsedControl"] > div > button,
        [data-testid="collapsedControl"] > div > div {
            background-color: #1a6fde !important;
            border: none !important;
            outline: none !important;
            border-radius: 12px !important;
            opacity: 1 !important;
            visibility: visible !important;
            box-shadow: none !important;
            filter: none !important;
        }
        [data-testid="collapsedControl"]:hover,
        [data-testid="collapsedControl"] button:hover,
        [data-testid="collapsedControl"] > div:hover,
        [data-testid="collapsedControl"] > div > button:hover {
            background-color: #155fc2 !important;
        }
        [data-testid="collapsedControl"] button:focus,
        [data-testid="collapsedControl"] button:active {
            outline: none !important;
            box-shadow: none !important;
        }
        [data-testid="collapsedControl"] svg {
            fill: #ffffff !important;
            stroke: #ffffff !important;
            opacity: 1 !important;
        }
        /* Sidebar collapse button inside the sidebar */
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="stSidebarCollapseButton"] > div,
        [data-testid="stSidebarCollapseButton"] > div > button {
            background: #1a6fde !important;
            border: none !important;
            outline: none !important;
            border-radius: 12px !important;
            box-shadow: 0 6px 16px rgba(26, 111, 222, 0.25) !important;
            opacity: 1 !important;
            visibility: visible !important;
        }
        [data-testid="stSidebarCollapseButton"] button:hover,
        [data-testid="stSidebarCollapseButton"] button:focus,
        [data-testid="stSidebarCollapseButton"] button:active {
            background: #155fc2 !important;
            outline: none !important;
            box-shadow: none !important;
            opacity: 1 !important;
        }
        [data-testid="stSidebarCollapseButton"] svg {
            fill: #ffffff !important;
            stroke: #ffffff !important;
            opacity: 1 !important;
        }
        button[aria-label="Open sidebar"],
        button[aria-label="Close sidebar"] {
            background: #1a6fde !important;
            border: none !important;
            outline: none !important;
            border-radius: 12px !important;
            box-shadow: 0 6px 16px rgba(26, 111, 222, 0.25) !important;
            opacity: 1 !important;
            visibility: visible !important;
        }
        button[aria-label="Open sidebar"]:hover,
        button[aria-label="Open sidebar"]:focus,
        button[aria-label="Open sidebar"]:active,
        button[aria-label="Close sidebar"]:hover,
        button[aria-label="Close sidebar"]:focus,
        button[aria-label="Close sidebar"]:active {
            background: #155fc2 !important;
            outline: none !important;
            box-shadow: none !important;
            opacity: 1 !important;
        }
        button[aria-label="Open sidebar"] svg,
        button[aria-label="Close sidebar"] svg {
            fill: #ffffff !important;
            stroke: #ffffff !important;
            opacity: 1 !important;
        }
        /* Fix 3-dot toolbar menu button square artifact */
        [data-testid="stToolbarActions"] button,
        [data-testid="stToolbarActions"] > div,
        [data-testid="stToolbarActionButton"] {
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
            background: transparent !important;
        }
        [data-testid="stToolbarActions"] button:focus,
        [data-testid="stToolbarActions"] button:focus-visible,
        [data-testid="stToolbarActionButton"]:focus,
        [data-testid="stToolbarActionButton"]:focus-visible {
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
        }
        [data-testid="stSidebar"] * {
            color: #e8eef7 !important;
        }
        [data-testid="stSidebar"] .stSelectbox > div > div {
            background: #1a2e4a !important;
            color: #e8eef7 !important;
            border: none !important;
            box-shadow: none !important;
            min-height: 52px !important;
        }
        [data-testid="stSidebar"] .stSelectbox svg {
            fill: #0d1f33 !important;
            color: #0d1f33 !important;
        }
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
            background: #1a2e4a !important;
            border-radius: 12px !important;
            border: 1px solid #3d5577 !important;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.04) !important;
            padding-left: 0.1rem !important;
            padding-right: 0.25rem !important;
            overflow: hidden !important;
        }
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
            color: #e8eef7 !important;
        }
        [data-testid="stSidebar"] .stSlider > div[data-baseweb="slider"] {
            background: transparent !important;
            border: none !important;
            padding: 0.15rem 0 !important;
        }
        [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
        [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
            margin-top: 0.25rem !important;
            display: inline-block !important;
            position: relative !important;
            color: #ecf3ff !important;
            font-weight: 800 !important;
            padding: 0 !important;
        }
        [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"] {
            left: 0 !important;
        }
        [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
            right: 0 !important;
        }
        [data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
            color: #ecf3ff !important;
            font-weight: 800 !important;
            font-size: 1.02rem !important;
        }
        [data-testid="stSidebar"] .stSlider p {
            color: #ecf3ff !important;
            font-weight: 700 !important;
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] .stCaption p {
            color: #ecf3ff !important;
            opacity: 1 !important;
            font-size: 0.82rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.06em !important;
        }
        [data-testid="stSidebar"] .stSlider [data-baseweb="slider"],
        [data-testid="stSidebar"] .stSlider [role="slider"] {
            pointer-events: auto !important;
        }
        [data-testid="stSidebar"] h2 {
            color: #ffffff !important;
            font-size: 1rem !important;
            font-weight: 700 !important;
        }
        /* Dataframe headers */
        [data-testid="stDataFrame"] th {
            background: #f0f4fa !important;
            color: #0d1f33 !important;
            font-weight: 600 !important;
            font-size: 0.8rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }
        [data-testid="stDataFrame"] td {
            color: #1a2e44 !important;
            font-size: 0.9rem !important;
        }
        [data-testid="stDataFrameResizeHandle"] {
            display: none !important;
        }
        [data-baseweb="tab-list"] {
            gap: 0.55rem !important;
            padding-bottom: 0.35rem !important;
        }
        [data-baseweb="tab-list"] button {
            color: #1f3147 !important;
            font-weight: 600 !important;
            background: #e8eef7 !important;
            border: 1px solid #d2deeb !important;
            border-radius: 10px 10px 0 0 !important;
            padding: 0.6rem 0.9rem !important;
            box-shadow: 0 2px 8px rgba(26, 46, 68, 0.05) !important;
            opacity: 1 !important;
        }
        [data-baseweb="tab-list"] button[aria-selected="true"] {
            color: #0d1f33 !important;
            background: #ffffff !important;
            border-bottom-color: #ffffff !important;
            box-shadow: 0 4px 12px rgba(26, 46, 68, 0.08) !important;
            opacity: 1 !important;
        }
        [data-baseweb="tab-panel"] {
            background: #ffffff;
            border: 1px solid #dce5f0;
            border-radius: 0 14px 14px 14px;
            padding: 1rem 1.1rem 0.9rem 1.1rem;
        }
        [data-baseweb="tab-panel"] [data-testid="stWidgetLabel"] p {
            color: #1f3147 !important;
            opacity: 1 !important;
            font-weight: 600 !important;
        }
        .caption-text {
            font-size: 0.78rem;
            color: #2b4361;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            font-weight: 600;
            margin-bottom: 0.4rem;
        }
        [data-testid="stAppViewContainer"] .stCaption,
        [data-testid="stAppViewContainer"] .stCaption p,
        [data-testid="stAppViewContainer"] .stCaption span,
        [data-testid="stAppViewContainer"] div[data-testid="stCaptionContainer"],
        [data-testid="stAppViewContainer"] div[data-testid="stCaptionContainer"] p,
        [data-testid="stAppViewContainer"] div[data-testid="stCaptionContainer"] span,
        [data-testid="stVerticalBlock"] .stCaption p,
        [data-testid="stVerticalBlock"] .stCaption span,
        [data-testid="stHorizontalBlock"] .stCaption p,
        [data-testid="stHorizontalBlock"] .stCaption span,
        .stMain .stCaption, .stMain .stCaption p, .stMain .stCaption span,
        .stCaption p, .stCaption span {
            color: #2b4361 !important;
            opacity: 1 !important;
            visibility: visible !important;
        }
        /* Sidebar captions override back to light */
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] .stCaption p,
        [data-testid="stSidebar"] .stCaption span {
            color: #ecf3ff !important;
        }
        .summary-note {
            font-size: 0.92rem;
            color: #2b4361;
            margin: 0.2rem 0 0.45rem 0;
        }
        .reasoning-card {
            border: 1px solid #d6e2f0;
            background: #f7faff;
            border-radius: 10px;
            padding: 0.65rem 0.85rem;
            margin: 0.45rem 0 0.75rem 0;
        }
        .reasoning-title {
            font-size: 0.83rem;
            font-weight: 700;
            color: #0d1f33;
            margin-bottom: 0.35rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .reasoning-item {
            color: #1f3147;
            font-size: 0.9rem;
            margin: 0.2rem 0;
        }
        .defense-card {
            border: 1px solid #d6e2f0;
            background: #f6f9ff;
            border-radius: 10px;
            padding: 0.7rem 0.85rem;
            margin: 0.45rem 0 0.75rem 0;
        }
        .defense-title {
            font-size: 0.83rem;
            font-weight: 700;
            color: #0d1f33;
            margin-bottom: 0.45rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .defense-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.35rem 0.75rem;
        }
        .defense-item {
            display: flex;
            justify-content: space-between;
            gap: 0.5rem;
            font-size: 0.86rem;
            color: #2a3f57;
            border-bottom: 1px dashed rgba(42, 63, 87, 0.16);
            padding-bottom: 0.2rem;
        }
        .defense-item strong {
            color: #0d1f33;
            font-weight: 700;
        }
        .defense-note {
            margin-top: 0.45rem;
            font-size: 0.8rem;
            color: #3d5674;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Helper Utilities
# ---------------------------------------------------------------------------

def info_caption(text: str) -> None:
    """Render a caption that is always visible regardless of Streamlit theme."""
    st.markdown(
        f"<p style='font-size:0.82rem;color:#2b4361;margin:0.15rem 0 0.3rem 0;opacity:1;'>{text}</p>",
        unsafe_allow_html=True,
    )


def render_metric(label: str, value: str, note: str, accent: str = "#1a6fde") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{accent};">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Player Detail Data Extraction
# ---------------------------------------------------------------------------

def player_history(history: pd.DataFrame, player_name: str) -> pd.DataFrame:
    if history.empty or "player_name" not in history.columns:
        return pd.DataFrame()
    player_rows = history[history["player_name"] == player_name].copy()
    if player_rows.empty:
        return player_rows
    if "game_date" in player_rows.columns:
        player_rows["game_date"] = pd.to_datetime(player_rows["game_date"], errors="coerce")
    player_rows["player_made_3pt"] = to_numeric(player_rows["player_made_3pt"])
    player_rows.sort_values(["game_date", "game_id"], inplace=True)
    player_rows["rolling_5_avg"] = player_rows["player_made_3pt"].rolling(5, min_periods=1).mean()
    return player_rows.tail(15)


def player_prop_rows(ranked: pd.DataFrame, player_name: str) -> pd.DataFrame:
    if ranked.empty or "player_name" not in ranked.columns:
        return pd.DataFrame()
    player_rows = ranked[ranked["player_name"] == player_name].copy()
    if player_rows.empty:
        return player_rows
    for column in ["edge", "expected_value_per_unit", "model_prob", "line", "estimated_3pm_mean"]:
        if column in player_rows.columns:
            player_rows[column] = to_numeric(player_rows[column])
    if "rank" in player_rows.columns:
        player_rows["rank"] = to_numeric(player_rows["rank"])
        player_rows.sort_values("rank", inplace=True)
    return player_rows


def normalize_team_key(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    upper = text.upper()
    if upper in TEAM_NAME_TO_ABBR.values():
        return upper
    return TEAM_NAME_TO_ABBR.get(text, upper)


def three_pt_defense_rank_context(
    primary_row: pd.Series,
    latest_history_row: pd.Series | None,
    history: pd.DataFrame | None,
) -> dict[str, object]:
    if history is None or history.empty or "opp_3pm_allowed" not in history.columns:
        return {}

    opp_series = history.copy()
    opp_series["opp_3pm_allowed"] = pd.to_numeric(opp_series["opp_3pm_allowed"], errors="coerce")
    if "opponent" in opp_series.columns:
        opp_series["opp_key"] = opp_series["opponent"].map(normalize_team_key)
    else:
        opp_series["opp_key"] = ""
    opp_series = opp_series[opp_series["opp_3pm_allowed"].notna() & opp_series["opp_key"].astype(str).ne("")].copy()
    if opp_series.empty:
        return {}

    team_table = opp_series.groupby("opp_key", dropna=False)["opp_3pm_allowed"].median().dropna()
    if team_table.empty:
        return {}

    opp_key = normalize_team_key(primary_row.get("opponent", ""))
    if not opp_key and latest_history_row is not None and not latest_history_row.empty:
        opp_key = normalize_team_key(latest_history_row.get("opponent", ""))

    if opp_key not in team_table.index:
        return {}

    sorted_allowed = team_table.sort_values(ascending=True)  # 1 = toughest (fewest 3PM allowed)
    rank = int(sorted_allowed.index.get_loc(opp_key) + 1)
    total_teams = int(len(sorted_allowed))
    opp_allowed = float(sorted_allowed.loc[opp_key])

    if rank <= max(3, int(round(total_teams * 0.33))):
        label = "Strong 3PT Defense"
    elif rank >= min(total_teams - 2, int(round(total_teams * 0.67))):
        label = "Weak 3PT Defense"
    else:
        label = "Middle-Tier 3PT Defense"

    return {
        "rank": rank,
        "total": total_teams,
        "label": label,
        "opp_3pm_allowed": opp_allowed,
    }


def pick_reasoning_lines(
    primary_row: pd.Series,
    latest_history_row: pd.Series | None = None,
    history: pd.DataFrame | None = None,
) -> list[str]:
    reasons: list[str] = []

    edge = pd.to_numeric(pd.Series([primary_row.get("edge")]), errors="coerce").iloc[0]
    model_prob = pd.to_numeric(pd.Series([primary_row.get("model_prob")]), errors="coerce").iloc[0]
    implied_prob = pd.to_numeric(pd.Series([primary_row.get("implied_prob")]), errors="coerce").iloc[0]
    ev_unit = pd.to_numeric(pd.Series([primary_row.get("expected_value_per_unit")]), errors="coerce").iloc[0]
    side = str(primary_row.get("side", "")).upper().strip()
    line = pd.to_numeric(pd.Series([primary_row.get("line")]), errors="coerce").iloc[0]
    lam = pd.to_numeric(pd.Series([primary_row.get("estimated_3pm_mean")]), errors="coerce").iloc[0]

    if pd.notna(edge):
        reasons.append(f"Model edge is {edge:.1%} versus market implied probability.")
    if pd.notna(model_prob) and pd.notna(implied_prob):
        gap = model_prob - implied_prob
        reasons.append(f"Model win probability {model_prob:.1%} vs market {implied_prob:.1%} (gap {gap:+.1%}).")
    if pd.notna(lam) and pd.notna(line) and side in {"OVER", "UNDER"}:
        if side == "OVER":
            reasons.append(f"Projected mean is {lam:.2f} threes against an OVER {line:.1f} line.")
        else:
            reasons.append(f"Projected mean is {lam:.2f} threes against an UNDER {line:.1f} line.")
    if pd.notna(ev_unit):
        reasons.append(f"Expected value is {ev_unit:.2f} units per 1 unit risked.")

    defense_rank = three_pt_defense_rank_context(primary_row, latest_history_row, history)
    if defense_rank:
        rank_num = int(defense_rank["rank"])
        rank_total = int(defense_rank["total"])
        defense_label = str(defense_rank["label"])
        reasons.append(
            f"Opponent 3PT defense rank is {rank_num}/{rank_total} by 3PM allowed ({defense_label})."
        )
        if side == "OVER" and "Weak" in defense_label:
            reasons.append("Opponent has allowed above-average threes recently, which supports the OVER angle.")
        if side == "UNDER" and "Strong" in defense_label:
            reasons.append("Opponent limits made threes well, which supports the UNDER angle.")

    if latest_history_row is not None and not latest_history_row.empty:
        last5_avg = pd.to_numeric(pd.Series([latest_history_row.get("last_5_games_3pm_avg")]), errors="coerce").iloc[0]
        last5_3pa_avg = pd.to_numeric(pd.Series([latest_history_row.get("last_5_games_3pa_avg")]), errors="coerce").iloc[0]
        last5_3pa_trend = pd.to_numeric(pd.Series([latest_history_row.get("last_5_3pa_trend")]), errors="coerce").iloc[0]
        season_avg = pd.to_numeric(pd.Series([latest_history_row.get("season_avg_3pm")]), errors="coerce").iloc[0]
        projected_minutes = pd.to_numeric(pd.Series([latest_history_row.get("projected_minutes")]), errors="coerce").iloc[0]
        last5_minutes = pd.to_numeric(pd.Series([latest_history_row.get("last_5_minutes_avg")]), errors="coerce").iloc[0]
        usage_rate = pd.to_numeric(pd.Series([latest_history_row.get("usage_rate")]), errors="coerce").iloc[0]
        opp_3pm_allowed = pd.to_numeric(pd.Series([latest_history_row.get("opp_3pm_allowed")]), errors="coerce").iloc[0]
        injury_usage_bump = pd.to_numeric(pd.Series([latest_history_row.get("injury_usage_bump")]), errors="coerce").iloc[0]
        teammate_minutes_vacated = pd.to_numeric(pd.Series([latest_history_row.get("teammate_minutes_vacated")]), errors="coerce").iloc[0]
        teammate_usage_vacated = pd.to_numeric(pd.Series([latest_history_row.get("teammate_usage_vacated")]), errors="coerce").iloc[0]
        position = str(latest_history_row.get("position", "") or "").strip()

        if pd.notna(last5_avg) and pd.notna(season_avg):
            delta = last5_avg - season_avg
            direction = "above" if delta >= 0 else "below"
            reasons.append(f"Recent 5-game 3PM average is {abs(delta):.2f} {direction} season baseline.")
        if pd.notna(projected_minutes):
            reasons.append(f"Projected minutes: {projected_minutes:.1f}.")
        if pd.notna(last5_3pa_avg):
            reasons.append(f"Average 3PA over last 5 games: {last5_3pa_avg:.2f}.")
        if pd.notna(last5_3pa_trend) and abs(last5_3pa_trend) >= 0.05:
            trend_word = "upward" if last5_3pa_trend > 0 else "downward"
            reasons.append(f"Recent 3PA trend is {trend_word} ({last5_3pa_trend:+.2f} per game window).")
        if pd.notna(projected_minutes) and pd.notna(last5_minutes):
            minute_delta = projected_minutes - last5_minutes
            if abs(minute_delta) >= 1.5:
                direction = "higher" if minute_delta > 0 else "lower"
                reasons.append(
                    f"Projected minutes are {abs(minute_delta):.1f} {direction} than recent 5-game average ({last5_minutes:.1f})."
                )
        if pd.notna(usage_rate):
            reasons.append(f"Usage rate context: {usage_rate:.3f}.")
        if pd.notna(opp_3pm_allowed):
            reasons.append(f"Opponent recent 3PM allowed profile: {opp_3pm_allowed:.2f}.")
        if pd.notna(injury_usage_bump) and abs(injury_usage_bump) >= 0.01:
            reasons.append(f"Injury-adjusted usage bump in feature set: {injury_usage_bump:+.3f}.")
        if pd.notna(teammate_minutes_vacated) and teammate_minutes_vacated >= 4:
            pos_text = f" ({position})" if position else ""
            reasons.append(
                f"Rotation context{pos_text}: about {teammate_minutes_vacated:.1f} teammate minutes vacated."
            )
        if pd.notna(teammate_usage_vacated) and teammate_usage_vacated >= 0.01:
            reasons.append(f"Teammate usage vacated indicator: {teammate_usage_vacated:.3f}.")

    # Keep it concise and readable in the UI.
    deduped: list[str] = []
    for reason in reasons:
        if reason and reason not in deduped:
            deduped.append(reason)
    return deduped[:7]


# ---------------------------------------------------------------------------
# Analytical Classification Functions
# ---------------------------------------------------------------------------

def _as_float(value: object) -> float:
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]


def confidence_tier(model_prob: float) -> str:
    if pd.isna(model_prob):
        return "Unrated"
    if model_prob >= 0.85:
        return "Elite"
    if model_prob >= 0.75:
        return "High"
    if model_prob >= 0.65:
        return "Solid"
    return "Lean"


def juice_tier(american_odds: float) -> str:
    if pd.isna(american_odds):
        return "Unknown"
    if american_odds <= -350:
        return "Heavy Juice"
    if american_odds <= -220:
        return "High Juice"
    if american_odds <= -170:
        return "Juiced"
    if american_odds >= 180:
        return "Long Plus"
    if american_odds >= 130:
        return "Plus Money"
    return "Standard"


def line_profile(main_line_distance: float) -> str:
    if pd.isna(main_line_distance):
        return "Unknown"
    if main_line_distance <= 0.12:
        return "Near Main Line"
    if main_line_distance <= 0.22:
        return "Alt Line"
    return "Deep Alt Line"


def alt_pick_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    model_prob = _as_float(row.get("model_prob"))
    market_prob = _as_float(row.get("market_prob"))
    edge = _as_float(row.get("edge"))
    ev_unit = _as_float(row.get("expected_value_per_unit"))
    line = _as_float(row.get("line"))
    lam = _as_float(row.get("estimated_3pm_mean"))
    side = str(row.get("side", "")).upper().strip()

    if pd.notna(model_prob) and pd.notna(market_prob):
        reasons.append(f"Model {model_prob:.1%} vs market {market_prob:.1%} ({model_prob - market_prob:+.1%}).")
    elif pd.notna(edge):
        reasons.append(f"Model edge {edge:.1%} vs market.")

    if pd.notna(lam) and pd.notna(line) and side in {"OVER", "UNDER"}:
        reasons.append(f"Projection {lam:.2f} vs {side} {line:.1f}.")

    if pd.notna(ev_unit):
        reasons.append(f"EV {ev_unit:+.2f} per unit.")

    return " ".join(reasons[:3])


def build_alt_line_juice_table(value_board: pd.DataFrame, confidence_board: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not value_board.empty:
        value_frame = value_board.copy()
        value_frame["board_source"] = "Value"
        frames.append(value_frame)
    if not confidence_board.empty:
        confidence_frame = confidence_board.copy()
        confidence_frame["board_source"] = "Confidence"
        frames.append(confidence_frame)
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    for column in [
        "model_prob",
        "no_vig_prob",
        "implied_prob",
        "edge",
        "expected_value_per_unit",
        "line",
        "american_odds",
        "estimated_3pm_mean",
    ]:
        if column in combined.columns:
            combined[column] = pd.to_numeric(combined[column], errors="coerce")

    if "no_vig_prob" in combined.columns:
        combined["market_prob"] = combined["no_vig_prob"]
    else:
        combined["market_prob"] = np.nan
    if "implied_prob" in combined.columns:
        combined["market_prob"] = combined["market_prob"].where(
            combined["market_prob"].notna(),
            combined["implied_prob"],
        )

    combined["main_line_distance"] = (combined["market_prob"] - 0.5).abs()
    combined["alt_line_flag"] = combined["main_line_distance"] > 0.12
    combined["juice_flag"] = combined["american_odds"] <= -170

    alt_juice = combined[combined["alt_line_flag"] | combined["juice_flag"]].copy()
    if alt_juice.empty:
        return alt_juice

    alt_juice["edge_score"] = alt_juice["edge"].fillna(0.0)
    alt_juice["ev_score"] = alt_juice["expected_value_per_unit"].clip(lower=-0.20, upper=0.50).fillna(0.0)
    # Value logic applied to the juice board only: edge-heavy score with EV support.
    alt_juice["juice_value_score"] = (0.75 * alt_juice["edge_score"]) + (0.25 * alt_juice["ev_score"])

    key_cols = [
        column
        for column in ["slate_date", "player_name", "side", "line", "bookmaker", "american_odds"]
        if column in alt_juice.columns
    ]
    if key_cols:
        source_map = (
            alt_juice.groupby(key_cols, dropna=False)["board_source"]
            .agg(lambda values: "/".join(sorted(set(values))))
            .reset_index()
        )
        alt_juice.sort_values(
            ["juice_value_score", "edge", "model_prob", "expected_value_per_unit"],
            ascending=[False, False, False, False],
            inplace=True,
        )
        alt_juice = alt_juice.drop_duplicates(subset=key_cols, keep="first")
        alt_juice = alt_juice.drop(columns=["board_source"]).merge(
            source_map,
            on=key_cols,
            how="left",
        )

    alt_juice["confidence_tier"] = alt_juice["model_prob"].map(confidence_tier)
    alt_juice["juice_tier"] = alt_juice["american_odds"].map(juice_tier)
    alt_juice["line_profile"] = alt_juice["main_line_distance"].map(line_profile)
    alt_juice["why"] = alt_juice.apply(alt_pick_reason, axis=1)

    alt_juice.sort_values(
        ["juice_value_score", "edge", "model_prob", "expected_value_per_unit"],
        ascending=[False, False, False, False],
        inplace=True,
    )
    alt_juice.insert(0, "alt_rank", np.arange(1, len(alt_juice) + 1))
    return alt_juice


def _metric_from_primary_or_history(
    primary_row: pd.Series,
    latest_history_row: pd.Series | None,
    column: str,
) -> tuple[float, str]:
    primary_value = pd.to_numeric(pd.Series([primary_row.get(column)]), errors="coerce").iloc[0]
    if pd.notna(primary_value):
        return float(primary_value), "snapshot"
    if latest_history_row is not None and not latest_history_row.empty:
        history_value = pd.to_numeric(pd.Series([latest_history_row.get(column)]), errors="coerce").iloc[0]
        if pd.notna(history_value):
            return float(history_value), "history"
    return np.nan, ""


def defense_focus_items(
    primary_row: pd.Series,
    latest_history_row: pd.Series | None = None,
    history: pd.DataFrame | None = None,
) -> tuple[list[tuple[str, str]], str]:
    fields = [
        ("opp_3pm_allowed", "Opp 3PM Allowed", "{:.2f}"),
        ("opp_3pa_allowed", "Opp 3PA Allowed", "{:.2f}"),
        ("opp_def_rating", "Opp Def Rating", "{:.1f}"),
        ("opp_pace", "Opp Pace", "{:.1f}"),
        ("opp_pace_differential", "Opp Pace Diff", "{:+.2f}"),
    ]

    items: list[tuple[str, str]] = []
    source_labels: set[str] = set()
    for column, label, value_format in fields:
        value, source = _metric_from_primary_or_history(primary_row, latest_history_row, column)
        if pd.notna(value):
            items.append((label, value_format.format(value)))
            if source:
                source_labels.add(source)

    scheme_raw = str(primary_row.get("defensive_scheme_tendency", "") or "").strip()
    if not scheme_raw and latest_history_row is not None and not latest_history_row.empty:
        scheme_raw = str(latest_history_row.get("defensive_scheme_tendency", "") or "").strip()
        if scheme_raw:
            source_labels.add("history")
    elif scheme_raw:
        source_labels.add("snapshot")
    if scheme_raw:
        items.append(("Scheme", scheme_raw.title()))

    defense_rank = three_pt_defense_rank_context(primary_row, latest_history_row, history)
    if defense_rank:
        rank_num = int(defense_rank["rank"])
        rank_total = int(defense_rank["total"])
        items.append(("3PT Defense Rank", f"{rank_num}/{rank_total}"))
        items.append(("Defense Class", str(defense_rank["label"])))

    if not items:
        return [], ""

    source_note = ""
    if source_labels == {"snapshot"}:
        source_note = "From the board snapshot used for this pick."
    elif source_labels == {"history"}:
        source_note = "From latest available player history."
    elif source_labels:
        source_note = "Combined from board snapshot and latest player history."
    return items, source_note


# ---------------------------------------------------------------------------
# Tracked Bets Analysis
# ---------------------------------------------------------------------------

def tracked_bet_summary(tracked: pd.DataFrame) -> tuple[dict[str, float | int | None], pd.DataFrame]:
    if tracked.empty:
        return {
            "tracked_count": 0,
            "settled_count": 0,
            "win_rate": None,
            "profit_units": None,
        }, pd.DataFrame()

    tracked = tracked.copy()
    tracked["bet_won_numeric"] = to_numeric(tracked["bet_won"]) if "bet_won" in tracked.columns else np.nan
    tracked["profit_units_numeric"] = (
        to_numeric(tracked["profit_units"]) if "profit_units" in tracked.columns else np.nan
    )

    settled = tracked[tracked["bet_won_numeric"].notna()].copy()
    summary = {
        "tracked_count": int(len(tracked)),
        "settled_count": int(len(settled)),
        "win_rate": float(settled["bet_won_numeric"].mean()) if not settled.empty else None,
        "profit_units": float(settled["profit_units_numeric"].sum()) if not settled.empty else None,
    }

    bucket_column = "tracking_bucket" if "tracking_bucket" in settled.columns else "edge_bucket"
    if settled.empty or bucket_column not in settled.columns:
        return summary, pd.DataFrame()

    bucket_summary = (
        settled.groupby(bucket_column, dropna=False)
        .agg(
            bets=("bet_won_numeric", "size"),
            wins=("bet_won_numeric", "sum"),
            win_rate=("bet_won_numeric", "mean"),
            profit_units=("profit_units_numeric", "sum"),
        )
        .reset_index()
    )
    bucket_summary.rename(columns={bucket_column: "tracking_bucket"}, inplace=True)
    bucket_order = {
        "Below 5%": 0,
        "5% to <8%": 1,
        "8% to <12%": 2,
        "12%+": 3,
        "8% to <10%": 2,
        "10%+": 3,
    }
    bucket_summary["sort_order"] = bucket_summary["tracking_bucket"].map(
        lambda value: bucket_order.get(value, 99)
    )
    bucket_summary.sort_values("sort_order", inplace=True)
    bucket_summary.drop(columns=["sort_order"], inplace=True)
    return summary, bucket_summary


def daily_success_summary(tracked: pd.DataFrame) -> pd.DataFrame:
    if tracked.empty or "slate_date" not in tracked.columns or "bet_won" not in tracked.columns:
        return pd.DataFrame()

    frame = tracked.copy()
    frame["bet_won_numeric"] = to_numeric(frame["bet_won"])
    settled = frame[frame["bet_won_numeric"].notna()].copy()
    if settled.empty:
        return pd.DataFrame()

    summary = (
        settled.groupby("slate_date", dropna=False)
        .agg(
            bets=("bet_won_numeric", "size"),
            wins=("bet_won_numeric", "sum"),
            win_rate=("bet_won_numeric", "mean"),
            profit_units=("profit_units", lambda s: pd.to_numeric(s, errors="coerce").sum()),
        )
        .reset_index()
        .sort_values("slate_date", ascending=False)
    )
    return summary


def merge_daily_summaries(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    if primary.empty and fallback.empty:
        return pd.DataFrame()
    if primary.empty:
        return fallback.copy().sort_values("slate_date", ascending=False)
    if fallback.empty:
        return primary.copy().sort_values("slate_date", ascending=False)

    merged = pd.concat([primary.copy(), fallback.copy()], ignore_index=True, sort=False)
    merged["slate_date"] = merged["slate_date"].fillna("").astype(str)
    merged.drop_duplicates(subset=["slate_date"], keep="first", inplace=True)
    return merged.sort_values("slate_date", ascending=False)


# ---------------------------------------------------------------------------
# Parlay Analysis & Pricing
# ---------------------------------------------------------------------------

def parlay_summary(parlays: pd.DataFrame) -> tuple[dict[str, float | int | None], pd.DataFrame]:
    if parlays.empty:
        return {
            "tracked_days": 0,
            "settled_days": 0,
            "hit_rate": None,
            "profit_units": None,
        }, pd.DataFrame()

    frame = parlays.copy()
    frame["parlay_hit_numeric"] = to_numeric(frame["parlay_hit"]) if "parlay_hit" in frame.columns else np.nan
    frame["profit_units_numeric"] = (
        to_numeric(frame["profit_units"]) if "profit_units" in frame.columns else np.nan
    )
    settled = frame[frame["bet_result"].fillna("").astype(str) != ""].copy() if "bet_result" in frame.columns else pd.DataFrame()
    summary = {
        "tracked_days": int(len(frame)),
        "settled_days": int(len(settled)),
        "hit_rate": float(settled["parlay_hit_numeric"].mean()) if not settled.empty else None,
        "profit_units": float(settled["profit_units_numeric"].sum()) if not settled.empty else None,
    }
    if settled.empty:
        return summary, pd.DataFrame()
    cols = [c for c in ["slate_date", "bet_result", "parlay_american_display", "profit_units_numeric", "leg_1", "leg_2", "leg_3"] if c in settled.columns]
    detail = settled[cols].copy().sort_values("slate_date", ascending=False)
    if "profit_units_numeric" in detail.columns:
        detail.rename(columns={"profit_units_numeric": "profit_units"}, inplace=True)
    return summary, detail


def build_backfill_parlay_rows(confidence_backfill: pd.DataFrame, legs: int) -> pd.DataFrame:
    if confidence_backfill.empty or "slate_date" not in confidence_backfill.columns or "rank" not in confidence_backfill.columns:
        return pd.DataFrame()
    if legs <= 1:
        return pd.DataFrame()

    frame = confidence_backfill.copy()
    frame["rank"] = pd.to_numeric(frame.get("rank"), errors="coerce")
    frame = frame[frame["rank"].notna()].copy()
    frame["rank"] = frame["rank"].astype(int)
    output_rows: list[dict[str, object]] = []

    for slate_date, slate_rows in frame.groupby("slate_date"):
        top_n = slate_rows[slate_rows["rank"] <= legs].copy().sort_values("rank")
        if len(top_n) < legs:
            continue
        leg_results = top_n.get("bet_result", "").fillna("").astype(str).str.upper().tolist()
        if any(result == "LOSS" for result in leg_results):
            overall = "LOSS"
            parlay_hit = 0
        elif all(result == "WIN" for result in leg_results):
            overall = "WIN"
            parlay_hit = 1
        elif all(result in {"WIN", "PUSH"} for result in leg_results):
            overall = "PUSH"
            parlay_hit = np.nan
        else:
            overall = ""
            parlay_hit = np.nan

        row: dict[str, object] = {
            "recommendation_date": slate_date,
            "slate_date": slate_date,
            "legs": legs,
            "parlay_decimal_odds": np.nan,
            "parlay_american_display": "",
            "parlay_hit": parlay_hit,
            "bet_result": overall,
            "profit_units": np.nan,
            "settled_date": slate_date if overall else np.nan,
            "verified_source": "confidence_model_backfill",
        }
        for leg_idx in range(1, legs + 1):
            leg_row = top_n[top_n["rank"] == leg_idx].head(1)
            if leg_row.empty:
                row[f"leg_{leg_idx}"] = ""
                continue
            leg_pick = leg_row.iloc[0]
            row[f"leg_{leg_idx}"] = (
                f"{leg_pick.get('player_name', '')} {leg_pick.get('side', '')} {leg_pick.get('line', '')}"
            ).strip()
        output_rows.append(row)

    if not output_rows:
        return pd.DataFrame()
    return pd.DataFrame(output_rows).sort_values("slate_date", ascending=False)


def merge_parlay_rows(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    if primary.empty and fallback.empty:
        return pd.DataFrame()
    if primary.empty:
        return fallback.copy()
    if fallback.empty:
        return primary.copy()
    merged = pd.concat([primary.copy(), fallback.copy()], ignore_index=True, sort=False)
    dedupe = [col for col in ["slate_date", "legs"] if col in merged.columns]
    if dedupe:
        merged.drop_duplicates(subset=dedupe, keep="first", inplace=True)
    return merged.sort_values("slate_date", ascending=False)


def parlay_leg_table_for_slate(
    slate_date: str,
    legs: int,
    tracked_confidence: pd.DataFrame,
    backfill_confidence: pd.DataFrame,
) -> pd.DataFrame:
    tracked_slice = pd.DataFrame()
    if not tracked_confidence.empty and "slate_date" in tracked_confidence.columns:
        tracked_slice = tracked_confidence[
            tracked_confidence["slate_date"].fillna("").astype(str) == str(slate_date)
        ].copy()
        if "tracked_top_rank" in tracked_slice.columns:
            tracked_slice["tracked_top_rank"] = pd.to_numeric(tracked_slice["tracked_top_rank"], errors="coerce")
            tracked_slice = tracked_slice[tracked_slice["tracked_top_rank"] <= legs].copy()
            tracked_slice.sort_values("tracked_top_rank", inplace=True)

    source = tracked_slice
    rank_col = "tracked_top_rank"
    if source.empty and not backfill_confidence.empty and "slate_date" in backfill_confidence.columns:
        source = backfill_confidence[
            backfill_confidence["slate_date"].fillna("").astype(str) == str(slate_date)
        ].copy()
        if "rank" in source.columns:
            source["rank"] = pd.to_numeric(source["rank"], errors="coerce")
            source = source[source["rank"] <= legs].copy()
            source.sort_values("rank", inplace=True)
        rank_col = "rank"

    if source.empty:
        return pd.DataFrame()

    source = with_player_display(source)
    source["bet_result_text"] = source.get("bet_result", "").fillna("").astype(str).str.upper()
    source["Status"] = np.where(
        source["bet_result_text"] == "WIN",
        "✅",
        np.where(source["bet_result_text"] == "LOSS", "❌", np.where(source["bet_result_text"] == "PUSH", "➖", "⏳")),
    )
    source["Leg"] = pd.to_numeric(source.get(rank_col), errors="coerce")
    side_series = (
        source["side"].fillna("").astype(str)
        if "side" in source.columns
        else pd.Series([""] * len(source), index=source.index)
    )
    line_series = (
        source["line"].astype(str)
        if "line" in source.columns
        else pd.Series([""] * len(source), index=source.index)
    )
    odds_series = (
        pd.to_numeric(source["american_odds"], errors="coerce")
        if "american_odds" in source.columns
        else pd.Series([np.nan] * len(source), index=source.index)
    )
    source["Pick"] = side_series.str.strip() + " " + line_series.str.strip()
    source["Pick"] = source["Pick"].str.strip()
    source["Odds"] = odds_series.map(lambda v: f"{int(v):+d}" if pd.notna(v) else "N/A")

    detail = source[
        [
            col
            for col in ["Status", "Leg", "player_display_name", "Pick", "Odds", "bet_result_text"]
            if col in source.columns
        ]
    ].copy()
    detail.rename(
        columns={
            "player_display_name": "Player",
            "player_name": "Player",
            "bet_result_text": "Result",
        },
        inplace=True,
    )
    detail["Leg"] = detail["Leg"].map(lambda value: f"Leg {int(value)}" if pd.notna(value) else "")
    detail["Result"] = detail["Result"].replace({"": "PENDING"})
    return detail


def american_to_decimal(odds: float | int | str | None) -> float | None:
    value = pd.to_numeric(pd.Series([odds]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    value = float(value)
    if value > 0:
        return 1.0 + (value / 100.0)
    if value < 0:
        return 1.0 + (100.0 / abs(value))
    return None


def decimal_to_american(decimal_odds: float | None) -> str:
    if decimal_odds is None or pd.isna(decimal_odds) or decimal_odds <= 1:
        return "N/A"
    if decimal_odds >= 2:
        return f"+{int(round((decimal_odds - 1.0) * 100))}"
    return f"{int(round(-100.0 / (decimal_odds - 1.0)))}"


def probability_to_decimal(probability: float | int | str | None) -> float | None:
    value = pd.to_numeric(pd.Series([probability]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    value = float(value)
    if value <= 0 or value >= 1:
        return None
    return 1.0 / value


def leg_decimal_with_source(row: pd.Series) -> tuple[float | None, str]:
    market_decimal = american_to_decimal(row.get("american_odds"))
    if market_decimal is not None:
        return market_decimal, "market"
    no_vig_decimal = probability_to_decimal(row.get("no_vig_prob"))
    if no_vig_decimal is not None:
        return no_vig_decimal, "no_vig_implied"
    model_decimal = probability_to_decimal(row.get("model_prob"))
    if model_decimal is not None:
        return model_decimal, "model_implied"
    return None, "unavailable"


def top_rank_rows_for_slate(
    slate_date: str,
    tracked_rows: pd.DataFrame,
    backfill_rows: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    tracked_slice = pd.DataFrame()
    if not tracked_rows.empty and "slate_date" in tracked_rows.columns:
        tracked_slice = tracked_rows[tracked_rows["slate_date"].fillna("").astype(str) == str(slate_date)].copy()
        if "tracked_top_rank" in tracked_slice.columns:
            tracked_slice["tracked_top_rank"] = pd.to_numeric(tracked_slice["tracked_top_rank"], errors="coerce")
            tracked_slice = tracked_slice[tracked_slice["tracked_top_rank"] <= top_n].copy()
            tracked_slice["result_priority"] = np.where(
                tracked_slice.get("bet_result", "").fillna("").astype(str).str.strip().ne(""),
                1,
                0,
            )
            if "edge" in tracked_slice.columns:
                tracked_slice["edge"] = pd.to_numeric(tracked_slice["edge"], errors="coerce")
            tracked_slice.sort_values(
                ["tracked_top_rank", "result_priority", "edge"],
                ascending=[True, False, False],
                inplace=True,
            )
            tracked_slice.drop_duplicates(subset=["tracked_top_rank"], keep="first", inplace=True)
            tracked_slice.drop(columns=["result_priority"], inplace=True, errors="ignore")
        elif "rank" in tracked_slice.columns:
            tracked_slice["rank"] = pd.to_numeric(tracked_slice["rank"], errors="coerce")
            tracked_slice = tracked_slice[tracked_slice["rank"] <= top_n].copy()
            tracked_slice["result_priority"] = np.where(
                tracked_slice.get("bet_result", "").fillna("").astype(str).str.strip().ne(""),
                1,
                0,
            )
            if "edge" in tracked_slice.columns:
                tracked_slice["edge"] = pd.to_numeric(tracked_slice["edge"], errors="coerce")
            tracked_slice.sort_values(
                ["rank", "result_priority", "edge"],
                ascending=[True, False, False],
                inplace=True,
            )
            tracked_slice.drop_duplicates(subset=["rank"], keep="first", inplace=True)
            tracked_slice.drop(columns=["result_priority"], inplace=True, errors="ignore")

    if not tracked_slice.empty:
        return tracked_slice.head(top_n).copy()

    backfill_slice = pd.DataFrame()
    if not backfill_rows.empty and "slate_date" in backfill_rows.columns:
        backfill_slice = backfill_rows[backfill_rows["slate_date"].fillna("").astype(str) == str(slate_date)].copy()
        if "rank" in backfill_slice.columns:
            backfill_slice["rank"] = pd.to_numeric(backfill_slice["rank"], errors="coerce")
            backfill_slice = backfill_slice[backfill_slice["rank"] <= top_n].copy()
            backfill_slice.sort_values("rank", inplace=True)
    return backfill_slice.head(top_n).copy()


def parlay_pricing_from_rows(rows: pd.DataFrame) -> dict[str, float | str | None]:
    if rows.empty:
        return {
            "decimal_odds": None,
            "american_odds": "N/A",
            "model_hit_prob": None,
            "implied_prob": None,
            "ev_per_unit": None,
        }
    odds_decimals = []
    for value in rows.get("american_odds", pd.Series([np.nan] * len(rows), index=rows.index)).tolist():
        dec = american_to_decimal(value)
        if dec is None:
            return {
                "decimal_odds": None,
                "american_odds": "N/A",
                "model_hit_prob": None,
                "implied_prob": None,
                "ev_per_unit": None,
            }
        odds_decimals.append(dec)
    parlay_decimal = float(np.prod(odds_decimals))

    model_prob_series = pd.to_numeric(rows.get("model_prob"), errors="coerce")
    if model_prob_series.isna().any():
        model_hit_prob = None
    else:
        model_hit_prob = float(np.prod(model_prob_series.tolist()))
    implied_prob = (1.0 / parlay_decimal) if parlay_decimal > 0 else None
    ev_per_unit = (parlay_decimal * model_hit_prob - 1.0) if model_hit_prob is not None else None
    return {
        "decimal_odds": parlay_decimal,
        "american_odds": decimal_to_american(parlay_decimal),
        "model_hit_prob": model_hit_prob,
        "implied_prob": implied_prob,
        "ev_per_unit": ev_per_unit,
    }


def parlay_stake_guide(
    rows: pd.DataFrame,
    bankroll: float,
    target_profit: float,
) -> dict[str, float | None]:
    pricing = parlay_pricing_from_rows(rows)
    decimal_odds = pricing.get("decimal_odds")
    ev_per_unit = pricing.get("ev_per_unit")
    model_prob = pricing.get("model_hit_prob")
    implied_prob = pricing.get("implied_prob")

    stake_to_win_target = None
    if decimal_odds is not None and not pd.isna(decimal_odds) and decimal_odds > 1 and target_profit > 0:
        stake_to_win_target = float(target_profit / (float(decimal_odds) - 1.0))

    recommended_stake = None
    if (
        ev_per_unit is not None
        and model_prob is not None
        and implied_prob is not None
        and ev_per_unit > 0
        and bankroll > 0
    ):
        edge = max(0.0, float(model_prob) - float(implied_prob))
        edge_scale = min(1.0, edge / 0.12)
        recommended_stake = float(bankroll * (0.01 + (0.02 * edge_scale)))  # 1% to 3% bankroll

    return {
        "stake_to_win_target": stake_to_win_target,
        "recommended_stake": recommended_stake,
    }


def round_robin_pairs_for_slate(
    slate_date: str,
    tracked_rows: pd.DataFrame,
    backfill_rows: pd.DataFrame,
    board_label: str,
    top_n: int = 5,
) -> pd.DataFrame:
    rows = top_rank_rows_for_slate(slate_date, tracked_rows, backfill_rows, top_n=top_n)
    if rows.empty or len(rows) < 2:
        return pd.DataFrame()

    rank_series = pd.to_numeric(
        rows["tracked_top_rank"] if "tracked_top_rank" in rows.columns else rows.get("rank"),
        errors="coerce",
    )
    rows = rows.copy()
    rows["pair_rank"] = rank_series
    rows = rows[rows["pair_rank"].notna()].copy().sort_values("pair_rank")
    if len(rows) < 2:
        return pd.DataFrame()

    output: list[dict[str, object]] = []
    for combo_idx, (idx1, idx2) in enumerate(combinations(rows.index.tolist(), 2), start=1):
        leg1 = rows.loc[idx1]
        leg2 = rows.loc[idx2]
        leg1_rank = int(leg1["pair_rank"])
        leg2_rank = int(leg2["pair_rank"])
        leg1_res = str(leg1.get("bet_result", "") or "").upper()
        leg2_res = str(leg2.get("bet_result", "") or "").upper()

        if "LOSS" in {leg1_res, leg2_res}:
            pair_result = "LOSS"
        elif leg1_res in {"", "NAN"} or leg2_res in {"", "NAN"}:
            pair_result = "PENDING"
        elif leg1_res == "WIN" and leg2_res == "WIN":
            pair_result = "WIN"
        elif leg1_res in {"WIN", "PUSH"} and leg2_res in {"WIN", "PUSH"}:
            pair_result = "PUSH"
        else:
            pair_result = "PENDING"

        dec1, src1 = leg_decimal_with_source(leg1)
        dec2, src2 = leg_decimal_with_source(leg2)
        pair_decimal = (dec1 * dec2) if dec1 is not None and dec2 is not None else None
        if src1 == "market" and src2 == "market":
            odds_source = "market"
        elif "unavailable" in {src1, src2}:
            odds_source = "unavailable"
        elif "model_implied" in {src1, src2}:
            odds_source = "model_implied"
        else:
            odds_source = "no_vig_implied"

        if pair_result == "WIN" and pair_decimal is not None:
            profit_units = float(pair_decimal - 1.0)
        elif pair_result == "LOSS":
            profit_units = -1.0
        elif pair_result == "PUSH":
            profit_units = 0.0
        else:
            profit_units = np.nan

        status = "⏳"
        if pair_result == "WIN":
            status = "✅"
        elif pair_result == "LOSS":
            status = "❌"
        elif pair_result == "PUSH":
            status = "➖"

        output.append(
            {
                "board": board_label,
                "slate_date": slate_date,
                "Ticket": combo_idx,
                "Status": status,
                "Leg 1": f"#{leg1_rank} {leg1.get('player_name', '')} {leg1.get('side', '')} {leg1.get('line', '')}",
                "Leg 2": f"#{leg2_rank} {leg2.get('player_name', '')} {leg2.get('side', '')} {leg2.get('line', '')}",
                "Odds": decimal_to_american(pair_decimal) if pair_decimal is not None else "N/A",
                "Odds Source": odds_source,
                "Result": pair_result,
                "profit_units": profit_units,
            }
        )
    return pd.DataFrame(output)


def round_robin_history(
    tracked_rows: pd.DataFrame,
    backfill_rows: pd.DataFrame,
    board_label: str,
    top_n: int = 5,
) -> pd.DataFrame:
    dates: list[str] = []
    for frame in [tracked_rows, backfill_rows]:
        if not frame.empty and "slate_date" in frame.columns:
            dates.extend(frame["slate_date"].dropna().astype(str).tolist())
    unique_dates = sorted(set([d for d in dates if d]), reverse=True)
    chunks: list[pd.DataFrame] = []
    for slate_date in unique_dates:
        pairs = round_robin_pairs_for_slate(
            slate_date=slate_date,
            tracked_rows=tracked_rows,
            backfill_rows=backfill_rows,
            board_label=board_label,
            top_n=top_n,
        )
        if not pairs.empty:
            chunks.append(pairs)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def round_robin_stats(frame: pd.DataFrame) -> dict[str, float | int | None]:
    if frame.empty:
        return {
            "tracked_days": 0,
            "settled_tickets": 0,
            "hit_rate": None,
            "profit_units": None,
        }
    settled = frame[frame["Result"].isin(["WIN", "LOSS", "PUSH"])].copy()
    win_loss = settled[settled["Result"].isin(["WIN", "LOSS"])].copy()
    return {
        "tracked_days": int(frame["slate_date"].nunique()) if "slate_date" in frame.columns else 0,
        "settled_tickets": int(len(settled)),
        "hit_rate": float((win_loss["Result"] == "WIN").mean()) if not win_loss.empty else None,
        "profit_units": float(pd.to_numeric(settled["profit_units"], errors="coerce").sum()) if not settled.empty else None,
    }


def value_history_stats(history: pd.DataFrame) -> dict[str, float | int | None]:
    if history.empty or "slate_date" not in history.columns:
        return {
            "all_time": None,
            "last_7": None,
            "last_30": None,
            "days": 0,
        }

    frame = history.copy()
    frame["slate_date_dt"] = pd.to_datetime(frame["slate_date"], errors="coerce")
    frame["wins"] = pd.to_numeric(frame.get("wins"), errors="coerce")
    frame["bets"] = pd.to_numeric(frame.get("bets"), errors="coerce")
    frame = frame.dropna(subset=["slate_date_dt", "wins", "bets"])
    if frame.empty:
        return {"all_time": None, "last_7": None, "last_30": None, "days": 0}

    def weighted_win_rate(subset: pd.DataFrame) -> float | None:
        total_bets = float(subset["bets"].sum())
        total_wins = float(subset["wins"].sum())
        if total_bets <= 0:
            return None
        return total_wins / total_bets

    latest_date = frame["slate_date_dt"].max()
    last_7_cutoff = latest_date - pd.Timedelta(days=6)
    last_30_cutoff = latest_date - pd.Timedelta(days=29)

    return {
        "all_time": weighted_win_rate(frame),
        "last_7": weighted_win_rate(frame[frame["slate_date_dt"] >= last_7_cutoff]),
        "last_30": weighted_win_rate(frame[frame["slate_date_dt"] >= last_30_cutoff]),
        "days": int(frame["slate_date_dt"].nunique()),
    }


def value_history_detail_for_slate(
    slate_date: str,
    tracked_rows: pd.DataFrame,
    backfill_rows: pd.DataFrame,
) -> pd.DataFrame:
    tracked_slice = pd.DataFrame()
    if not tracked_rows.empty and "slate_date" in tracked_rows.columns:
        tracked_slice = tracked_rows[tracked_rows["slate_date"].fillna("").astype(str) == slate_date].copy()
        if "tracked_top_rank" in tracked_slice.columns:
            tracked_slice["tracked_top_rank"] = pd.to_numeric(tracked_slice["tracked_top_rank"], errors="coerce")
            tracked_slice = tracked_slice[tracked_slice["tracked_top_rank"] <= 10].copy()
            tracked_slice["result_priority"] = np.where(
                tracked_slice.get("bet_result", "").fillna("").astype(str).str.strip().ne(""),
                1,
                0,
            )
            if "edge" in tracked_slice.columns:
                tracked_slice["edge"] = pd.to_numeric(tracked_slice["edge"], errors="coerce")
            tracked_slice.sort_values(
                ["tracked_top_rank", "result_priority", "edge"],
                ascending=[True, False, False],
                inplace=True,
            )
            tracked_slice.drop_duplicates(subset=["tracked_top_rank"], keep="first", inplace=True)
            tracked_slice.drop(columns=["result_priority"], inplace=True, errors="ignore")
        elif "rank" in tracked_slice.columns:
            tracked_slice["rank"] = pd.to_numeric(tracked_slice["rank"], errors="coerce")
            tracked_slice["result_priority"] = np.where(
                tracked_slice.get("bet_result", "").fillna("").astype(str).str.strip().ne(""),
                1,
                0,
            )
            if "edge" in tracked_slice.columns:
                tracked_slice["edge"] = pd.to_numeric(tracked_slice["edge"], errors="coerce")
            tracked_slice.sort_values(
                ["rank", "result_priority", "edge"],
                ascending=[True, False, False],
                inplace=True,
            )
            tracked_slice.drop_duplicates(subset=["rank"], keep="first", inplace=True)
            tracked_slice.drop(columns=["result_priority"], inplace=True, errors="ignore")
            tracked_slice = tracked_slice.head(10)

    if tracked_slice.empty and not backfill_rows.empty and "slate_date" in backfill_rows.columns:
        tracked_slice = backfill_rows[backfill_rows["slate_date"].fillna("").astype(str) == slate_date].copy()
        if "rank" in tracked_slice.columns:
            tracked_slice["rank"] = pd.to_numeric(tracked_slice["rank"], errors="coerce")
            tracked_slice.sort_values("rank", inplace=True)
            tracked_slice = tracked_slice.head(10)

    if tracked_slice.empty:
        return tracked_slice

    tracked_slice["bet_result_text"] = tracked_slice.get("bet_result", "").fillna("").astype(str).str.upper()
    tracked_slice["bet_won_numeric"] = pd.to_numeric(tracked_slice.get("bet_won"), errors="coerce")
    tracked_slice["Status"] = np.where(
        tracked_slice["bet_result_text"] == "WIN",
        "✅",
        np.where(
            tracked_slice["bet_result_text"] == "LOSS",
            "❌",
            np.where(tracked_slice["bet_result_text"] == "PUSH", "➖", "⏳"),
        ),
    )

    slice_model_implied = _is_model_implied(tracked_slice)
    rank_column = "tracked_top_rank" if "tracked_top_rank" in tracked_slice.columns else "rank"
    tracked_slice = with_player_display(tracked_slice)
    details = tracked_slice[
        [
            col
            for col in [
                "Status",
                rank_column,
                "player_display_name",
                "side",
                "line",
                "american_odds",
                "model_prob",
                "edge",
                "actual_made_3pt",
                "bet_result_text",
            ]
            if col in tracked_slice.columns
        ]
    ].copy()
    details.rename(
        columns={
            rank_column: "Rank",
            "player_display_name": "Player",
            "player_name": "Player",
            "side": "Side",
            "line": "Line",
            "american_odds": "Odds",
            "model_prob": "Model Prob",
            "edge": "Edge",
            "actual_made_3pt": "Actual 3PM",
            "bet_result_text": "Result",
        },
        inplace=True,
    )
    if "Odds" in details.columns:
        details["Odds"] = details["Odds"].map(
            lambda v: (f"{int(v):+d} ★" if slice_model_implied else f"{int(v):+d}") if pd.notna(v) else ""
        )
    if "Model Prob" in details.columns:
        details["Model Prob"] = details["Model Prob"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
    if "Edge" in details.columns:
        details["Edge"] = details["Edge"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
    return details


# ---------------------------------------------------------------------------
# Charting & Data Visualization
# ---------------------------------------------------------------------------

def slate_date_options(ranked: pd.DataFrame, tracked: pd.DataFrame, *extra_frames: pd.DataFrame) -> list[str]:
    options: list[str] = []
    for frame in [tracked, ranked, *extra_frames]:
        if frame is not None and "slate_date" in frame.columns:
            values = frame["slate_date"].dropna().astype(str)
            options.extend([value for value in values if value])
    return sorted(set(options), reverse=True)


def plot_edge_bar(df: pd.DataFrame) -> go.Figure:
    """Bar chart: player name on x-axis, edge % on y-axis."""
    chart = df[["player_name", "edge"]].dropna().copy()
    chart["edge_pct"] = chart["edge"] * 100
    colors = [ACCENT if i % 2 == 0 else "#4a9ae8" for i in range(len(chart))]
    fig = go.Figure(
        go.Bar(
            x=chart["player_name"],
            y=chart["edge_pct"],
            marker_color=colors,
            marker_line_width=0,
            hovertemplate="%{x}<br>Edge: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        **CHART_LAYOUT,
        xaxis_title="Player",
        yaxis_title="Edge (%)",
        height=280,
    )
    fig.update_xaxes(tickangle=-30)
    return fig


def _is_model_implied(frame: pd.DataFrame) -> bool:
    """Return True if the odds in this frame are model-implied (not real sportsbook lines)."""
    if "odds_source" in frame.columns:
        return frame["odds_source"].fillna("").astype(str).str.contains("model_implied").any()
    # Fallback: backfill rows have no bookmaker column or all-null bookmaker
    if "bookmaker" not in frame.columns:
        return True
    return frame["bookmaker"].isna().all()


# ---------------------------------------------------------------------------
# Board & Table Rendering
# ---------------------------------------------------------------------------

def render_board_table(frame: pd.DataFrame, title: str, height: int | None = None) -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if frame.empty:
        st.info(f"No ranked props match the current filters for {title.lower()}.")
        return

    model_implied = _is_model_implied(frame)

    displayed_frame = with_player_display(frame)
    board_table = displayed_frame[
        [
            c
            for c in [
                "rank",
                "player_display_name",
                "team",
                "opponent",
                "side",
                "line",
                "bookmaker",
                "american_odds",
                "edge",
                "model_prob",
                "expected_value_per_unit",
            ]
            if c in displayed_frame.columns
        ]
    ].copy()
    board_table.rename(columns={"player_display_name": "Player", "player_name": "Player"}, inplace=True)
    if "rank" in board_table.columns:
        board_table["rank"] = np.arange(1, len(board_table) + 1)
    if "edge" in board_table.columns:
        board_table["edge"] = board_table["edge"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
    if "model_prob" in board_table.columns:
        board_table["model_prob"] = board_table["model_prob"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
    if "american_odds" in board_table.columns:
        board_table["american_odds"] = board_table["american_odds"].map(
            lambda v: (f"{int(v):+d} ★" if model_implied else f"{int(v):+d}") if pd.notna(v) else ""
        )
    if "expected_value_per_unit" in board_table.columns:
        board_table["expected_value_per_unit"] = board_table["expected_value_per_unit"].map(
            lambda v: f"{v:.2f}" if pd.notna(v) else ""
        )
    st.dataframe(
        board_table,
        use_container_width=True,
        hide_index=True,
        height=height or min(520, 56 + (len(board_table) * 35)),
    )
    captions = []
    if model_implied:
        captions.append("★ Odds are model-implied (no sportsbook line available for this date). Edge figures do not apply.")
    if "Player" in board_table.columns and board_table["Player"].astype(str).str.endswith(" *").any():
        captions.append("* marks players the model has historically predicted well on the holdout window.")
    for caption in captions:
        info_caption(caption)


def plot_trend(trend_df: pd.DataFrame, player_name: str) -> go.Figure:
    """Line chart: actual made 3s + rolling avg."""
    fig = go.Figure()
    if "made_3pt" in trend_df.columns:
        fig.add_trace(
            go.Scatter(
                x=trend_df.index,
                y=trend_df["made_3pt"],
                name="3PT Made",
                mode="lines+markers",
                line=dict(color=ACCENT, width=2),
                marker=dict(size=6, color=ACCENT),
                hovertemplate="%{x|%b %d}<br>Made: %{y}<extra></extra>",
            )
        )
    if "rolling_5" in trend_df.columns:
        fig.add_trace(
            go.Scatter(
                x=trend_df.index,
                y=trend_df["rolling_5"],
                name="5-Game Avg",
                mode="lines",
                line=dict(color=ACCENT2, width=2.5, dash="dot"),
                hovertemplate="%{x|%b %d}<br>Avg: %{y:.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        **CHART_LAYOUT,
        xaxis_title="Game Date",
        yaxis_title="3-Pointers Made",
        title=dict(
            text=f"{player_name} — Recent 3PT Output (Last 15 Games)",
            font=dict(color="#0d1f33", size=15, family="'DM Sans', sans-serif"),
            x=0,
        ),
        height=280,
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Main Streamlit Application Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Configure Streamlit page settings
    st.set_page_config(
        page_title="ML3pt Dashboard",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    style_app()

    # 2. Auto-settle tracked bets on first app load
    if not st.session_state.get("tracked_bets_auto_settled", False):
        settle_tracked_bets_from_history(TRACKED_CONFIDENCE_PATH, HISTORY_PATH)
        settle_top3_parlays(TOP3_PARLAY_PATH, TRACKED_CONFIDENCE_PATH)
        settle_top5_parlays(TOP5_PARLAY_PATH, TRACKED_CONFIDENCE_PATH)
        st.session_state["tracked_bets_auto_settled"] = True

    # 3. Load all data files from disk
    (
        _ranked,
        confident,
        history,
        _tracked_value,
        tracked_confidence,
        top3_parlays,
        top5_parlays,
        _backfill_top10,
        _backfill_summary,
        backfill_conf_top10,
        backfill_conf_summary,
    ) = load_data()

    st.markdown(
        """
        <div class="hero">
            <h1>🏀 ML3pt Confidence Dashboard</h1>
            <p>Confidence-first picks, hit-rate tracking, and player trend views powered by model outputs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if (
        confident.empty
        and history.empty
        and tracked_confidence.empty
        and top3_parlays.empty
        and top5_parlays.empty
        and backfill_conf_top10.empty
    ):
        st.error("No data found. Build the CSVs first, then rerun the app.")
        st.code(
            "python3 update_features.py --season 2025-26 --as-of-date 2026-03-06\n"
            "python3 rank_props.py --history data/player_game_3pt_features.csv "
            "--props data/player_threes_props.csv --confidence-output data/most_confident_3pt_props.csv"
        )
        return

    # 4. Clean and validate data: convert to numeric, normalize dates
    confident = confident.copy()
    history = history.copy()
    tracked_confidence = tracked_confidence.copy()
    top3_parlays = top3_parlays.copy()
    top5_parlays = top5_parlays.copy()
    backfill_conf_top10 = backfill_conf_top10.copy()
    backfill_conf_summary = backfill_conf_summary.copy()

    for frame in [
        confident,
        tracked_confidence,
        top3_parlays,
        top5_parlays,
        backfill_conf_top10,
        backfill_conf_summary,
    ]:
        if "slate_date" in frame.columns:
            frame["slate_date"] = frame["slate_date"].fillna("").astype(str)

    for column in ["edge", "expected_value_per_unit", "model_prob", "line", "estimated_3pm_mean"]:
        if column in confident.columns:
            confident[column] = to_numeric(confident[column])
        if column in tracked_confidence.columns:
            tracked_confidence[column] = to_numeric(tracked_confidence[column])

    # 5. Build sidebar controls: slate date filter and player selector
    slate_options = slate_date_options(confident, tracked_confidence, backfill_conf_top10)

    with st.sidebar:
        st.header("Controls")
        selected_slate = (
            st.selectbox(
                "Slate Date",
                options=slate_options,
                index=0,
            )
            if slate_options
            else ""
        )
        confidence_source = confident.copy()
        if selected_slate and "slate_date" in confidence_source.columns:
            confidence_source = confidence_source[confidence_source["slate_date"] == selected_slate].copy()

        available_players: list[str] = []
        if "player_name" in confidence_source.columns:
            available_players.extend(confidence_source["player_name"].dropna().astype(str).tolist())
        if "player_name" in history.columns:
            available_players.extend(history["player_name"].dropna().astype(str).tolist())
        available_players = sorted(set(available_players))

        default_player = available_players[0] if available_players else ""
        if not confidence_source.empty and {"player_name", "rank"}.issubset(confidence_source.columns):
            confidence_source["rank"] = to_numeric(confidence_source["rank"])
            top_ranked = confidence_source.sort_values("rank").iloc[0]
            top_player = str(top_ranked.get("player_name", ""))
            if top_player:
                default_player = top_player

        confidence_display_lookup: dict[str, str] = {}
        if not confidence_source.empty and "player_name" in confidence_source.columns:
            display_source = with_player_display(confidence_source)
            confidence_display_lookup = (
                display_source.drop_duplicates(subset=["player_name"], keep="first")
                .set_index("player_name")["player_display_name"]
                .to_dict()
            )

        selected_player = (
            st.selectbox(
                "Player",
                options=available_players,
                index=available_players.index(default_player) if default_player in available_players else 0,
                format_func=lambda name: confidence_display_lookup.get(name, name),
            )
            if available_players
            else ""
        )
        board_size_key = "confidence_board_size_slider"
        max_confidence_rows = 25
        default_confidence_rows = 10
        existing_confidence_rows = st.session_state.get(board_size_key, default_confidence_rows)
        try:
            existing_confidence_rows = int(existing_confidence_rows)
        except (TypeError, ValueError):
            existing_confidence_rows = default_confidence_rows
        existing_confidence_rows = max(1, min(existing_confidence_rows, max_confidence_rows))
        max_rows = st.slider(
            "Top Confidence Board Size (Rows)",
            min_value=1,
            max_value=max_confidence_rows,
            value=existing_confidence_rows,
            key=board_size_key,
        )
        st.caption(f"Controls how many ranked rows are shown (1 to {max_confidence_rows}).")
        apply_edge_filter = st.checkbox(
            "Apply Edge Filter",
            value=False,
            key="confidence_apply_edge_filter",
            help="Off by default. Turn on only if you want to exclude lower-edge confidence picks.",
        )
        min_edge = 0.0
        if apply_edge_filter:
            min_edge = st.slider(
                "Minimum Edge Filter",
                min_value=0.0,
                max_value=0.3,
                value=0.03,
                step=0.01,
                key="confidence_min_edge_slider",
            )
            st.caption("Filters out picks below this edge threshold (0.00 to 0.30).")
        else:
            st.caption("Confidence-first mode: edge filter is off.")

    filtered_confidence_board = confidence_source.copy()
    if apply_edge_filter and not filtered_confidence_board.empty and "edge" in filtered_confidence_board.columns:
        filtered_confidence_board.drop(
            filtered_confidence_board[filtered_confidence_board["edge"].fillna(-1) < min_edge].index,
            inplace=True,
        )
    if not filtered_confidence_board.empty and "rank" in filtered_confidence_board.columns:
        filtered_confidence_board.sort_values("rank", inplace=True)
    filtered_confidence_board = filtered_confidence_board.head(max_rows).copy()
    if not filtered_confidence_board.empty and "rank" in filtered_confidence_board.columns:
        filtered_confidence_board["rank"] = np.arange(1, len(filtered_confidence_board) + 1)
    source_count = len(confidence_source.index)
    shown_count = len(filtered_confidence_board.index)

    # 6. Compute metrics for the filtered confidence board display
    metric_source = filtered_confidence_board
    top_edge = metric_source["edge"].max() if "edge" in metric_source.columns and not metric_source.empty else None
    avg_ev = (
        metric_source["expected_value_per_unit"].mean()
        if "expected_value_per_unit" in metric_source.columns and not metric_source.empty
        else None
    )
    slate_count = len(metric_source.index)
    # 7. Aggregate tracked bets, parlay performance, and round-robin history
    full_confidence_tracking_summary, _ = tracked_bet_summary(tracked_confidence)
    confidence_daily_tracked = daily_success_summary(tracked_confidence)
    confidence_daily_history = merge_daily_summaries(confidence_daily_tracked, backfill_conf_summary)
    top3_backfill = build_backfill_parlay_rows(backfill_conf_top10, legs=3)
    top5_backfill = build_backfill_parlay_rows(backfill_conf_top10, legs=5)
    top3_parlays_display = merge_parlay_rows(top3_parlays, top3_backfill)
    top5_parlays_display = merge_parlay_rows(top5_parlays, top5_backfill)
    top3_parlay_totals, _top3_parlay_history = parlay_summary(top3_parlays_display)
    top5_parlay_totals, _top5_parlay_history = parlay_summary(top5_parlays_display)
    rr_conf_history = round_robin_history(tracked_confidence, backfill_conf_top10, board_label="Confidence", top_n=5)
    rr_conf_stats = round_robin_stats(rr_conf_history)
    alt_juice_board = build_alt_line_juice_table(pd.DataFrame(), confidence_source)
    using_backfill_confidence = (
        full_confidence_tracking_summary["settled_count"] == 0 and not backfill_conf_summary.empty
    )
    confidence_history_includes_backfill = (
        not backfill_conf_summary.empty and len(confidence_daily_history) > len(confidence_daily_tracked)
    )
    if using_backfill_confidence:
        backfill_conf_metric_win_rate = pd.to_numeric(backfill_conf_summary.get("win_rate"), errors="coerce").mean()
        backfill_conf_metric_settled = int(pd.to_numeric(backfill_conf_summary.get("bets"), errors="coerce").sum())
        full_confidence_tracking_summary["win_rate"] = (
            float(backfill_conf_metric_win_rate) if pd.notna(backfill_conf_metric_win_rate) else None
        )
        full_confidence_tracking_summary["settled_count"] = backfill_conf_metric_settled
        if confidence_daily_history.empty and not backfill_conf_summary.empty:
            confidence_daily_history = backfill_conf_summary.copy()

    # 8. Render summary metric cards: filtered props, edge, EV, and hit rate
    metric_columns = st.columns(4)
    with metric_columns[0]:
        render_metric("Filtered Props", str(slate_count), "Rows shown on the confidence board.", accent="#0d1f33")
    with metric_columns[1]:
        render_metric(
            "Best Edge",
            f"{top_edge:.1%}" if top_edge is not None and pd.notna(top_edge) else "N/A",
            "Highest model-vs-market gap after filters.",
            accent=ACCENT,
        )
    with metric_columns[2]:
        render_metric(
            "Avg EV / Unit",
            f"{avg_ev:.2f}" if avg_ev is not None and pd.notna(avg_ev) else "N/A",
            "Average expected return for displayed props.",
            accent=ACCENT3,
        )
    with metric_columns[3]:
        render_metric(
            "Confidence Top 10 Hit Rate",
            (
                f"{full_confidence_tracking_summary['win_rate']:.1%}"
                if full_confidence_tracking_summary["win_rate"] is not None
                and pd.notna(full_confidence_tracking_summary["win_rate"])
                else "N/A"
            ),
            (
                f"{full_confidence_tracking_summary['settled_count']} settled top-10 model-backfill picks stored locally since 2026-03-01."
                if using_backfill_confidence
                else f"{full_confidence_tracking_summary['settled_count']} settled top-10 confidence picks stored since 2026-03-01."
            ),
            accent=ACCENT2,
        )

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # 9. Render tabbed sections: tracking history, parlay details, round-robin, and alt-line picks
    history_tab_conf, history_tab_parlay, history_tab_rr, history_tab_alt = st.tabs(
        [
            "Confidence Top 10 History",
            "Confidence Parlay History",
            "Round Robin History",
            "Alt-Line Juice",
        ]
    )
    with history_tab_conf:
        if using_backfill_confidence or confidence_history_includes_backfill:
            info_caption("Showing local confidence model backfill history generated from historical game rows.")

        conf_stats = value_history_stats(confidence_daily_history)
        conf_cols = st.columns(4)
        with conf_cols[0]:
            render_metric(
                "All-Time Win Rate",
                f"{conf_stats['all_time']:.1%}" if conf_stats["all_time"] is not None and pd.notna(conf_stats["all_time"]) else "N/A",
                "Across all settled confidence top-10 slates.",
                accent=ACCENT,
            )
        with conf_cols[1]:
            render_metric(
                "Last 7 Days",
                f"{conf_stats['last_7']:.1%}" if conf_stats["last_7"] is not None and pd.notna(conf_stats["last_7"]) else "N/A",
                "Weighted by settled bet count.",
                accent=ACCENT3,
            )
        with conf_cols[2]:
            render_metric(
                "Last 30 Days",
                f"{conf_stats['last_30']:.1%}" if conf_stats["last_30"] is not None and pd.notna(conf_stats["last_30"]) else "N/A",
                "Weighted by settled bet count.",
                accent=ACCENT2,
            )
        with conf_cols[3]:
            tracked_days_conf = (
                tracked_confidence["slate_date"].fillna("").astype(str).replace("", np.nan).dropna().nunique()
                if "slate_date" in tracked_confidence.columns
                else 0
            )
            render_metric(
                "Tracked Days",
                str(int(max(conf_stats["days"], tracked_days_conf))),
                "Includes pending and settled confidence slates.",
                accent="#0d1f33",
            )

        st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)
        conf_history_table = confidence_daily_history.copy()
        if "slate_date" in conf_history_table.columns:
            conf_history_table["slate_display"] = pd.to_datetime(
                conf_history_table["slate_date"], errors="coerce"
            ).dt.strftime("%m/%d/%Y")
            conf_history_table["win_rate_display"] = conf_history_table["win_rate"].map(
                lambda v: f"{v:.1%}" if pd.notna(v) else "N/A"
            )
        else:
            conf_history_table["slate_date"] = ""
            conf_history_table["slate_display"] = ""
            conf_history_table["win_rate_display"] = "N/A"

        conf_slate_options: list[str] = []
        if "slate_date" in conf_history_table.columns:
            conf_slate_options.extend(conf_history_table["slate_date"].dropna().astype(str).tolist())
        if "slate_date" in tracked_confidence.columns:
            conf_slate_options.extend(tracked_confidence["slate_date"].dropna().astype(str).tolist())
        if "slate_date" in backfill_conf_top10.columns:
            conf_slate_options.extend(backfill_conf_top10["slate_date"].dropna().astype(str).tolist())
        conf_slate_options = sorted(set([value for value in conf_slate_options if value]), reverse=True)

        if not conf_slate_options:
            st.info("No confidence top-10 slates available yet.")
        else:
            selected_conf_slate = st.selectbox(
                "Confidence History Slate Date",
                options=conf_slate_options,
                format_func=lambda value: pd.to_datetime(value, errors="coerce").strftime("%m/%d/%Y")
                if pd.notna(pd.to_datetime(value, errors="coerce"))
                else value,
                key="confidence_history_slate_dropdown",
            )

            selected_conf_row = conf_history_table[conf_history_table["slate_date"] == selected_conf_slate].head(1)
            if not selected_conf_row.empty:
                conf_pick = selected_conf_row.iloc[0]
                info_caption(
                    f"Slate {conf_pick['slate_display']}: {int(conf_pick['wins'])}/{int(conf_pick['bets'])} wins "
                    f"({conf_pick['win_rate_display']})"
                )
            else:
                info_caption(
                    f"Slate {pd.to_datetime(selected_conf_slate, errors='coerce').strftime('%m/%d/%Y') if pd.notna(pd.to_datetime(selected_conf_slate, errors='coerce')) else selected_conf_slate}: pending settlement."
                )

            info_caption("Selected slate top-10 confidence picks with hit/miss status:")
            conf_detail_table = value_history_detail_for_slate(
                selected_conf_slate, tracked_confidence, backfill_conf_top10
            )
            if conf_detail_table.empty:
                st.info("No confidence top-10 detail rows available for this slate yet.")
            else:
                st.dataframe(conf_detail_table, use_container_width=True, hide_index=True)

    with history_tab_parlay:
        if top3_parlays_display.empty and top5_parlays_display.empty:
            st.info("No confidence parlay history yet.")
        else:
            control_left, control_right = st.columns(2)
            with control_left:
                parlay_bankroll = st.number_input(
                    "Parlay Bankroll ($)",
                    min_value=50.0,
                    max_value=1_000_000.0,
                    value=1000.0,
                    step=50.0,
                    key="parlay_bankroll_input",
                )
            with control_right:
                target_profit = st.number_input(
                    "Target Profit ($)",
                    min_value=10.0,
                    max_value=10_000.0,
                    value=100.0,
                    step=10.0,
                    key="parlay_target_profit_input",
                )

            mcol1, mcol2 = st.columns(2)
            with mcol1:
                render_metric(
                    "Top 3 Parlay Hit Rate",
                    (
                        f"{top3_parlay_totals['hit_rate']:.1%}"
                        if top3_parlay_totals["hit_rate"] is not None and pd.notna(top3_parlay_totals["hit_rate"])
                        else "N/A"
                    ),
                    f"All 3 must hit. {top3_parlay_totals['settled_days']} settled of {top3_parlay_totals['tracked_days']} tracked.",
                    accent=ACCENT,
                )
            with mcol2:
                render_metric(
                    "Top 5 Parlay Hit Rate",
                    (
                        f"{top5_parlay_totals['hit_rate']:.1%}"
                        if top5_parlay_totals["hit_rate"] is not None and pd.notna(top5_parlay_totals["hit_rate"])
                        else "N/A"
                    ),
                    f"All 5 must hit. {top5_parlay_totals['settled_days']} settled of {top5_parlay_totals['tracked_days']} tracked.",
                    accent=ACCENT3,
                )

            all_parlay_dates: list[str] = []
            for frame in [top3_parlays_display, top5_parlays_display]:
                if not frame.empty and "slate_date" in frame.columns:
                    all_parlay_dates.extend(frame["slate_date"].dropna().astype(str).tolist())
            all_parlay_dates = sorted(set([d for d in all_parlay_dates if d]), reverse=True)

            if not all_parlay_dates:
                st.info("No confidence parlay dates available yet.")
            else:
                selected_parlay_slate = st.selectbox(
                    "Confidence Parlay Slate Date",
                    options=all_parlay_dates,
                    key="confidence_parlay_slate_dropdown",
                )

                pleft, pright = st.columns(2, gap="large")
                with pleft:
                    st.markdown("<div class='section-title'>Top 3 Parlay (All Must Hit)</div>", unsafe_allow_html=True)
                    top3_row = top3_parlays_display[
                        top3_parlays_display["slate_date"].fillna("").astype(str) == str(selected_parlay_slate)
                    ].head(1)
                    if not top3_row.empty:
                        top3_odds = str(top3_row.iloc[0].get("parlay_american_display", "") or "").strip() or "N/A"
                        top3_result = str(top3_row.iloc[0].get("bet_result", "") or "").strip().upper() or "PENDING"
                        info_caption(f"Parlay odds: {top3_odds} | Parlay result: {top3_result}")
                    top3_detail = parlay_leg_table_for_slate(
                        selected_parlay_slate,
                        legs=3,
                        tracked_confidence=tracked_confidence,
                        backfill_confidence=backfill_conf_top10,
                    )
                    if top3_detail.empty:
                        st.info("No top-3 parlay row for this slate.")
                    else:
                        top3_legs = top_rank_rows_for_slate(
                            selected_parlay_slate,
                            tracked_confidence,
                            backfill_conf_top10,
                            top_n=3,
                        )
                        top3_pricing = parlay_pricing_from_rows(top3_legs)
                        top3_stake = parlay_stake_guide(top3_legs, bankroll=parlay_bankroll, target_profit=target_profit)
                        info_caption(
                            f"Whole parlay odds: {top3_pricing['american_odds']} | "
                            f"Model hit: {top3_pricing['model_hit_prob']:.1%} | "
                            f"EV/$1: {top3_pricing['ev_per_unit']:.2f}"
                            if top3_pricing["model_hit_prob"] is not None and top3_pricing["ev_per_unit"] is not None
                            else f"Whole parlay odds: {top3_pricing['american_odds']} | Model hit: N/A | EV/$1: N/A"
                        )
                        info_caption(
                            f"Stake to win ${target_profit:,.0f}: "
                            f"${top3_stake['stake_to_win_target']:,.2f}"
                            if top3_stake["stake_to_win_target"] is not None
                            else f"Stake to win ${target_profit:,.0f}: N/A"
                        )
                        info_caption(
                            f"Suggested whole-parlay stake (aggressive, 1%-3% cap): "
                            f"${top3_stake['recommended_stake']:,.2f}"
                            if top3_stake["recommended_stake"] is not None
                            else "Suggested whole-parlay stake (aggressive, 1%-3% cap): $0.00"
                        )
                        st.dataframe(top3_detail, use_container_width=True, hide_index=True)
                with pright:
                    st.markdown("<div class='section-title'>Top 5 Parlay (All Must Hit)</div>", unsafe_allow_html=True)
                    top5_row = top5_parlays_display[
                        top5_parlays_display["slate_date"].fillna("").astype(str) == str(selected_parlay_slate)
                    ].head(1)
                    if not top5_row.empty:
                        top5_odds = str(top5_row.iloc[0].get("parlay_american_display", "") or "").strip() or "N/A"
                        top5_result = str(top5_row.iloc[0].get("bet_result", "") or "").strip().upper() or "PENDING"
                        info_caption(f"Parlay odds: {top5_odds} | Parlay result: {top5_result}")
                    top5_detail = parlay_leg_table_for_slate(
                        selected_parlay_slate,
                        legs=5,
                        tracked_confidence=tracked_confidence,
                        backfill_confidence=backfill_conf_top10,
                    )
                    if top5_detail.empty:
                        st.info("No top-5 parlay row for this slate.")
                    else:
                        top5_legs = top_rank_rows_for_slate(
                            selected_parlay_slate,
                            tracked_confidence,
                            backfill_conf_top10,
                            top_n=5,
                        )
                        top5_pricing = parlay_pricing_from_rows(top5_legs)
                        top5_stake = parlay_stake_guide(top5_legs, bankroll=parlay_bankroll, target_profit=target_profit)
                        info_caption(
                            f"Whole parlay odds: {top5_pricing['american_odds']} | "
                            f"Model hit: {top5_pricing['model_hit_prob']:.1%} | "
                            f"EV/$1: {top5_pricing['ev_per_unit']:.2f}"
                            if top5_pricing["model_hit_prob"] is not None and top5_pricing["ev_per_unit"] is not None
                            else f"Whole parlay odds: {top5_pricing['american_odds']} | Model hit: N/A | EV/$1: N/A"
                        )
                        info_caption(
                            f"Stake to win ${target_profit:,.0f}: "
                            f"${top5_stake['stake_to_win_target']:,.2f}"
                            if top5_stake["stake_to_win_target"] is not None
                            else f"Stake to win ${target_profit:,.0f}: N/A"
                        )
                        info_caption(
                            f"Suggested whole-parlay stake (aggressive, 1%-3% cap): "
                            f"${top5_stake['recommended_stake']:,.2f}"
                            if top5_stake["recommended_stake"] is not None
                            else "Suggested whole-parlay stake (aggressive, 1%-3% cap): $0.00"
                        )
                        st.dataframe(top5_detail, use_container_width=True, hide_index=True)

    with history_tab_rr:
        if rr_conf_history.empty:
            st.info("No confidence round robin history yet.")
        else:
            render_metric(
                "Confidence Top 5 RR Hit Rate",
                f"{rr_conf_stats['hit_rate']:.1%}" if rr_conf_stats["hit_rate"] is not None else "N/A",
                (
                    f"{rr_conf_stats['settled_tickets']} settled tickets across "
                    f"{rr_conf_stats['tracked_days']} days, "
                    f"{rr_conf_stats['profit_units']:.2f} units."
                    if rr_conf_stats["profit_units"] is not None
                    else f"{rr_conf_stats['settled_tickets']} settled tickets across {rr_conf_stats['tracked_days']} days."
                ),
                accent=ACCENT3,
            )

            rr_dates: list[str] = []
            if "slate_date" in rr_conf_history.columns:
                rr_dates.extend(rr_conf_history["slate_date"].dropna().astype(str).tolist())
            rr_dates = sorted(set([d for d in rr_dates if d]), reverse=True)
            if not rr_dates:
                st.info("No round robin slate dates available yet.")
            else:
                selected_rr_slate = st.selectbox(
                    "Round Robin Slate Date",
                    options=rr_dates,
                    key="round_robin_slate_dropdown",
                )
                st.markdown("<div class='section-title'>Confidence Top 5 (2-leg RR)</div>", unsafe_allow_html=True)
                rr_conf_slice = rr_conf_history[
                    rr_conf_history["slate_date"].fillna("").astype(str) == str(selected_rr_slate)
                ].copy()
                if rr_conf_slice.empty:
                    st.info("No confidence round robin rows for this slate.")
                else:
                    rr_conf_view = rr_conf_slice[
                        [
                            c
                            for c in ["Status", "Ticket", "Leg 1", "Leg 2", "Odds", "Odds Source", "Result"]
                            if c in rr_conf_slice.columns
                        ]
                    ].copy()
                    st.dataframe(rr_conf_view, use_container_width=True, hide_index=True)

    with history_tab_alt:
        if alt_juice_board.empty:
            st.info("No alternate-line or juiced picks found on the currently selected slate.")
        else:
            alt_metrics = st.columns(4)
            avg_alt_conf = alt_juice_board["model_prob"].mean() if "model_prob" in alt_juice_board.columns else np.nan
            avg_alt_edge = alt_juice_board["edge"].mean() if "edge" in alt_juice_board.columns else np.nan
            best_alt_edge = alt_juice_board["edge"].max() if "edge" in alt_juice_board.columns else np.nan
            best_alt_player = (
                str(alt_juice_board.iloc[0].get("player_name", "N/A")) if not alt_juice_board.empty else "N/A"
            )
            with alt_metrics[0]:
                render_metric(
                    "Alt/Juice Picks",
                    str(len(alt_juice_board)),
                    "Rows identified as alternate line or high-juice.",
                    accent="#0d1f33",
                )
            with alt_metrics[1]:
                render_metric(
                    "Avg Confidence",
                    f"{avg_alt_conf:.1%}" if pd.notna(avg_alt_conf) else "N/A",
                    "Mean model hit probability for these picks.",
                    accent=ACCENT,
                )
            with alt_metrics[2]:
                render_metric(
                    "Avg Edge",
                    f"{avg_alt_edge:.1%}" if pd.notna(avg_alt_edge) else "N/A",
                    "Average model-vs-market gap.",
                    accent=ACCENT3,
                )
            with alt_metrics[3]:
                render_metric(
                    "Best Odds Beat",
                    f"{best_alt_edge:.1%}" if pd.notna(best_alt_edge) else "N/A",
                    f"Top row: {best_alt_player}",
                    accent=ACCENT2,
                )

            info_caption("Sorted by juice value score (edge-heavy with EV support) so the strongest market-beating opportunities show first.")
            alt_best = alt_juice_board.head(5).copy()
            alt_best_view = alt_best[
                [
                    c
                    for c in [
                        "alt_rank",
                        "player_name",
                        "side",
                        "line",
                        "american_odds",
                        "juice_value_score",
                        "model_prob",
                        "market_prob",
                        "edge",
                        "confidence_tier",
                        "line_profile",
                    ]
                    if c in alt_best.columns
                ]
            ].copy()
            if "american_odds" in alt_best_view.columns:
                alt_best_view["american_odds"] = alt_best_view["american_odds"].map(
                    lambda v: f"{int(v):+d}" if pd.notna(v) else ""
                )
            for percent_col in ["model_prob", "market_prob", "edge"]:
                if percent_col in alt_best_view.columns:
                    alt_best_view[percent_col] = alt_best_view[percent_col].map(
                        lambda v: f"{v:.1%}" if pd.notna(v) else ""
                    )
            if "juice_value_score" in alt_best_view.columns:
                alt_best_view["juice_value_score"] = alt_best_view["juice_value_score"].map(
                    lambda v: f"{v:.3f}" if pd.notna(v) else ""
                )
            st.dataframe(alt_best_view, use_container_width=True, hide_index=True)

            info_caption("Classification + reasoning for each alternate-line/juice pick:")
            alt_full = alt_juice_board[
                [
                    c
                    for c in [
                        "alt_rank",
                        "board_source",
                        "player_name",
                        "team",
                        "opponent",
                        "side",
                        "line",
                        "bookmaker",
                        "american_odds",
                        "juice_value_score",
                        "model_prob",
                        "market_prob",
                        "edge",
                        "expected_value_per_unit",
                        "confidence_tier",
                        "juice_tier",
                        "line_profile",
                        "why",
                    ]
                    if c in alt_juice_board.columns
                ]
            ].copy()
            if "american_odds" in alt_full.columns:
                alt_full["american_odds"] = alt_full["american_odds"].map(
                    lambda v: f"{int(v):+d}" if pd.notna(v) else ""
                )
            for percent_col in ["model_prob", "market_prob", "edge"]:
                if percent_col in alt_full.columns:
                    alt_full[percent_col] = alt_full[percent_col].map(
                        lambda v: f"{v:.1%}" if pd.notna(v) else ""
                    )
            if "juice_value_score" in alt_full.columns:
                alt_full["juice_value_score"] = alt_full["juice_value_score"].map(
                    lambda v: f"{v:.3f}" if pd.notna(v) else ""
                )
            if "expected_value_per_unit" in alt_full.columns:
                alt_full["expected_value_per_unit"] = alt_full["expected_value_per_unit"].map(
                    lambda v: f"{v:+.2f}" if pd.notna(v) else ""
                )
            st.dataframe(alt_full, use_container_width=True, hide_index=True)

    info_caption(
        f"Showing {shown_count} of {source_count} available picks for {selected_slate or 'selected slate'} "
        f"({'min edge ' + format(min_edge, '.2f') if apply_edge_filter else 'edge filter off'}, board size {max_rows})."
    )

    # 10. Render main confidence board and edge distribution chart
    render_board_table(filtered_confidence_board, "Confidence Board")

    chart_source = filtered_confidence_board
    edge_chart_data = chart_source[["player_name", "edge"]].dropna() if not chart_source.empty else pd.DataFrame()
    if not edge_chart_data.empty:
        st.markdown("<div class='caption-text' style='margin-top:1rem;'>Top Edges by Player</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_edge_bar(chart_source), use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # 11. Render player detail panel with trend charts and game log
    st.markdown("<div class='section-title'>Player Focus</div>", unsafe_allow_html=True)
    if not selected_player:
        st.info("Add player data to use the player detail view.")
    else:
        st.markdown(f"<div style='font-size:1.2rem; font-weight:700; color:#0d1f33; margin-bottom:0.55rem;'>{selected_player}</div>", unsafe_allow_html=True)
        focus_props = player_prop_rows(confidence_source, selected_player)
        focus_history = player_history(history, selected_player)

        if not focus_props.empty:
            primary_row = focus_props.iloc[0]
            side = primary_row.get("side", "")
            line = primary_row.get("line", "")
            edge = primary_row.get("edge")
            book = primary_row.get("bookmaker", "")
            team = str(primary_row.get("team", "") or "").strip()
            opponent = str(primary_row.get("opponent", "") or "").strip()
            matchup = f"{team} vs {opponent}" if team and opponent else team or opponent
            edge_str = f" · **{edge:.1%} edge**" if pd.notna(edge) else ""
            st.markdown(
                f"<div style='font-size:0.92rem; color:#3a5270; margin-bottom:0.8rem; padding:0.8rem 1rem; background:#f0f5fc; border-radius:10px; border-left:3px solid {ACCENT};'>"
                f"<strong>{side} {line}</strong> at {book} &nbsp;|&nbsp; {matchup}{edge_str}</div>",
                unsafe_allow_html=True,
            )

            mean_projection = primary_row.get("estimated_3pm_mean")
            latest_history_row = focus_history.iloc[-1] if not focus_history.empty else None
            if pd.notna(mean_projection):
                info_caption(f"Model mean projection: {float(mean_projection):.2f} made threes")
            avg_3pa_last5 = (
                pd.to_numeric(pd.Series([latest_history_row.get("last_5_games_3pa_avg")]), errors="coerce").iloc[0]
                if latest_history_row is not None and not latest_history_row.empty
                else np.nan
            )
            if pd.notna(avg_3pa_last5):
                info_caption(f"Average 3-point attempts (last 5 games): {float(avg_3pa_last5):.2f}")

            reasoning = pick_reasoning_lines(primary_row, latest_history_row, history)
            if reasoning:
                reasoning_html = "".join([f"<div class='reasoning-item'>• {line}</div>" for line in reasoning])
                st.markdown(
                    "<div class='reasoning-card'>"
                    "<div class='reasoning-title'>Why This Pick</div>"
                    f"{reasoning_html}"
                    "</div>",
                    unsafe_allow_html=True,
                )
            defense_items, defense_note = defense_focus_items(primary_row, latest_history_row, history)
            if defense_items:
                defense_html = "".join(
                    [
                        f"<div class='defense-item'><span>{label}</span><strong>{value}</strong></div>"
                        for label, value in defense_items
                    ]
                )
                note_html = f"<div class='defense-note'>{defense_note}</div>" if defense_note else ""
                st.markdown(
                    "<div class='defense-card'>"
                    "<div class='defense-title'>Opponent Defense Snapshot</div>"
                    f"<div class='defense-grid'>{defense_html}</div>"
                    f"{note_html}"
                    "</div>",
                    unsafe_allow_html=True,
                )

            prop_table = focus_props[
                [c for c in ["rank", "side", "line", "bookmaker", "american_odds", "edge", "expected_value_per_unit", "model_prob"]
                 if c in focus_props.columns]
            ].copy()
            for col in ["edge", "model_prob"]:
                if col in prop_table.columns:
                    prop_table[col] = prop_table[col].map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
            if "american_odds" in prop_table.columns:
                prop_table["american_odds"] = prop_table["american_odds"].map(
                    lambda v: f"{int(v):+d}" if pd.notna(v) else ""
                )
            if "expected_value_per_unit" in prop_table.columns:
                prop_table["expected_value_per_unit"] = prop_table["expected_value_per_unit"].map(
                    lambda v: f"{v:.2f}" if pd.notna(v) else ""
                )
            st.dataframe(prop_table, use_container_width=True, hide_index=True)
        else:
            st.info("This player is not currently on the confidence board.")

        if not focus_history.empty:
            trend = focus_history[
                [c for c in ["game_date", "player_made_3pt", "rolling_5_avg"] if c in focus_history.columns]
            ].copy()
            trend = trend.rename(columns={"game_date": "date", "player_made_3pt": "made_3pt", "rolling_5_avg": "rolling_5"})
            trend = trend.set_index("date")
            st.plotly_chart(
                plot_trend(trend, selected_player),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            if "opponent" in focus_history.columns and "player_made_3pt" in focus_history.columns:
                recent_log = focus_history[["game_date", "opponent", "player_made_3pt"]].copy()
                recent_log["game_date"] = recent_log["game_date"].dt.strftime("%Y-%m-%d")
                recent_log.rename(columns={"game_date": "Date", "opponent": "Opponent", "player_made_3pt": "3PM"}, inplace=True)
                st.dataframe(recent_log.sort_values("Date", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("No historical game log found for this player.")


if __name__ == "__main__":
    main()
