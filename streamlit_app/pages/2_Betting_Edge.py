import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Betting Edge — Augusta Model", layout="wide", page_icon="⛳")
PROCESSED = Path(__file__).parent.parent.parent / "data" / "processed"

@st.cache_data(ttl=1800)
def load_edge():
    p = PROCESSED / "edge_2026.csv"
    if not p.exists(): return None
    return pd.read_csv(p)

edge = load_edge()
if edge is None:
    st.error("Edge data not found. Run: `python run_production.py`")
    st.stop()

st.title("Betting Edge — Value vs Market")
st.caption("Market odds source: **DraftKings**")

threshold = st.sidebar.slider("Edge threshold (%)", 10, 50, 20, 5)
st.sidebar.markdown("Only shows plays with edge above this threshold")

# Detect column names (handle different CSV formats)
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

model_t10 = find_col(edge, ["model_top10_pct", "model_top10"])
mkt_t10 = find_col(edge, ["market_top10_pct", "dk_fair_top10_pct"])
edge_t10 = find_col(edge, ["top10_edge_pct", "top10_kelly_edge"])
model_win = find_col(edge, ["model_win_pct", "model_win"])
mkt_win = find_col(edge, ["market_win_pct", "dk_fair_win_pct"])
edge_win = find_col(edge, ["win_edge_pct", "win_kelly_edge"])
model_t20 = find_col(edge, ["model_top20_pct", "model_top20"])

tab_t10, tab_win = st.tabs(["Top-10 Market", "Win Market"])

def show_market_tab(df, model_col, market_col, edge_col, threshold_pct):
    if not all([model_col, market_col, edge_col]):
        st.info("Required columns not found in edge data")
        return

    filtered = df[df[edge_col] > threshold_pct / 100].sort_values(edge_col, ascending=False).copy()
    display_cols = ["player_name", model_col, market_col, edge_col]
    if "fade_reason" in df.columns:
        display_cols.append("fade_reason")

    display = filtered[display_cols].copy()
    names = ["Player", "Model %", "Market %", "Edge %"]
    if "fade_reason" in display.columns:
        names.append("Notes")
    display.columns = names
    display["Model %"] = display["Model %"].map("{:.1%}".format)
    display["Market %"] = display["Market %"].map("{:.1%}".format)
    display["Edge %"] = display["Edge %"].map("{:+.0%}".format)

    if len(display) == 0:
        st.info(f"No plays above {threshold_pct}% edge threshold")
    else:
        st.dataframe(display, use_container_width=True, height=600)
        st.caption(f"{len(display)} plays above {threshold_pct}% edge")

    # Fades
    fades = df[df[edge_col] < -0.30].sort_values(edge_col).head(10).copy()
    if len(fades) > 0:
        st.markdown("#### Fade Candidates (model below market)")
        fade_disp = fades[["player_name", model_col, market_col, edge_col]].copy()
        fade_disp.columns = ["Player", "Model %", "Market %", "Edge %"]
        fade_disp["Model %"] = fade_disp["Model %"].map("{:.1%}".format)
        fade_disp["Market %"] = fade_disp["Market %"].map("{:.1%}".format)
        fade_disp["Edge %"] = fade_disp["Edge %"].map("{:+.0%}".format)
        st.dataframe(fade_disp, use_container_width=True)

    st.markdown("---")
    st.caption("This model is for research and entertainment purposes only. Not financial advice. "
               "Past backtest performance does not guarantee future results.")

with tab_t10:
    show_market_tab(edge, model_t10, mkt_t10, edge_t10, threshold)

with tab_win:
    show_market_tab(edge, model_win, mkt_win, edge_win, threshold)
