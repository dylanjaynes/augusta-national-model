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
    st.error("Run: `python run_2026_predictions.py`")
    st.stop()

st.title("Betting Edge — Value vs Market")

odds_src = edge["odds_source"].iloc[0] if "odds_source" in edge.columns else "Unknown"
st.caption(f"Market odds source: **{odds_src}**")

# Sidebar filter
threshold = st.sidebar.slider("Edge threshold (%)", 10, 50, 20, 5)
st.sidebar.markdown("Only shows plays with edge above this threshold")

tab_t10, tab_t20, tab_win = st.tabs(["Top-10 Market", "Top-20 Market", "Win Market"])

def show_market_tab(df, model_col, market_col, edge_col, threshold_pct):
    filtered = df[df[edge_col] > threshold_pct/100].sort_values(edge_col, ascending=False).copy()

    display = filtered[["player_name", model_col, market_col, edge_col, "recommendation"]].copy()
    display.columns = ["Player", "Model %", "Market %", "Edge %", "Recommendation"]
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
    st.caption("⚠️ This model is for research and entertainment purposes only. Not financial advice. "
               "Past backtest performance does not guarantee future results. Gamble responsibly.")

with tab_t10:
    show_market_tab(edge, "model_top10_pct", "market_top10_pct", "top10_edge_pct", threshold)

with tab_t20:
    show_market_tab(edge, "model_top20_pct", "market_top20_pct", "top20_edge_pct", threshold)

with tab_win:
    show_market_tab(edge, "model_win_pct", "market_win_pct", "win_edge_pct", threshold)
