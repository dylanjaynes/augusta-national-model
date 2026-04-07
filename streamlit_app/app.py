import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Augusta Model — 2026 Masters", layout="wide", page_icon="⛳")

PROCESSED = Path(__file__).parent.parent / "data" / "processed"

@st.cache_data(ttl=1800)
def load_predictions():
    p = PROCESSED / "predictions_2026.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)

@st.cache_data(ttl=1800)
def load_edge():
    p = PROCESSED / "edge_2026.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)

preds = load_predictions()
edge = load_edge()

if preds is None:
    st.error("Predictions not found. Run: `python run_2026_predictions.py`")
    st.stop()

st.sidebar.title("⛳ Augusta Model")
st.sidebar.markdown("**2026 Masters — April 9-12**")
st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%b %d, %H:%M')}")
st.sidebar.markdown(f"Field size: {len(preds)}")
st.sidebar.markdown("---")
st.sidebar.page_link("app.py", label="Home")
st.sidebar.page_link("pages/1_Model_Picks.py", label="Model Picks")
st.sidebar.page_link("pages/2_Betting_Edge.py", label="Betting Edge")
st.sidebar.page_link("pages/3_H2H_Matchups.py", label="H2H Matchups")
st.sidebar.page_link("pages/4_Backtest.py", label="Backtest")

st.title("⛳ 2026 Masters — Augusta National Model")
st.markdown("Two-stage XGBoost + Monte Carlo model with Augusta-specific experience features. "
            "Backtested 2021-2025: **42% top-10 precision**, **+55% ROI** on top-10 market.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Field Size", len(preds))
debutants = (preds["augusta_experience_tier"]==0).sum()
col2.metric("Debutants", debutants)
top1 = preds.iloc[0]
col3.metric("Model #1 (Top-10)", top1["player_name"], f"{top1['top10_prob_calibrated']:.1%}")
if edge is not None and len(edge)>0:
    fav = edge.sort_values("market_win_pct", ascending=False).iloc[0]
    col4.metric("Odds Favourite", fav["player_name"], f"{fav['market_win_pct']:.1%} mkt")

st.markdown("---")
st.subheader("Quick Look — Top 10")
top10 = preds.head(10)[["player_name","win_prob","top10_prob_calibrated","top20_prob",
                         "augusta_experience_tier","augusta_made_cut_prev_year","augusta_scoring_trajectory"]].copy()
top10.columns = ["Player","Win %","Top-10 %","Top-20 %","Exp Tier","Cut Prev Yr","Trajectory"]
top10["Win %"] = top10["Win %"].map("{:.1%}".format)
top10["Top-10 %"] = top10["Top-10 %"].map("{:.1%}".format)
top10["Top-20 %"] = top10["Top-20 %"].map("{:.1%}".format)
top10["Trajectory"] = top10["Trajectory"].map("{:+.2f}".format)
top10.index = range(1, len(top10)+1)
st.dataframe(top10, use_container_width=True)
