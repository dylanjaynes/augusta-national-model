import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from market_odds import fetch_market_odds, merge_with_model

st.set_page_config(page_title="Betting Edge — Augusta Model", layout="wide", page_icon="⛳")

PROCESSED = Path(__file__).parent.parent.parent / "data" / "processed"


@st.cache_data(ttl=1800)
def load_model_preds():
    p = PROCESSED / "predictions_2026.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None


model_df = load_model_preds()
if model_df is None:
    st.error("Predictions not found. Run: `python run_production.py`")
    st.stop()

model_df = model_df.sort_values("win_prob", ascending=False).reset_index(drop=True)

market_df = fetch_market_odds()
if market_df is None:
    st.warning("Could not fetch live market odds. Set DATAGOLF_API_KEY in secrets.")
    st.stop()

source_label = market_df["_source"].iloc[0] if "_source" in market_df.columns else "DataGolf"
df = merge_with_model(model_df, market_df)
# Only players with real market lines
df = df[df["market_win"].fillna(0) >= 0.005].copy()

st.title("Betting Edge — Value vs Market")
st.caption(f"Market odds: **{source_label}**")

threshold = st.sidebar.slider("Edge threshold (%)", 10, 50, 20, 5)
st.sidebar.markdown("Only shows plays with edge above this threshold")

tab_t10, tab_win, tab_t20 = st.tabs(["Top-10 Market", "Win Market", "Top-20 Market"])


def show_tab(df, model_col, mkt_col, edge_col, threshold_pct, label):
    if model_col not in df.columns or mkt_col not in df.columns:
        st.info(f"No data for {label}")
        return

    df2 = df[[c for c in ["player_name", model_col, mkt_col, edge_col] if c in df.columns]].copy()
    df2.columns = ["Player", "Model %", "Market %", "Edge %"][:len(df2.columns)]
    df2 = df2.dropna(subset=["Market %"])

    positive = df2[df2["Edge %"].fillna(0) > threshold_pct / 100].sort_values(
        "Edge %", ascending=False).copy()
    for col in ["Model %", "Market %", "Edge %"]:
        if col in positive.columns:
            positive[col] = positive[col] * 100

    _edge_cfg = {
        "Model %":  st.column_config.NumberColumn("Model %",  format="%.1f%%"),
        "Market %": st.column_config.NumberColumn("Market %", format="%.1f%%"),
        "Edge %":   st.column_config.NumberColumn("Edge %",   format="%+.0f%%"),
    }
    if len(positive):
        st.dataframe(positive, width="stretch", height=500, hide_index=True,
                     column_config=_edge_cfg)
        st.caption(f"{len(positive)} plays above {threshold_pct}% edge")
    else:
        st.info(f"No plays above {threshold_pct}% edge")

    fades = df2[df2["Edge %"].fillna(0) < -0.30].sort_values("Edge %").head(10).copy()
    if len(fades):
        st.markdown("#### Fades (model below market)")
        for col in ["Model %", "Market %", "Edge %"]:
            if col in fades.columns:
                fades[col] = fades[col] * 100
        st.dataframe(fades, width="stretch", hide_index=True, column_config=_edge_cfg)

    st.markdown("---")
    st.caption("For research and entertainment only. Not financial advice.")


with tab_t10:
    show_tab(df, "top10_prob", "market_top10", "kelly_edge_top10", threshold, "Top-10")

with tab_win:
    show_tab(df, "win_prob", "market_win", "kelly_edge_win", threshold, "Win")

with tab_t20:
    show_tab(df, "top20_prob", "market_top20", "kelly_edge_top20", threshold, "Top-20")
