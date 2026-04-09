import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from market_odds import fetch_market_odds, merge_with_model

st.set_page_config(page_title="Best Bets — Augusta Model", layout="wide", page_icon="⛳")

PROCESSED = Path(__file__).parent.parent.parent / "data" / "processed"


@st.cache_data(ttl=1800)
def load_model_preds():
    p = PROCESSED / "predictions_2026.parquet"
    if p.exists():
        return pd.read_parquet(p)
    p = PROCESSED / "predictions_2026.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


# ── Load data ────────────────────────────────────────────────────────────────
model_df = load_model_preds()
if model_df is None:
    st.error("Predictions not found. Run: `python run_production.py`")
    st.stop()

model_df = model_df.sort_values("win_prob", ascending=False).reset_index(drop=True)

market_df = fetch_market_odds()
if market_df is None:
    st.warning("Could not fetch live market odds from DataGolf. Set DATAGOLF_API_KEY in secrets.")
    st.stop()

source_label = market_df["_source"].iloc[0] if "_source" in market_df.columns else "DataGolf"
df = merge_with_model(model_df, market_df)

# Filter to players with real market lines (> 0.5% win implied)
df = df[df["market_win"].fillna(0) >= 0.005].copy()

# ── Controls ─────────────────────────────────────────────────────────────────
st.title("Best Bets — Value vs Market")
st.caption(f"Market odds: **{source_label}** — Kelly edge = (model − book) / book")

col_ctrl1, col_ctrl2 = st.columns([1, 1])
with col_ctrl1:
    min_edge = st.slider("Min Win Edge (Kelly)", min_value=-2.0, max_value=5.0,
                         value=0.0, step=0.1,
                         help="0 = model beats the market; positive = larger edge required")
with col_ctrl2:
    sort_by = st.selectbox("Sort by", ["Win Edge", "T10 Edge", "Win %", "T10 %"])

# ── Summary metrics ───────────────────────────────────────────────────────────
n_pos_win  = (df["kelly_edge_win"].fillna(0) > 0).sum()
n_pos_t10  = (df["kelly_edge_top10"].fillna(0) > 0).sum()
n_strong   = (df["kelly_edge_win"].fillna(0) > 1.0).sum()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Players with market lines", len(df))
m2.metric("Win+ edges",  n_pos_win)
m3.metric("T10+ edges",  n_pos_t10)
m4.metric("Strong win edges (>1.0)", n_strong)
st.markdown("---")

# ── Build table ───────────────────────────────────────────────────────────────
sort_map = {"Win Edge": "kelly_edge_win", "T10 Edge": "kelly_edge_top10",
            "Win %": "win_prob", "T10 %": "top10_prob"}

table = df[df["kelly_edge_win"].fillna(-99) >= min_edge].copy()
table = table.sort_values(sort_map[sort_by], ascending=False).reset_index(drop=True)

display_cols = ["player_name", "win_prob", "market_win", "kelly_edge_win",
                "top10_prob", "market_top10", "kelly_edge_top10"]
if "top20_prob" in table.columns:
    display_cols.append("top20_prob")

rename = {
    "player_name":      "Player",
    "win_prob":         "Model Win%",
    "market_win":       "Book Win%",
    "kelly_edge_win":   "Win Edge",
    "top10_prob":       "Model T10%",
    "market_top10":     "Book T10%",
    "kelly_edge_top10": "T10 Edge",
    "top20_prob":       "Model T20%",
}
display = table[[c for c in display_cols if c in table.columns]].rename(columns=rename)

for col in ["Model Win%", "Book Win%", "Model T10%", "Book T10%"]:
    if col in display.columns:
        display[col] = display[col].apply(lambda v: f"{float(v):.1%}" if pd.notna(v) else "—")
if "Model T20%" in display.columns:
    display["Model T20%"] = display["Model T20%"].apply(
        lambda v: f"{float(v):.1%}" if pd.notna(v) else "—")
for col in ["Win Edge", "T10 Edge"]:
    if col in display.columns:
        display[col] = display[col].apply(
            lambda v: f"{float(v):+.2f}x" if pd.notna(v) else "—")


def style_edge(val):
    try:
        v = float(str(val).replace("x", "").replace("+", ""))
        if v > 0.5:
            return "background-color: #1a472a; color: #7dcea0; font-weight: bold"
        elif v > 0:
            return "background-color: #145a32; color: #a9dfbf"
        elif v < -0.5:
            return "background-color: #641e16; color: #f1948a; font-weight: bold"
        else:
            return "background-color: #7b241c; color: #f5cba7"
    except (ValueError, TypeError):
        return ""


edge_cols = [c for c in ["Win Edge", "T10 Edge"] if c in display.columns]
styled = display.style.map(style_edge, subset=edge_cols)
st.dataframe(styled, width="stretch", height=600, hide_index=True)

if len(table) == 0:
    st.info(f"No players with Win Edge ≥ {min_edge:.1f}. Lower the threshold.")

# ── Top plays callout ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Top Value Plays")
col_plays, col_fades = st.columns(2)

with col_plays:
    st.markdown("**Backs (positive win edge)**")
    backs = df[df["kelly_edge_win"].fillna(0) > 0].sort_values(
        "kelly_edge_win", ascending=False).head(8)
    for _, row in backs.iterrows():
        e = row["kelly_edge_win"]
        bar = "🟢" if e > 1 else "🔵"
        st.markdown(
            f"{bar} **{row['player_name']}** — "
            f"Model {row['win_prob']:.1%} vs Book {row['market_win']:.1%} "
            f"_(edge {e:+.2f}x)_"
        )

with col_fades:
    st.markdown("**Fades (model below market)**")
    fades = df[df["kelly_edge_win"].fillna(0) < -0.3].sort_values("kelly_edge_win").head(8)
    for _, row in fades.iterrows():
        e = row["kelly_edge_win"]
        st.markdown(
            f"🔴 **{row['player_name']}** — "
            f"Model {row['win_prob']:.1%} vs Book {row['market_win']:.1%} "
            f"_(edge {e:+.2f}x)_"
        )

st.markdown("---")
st.caption(
    "Kelly edge = (model_prob − book_prob) / book_prob. "
    "Values > 0 mean model gives higher probability than the market implies. "
    "This model is for research and entertainment only. Not financial advice."
)
