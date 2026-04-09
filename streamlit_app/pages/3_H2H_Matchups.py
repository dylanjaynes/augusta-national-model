import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="H2H Matchups — Augusta Model", layout="wide", page_icon="⛳")
PROCESSED = Path(__file__).parent.parent.parent / "data" / "processed"

@st.cache_data(ttl=1800)
def load_predictions():
    p = PROCESSED / "predictions_2026.parquet"
    if not p.exists(): return None
    return pd.read_parquet(p)

preds = load_predictions()
if preds is None:
    st.error("Run: `python run_2026_predictions.py`")
    st.stop()

st.title("Head-to-Head Matchups")

t10_col = "top10_prob" if "top10_prob" in preds.columns else "top10_prob_calibrated"
players = preds["player_name"].tolist()

col1, col2 = st.columns(2)
with col1:
    player_a = st.selectbox("Player A", players, index=0)
with col2:
    player_b = st.selectbox("Player B", players, index=min(1, len(players)-1))

if player_a == player_b:
    st.warning("Select two different players")
    st.stop()

a = preds[preds["player_name"]==player_a].iloc[0]
b = preds[preds["player_name"]==player_b].iloc[0]

st.markdown("---")

# Probability comparison
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader(player_a)
    st.metric("Win %", f"{a['win_prob']:.1%}")
    st.metric("Top-10 %", f"{a[t10_col]:.1%}")
    st.metric("Top-20 %", f"{a['top20_prob']:.1%}")
with col2:
    st.subheader("vs")
    # H2H approximation: normalize model_scores
    score_a = a["model_score"]
    score_b = b["model_score"]
    # Lower finish_pct = better, so invert for H2H
    if score_a + score_b > 0:
        h2h_a = (1 - score_a) / ((1-score_a) + (1-score_b))
    else:
        h2h_a = 0.5
    h2h_b = 1 - h2h_a
    st.metric("H2H Win %", f"{h2h_a:.0%} — {h2h_b:.0%}")
with col3:
    st.subheader(player_b)
    st.metric("Win %", f"{b['win_prob']:.1%}")
    st.metric("Top-10 %", f"{b[t10_col]:.1%}")
    st.metric("Top-20 %", f"{b['top20_prob']:.1%}")

# Augusta profile comparison
st.markdown("---")
st.subheader("Augusta Profile Comparison")

tier_map = {0: "Debutant", 1: "Learning", 2: "Established", 3: "Veteran", 4: "Deep Vet"}
comp = pd.DataFrame({
    "Metric": ["Competitive Rounds", "Made Cut Last Year", "Experience Tier",
               "Best Recent Finish", "Scoring Trajectory", "Divergence Flag"],
    player_a: [
        f"{a['augusta_competitive_rounds']:.0f}",
        "Yes" if a["augusta_made_cut_prev_year"]==1 else "No",
        tier_map.get(a["augusta_experience_tier"], "?"),
        f"{a.get('augusta_best_finish_recent','?')}",
        f"{a['augusta_scoring_trajectory']:+.2f}",
        "Yes" if a.get("tour_vs_augusta_divergence",0) > 0.35 else "No",
    ],
    player_b: [
        f"{b['augusta_competitive_rounds']:.0f}",
        "Yes" if b["augusta_made_cut_prev_year"]==1 else "No",
        tier_map.get(b["augusta_experience_tier"], "?"),
        f"{b.get('augusta_best_finish_recent','?')}",
        f"{b['augusta_scoring_trajectory']:+.2f}",
        "Yes" if b.get("tour_vs_augusta_divergence",0) > 0.35 else "No",
    ],
})
st.dataframe(comp, width="stretch", hide_index=True)
