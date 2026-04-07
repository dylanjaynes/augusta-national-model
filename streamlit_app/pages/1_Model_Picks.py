import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Model Picks — Augusta Model", layout="wide", page_icon="⛳")
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

st.title("2026 Masters — Model Predictions")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Field Size", len(preds))
col2.metric("Debutants", (preds["augusta_experience_tier"]==0).sum())
top1 = preds.iloc[0]
col3.metric("Top Pick", top1["player_name"])
col4.metric("Top Pick T10%", f"{top1['top10_prob_calibrated']:.1%}")

st.markdown("---")

# Build display table
display = preds.copy()
display.insert(0, "Rank", range(1, len(display)+1))

# Flags
def make_flags(row):
    flags = []
    if row["tour_vs_augusta_divergence"] > 0.35:
        flags.append("⚠ FADE?")
    if row["augusta_experience_tier"] == 0:
        flags.append("🆕 DEBUT")
    if row["augusta_scoring_trajectory"] < -0.5:
        flags.append("📉")
    return " ".join(flags)

display["Flags"] = display.apply(make_flags, axis=1)
display["Cut Last Yr"] = display["augusta_made_cut_prev_year"].map({1: "Yes", 0: "—"})

tier_map = {0: "Debutant", 1: "Learning", 2: "Established", 3: "Veteran", 4: "Deep Vet"}
display["Experience"] = display["augusta_experience_tier"].map(tier_map)

show_cols = {
    "Rank": "Rank", "player_name": "Player",
    "win_prob": "Win %", "top10_prob_calibrated": "Top-10 %",
    "top20_prob": "Top-20 %", "Experience": "Experience",
    "Cut Last Yr": "Cut Last Yr",
    "augusta_scoring_trajectory": "Trajectory", "Flags": "Flags",
}

table = display[list(show_cols.keys())].rename(columns=show_cols)
table["Win %"] = table["Win %"].map("{:.1%}".format)
table["Top-10 %"] = table["Top-10 %"].map("{:.1%}".format)
table["Top-20 %"] = table["Top-20 %"].map("{:.1%}".format)
table["Trajectory"] = table["Trajectory"].map("{:+.2f}".format)

st.dataframe(table, use_container_width=True, height=800)

with st.expander("About the model"):
    st.markdown("""
**Win %** — Probability of winning outright (Monte Carlo, 50K simulations)

**Top-10 %** — Calibrated probability of finishing top-10 (Stage 2 XGBoost + MC blend)

**Top-20 %** — Probability of finishing top-20

**Experience** — Augusta course experience tier:
- Debutant: 0 prior rounds
- Learning: 1-7 rounds (1-2 starts)
- Established: 8-19 rounds
- Veteran: 20-35 rounds
- Deep Vet: 36+ rounds

**Cut Last Yr** — Did this player make the cut at Augusta last year? 9 of the last 10 winners did.

**Trajectory** — Is the player improving (+) or declining (-) at Augusta? Compares recent 2 appearances to career average.

**Flags:**
- ⚠ FADE? — Tour form may not translate to Augusta (high divergence)
- 🆕 DEBUT — First Masters appearance
- 📉 — Scoring trajectory declining
""")
