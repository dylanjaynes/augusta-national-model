import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from market_odds import fetch_market_odds, merge_with_model

st.set_page_config(page_title="Model Picks — Augusta Model", layout="wide", page_icon="⛳")
PROCESSED = Path(__file__).parent.parent.parent / "data" / "processed"


def prob_to_american(p):
    """Convert decimal probability to American odds string."""
    try:
        p = float(p)
        if p <= 0 or p >= 1:
            return "—"
        if p >= 0.5:
            return f"{round(-p / (1 - p) * 100):+d}"
        else:
            return f"+{round((1 - p) / p * 100)}"
    except (TypeError, ValueError):
        return "—"


@st.cache_data(ttl=1800)
def load_predictions():
    p = PROCESSED / "predictions_2026.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


preds = load_predictions()
if preds is None:
    st.error("Run: `python run_production.py`")
    st.stop()

preds = preds.sort_values("win_prob", ascending=False).reset_index(drop=True)

# ── Fetch live market odds ────────────────────────────────────────────────────
market_df = fetch_market_odds()
if market_df is not None:
    preds = merge_with_model(preds, market_df)
    source_label = market_df["_source"].iloc[0] if "_source" in market_df.columns else "DataGolf"
else:
    source_label = None

# ── Header metrics ────────────────────────────────────────────────────────────
st.title("2026 Masters — Model Predictions")

top1 = preds.iloc[0]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Field Size", len(preds))
col2.metric("Debutants", (preds["augusta_experience_tier"] == 0).sum())
col3.metric("Top Pick", top1["player_name"])
col4.metric("Top Pick T10%", f"{top1['top10_prob']:.1%}")

if source_label:
    st.caption(f"Market odds: **{source_label}**")

st.markdown("---")

# ── Build display table ───────────────────────────────────────────────────────
display = preds.copy()
display.insert(0, "Rank", range(1, len(display) + 1))

tier_map = {0: "Debutant", 1: "Learning", 2: "Established", 3: "Veteran", 4: "Deep Vet"}
display["Experience"] = display["augusta_experience_tier"].map(tier_map)
display["Cut Last Yr"] = display["augusta_made_cut_prev_year"].map({1: "Yes", 0: "—"})


def make_flags(row):
    flags = []
    if row.get("tour_vs_augusta_divergence", 0) > 0.35:
        flags.append("⚠ FADE?")
    if row.get("augusta_experience_tier", -1) == 0:
        flags.append("🆕 DEBUT")
    if row.get("augusta_scoring_trajectory", 0) < -0.5:
        flags.append("📉")
    return " ".join(flags)


display["Flags"] = display.apply(make_flags, axis=1)

# American odds columns
display["Book Win"] = display["market_win"].apply(prob_to_american) \
    if "market_win" in display.columns else "—"
display["Book T10"] = display["market_top10"].apply(prob_to_american) \
    if "market_top10" in display.columns else "—"
display["Book T20"] = display["market_top20"].apply(prob_to_american) \
    if "market_top20" in display.columns else "—"

# Formatted model probs
display["Win %"]    = display["win_prob"].map("{:.1%}".format)
display["Top-10 %"] = display["top10_prob"].map("{:.1%}".format)
display["Top-20 %"] = display["top20_prob"].map("{:.1%}".format)
display["Trajectory"] = display["augusta_scoring_trajectory"].map("{:+.2f}".format)

show_cols = [
    "Rank", "player_name",
    "Win %", "Book Win",
    "Top-10 %", "Book T10",
    "Top-20 %", "Book T20",
    "Experience", "Cut Last Yr", "Trajectory", "Flags",
]
show_cols = [c for c in show_cols if c in display.columns]

table = display[show_cols].rename(columns={"player_name": "Player"})

st.dataframe(table, width="stretch", height=800, hide_index=True)

with st.expander("About the model"):
    st.markdown("""
**Win % / Top-10 % / Top-20 %** — Model probabilities (Monte Carlo 50K sims, Stage 2 XGBoost blend)

**Book Win / Book T10 / Book T20** — Current market American odds from DataGolf (consensus across 11 sportsbooks).
Positive = underdog, negative = favourite.

**Experience** — Augusta course experience tier (Debutant → Deep Vet)

**Cut Last Yr** — Made cut at Augusta last year? 9 of 10 recent winners did.

**Trajectory** — Improving (+) or declining (−) at Augusta vs career average.

**Flags:** ⚠ FADE? = tour form may not translate · 🆕 DEBUT = first Masters · 📉 = declining trajectory
""")
