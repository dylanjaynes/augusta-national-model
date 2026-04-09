import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from market_odds import fetch_market_odds, merge_with_model

st.set_page_config(page_title="Model Picks — Augusta Model", layout="wide", page_icon="⛳")

PROCESSED = Path(__file__).parent.parent.parent / "data" / "processed"

st.markdown("""
<style>
.ev-pos  { background:#1a472a; color:#7dcea0; font-weight:700; padding:2px 6px; border-radius:4px; }
.ev-neg  { background:#641e16; color:#f1948a; font-weight:700; padding:2px 6px; border-radius:4px; }
.ev-zero { color:#888; }
.my-odds { font-weight:700; font-size:1.05em; }
</style>
""", unsafe_allow_html=True)


def prob_to_american(p):
    try:
        p = float(p)
        if p <= 0 or p >= 1:
            return "—"
        return f"{round(-p/(1-p)*100):+d}" if p >= 0.5 else f"+{round((1-p)/p*100)}"
    except (TypeError, ValueError):
        return "—"


def fmt_pct(v):
    try:
        return f"{float(v):.1%}"
    except (TypeError, ValueError):
        return "—"


def fmt_ev(v):
    try:
        v = float(v)
        if abs(v) < 0.01:
            return "—"
        return f"{v:+.2f}"
    except (TypeError, ValueError):
        return "—"


@st.cache_data(ttl=1800)
def load_predictions():
    p = PROCESSED / "predictions_2026.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None


preds = load_predictions()
if preds is None:
    st.error("Run: `python run_production.py`")
    st.stop()

preds = preds.sort_values("win_prob", ascending=False).reset_index(drop=True)

market_df = fetch_market_odds()
if market_df is not None:
    preds = merge_with_model(preds, market_df)
    source = market_df["_source"].iloc[0] if "_source" in market_df.columns else "DataGolf"
else:
    source = None

# ── Header ───────────────────────────────────────────────────────────────────
st.title("2026 Masters — Model vs Market")

top1 = preds.iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Field", len(preds))
c2.metric("Top Pick", top1["player_name"])
c3.metric("Model Win%", fmt_pct(top1["win_prob"]))
c4.metric("Book Win%", fmt_pct(top1.get("market_win")))
if source:
    st.caption(f"Market: **{source}** — EV = (Model − Book) / Book")
st.markdown("---")

# ── Market tabs ───────────────────────────────────────────────────────────────
MARKETS = [
    ("Win",    "win_prob",    "market_win",   "kelly_edge_win"),
    ("Top 5",  "top5_prob",   "market_top5",  None),
    ("Top 10", "top10_prob",  "market_top10", "kelly_edge_top10"),
    ("Top 20", "top20_prob",  "market_top20", "kelly_edge_top20"),
]

tabs = st.tabs([m[0] for m in MARKETS])

for tab, (label, model_col, mkt_col, edge_col) in zip(tabs, MARKETS):
    with tab:
        df = preds.copy()

        # If no dedicated edge col, compute on the fly
        if edge_col is None or edge_col not in df.columns:
            if mkt_col in df.columns:
                mkt = df[mkt_col].fillna(0)
                df["_edge"] = np.where(mkt > 0, (df[model_col] - mkt) / mkt, np.nan)
            else:
                df["_edge"] = np.nan
            edge_col = "_edge"

        df = df.sort_values(model_col, ascending=False).reset_index(drop=True)

        # Experience / flags helpers
        tier_map = {0: "🆕 Debut", 1: "Learning", 2: "Established",
                    3: "Veteran", 4: "Deep Vet"}

        rows = []
        for i, row in df.iterrows():
            name = row["player_name"]
            model_p = row.get(model_col, np.nan)
            mkt_p   = row.get(mkt_col,   np.nan)
            ev      = row.get(edge_col,  np.nan)

            flags = []
            if row.get("tour_vs_augusta_divergence", 0) > 0.35:
                flags.append("⚠")
            if row.get("augusta_experience_tier", -1) == 0:
                flags.append("🆕")
            if row.get("augusta_scoring_trajectory", 0) < -0.5:
                flags.append("📉")

            rows.append({
                "#":          i + 1,
                "Player":     name,
                "My Odds":    fmt_pct(model_p),
                "Book Odds":  fmt_pct(mkt_p),
                "Book (Amer)":prob_to_american(mkt_p),
                "EV":         fmt_ev(ev),
                "_ev_raw":    float(ev) if pd.notna(ev) else 0.0,
                "Tier":       tier_map.get(int(row.get("augusta_experience_tier", 2)), "—"),
                "Cut '25":    "✓" if row.get("augusta_made_cut_prev_year") == 1 else "—",
                "Flags":      " ".join(flags),
            })

        display = pd.DataFrame(rows)

        # Style EV column
        def style_row(row):
            styles = [""] * len(row)
            ev_idx = display.columns.get_loc("EV")
            ev = row["_ev_raw"]
            if ev > 0.05:
                styles[ev_idx] = "background-color:#1a472a; color:#7dcea0; font-weight:bold"
            elif ev < -0.05:
                styles[ev_idx] = "background-color:#641e16; color:#f1948a"
            return styles

        show = display.drop(columns=["_ev_raw"])

        styled = show.style.apply(style_row, axis=1)

        st.dataframe(styled, width="stretch", height=900, hide_index=True)

        # Quick summary
        if "_ev_raw" in display.columns:
            n_pos = (display["_ev_raw"] > 0.05).sum()
            n_neg = (display["_ev_raw"] < -0.05).sum()
            st.caption(f"**{n_pos}** positive EV plays · **{n_neg}** fades · threshold ±0.05")

# ── Expander ─────────────────────────────────────────────────────────────────
with st.expander("Column guide"):
    st.markdown("""
| Column | Meaning |
|--------|---------|
| **My Odds** | Model implied probability (%) |
| **Book Odds** | Market implied % (DataGolf consensus, no-vig) |
| **Book (Amer)** | Same expressed as American odds |
| **EV** | (Model − Book) / Book. Green = model sees value vs book |
| **Tier** | Augusta experience: 🆕 Debut → Deep Vet |
| **Cut '25** | Made the cut at Augusta last year |
| **Flags** | ⚠ Tour form may not translate · 🆕 First Masters · 📉 Declining trajectory |
""")
