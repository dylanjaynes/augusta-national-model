import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Best Bets — Augusta Model", layout="wide", page_icon="⛳")

PROCESSED = Path(__file__).parent.parent.parent / "data" / "processed"


@st.cache_data(ttl=1800)
def load_data():
    edge_path = PROCESSED / "edge_2026.csv"
    preds_path = PROCESSED / "predictions_2026.parquet"

    if not edge_path.exists():
        return None

    edge = pd.read_csv(edge_path)

    # Merge top-20 model probs from predictions if available
    if preds_path.exists():
        preds = pd.read_parquet(preds_path)
        t20_cols = [c for c in preds.columns if "top20" in c.lower() or "top_20" in c.lower()]
        if t20_cols and "player_name" in preds.columns:
            edge = edge.merge(
                preds[["player_name"] + t20_cols],
                on="player_name",
                how="left",
                suffixes=("", "_pred"),
            )

    return edge


df = load_data()

if df is None:
    st.error("Edge data not found. Run: `python run_production.py`")
    st.stop()

# ── Column detection ──────────────────────────────────────────────────────────
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


model_win_col = find_col(df, ["model_win_pct", "model_win"])
mkt_win_col   = find_col(df, ["dk_fair_win_pct", "market_win_pct"])
edge_win_col  = find_col(df, ["win_kelly_edge", "win_edge_pct"])

model_t10_col = find_col(df, ["model_top10_pct", "model_top10"])
mkt_t10_col   = find_col(df, ["dk_fair_top10_pct", "market_top10_pct"])
edge_t10_col  = find_col(df, ["top10_kelly_edge", "top10_edge_pct"])

model_t20_col = find_col(df, ["model_top20_pct", "model_top20", "top20_prob"])

# ── Controls ──────────────────────────────────────────────────────────────────
st.title("Best Bets — Value vs Market")
st.caption("Edges vs **DraftKings** closing lines. Kelly edge = (model − book) / book.")

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
with col_ctrl1:
    min_edge = st.slider(
        "Min Win Edge (Kelly)",
        min_value=-2.0,
        max_value=5.0,
        value=0.0,
        step=0.1,
        help="Show only players where win Kelly edge ≥ this value. 0 = model beats the market.",
    )
with col_ctrl2:
    sort_by = st.selectbox(
        "Sort by",
        ["Win Edge", "T10 Edge", "Win %", "T10 %"],
    )

# ── Build display table ───────────────────────────────────────────────────────
required = [model_win_col, mkt_win_col, edge_win_col, model_t10_col, mkt_t10_col, edge_t10_col]
if not all(required):
    missing = [c for c, v in zip(
        ["model_win", "mkt_win", "edge_win", "model_t10", "mkt_t10", "edge_t10"], required
    ) if v is None]
    st.error(f"Missing columns in edge_2026.csv: {missing}")
    st.stop()

cols_needed = ["player_name", model_win_col, mkt_win_col, edge_win_col,
               model_t10_col, mkt_t10_col, edge_t10_col]
if model_t20_col:
    cols_needed.append(model_t20_col)

table = df[cols_needed].copy()

# Rename for display
rename = {
    "player_name": "Player",
    model_win_col: "Model Win%",
    mkt_win_col:   "Book Win%",
    edge_win_col:  "Win Edge",
    model_t10_col: "Model T10%",
    mkt_t10_col:   "Book T10%",
    edge_t10_col:  "T10 Edge",
}
if model_t20_col:
    rename[model_t20_col] = "Model T20%"

table = table.rename(columns=rename)

sort_map = {
    "Win Edge": "Win Edge",
    "T10 Edge": "T10 Edge",
    "Win %": "Model Win%",
    "T10 %": "Model T10%",
}
table = table.sort_values(sort_map[sort_by], ascending=False).reset_index(drop=True)

# Filter
table = table[table["Win Edge"] >= min_edge]

# ── Summary metrics ───────────────────────────────────────────────────────────
n_positive_win  = (df[edge_win_col] > 0).sum()
n_positive_t10  = (df[edge_t10_col] > 0).sum()
n_strong_win    = (df[edge_win_col] > 1.0).sum()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Players shown", len(table))
m2.metric("Win+ edges (full field)", n_positive_win)
m3.metric("T10+ edges (full field)", n_positive_t10)
m4.metric("Strong win edges (>1.0)", n_strong_win)
st.markdown("---")

# ── Styling ───────────────────────────────────────────────────────────────────
def style_edge(val):
    """Green for positive edge, red for negative."""
    try:
        v = float(val)
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


def format_pct(val):
    try:
        return f"{float(val):.1%}"
    except (ValueError, TypeError):
        return str(val)


def format_edge(val):
    try:
        v = float(val)
        return f"{v:+.2f}x"
    except (ValueError, TypeError):
        return str(val)


# Build a formatted copy for display
display = table.copy()
for col in ["Model Win%", "Book Win%", "Model T10%", "Book T10%"]:
    if col in display.columns:
        display[col] = display[col].apply(format_pct)
if "Model T20%" in display.columns:
    display["Model T20%"] = display["Model T20%"].apply(format_pct)

edge_cols_display = [c for c in ["Win Edge", "T10 Edge"] if c in display.columns]
for col in edge_cols_display:
    display[col] = display[col].apply(format_edge)

styled = display.style.applymap(style_edge, subset=edge_cols_display)

st.dataframe(styled, use_container_width=True, height=600, hide_index=True)

if len(table) == 0:
    st.info(f"No players with Win Edge ≥ {min_edge:.1f}. Lower the threshold.")

# ── Top plays callout ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Top Value Plays")
col_plays, col_fades = st.columns(2)

with col_plays:
    st.markdown("**Backs (positive win edge)**")
    backs = df[df[edge_win_col] > 0].sort_values(edge_win_col, ascending=False).head(8)
    for _, row in backs.iterrows():
        edge_val = row[edge_win_col]
        model_pct = row[model_win_col]
        book_pct  = row[mkt_win_col]
        bar = "🟢" if edge_val > 1 else "🔵"
        st.markdown(
            f"{bar} **{row['player_name']}** — "
            f"Model {model_pct:.1%} vs Book {book_pct:.1%} "
            f"_(edge {edge_val:+.2f}x)_"
        )

with col_fades:
    st.markdown("**Fades (model below market)**")
    fades = df[df[edge_win_col] < -0.3].sort_values(edge_win_col).head(8)
    for _, row in fades.iterrows():
        edge_val = row[edge_win_col]
        model_pct = row[model_win_col]
        book_pct  = row[mkt_win_col]
        st.markdown(
            f"🔴 **{row['player_name']}** — "
            f"Model {model_pct:.1%} vs Book {book_pct:.1%} "
            f"_(edge {edge_val:+.2f}x)_"
        )

st.markdown("---")
st.caption(
    "Kelly edge = (model_prob − book_prob) / book_prob. "
    "Values >0 mean model gives higher probability than the market implies. "
    "This model is for research and entertainment only. Not financial advice."
)
