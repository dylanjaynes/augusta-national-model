"""Best Bets — Betting value comparison across Win / Top-10 / Top-20 markets."""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Best Bets — Augusta Model", layout="wide", page_icon="⛳")

PROCESSED = Path(__file__).parent.parent.parent / "data" / "processed"

DISCLAIMER = (
    "This model is for research and entertainment purposes only. "
    "Not financial advice. Past backtest performance does not guarantee future results."
)


@st.cache_data(ttl=1800)
def load_preds():
    p = PROCESSED / "predictions_2026.parquet"
    if p.exists():
        return pd.read_parquet(p)
    p = PROCESSED / "predictions_2026.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


preds = load_preds()
if preds is None:
    st.error("Predictions not found. Run: `python run_production.py`")
    st.stop()

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.title("⛳ Best Bets")
min_edge_pct = st.sidebar.slider(
    "Min edge threshold (%)",
    min_value=-50, max_value=50, value=5, step=1,
    help="Filter to players where model edge exceeds this % (positive = model likes them more than books)"
)
show_fades = st.sidebar.checkbox("Show fade candidates (model < book)", value=True)
top_n = st.sidebar.slider("Max players to display", 10, 80, 30, 5)

st.title("Best Bets — Betting Value vs Market")
st.caption("Market odds source: **DraftKings**  |  Edge = Model % − Book %  |  Kelly = (Model − Book) / Book")

# ── Build edge tables ─────────────────────────────────────────────────────────

def _kelly(model_p, book_p):
    """Kelly edge fraction: (edge / book odds)."""
    valid = (book_p > 0.01) & np.isfinite(book_p)
    result = np.where(valid, (model_p - book_p) / book_p, np.nan)
    return result


def build_win_table(df):
    needed = ["player_name", "win_prob"]
    if not all(c in df.columns for c in needed):
        return None
    t = df[needed].copy()
    t["model_win_pct"] = t["win_prob"]
    has_market = "dk_fair_prob_win" in df.columns and df["dk_fair_prob_win"].notna().any()
    if has_market:
        t["book_win_pct"] = df["dk_fair_prob_win"]
        t["win_edge"] = t["model_win_pct"] - t["book_win_pct"]
        t["kelly_win"] = _kelly(t["model_win_pct"].values, t["book_win_pct"].values)
    else:
        t["book_win_pct"] = np.nan
        t["win_edge"] = np.nan
        t["kelly_win"] = np.nan
    if "dg_rank" in df.columns:
        t["dg_rank"] = df["dg_rank"]
    return t.dropna(subset=["model_win_pct"])


def build_t10_table(df):
    needed = ["player_name", "top10_prob"]
    if not all(c in df.columns for c in needed):
        return None
    t = df[needed].copy()
    t["model_t10_pct"] = t["top10_prob"]
    has_market = "dk_fair_prob_top10" in df.columns and df["dk_fair_prob_top10"].notna().any()
    if has_market:
        t["book_t10_pct"] = df["dk_fair_prob_top10"]
        t["t10_edge"] = t["model_t10_pct"] - t["book_t10_pct"]
        t["kelly_t10"] = _kelly(t["model_t10_pct"].values, t["book_t10_pct"].values)
    else:
        t["book_t10_pct"] = np.nan
        t["t10_edge"] = np.nan
        t["kelly_t10"] = np.nan
    if "dg_rank" in df.columns:
        t["dg_rank"] = df["dg_rank"]
    return t.dropna(subset=["model_t10_pct"])


def build_t20_table(df):
    needed = ["player_name", "top20_prob"]
    if not all(c in df.columns for c in needed):
        return None
    t = df[needed].copy()
    t["model_t20_pct"] = t["top20_prob"]
    # No DK market for T20 yet — show model-only
    t["book_t20_pct"] = np.nan
    t["t20_edge"] = np.nan
    t["kelly_t20"] = np.nan
    if "dg_rank" in df.columns:
        t["dg_rank"] = df["dg_rank"]
    return t.dropna(subset=["model_t20_pct"])


# ── Display helpers ────────────────────────────────────────────────────────────

def _color_edge(val):
    """Green for positive edge, red for negative."""
    if pd.isna(val):
        return ""
    if val > 0.02:
        return "background-color: #d4edda; color: #155724"
    if val < -0.02:
        return "background-color: #f8d7da; color: #721c24"
    return "background-color: #fff3cd; color: #856404"


def show_market_tab(table, model_col, book_col, edge_col, kelly_col, market_label):
    if table is None:
        st.info("No data available.")
        return

    no_market = table[edge_col].isna().all() if edge_col in table.columns else True

    if no_market:
        st.info(f"No {market_label} market data available — showing model probabilities only.")
        display = table.sort_values(model_col, ascending=False).head(top_n).copy()
        display = display.reset_index(drop=True)
        display.index += 1
        fmt = {model_col: "{:.1%}"}
        st.dataframe(display[["player_name", model_col]].rename(
            columns={"player_name": "Player", model_col: "Model %"}
        ).style.format(fmt), use_container_width=True)
        st.caption(DISCLAIMER)
        return

    # Filter by edge threshold
    threshold = min_edge_pct / 100
    value_bets = table[table[edge_col] >= threshold].copy()
    value_bets = value_bets.sort_values(edge_col, ascending=False).head(top_n)

    if len(value_bets) == 0:
        st.info(f"No value bets above {min_edge_pct}% edge threshold. Try lowering the slider.")
    else:
        st.markdown(f"#### Value Bets — edge ≥ {min_edge_pct}%")
        _render_edge_table(value_bets, model_col, book_col, edge_col, kelly_col)
        st.caption(f"{len(value_bets)} player(s) above threshold")

    if show_fades:
        fades = table[table[edge_col] < -0.10].sort_values(edge_col).head(10).copy()
        if len(fades) > 0:
            st.markdown("#### Fade Candidates — model significantly below market")
            _render_edge_table(fades, model_col, book_col, edge_col, kelly_col)

    st.markdown("---")
    st.caption(DISCLAIMER)


def _render_edge_table(df, model_col, book_col, edge_col, kelly_col):
    display = df.copy()
    display = display.reset_index(drop=True)
    display.index += 1

    cols = {"player_name": "Player", model_col: "Model %", book_col: "Book %", edge_col: "Edge"}
    has_kelly = kelly_col in display.columns and display[kelly_col].notna().any()
    has_rank = "dg_rank" in display.columns

    select = ["player_name", model_col, book_col, edge_col]
    if has_kelly:
        select.append(kelly_col)
        cols[kelly_col] = "Kelly"
    if has_rank:
        select.append("dg_rank")
        cols["dg_rank"] = "DG Rank"
    select = [c for c in select if c in display.columns]

    display = display[select].rename(columns=cols)

    fmt = {"Model %": "{:.1%}", "Book %": "{:.1%}", "Edge": "{:+.1%}"}
    if "Kelly" in display.columns:
        fmt["Kelly"] = "{:+.0%}"
    if "DG Rank" in display.columns:
        fmt["DG Rank"] = lambda v: f"#{int(v)}" if pd.notna(v) and v > 0 else "—"

    styled = display.style.format(fmt)
    if "Edge" in display.columns:
        styled = styled.applymap(_color_edge, subset=["Edge"])

    st.dataframe(styled, use_container_width=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_win, tab_t10, tab_t20 = st.tabs(["Win Market", "Top-10 Market", "Top-20 (Model Only)"])

win_table = build_win_table(preds)
t10_table = build_t10_table(preds)
t20_table = build_t20_table(preds)

with tab_win:
    show_market_tab(win_table, "model_win_pct", "book_win_pct", "win_edge", "kelly_win", "Win")

with tab_t10:
    show_market_tab(t10_table, "model_t10_pct", "book_t10_pct", "t10_edge", "kelly_t10", "Top-10")

with tab_t20:
    st.info("DraftKings doesn't offer a standard Top-20 market — showing model probabilities only.")
    if t20_table is not None:
        t20_disp = t20_table.sort_values("model_t20_pct", ascending=False).head(top_n).copy()
        t20_disp = t20_disp.reset_index(drop=True)
        t20_disp.index += 1
        show_cols = ["player_name", "model_t20_pct"]
        if "dg_rank" in t20_disp.columns:
            show_cols.append("dg_rank")
        t20_disp = t20_disp[show_cols].rename(columns={
            "player_name": "Player", "model_t20_pct": "Model Top-20 %", "dg_rank": "DG Rank"
        })
        fmt = {"Model Top-20 %": "{:.1%}"}
        if "DG Rank" in t20_disp.columns:
            fmt["DG Rank"] = lambda v: f"#{int(v)}" if pd.notna(v) and v > 0 else "—"
        st.dataframe(t20_disp.style.format(fmt), use_container_width=True)
    st.caption(DISCLAIMER)
