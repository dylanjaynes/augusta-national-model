"""
Live Tournament page — Augusta National Model

Three tabs: Outright | Top 10 | Other
Shows value bets, Kelly sizing, expected winnings, and historical model performance.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(
    page_title="Live Tournament — Augusta Model",
    layout="wide",
    page_icon="🏌️",
)

ROOT = Path(__file__).parent.parent.parent
PROCESSED = ROOT / "data" / "processed"
LIVE_DIR = ROOT / "data" / "live"
LIVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Backtest data (from CLAUDE.md V3 results) ────────────────────────────────
BACKTEST = {
    2021: {"t10_roi": 0.00,   "t10_prec": 0.30, "spearman": 0.227,  "brier": 0.0134},
    2022: {"t10_roi": 0.55,   "t10_prec": 0.50, "spearman": -0.009, "brier": 0.0141},
    2023: {"t10_roi": 1.04,   "t10_prec": 0.40, "spearman": -0.015, "brier": 0.0145},
    2024: {"t10_roi": 0.36,   "t10_prec": 0.60, "spearman": 0.177,  "brier": 0.0134},
    2025: {"t10_roi": 0.78,   "t10_prec": 0.30, "spearman": 0.173,  "brier": 0.0144},
}

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=120)
def load_live_predictions() -> pd.DataFrame | None:
    latest = LIVE_DIR / "live_predictions_latest.csv"
    if latest.exists():
        return pd.read_csv(latest)
    return None


@st.cache_data(ttl=300)
def load_pretournament() -> pd.DataFrame | None:
    p = PROCESSED / "predictions_2026.parquet"
    if p.exists():
        return pd.read_parquet(p)
    p = PROCESSED / "predictions_2026.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data(ttl=3600)
def find_historical_match(leader_score: float, margin_over_2nd: float, within_5: int) -> dict | None:
    hist_path = PROCESSED / "masters_unified.parquet"
    if not hist_path.exists():
        return None
    hist = pd.read_parquet(hist_path)
    hist = hist[hist["data_source"] == "dg_sg"].copy() if "data_source" in hist.columns else hist[hist["season"] >= 2021].copy()
    results = []
    for year in sorted(hist["season"].unique()):
        yr = hist[hist["season"] == year].dropna(subset=["r1_score", "r2_score"]).copy()
        yr["r2_cum"] = yr["r1_score"] + yr["r2_score"]
        yr = yr.sort_values("r2_cum").reset_index(drop=True)
        if len(yr) < 2:
            continue
        h_leader_score = float(yr["r2_cum"].iloc[0])
        h_margin = float(yr["r2_cum"].iloc[1] - yr["r2_cum"].iloc[0])
        h_within_5 = int((yr["r2_cum"] <= h_leader_score + 5).sum())
        h_leader_name = yr.iloc[0]["player_name"]
        h_leader_won = bool(yr.iloc[0].get("finish_num", 999) == 1)
        winner_row = yr[yr["finish_num"] == 1]
        h_winner_name = winner_row.iloc[0]["player_name"] if not winner_row.empty else "Unknown"
        h_final_margin = None
        if not winner_row.empty:
            ru = yr[yr["finish_num"] == 2]
            if not ru.empty:
                try:
                    w_tot = winner_row.iloc[0][["r1_score", "r2_score", "r3_score", "r4_score"]].sum()
                    ru_tot = ru.iloc[0][["r1_score", "r2_score", "r3_score", "r4_score"]].sum()
                    h_final_margin = int(round(ru_tot - w_tot))
                except Exception:
                    pass
        top6 = yr.head(6)[["player_name", "r2_cum", "finish_pos", "finish_num"]].copy()
        top6["Score"] = top6["r2_cum"].apply(lambda x: "E" if int(x) == 0 else f"{int(x):+d}")
        top6["Finish"] = top6["finish_pos"].fillna(
            top6["finish_num"].apply(lambda x: f"T{int(x)}" if pd.notna(x) and x < 900 else "MC")
        )
        sim = (abs(h_leader_score - leader_score) * 1.0
               + abs(h_margin - margin_over_2nd) * 1.5
               + abs(h_within_5 - within_5) * 0.5)
        results.append(dict(
            year=int(year), leader_name=h_leader_name, leader_score=h_leader_score,
            margin=h_margin, within_5=h_within_5, leader_won=h_leader_won,
            winner_name=h_winner_name, final_margin=h_final_margin,
            similarity=sim, top6=top6,
        ))
    if not results:
        return None
    return min(results, key=lambda x: x["similarity"])


@st.cache_data(ttl=3600)
def load_course_stats() -> pd.DataFrame | None:
    p = PROCESSED / "course_hole_stats.parquet"
    if p.exists():
        return pd.read_parquet(p)
    hbh_p = PROCESSED / "masters_hole_by_hole.parquet"
    if hbh_p.exists():
        try:
            sys.path.insert(0, str(ROOT))
            from augusta_model.features.live_features import _load_course_stats
            hbh = pd.read_parquet(hbh_p)
            return _load_course_stats(hbh)
        except Exception:
            pass
    return None


def score_str(v) -> str:
    if pd.isna(v):
        return "E"
    v = int(v)
    if v == 0:
        return "E"
    return f"+{v}" if v > 0 else str(v)


def pct_num(v):
    """Return v * 100 as float for NumberColumn display (NaN → None)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return round(float(v) * 100, 1)
    except (TypeError, ValueError):
        return None


def format_american(v) -> str:
    if pd.isna(v):
        return "—"
    try:
        v = int(round(float(v)))
    except (ValueError, TypeError):
        return str(v)
    return f"+{v}" if v >= 0 else str(v)


def prob_to_american(p) -> str:
    try:
        p = float(p)
        if p <= 0 or p >= 1:
            return "—"
        if p >= 0.5:
            return f"{round(-p / (1 - p) * 100):+d}"
        return f"+{round((1 - p) / p * 100)}"
    except (TypeError, ValueError):
        return "—"


def movement_arrow(rank_change):
    if pd.isna(rank_change):
        return "—"
    rank_change = int(rank_change)
    if rank_change > 5:
        return f"▲{rank_change}"
    if rank_change > 0:
        return f"↑{rank_change}"
    if rank_change < -5:
        return f"▼{abs(rank_change)}"
    if rank_change < 0:
        return f"↓{abs(rank_change)}"
    return "—"


def _kelly_bet(model_prob: float, market_implied: float,
               bankroll: float, kelly_frac: float) -> float | None:
    try:
        if market_implied <= 0 or model_prob <= 0 or model_prob >= 1:
            return None
        decimal = 1.0 / market_implied
        k = (model_prob * decimal - 1.0) / (decimal - 1.0)
        if k <= 0:
            return None
        return round(bankroll * k * kelly_frac, 2)
    except Exception:
        return None


def _expected_win(kelly_bet: float | None, market_implied: float) -> float | None:
    """Profit if bet hits: kelly_bet × (decimal_odds − 1)."""
    if kelly_bet is None or kelly_bet <= 0:
        return None
    try:
        if market_implied <= 0:
            return None
        decimal = 1.0 / market_implied
        return round(kelly_bet * (decimal - 1.0), 2)
    except Exception:
        return None


def _detect_round_label(df: pd.DataFrame | None) -> str:
    if df is None:
        return ""
    if "current_round" in df.columns:
        r = int(df["current_round"].mode().iloc[0])
        thru = df["thru"].fillna(18).median() if "thru" in df.columns else 18
        if thru >= 18:
            return f"Through Round {r}"
        return f"Round {r} — {int(thru)} holes"
    max_holes = df["holes_completed"].max() if "holes_completed" in df.columns else 0
    if max_holes > 36:
        return "Through Round 3+"
    if max_holes > 18:
        return "Through Round 2"
    if "current_score_to_par" in df.columns:
        leader = df["current_score_to_par"].min()
        if leader < -10:
            return "Through Round 2"
    return "Round 1 in progress"


def run_demo_inference():
    import subprocess
    result = subprocess.run(
        ["python3", str(ROOT / "scripts" / "run_live_inference.py"), "--demo", "--round", "2"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    return result.returncode == 0, result.stdout, result.stderr


# ── Build tab dataframes ──────────────────────────────────────────────────────

def build_outright_df(df: pd.DataFrame, bankroll: float, kelly_frac: float) -> pd.DataFrame:
    """
    Returns display DataFrame for the Outright tab.
    Columns: Player | Score | Model Odds | Market Odds | Model % | Market % | Edge | Kelly Bet | Expected Win
    """
    d = df.copy()

    has_mc_win = "mc_win_prob" in d.columns and d["mc_win_prob"].notna().any()
    win_col = "mc_win_prob" if has_mc_win else "blended_win_prob"

    if win_col not in d.columns:
        d[win_col] = 0.0

    d = d.sort_values(win_col, ascending=False).reset_index(drop=True)

    model_prob = d[win_col].fillna(0)
    market_implied = d["book_implied_win"].fillna(0) if "book_implied_win" in d.columns else pd.Series(0.0, index=d.index)

    kelly_bets = [_kelly_bet(float(mp), float(mi), bankroll, kelly_frac)
                  for mp, mi in zip(model_prob, market_implied)]
    expected_wins = [_expected_win(kb, float(mi))
                     for kb, mi in zip(kelly_bets, market_implied)]

    out = pd.DataFrame()
    out["Player"] = d["player_name"]
    out["Score"] = d["current_score_to_par"].apply(score_str) if "current_score_to_par" in d.columns else "—"
    out["Model Odds"] = model_prob.apply(prob_to_american)
    out["Market Odds"] = (d["book_american_win"].apply(format_american)
                          if "book_american_win" in d.columns
                          else market_implied.apply(prob_to_american))
    out["Model %"] = (model_prob * 100).round(1)
    out["Market %"] = (market_implied * 100).round(1)
    out["Edge"] = ((model_prob - market_implied) * 100).round(1)
    out["Kelly Bet"] = kelly_bets
    out["Expected Win"] = expected_wins
    return out


def build_top10_df(df: pd.DataFrame, bankroll: float, kelly_frac: float) -> tuple[pd.DataFrame, str]:
    """
    Returns (display DataFrame, market_label) for the Top 10 tab.
    Columns: Player | Score | Model Odds | Market Odds | Model % | Market % | Edge | Kelly Bet | Expected Win
    """
    d = df.copy()

    has_mc_t10 = "mc_top10_prob" in d.columns and d["mc_top10_prob"].notna().any()
    t10_col = "mc_top10_prob" if has_mc_t10 else "blended_top10_prob"

    if t10_col not in d.columns:
        d[t10_col] = 0.0

    d = d.sort_values(t10_col, ascending=False).reset_index(drop=True)

    model_prob = d[t10_col].fillna(0)
    has_real_t10 = "book_top_10_implied" in d.columns and d["book_top_10_implied"].notna().any()
    if has_real_t10:
        market_implied = d["book_top_10_implied"].fillna(0)
        market_label = "Market (sportsbook)"
    elif "dg_top10_prob" in d.columns:
        market_implied = d["dg_top10_prob"].fillna(0)
        market_label = "Market (DG model)"
    else:
        market_implied = pd.Series(0.0, index=d.index)
        market_label = "Market %"

    kelly_bets = [_kelly_bet(float(mp), float(mi), bankroll, kelly_frac)
                  for mp, mi in zip(model_prob, market_implied)]
    expected_wins = [_expected_win(kb, float(mi))
                     for kb, mi in zip(kelly_bets, market_implied)]

    if has_real_t10 and "book_top_10_american" in d.columns:
        market_odds_col = d["book_top_10_american"].apply(format_american)
    else:
        market_odds_col = market_implied.apply(prob_to_american)

    out = pd.DataFrame()
    out["Player"] = d["player_name"]
    out["Score"] = d["current_score_to_par"].apply(score_str) if "current_score_to_par" in d.columns else "—"
    out["Model Odds"] = model_prob.apply(prob_to_american)
    out["Market Odds"] = market_odds_col
    out["Model %"] = (model_prob * 100).round(1)
    out["Market %"] = (market_implied * 100).round(1)
    out["Edge"] = ((model_prob - market_implied) * 100).round(1)
    out["Kelly Bet"] = kelly_bets
    out["Expected Win"] = expected_wins
    return out, market_label


# ── Column config for standardised tab table ─────────────────────────────────

def tab_column_config(kelly_frac: float, bankroll: float, market_label: str = "Market %") -> dict:
    pct = "%.1f%%"
    epct = "%+.1f%%"
    return {
        "Model %": st.column_config.NumberColumn("Model %", format=pct),
        "Market %": st.column_config.NumberColumn(market_label, format=pct),
        "Edge": st.column_config.NumberColumn("Edge", format=epct),
        "Kelly Bet": st.column_config.NumberColumn(
            f"Kelly Bet ({kelly_frac:.0%}×)",
            format="$%.2f",
            help=f"{kelly_frac:.0%} Kelly on ${bankroll:.0f} bankroll",
        ),
        "Expected Win": st.column_config.NumberColumn(
            "Expected Win",
            format="+$%.2f",
            help="Profit if bet wins: Kelly Bet × (decimal odds − 1)",
        ),
    }


# ── Kelly explainer ───────────────────────────────────────────────────────────

def render_kelly_explainer(value_rows: pd.DataFrame, market: str, kelly_frac: float) -> None:
    """Show one example explainer row below a value bets table."""
    if value_rows.empty:
        return
    row = value_rows.iloc[0]
    player = row["Player"]
    model_pct = row["Model %"]
    market_pct = row["Market %"]
    market_odds = row["Market Odds"]
    kelly_bet = row.get("Kelly Bet")
    exp_win = row.get("Expected Win")

    if kelly_bet is None or pd.isna(kelly_bet) or kelly_bet <= 0:
        return

    exp_str = f"+${exp_win:.2f}" if exp_win is not None and not pd.isna(exp_win) else "—"
    frac_pct = f"{kelly_frac:.0%}"

    st.caption(
        f"**Kelly explained:** Kelly suggests betting **${kelly_bet:.0f}** on {player} "
        f"({market}) because our model sees **{model_pct:.1f}%** probability vs the market's "
        f"**{market_pct:.1f}%** ({market_odds}). "
        f"At those odds, a ${kelly_bet:.0f} bet returns **{exp_str} profit** if it hits. "
        f"Quarter-Kelly sizing ({frac_pct}) limits risk to ~{kelly_frac * 100:.0f}% of full Kelly on any single bet."
    )


# ── Historical model performance ─────────────────────────────────────────────

def render_historical_performance(tab: str) -> None:
    """Show backtest ROI summary at the bottom of a tab."""
    rows = []
    for yr, d in sorted(BACKTEST.items()):
        roi = d["t10_roi"] if tab == "top10" else None
        prec = d["t10_prec"] if tab == "top10" else None
        rows.append({
            "Year": yr,
            "T10 Precision": f"{prec:.0%}" if prec is not None else "—",
            "T10 ROI": f"{'+' if roi and roi > 0 else ''}{roi:.0%}" if roi is not None else "—",
            "Spearman": f"{d['spearman']:+.3f}",
        })

    df_bt = pd.DataFrame(rows)
    t10_rois = [d["t10_roi"] for d in BACKTEST.values() if d["t10_roi"] is not None]
    avg_roi = sum(t10_rois) / len(t10_rois)
    best_roi = max(t10_rois)
    worst_roi = min(t10_rois)
    profitable_years = sum(1 for r in t10_rois if r > 0)
    avg_prec = sum(d["t10_prec"] for d in BACKTEST.values()) / len(BACKTEST)

    with st.expander("📈 Historical Model Performance (2021–2025 Backtests)"):
        if tab == "top10":
            st.markdown(
                f"**If you placed all recommended Kelly bets from our model's backtest (2021–2025):**\n\n"
                f"- Avg T10 Precision: **{avg_prec:.0%}** — our top-10 picks hit at 42% (10 bets/year, ~4.2 hits)\n"
                f"- Avg ROI: **+{avg_roi:.0%}** across 5 years\n"
                f"- Best year: **+{best_roi:.0%}** (2023) · Worst year: **{'+' if worst_roi > 0 else ''}{worst_roi:.0%}** (2021)\n"
                f"- Profitable in **{profitable_years}/5 years**"
            )
        else:
            st.markdown(
                f"**Model outright performance is tracked via Spearman rank correlation and Brier score.**\n\n"
                f"- Avg Spearman: **{sum(d['spearman'] for d in BACKTEST.values()) / len(BACKTEST):+.3f}** (positive = model ranks align with outcomes)\n"
                f"- Avg Weighted Brier: **{sum(d['brier'] for d in BACKTEST.values()) / len(BACKTEST):.4f}** (lower = better calibration)\n"
                f"- Note: Outright ROI requires closing win odds history — not yet tracked. Use T10 tab for ROI context."
            )

        st.dataframe(df_bt, use_container_width=True, hide_index=True)
        st.caption(
            "Backtest uses leave-one-year-out training (no lookahead). "
            "Stage 2 trained on Augusta-only data. Blend = 0.9 × Stage2 + 0.1 × MC."
        )


# ── Page ──────────────────────────────────────────────────────────────────────

st.title("🏌️ Live Tournament — Augusta National 2026")

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")

    live_mode = st.selectbox(
        "Data source",
        ["Live File (auto-refresh)", "Demo Mode", "Pre-Tournament Only"],
        index=0,
    )

    st.divider()

    show_cols = st.multiselect(
        "Extra columns",
        ["Rank Change", "Confidence"],
        default=["Rank Change", "Confidence"],
    )

    min_holes = st.slider("Min holes completed", 0, 18, 0)
    top_n = st.slider("Players to show", 5, 88, 30)

    st.divider()
    st.markdown("**💰 Kelly Bet Sizing**")
    bankroll = st.number_input(
        "Bankroll ($)", min_value=10, max_value=100_000, value=200, step=10,
        help="Total bankroll for Kelly sizing calculations",
    )
    kelly_frac = st.slider(
        "Kelly fraction", min_value=0.10, max_value=1.0, value=0.25, step=0.05,
        help="Fraction of full Kelly. 0.25 = quarter Kelly (standard for variance management).",
    )

    st.divider()
    with st.expander("Run Inference"):
        if st.button("🔄 Run Demo Inference"):
            with st.spinner("Running live inference..."):
                ok, out, err = run_demo_inference()
            if ok:
                st.success("Done! Refresh to see results.")
                st.text(out[-500:] if len(out) > 500 else out)
            else:
                st.error("Inference failed")
                st.text(err[-500:] if len(err) > 500 else err)

# Load data
pre_df = load_pretournament()
live_df = load_live_predictions()
course_stats = load_course_stats()

if live_mode == "Pre-Tournament Only":
    live_df = None
if live_mode == "Demo Mode" and live_df is None:
    st.info("No live data found. Click 'Run Demo Inference' in the sidebar to generate demo predictions.")

round_label = _detect_round_label(live_df)
st.caption(
    f"{'**' + round_label + '** — p' if round_label else 'P'}robabilities updated with hole-by-hole scoring data"
)

# ── Metrics row ───────────────────────────────────────────────────────────────

if live_df is not None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Players tracked", len(live_df))
    if "holes_completed" in live_df.columns:
        col2.metric("Avg holes completed", f"{live_df['holes_completed'].mean():.1f}")
    else:
        col2.metric("Avg holes completed", "—")
    if "current_score_to_par" in live_df.columns:
        col3.metric("Leader score", score_str(live_df["current_score_to_par"].min()))
    else:
        col3.metric("Leader score", "—")
    col4.metric("Data refresh", "~120s (auto)")
    st.divider()
elif pre_df is not None:
    st.info("Showing pre-tournament predictions only. Start live inference to see updated probabilities.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Players", len(pre_df))
    c2.metric("Data source", "Pre-tournament model")
    c3.metric("Mode", "Baseline")
    st.divider()

# ── Main tabs ─────────────────────────────────────────────────────────────────

tab_outright, tab_top10, tab_other = st.tabs(["Outright", "Top 10", "Other"])

# ─────────────────────────────────────────────────────────────────────────────
# OUTRIGHT TAB
# ─────────────────────────────────────────────────────────────────────────────
with tab_outright:
    if live_df is not None:
        display_raw = live_df.copy()
        if "holes_completed" in display_raw.columns and min_holes > 0:
            display_raw = display_raw[display_raw["holes_completed"] >= min_holes]

        out_df = build_outright_df(display_raw, bankroll=bankroll, kelly_frac=kelly_frac)
        col_cfg = tab_column_config(kelly_frac, bankroll, market_label="Market %")

        # Value bets at top
        WIN_EDGE_THRESH = 1.0
        value_win = out_df[out_df["Edge"].fillna(0) > WIN_EDGE_THRESH].sort_values("Edge", ascending=False)

        st.subheader("🎯 Outright Value Bets")
        if value_win.empty:
            st.info(f"No outright edges above +{WIN_EDGE_THRESH:.0f}% threshold.")
        else:
            st.dataframe(
                value_win[["Player", "Score", "Model Odds", "Market Odds",
                            "Model %", "Market %", "Edge", "Kelly Bet", "Expected Win"]],
                use_container_width=True,
                hide_index=True,
                column_config=col_cfg,
            )
            render_kelly_explainer(value_win, "outright win", kelly_frac)
        st.caption("Edge = Model Win% minus Market implied win% (DK/FD/BetMGM best odds). Only positive Kelly bets shown.")

        st.divider()

        # Full leaderboard
        st.subheader(f"📋 Full Leaderboard — Top {min(top_n, len(out_df))} Players")
        _win_src = "MC sim" if ("mc_win_prob" in display_raw.columns and display_raw["mc_win_prob"].notna().any()) else "XGBoost blend"
        st.caption(f"Sorted by Model Win% — source: **{_win_src}**. Blank Kelly/Expected Win = no edge detected.")
        st.dataframe(
            out_df.head(top_n)[["Player", "Score", "Model Odds", "Market Odds",
                                 "Model %", "Market %", "Edge", "Kelly Bet", "Expected Win"]],
            use_container_width=True,
            hide_index=True,
            column_config=col_cfg,
        )

        render_historical_performance("outright")

    elif pre_df is not None:
        st.subheader("Pre-Tournament Outright Predictions")
        disp = pre_df.sort_values("win_prob", ascending=False).head(top_n).copy()
        disp["Model %"] = (disp["win_prob"] * 100).round(1)
        disp["Rank"] = range(1, len(disp) + 1)
        st.dataframe(
            disp[["Rank", "player_name", "Model %"]].rename(columns={"player_name": "Player"}),
            use_container_width=True,
            hide_index=True,
            column_config={"Model %": st.column_config.NumberColumn("Model %", format="%.1f%%")},
        )
    else:
        st.error("No data available. Run `python3 run_production.py` first.")

# ─────────────────────────────────────────────────────────────────────────────
# TOP 10 TAB
# ─────────────────────────────────────────────────────────────────────────────
with tab_top10:
    if live_df is not None:
        display_raw = live_df.copy()
        if "holes_completed" in display_raw.columns and min_holes > 0:
            display_raw = display_raw[display_raw["holes_completed"] >= min_holes]

        t10_df, mkt_label = build_top10_df(display_raw, bankroll=bankroll, kelly_frac=kelly_frac)
        col_cfg_t10 = tab_column_config(kelly_frac, bankroll, market_label=mkt_label)

        # Value bets at top
        T10_EDGE_THRESH = 3.0
        value_t10 = t10_df[t10_df["Edge"].fillna(0) > T10_EDGE_THRESH].sort_values("Edge", ascending=False)

        st.subheader("🎯 Top 10 Value Bets")
        if value_t10.empty:
            st.info(f"No T10 edges above +{T10_EDGE_THRESH:.0f}% threshold.")
        else:
            st.dataframe(
                value_t10[["Player", "Score", "Model Odds", "Market Odds",
                            "Model %", "Market %", "Edge", "Kelly Bet", "Expected Win"]],
                use_container_width=True,
                hide_index=True,
                column_config=col_cfg_t10,
            )
            render_kelly_explainer(value_t10, "top-10 finish", kelly_frac)

        if mkt_label == "Market (sportsbook)":
            st.caption("Edge = Model T10% minus real sportsbook implied probability (DK/FD/MGM).")
        else:
            st.caption("Edge = Model T10% vs DataGolf live model — no real T10 sportsbook lines available.")

        st.divider()

        # Full leaderboard
        _t10_src = "MC sim" if ("mc_top10_prob" in display_raw.columns and display_raw["mc_top10_prob"].notna().any()) else "XGBoost blend"
        st.subheader(f"📋 Full Top 10 Table — Top {min(top_n, len(t10_df))} Players")
        st.caption(f"Sorted by Model T10% — source: **{_t10_src}**. Blank Kelly/Expected Win = no edge detected.")
        st.dataframe(
            t10_df.head(top_n)[["Player", "Score", "Model Odds", "Market Odds",
                                  "Model %", "Market %", "Edge", "Kelly Bet", "Expected Win"]],
            use_container_width=True,
            hide_index=True,
            column_config=col_cfg_t10,
        )

        render_historical_performance("top10")

    elif pre_df is not None:
        st.subheader("Pre-Tournament T10 Predictions")
        disp = pre_df.sort_values("top10_prob", ascending=False).head(top_n).copy() if "top10_prob" in pre_df.columns else pre_df.head(top_n).copy()
        disp["Model %"] = (disp["top10_prob"] * 100).round(1) if "top10_prob" in disp.columns else None
        disp["Rank"] = range(1, len(disp) + 1)
        st.dataframe(
            disp[["Rank", "player_name", "Model %"]].rename(columns={"player_name": "Player"}),
            use_container_width=True,
            hide_index=True,
            column_config={"Model %": st.column_config.NumberColumn("Model %", format="%.1f%%")},
        )
    else:
        st.error("No data available. Run `python3 run_production.py` first.")

# ─────────────────────────────────────────────────────────────────────────────
# OTHER TAB
# ─────────────────────────────────────────────────────────────────────────────
with tab_other:
    st.subheader("Other Markets")
    st.info(
        "**Coming soon:** Top 5, Top 20, and Make Cut probability tables with Kelly sizing.\n\n"
        "These markets require additional sportsbook line feeds (The Odds API). "
        "Model probabilities are already computed — `model_top5_prob`, `model_top20_prob`, "
        "`model_make_cut` — and will populate once market lines are wired in."
    )

    # Show raw model probabilities if available
    if live_df is not None and any(c in live_df.columns for c in ["mc_top5_prob", "blended_top5_prob", "blended_make_cut_prob"]):
        d = live_df.sort_values("mc_win_prob" if "mc_win_prob" in live_df.columns else live_df.columns[0], ascending=False).head(top_n).copy()
        preview_cols = {"Player": "player_name"}
        for col, label in [("mc_top5_prob", "Model Top5%"), ("blended_top5_prob", "Model Top5%"),
                           ("mc_top20_prob", "Model Top20%"), ("blended_top20_prob", "Model Top20%"),
                           ("blended_make_cut_prob", "Model Cut%")]:
            if col in d.columns:
                d[label] = (d[col] * 100).round(1)
                preview_cols[label] = label
        d["Player"] = d["player_name"]
        show_cols_list = ["Player"] + [v for v in preview_cols if v != "Player" and v in d.columns]
        cfg = {c: st.column_config.NumberColumn(c, format="%.1f%%") for c in show_cols_list if c != "Player"}
        st.dataframe(d[show_cols_list], use_container_width=True, hide_index=True, column_config=cfg)

# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO ANALYSIS + MOVERS (below tabs, live only)
# ─────────────────────────────────────────────────────────────────────────────

if live_df is not None:
    display = live_df.copy()
    if "holes_completed" in display.columns and min_holes > 0:
        display = display[display["holes_completed"] >= min_holes]

    _sort_col = "mc_win_prob" if ("mc_win_prob" in display.columns and display["mc_win_prob"].notna().any()) else "blended_win_prob"
    if _sort_col in display.columns:
        display = display.sort_values(_sort_col, ascending=False).reset_index(drop=True)
    display["live_rank"] = range(1, len(display) + 1)
    rows_raw = display.head(top_n)

    if "holes_completed" in rows_raw.columns:
        avg_holes = rows_raw["holes_completed"].mean()
        rounds_remaining = max(0.25, 4.0 - avg_holes / 18.0)
    else:
        rounds_remaining = 2.0

    has_mc = "mc_win_prob" in rows_raw.columns and rows_raw["mc_win_prob"].notna().any()

    st.divider()
    st.subheader("📊 Scenario Analysis")

    if has_mc and "current_score_to_par" in rows_raw.columns:
        import plotly.graph_objects as go

        all_rows = display.copy()
        sc_sorted = all_rows.sort_values("current_score_to_par").reset_index(drop=True)
        leader_r = sc_sorted.iloc[0]
        leader_name = leader_r["player_name"]
        leader_score = int(leader_r["current_score_to_par"])
        leader_win_pct = float(leader_r.get("mc_win_prob", 0))
        collapse_pct = float(leader_r.get("mc_collapse_prob", 0) or 0)
        leader_p25 = leader_r.get("mc_proj_p25")
        leader_p75 = leader_r.get("mc_proj_p75")
        leader_p90 = leader_r.get("mc_proj_p90")
        leader_proj = leader_r.get("mc_projected_total")
        second_r = sc_sorted.iloc[1] if len(sc_sorted) > 1 else None
        shots_ahead = int(second_r["current_score_to_par"]) - leader_score if second_r is not None else 0

        if collapse_pct < 0.05:
            risk_color, risk_label, risk_emoji = "#28a745", "LOW RISK", "🟢"
        elif collapse_pct < 0.15:
            risk_color, risk_label, risk_emoji = "#e6a817", "MODERATE RISK", "🟡"
        else:
            risk_color, risk_label, risk_emoji = "#dc3545", "HIGH RISK", "🔴"

        second_best_case = second_r.get("mc_proj_p25") if second_r is not None else None
        if pd.notna(leader_p90) and second_best_case is not None and pd.notna(second_best_case):
            if leader_p90 < second_best_case:
                floor_txt = (f"Even in {leader_name}'s worst 10% of scenarios he finishes "
                             f"at **{score_str(leader_p90)}** — better than anyone else can "
                             f"realistically reach ({score_str(second_best_case)} is the best "
                             f"case for {second_r['player_name']}).")
            else:
                floor_txt = (f"In {leader_name}'s worst 10% of scenarios he finishes "
                             f"**{score_str(leader_p90)}**. "
                             f"{second_r['player_name']} could reach "
                             f"**{score_str(second_best_case)}** in their best case — "
                             f"a genuine threat if {leader_name} struggles.")
        elif pd.notna(leader_p90):
            floor_txt = f"Worst 10% scenario: {leader_name} finishes at **{score_str(leader_p90)}**."
        else:
            floor_txt = ""

        lc1, lc2 = st.columns([1, 2])
        with lc1:
            st.metric(f"🏆 {leader_name} to win", f"{leader_win_pct:.0%}")
            st.markdown(
                f"<div style='background:{risk_color}22; border-left:4px solid {risk_color};"
                f"padding:8px 12px; border-radius:4px; font-size:0.85em;'>"
                f"<b style='color:{risk_color}'>{risk_emoji} {risk_label}</b><br>"
                f"<span style='color:#666'>{collapse_pct:.1%} chance of blowing the lead</span>"
                f"</div>", unsafe_allow_html=True,
            )
        with lc2:
            st.markdown(f"**Leads by {shots_ahead} shots** with {rounds_remaining:.0f} rounds to play.")
            if floor_txt:
                st.markdown(floor_txt)
            if pd.notna(leader_p25) and pd.notna(leader_p75):
                st.markdown(
                    f"Projected finish range: **{score_str(leader_p25)}** *(best case)* "
                    f"→ **{score_str(leader_proj)}** *(most likely)* "
                    f"→ **{score_str(leader_p75)}** *(bad day)*"
                )

        st.markdown("")

        prob_rows = sc_sorted[["player_name", "mc_win_prob"]].head(8).copy()
        others = max(0.0, 1.0 - prob_rows["mc_win_prob"].sum())
        if others > 0.005:
            prob_rows = pd.concat([prob_rows,
                pd.DataFrame([{"player_name": "Rest of field", "mc_win_prob": others}])],
                ignore_index=True)
        bar_colors = ["#C9A84C", "#4A90D9", "#6BB3E0", "#8CBFE0", "#A8CBE0",
                      "#C4D8E0", "#B0BEC5", "#CFD8DC", "#E0E0E0"]
        fig_bar = go.Figure()
        for i, row in prob_rows.iterrows():
            p = row["mc_win_prob"]
            if p < 0.003:
                continue
            label = f"{row['player_name']} {p:.0%}" if p > 0.06 else f"{p:.0%}"
            fig_bar.add_trace(go.Bar(
                x=[p], y=[""], orientation="h", name=row["player_name"],
                marker_color=bar_colors[min(i, len(bar_colors) - 1)],
                text=label, textposition="inside", insidetextanchor="middle",
                hovertemplate=f"{row['player_name']}: {p:.1%}<extra></extra>",
            ))
        fig_bar.update_layout(
            barmode="stack", height=70,
            margin=dict(l=0, r=0, t=4, b=4), showlegend=False,
            xaxis=dict(range=[0, 1], visible=False),
            yaxis=dict(visible=False),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        st.markdown(f"**In {1 - leader_win_pct:.0%} of scenarios {leader_name} doesn't win — what would it take?**")

        chal_rows_out = []
        challengers = sc_sorted[sc_sorted["player_name"] != leader_name].head(9)
        for _, r in challengers.iterrows():
            wp = float(r.get("mc_win_prob", 0))
            shots_bk = int(r["current_score_to_par"]) - leader_score
            proj_rd = r.get("expected_score_per_round")

            if pd.notna(leader_proj) and rounds_remaining > 0:
                need_total = (leader_proj - 1) - r["current_score_to_par"]
                need_rd = need_total / rounds_remaining
            else:
                need_rd = np.nan

            if pd.notna(need_rd) and pd.notna(proj_rd):
                gap = need_rd - proj_rd
                if gap > -0.5:
                    difficulty = "🟢 Realistic"
                elif gap > -2.0:
                    difficulty = "🟡 Stretch"
                else:
                    difficulty = "🔴 Requires hot streak"
            else:
                difficulty = "—"

            gain_needed = abs(shots_bk) + 1
            gain_desc = (f"needs to gain {gain_needed} shots on {leader_name} "
                         f"over {rounds_remaining:.0f} round{'s' if rounds_remaining != 1 else ''}"
                         if rounds_remaining > 0 else "—")

            chal_rows_out.append({
                "Player": r["player_name"],
                "Shots back": abs(shots_bk),
                "Win chance": f"{wp:.1%}",
                "What they need": (f"{gain_desc} — avg {need_rd:+.1f}/rd vs projecting {proj_rd:+.1f}/rd"
                                   if pd.notna(need_rd) and pd.notna(proj_rd) else gain_desc),
                "Realistic?": difficulty,
            })

        if chal_rows_out:
            st.dataframe(pd.DataFrame(chal_rows_out), use_container_width=True, hide_index=True)

        # Historical comparison
        st.divider()
        st.subheader("📖 Closest Historical Match")
        st.caption("Which past Masters did this leaderboard most resemble after Round 2?")

        within_5_count = int((sc_sorted["current_score_to_par"] <= leader_score + 5).sum())
        margin_over_2nd = float(second_r["current_score_to_par"] - leader_score) if second_r is not None else 0.0
        match = find_historical_match(float(leader_score), margin_over_2nd, within_5_count)

        if match is not None:
            won_icon = "✅" if match["leader_won"] else "❌"
            outcome = (f"went on to **win by {match['final_margin']}**" if match["leader_won"]
                       else f"**did not win** — **{match['winner_name']}** came from behind")
            margin_s = (f"led by {int(match['margin'])} shot{'s' if int(match['margin']) != 1 else ''}"
                        if match["margin"] > 0 else "was tied for the lead")
            st.info(
                f"{won_icon} **{match['year']} Masters** — {match['leader_name']} {margin_s} "
                f"after R2 at **{score_str(match['leader_score'])}**, and {outcome}."
            )
            hcol1, hcol2 = st.columns(2)
            with hcol1:
                st.markdown(f"**{match['year']} R2 leaderboard** (historical)")
                hist_tbl = match["top6"][["player_name", "Score", "Finish"]].rename(
                    columns={"player_name": "Player", "Score": "Score after R2", "Finish": "Final finish"})
                st.dataframe(hist_tbl, use_container_width=True, hide_index=True)
            with hcol2:
                st.markdown("**Today's leaderboard**")
                today_tbl = sc_sorted.head(6)[["player_name", "current_score_to_par", "mc_win_prob"]].copy()
                today_tbl["Score"] = today_tbl["current_score_to_par"].apply(score_str)
                today_tbl["Win%"] = (today_tbl["mc_win_prob"] * 100).round(1)
                st.dataframe(
                    today_tbl[["player_name", "Score", "Win%"]].rename(columns={"player_name": "Player"}),
                    use_container_width=True, hide_index=True,
                    column_config={"Win%": st.column_config.NumberColumn("Win%", format="%.1f%%")},
                )
        else:
            st.info("Historical data not available for comparison.")

    else:
        st.info("Run live inference to see scenario analysis.")

    # Movers
    if "rank_change" in display.columns:
        st.divider()
        st.subheader("📈 Biggest Movers")
        mc1, mc2 = st.columns(2)
        _t10_col = "blended_top10_prob" if "blended_top10_prob" in display.columns else (
            "mc_top10_prob" if "mc_top10_prob" in display.columns else None)

        with mc1:
            st.markdown("**Gaining probability**")
            g_cols = ["player_name", "live_rank", "rank_change"] + ([_t10_col] if _t10_col else []) + (["current_score_to_par"] if "current_score_to_par" in display.columns else [])
            gainers = display.nlargest(5, "rank_change")[g_cols].copy()
            gainers["Move"] = gainers["rank_change"].apply(movement_arrow)
            if _t10_col:
                gainers["T10%"] = (gainers[_t10_col] * 100).round(1)
            if "current_score_to_par" in gainers.columns:
                gainers["Score"] = gainers["current_score_to_par"].apply(score_str)
            show = ["player_name", "live_rank", "Move"] + (["T10%"] if _t10_col else []) + (["Score"] if "Score" in gainers.columns else [])
            cfg = {"T10%": st.column_config.NumberColumn("T10%", format="%.1f%%")} if _t10_col else {}
            st.dataframe(gainers[show].rename(columns={"player_name": "Player", "live_rank": "Rank"}),
                         use_container_width=True, hide_index=True, column_config=cfg)

        with mc2:
            st.markdown("**Losing probability**")
            losers = display.nsmallest(5, "rank_change")[g_cols].copy()
            losers["Move"] = losers["rank_change"].apply(movement_arrow)
            if _t10_col:
                losers["T10%"] = (losers[_t10_col] * 100).round(1)
            if "current_score_to_par" in losers.columns:
                losers["Score"] = losers["current_score_to_par"].apply(score_str)
            st.dataframe(losers[show].rename(columns={"player_name": "Player", "live_rank": "Rank"}),
                         use_container_width=True, hide_index=True, column_config=cfg)

# ── Course Hole Difficulty ────────────────────────────────────────────────────

with st.expander("📊 Augusta Hole Difficulty (Historical)"):
    if course_stats is not None:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=course_stats["hole_number"],
            y=course_stats["avg_score_to_par"],
            marker_color=["#28a745" if v < 0 else "#dc3545" if v > 0.2 else "#ffc107"
                          for v in course_stats["avg_score_to_par"]],
            name="Avg Score vs Par",
            hovertemplate="Hole %{x}<br>Avg: %{y:+.3f}<br><extra></extra>",
        ))
        for hole in [11, 12, 13]:
            fig.add_vline(x=hole, line_dash="dot", line_color="rgba(200,100,0,0.4)")
        fig.update_layout(
            title="Scoring difficulty per hole (+ = harder, − = easier)",
            xaxis_title="Hole", yaxis_title="Avg score vs par",
            xaxis=dict(tickmode="linear", dtick=1), height=300, showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Orange dotted lines = Amen Corner (holes 11-13). Green = birdie opportunity, red = bogey-prone.")
    else:
        st.info("Course stats not available. Run training pipeline first.")

with st.expander("ℹ️ How confidence works"):
    st.markdown("""
**Confidence weight** = √(holes_completed / 18)

| Holes done | Confidence | Blend |
|------------|------------|-------|
| 0          | 0%         | 100% pre-tournament |
| 3          | 41%        | 41% live + 59% pre |
| 9          | 71%        | 71% live + 29% pre |
| 18         | 100%       | 100% live model |

The model blends live scoring data with pre-tournament predictions.
Early in the round, most of the probability comes from the baseline model.
By the back nine, live strokes-gained patterns dominate.

**Amen Corner (11-13)** has outsized importance — scoring here often predicts
final position better than anywhere else on the course.
""")
