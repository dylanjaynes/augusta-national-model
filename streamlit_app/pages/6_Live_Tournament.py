"""
Live Tournament page — Augusta National Model

Shows real-time (or demo) probability updates as players complete holes.
Highlights movers, confidence scores, and edge vs pre-tournament odds.

Refreshes automatically every 2 minutes when live data is available.
"""

import sys
import time
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

# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_live_predictions() -> pd.DataFrame | None:
    """
    Fetches live predictions from the data-live branch on GitHub (updated every 5 min
    by GitHub Actions — pushes go there, NOT to main, so Streamlit never redeploys
    just because data changed).

    Falls back to a local file for development / offline use.
    TTL=60s so the page auto-refreshes data within 1 minute of a new push.
    """
    # Bust the CDN cache by rotating a token every 60 seconds
    cache_bust = int(time.time() // 60)
    url = (
        "https://raw.githubusercontent.com/"
        "dylanjaynes/augusta-national-model/"
        f"data-live/data/live/live_predictions_latest.csv"
        f"?t={cache_bust}"
    )
    try:
        df = pd.read_csv(url)
        return df
    except Exception:
        pass
    # Local fallback (dev mode / no internet)
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
def load_course_stats() -> pd.DataFrame | None:
    p = PROCESSED / "course_hole_stats.parquet"
    if p.exists():
        return pd.read_parquet(p)
    # Fall back to computing from hole_by_hole
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
    """Format score_to_par as +/-."""
    if pd.isna(v):
        return "E"
    v = int(v)
    if v == 0:
        return "E"
    return f"+{v}" if v > 0 else str(v)


def pct_str(v, decimals=1) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:.{decimals}%}"


def format_american(v) -> str:
    """Format raw American odds integer/float (e.g. 280.0 → '+280', -110.0 → '-110')."""
    if pd.isna(v):
        return "—"
    try:
        v = int(round(float(v)))
    except (ValueError, TypeError):
        return str(v)
    return f"+{v}" if v >= 0 else str(v)


def color_edge(val):
    """Return CSS color for edge value."""
    if pd.isna(val):
        return "color: gray"
    if val > 0.03:
        return "color: #28a745; font-weight: bold"
    if val < -0.03:
        return "color: #dc3545"
    return "color: inherit"


def confidence_badge(pct: float) -> str:
    """Return colored badge string for confidence level."""
    if pct < 0.17:
        return "🔴 Low"
    if pct < 0.5:
        return "🟡 Medium"
    if pct < 0.83:
        return "🟢 Good"
    return "✅ Full"


def movement_arrow(rank_change):
    """Arrow indicator for rank movement."""
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


# ── Demo mode: run inference with synthetic data ──────────────────────────────

def run_demo_inference():
    """Run live inference in demo mode and save results."""
    import subprocess
    result = subprocess.run(
        ["python3", str(ROOT / "scripts" / "run_live_inference.py"), "--demo", "--round", "2"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    return result.returncode == 0, result.stdout, result.stderr


# ── Page Layout ───────────────────────────────────────────────────────────────

st.title("🏌️ Live Tournament — Augusta National 2026")
st.caption("Probabilities updated with hole-by-hole scoring data")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")

    live_mode = st.selectbox(
        "Data source",
        ["Live File (auto-refresh)", "Demo Mode", "Pre-Tournament Only"],
        index=0,
    )

    auto_refresh = st.checkbox("Auto-refresh every 2 minutes", value=False)
    if auto_refresh:
        refresh_interval = st.slider("Refresh interval (sec)", 30, 600, 120)
        st.info(f"Will refresh every {refresh_interval}s")
        time.sleep(0.1)
        st.rerun() if st.session_state.get("_refresh_count", 0) > 0 else None

    st.divider()

    show_cols = st.multiselect(
        "Extra columns",
        ["Win Edge vs Market", "Amen Corner", "Par 5 Scoring", "Rank Change", "Confidence"],
        default=["Rank Change", "Confidence"],
    )

    min_holes = st.slider("Min holes completed", 0, 18, 0)
    top_n = st.slider("Players to show", 5, 88, 30)

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

# Handle data modes
if live_mode == "Pre-Tournament Only":
    live_df = None

if live_mode == "Demo Mode" and live_df is None:
    st.info("No live data found. Click 'Run Demo Inference' in the sidebar to generate demo predictions.")

# ── Metrics Row ──────────────────────────────────────────────────────────────

if live_df is not None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Players tracked", len(live_df))

    if "holes_completed" in live_df.columns:
        avg_holes = live_df["holes_completed"].mean()
        col2.metric("Avg holes completed", f"{avg_holes:.1f}")
    else:
        col2.metric("Avg holes completed", "—")

    if "current_score_to_par" in live_df.columns:
        leader_score = live_df["current_score_to_par"].min()
        col3.metric("Leader score", score_str(leader_score))
    else:
        col3.metric("Leader score", "—")

    # Data freshness: Streamlit re-fetches from GitHub every 60s (ttl=60)
    col4.metric("Data refresh", "~60s (auto)")

    st.divider()

elif pre_df is not None:
    st.info("Showing pre-tournament predictions only. Start live inference to see updated probabilities.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Players", len(pre_df))
    col2.metric("Data source", "Pre-tournament model")
    col3.metric("Mode", "Baseline")
    st.divider()

# ── Main Table ────────────────────────────────────────────────────────────────

if live_df is not None:
    display = live_df.copy()

    # Apply holes filter
    if "holes_completed" in display.columns and min_holes > 0:
        display = display[display["holes_completed"] >= min_holes]

    # Ensure key columns exist
    for col in ["blended_top10_prob", "blended_win_prob", "live_rank"]:
        if col not in display.columns:
            display[col] = np.nan

    # Always sort by win probability (descending) and reassign ranks
    display = display.sort_values("blended_win_prob", ascending=False).reset_index(drop=True)
    display["live_rank"] = range(1, len(display) + 1)

    # ── Single unified table ──────────────────────────────────────────────────
    # Player | Score | M Win% | M Odds | Bk Odds | M T10% | DG T10% | Edge
    # Sorted by M Win% descending. Simple st.dataframe — no column_config tricks.

    rows = display.head(top_n).copy()

    # ── Compute rounds remaining (for Need/Rd) ───────────────────────────────
    if "holes_completed" in rows.columns:
        avg_holes = rows["holes_completed"].mean()
        rounds_remaining = max(0.25, 4.0 - avg_holes / 18.0)
    else:
        rounds_remaining = 2.0  # fallback: assume mid-tournament

    # Leader's projected final total (lowest mc_projected_total wins)
    if "mc_projected_total" in rows.columns and rows["mc_projected_total"].notna().any():
        leader_proj_total = rows["mc_projected_total"].min()
    elif "current_score_to_par" in rows.columns:
        leader_proj_total = rows["current_score_to_par"].min()
    else:
        leader_proj_total = None

    tbl = pd.DataFrame()
    tbl["Player"]    = rows["player_name"].values
    tbl["Score"]     = rows["current_score_to_par"].apply(score_str).values \
                       if "current_score_to_par" in rows.columns else "—"
    # MC projected finish (what the model expects them to shoot total)
    if "mc_projected_total" in rows.columns:
        tbl["Proj Total"] = rows["mc_projected_total"].apply(
            lambda x: f"{int(x):+d}" if pd.notna(x) else "—"
        ).values
    # Projected scoring rate per remaining round (player skill + form)
    if "expected_score_per_round" in rows.columns:
        tbl["Proj/Rd"] = rows["expected_score_per_round"].apply(
            lambda x: f"{x:+.2f}" if pd.notna(x) else "—"
        ).values
    # What the player needs to average per remaining round to win
    if leader_proj_total is not None and "current_score_to_par" in rows.columns and rounds_remaining > 0:
        need_vals = (leader_proj_total - 1 - rows["current_score_to_par"]) / rounds_remaining
        tbl["Need/Rd"] = need_vals.apply(
            lambda x: f"{x:+.1f}" if pd.notna(x) else "—"
        ).values
    # Projected finish range (P25–P75): the realistic band of outcomes
    if "mc_proj_p25" in rows.columns and "mc_proj_p75" in rows.columns:
        def _range_str(row):
            p25, p75 = row["mc_proj_p25"], row["mc_proj_p75"]
            if pd.isna(p25) or pd.isna(p75):
                return "—"
            return f"{int(p25):+d} / {int(p75):+d}"
        tbl["Range (25/75)"] = rows.apply(_range_str, axis=1).values

    tbl["MC Win%"] = rows["mc_win_prob"].apply(pct_str).values \
                     if "mc_win_prob" in rows.columns else "—"
    tbl["M Win%"]  = rows["blended_win_prob"].apply(pct_str).values
    tbl["M Odds"]  = rows["model_american_win"].apply(format_american).values \
                     if "model_american_win" in rows.columns else "—"
    tbl["Bk Odds"] = rows["book_american_win"].apply(format_american).values \
                     if "book_american_win" in rows.columns else "—"
    tbl["M T10%"]  = rows["blended_top10_prob"].apply(pct_str).values \
                     if "blended_top10_prob" in rows.columns else "—"
    tbl["DG T10%"] = rows["dg_top10_prob"].apply(pct_str).values \
                     if "dg_top10_prob" in rows.columns else "—"
    tbl["Edge"]    = rows["live_edge_vs_book"].apply(
                         lambda x: f"{x:+.1%}" if pd.notna(x) else "—"
                     ).values if "live_edge_vs_book" in rows.columns else "—"

    st.subheader(f"Live Leaderboard — Top {min(top_n, len(tbl))} Players")
    st.caption(
        "Sorted by Model Win% (descending). "
        "**Proj/Rd** = projected scoring rate per remaining round. "
        "**Need/Rd** = pace needed to beat the leader's projected finish. "
        "**Range (25/75)** = optimistic / pessimistic projected total. "
        "Bk Odds = best of DK/FD/BetMGM. Edge = M Win% minus Book implied win%."
    )
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    # ── Scenario Analysis ────────────────────────────────────────────────────
    has_mc = "mc_win_prob" in rows.columns and rows["mc_win_prob"].notna().any()
    with st.expander("📊 Scenario Analysis", expanded=False):
        if has_mc:
            # Identify the leaderboard leader by current score
            if "current_score_to_par" in rows.columns:
                leader_idx  = rows["current_score_to_par"].idxmin()
                leader_name = rows.loc[leader_idx, "player_name"]
                leader_score_str = score_str(rows.loc[leader_idx, "current_score_to_par"])
            else:
                leader_idx  = rows.index[0]
                leader_name = rows.iloc[0]["player_name"]
                leader_score_str = "—"

            leader_win_pct   = rows.loc[leader_idx, "mc_win_prob"]
            collapse_pct     = rows.loc[leader_idx, "mc_collapse_prob"] \
                               if "mc_collapse_prob" in rows.columns else np.nan
            leader_p90_total = rows.loc[leader_idx, "mc_proj_p90"] \
                               if "mc_proj_p90" in rows.columns else np.nan

            # ── Section 1: Leader Win Distribution ───────────────────────────
            st.markdown(f"### 🏆 Leader: {leader_name} ({leader_score_str})")

            lc1, lc2, lc3 = st.columns(3)
            lc1.metric("Win probability", f"{leader_win_pct:.1%}")
            lc2.metric(
                "Collapses (finishes outside top 3)",
                f"{collapse_pct:.1%}" if pd.notna(collapse_pct) else "—",
            )
            lc3.metric(
                "Worst-case finish (90th pct)",
                f"{int(leader_p90_total):+d}" if pd.notna(leader_p90_total) else "—",
            )

            st.markdown(
                f"**In {1 - leader_win_pct:.0%} of scenarios {leader_name} doesn't win — "
                f"who does?**"
            )

            # Top 8 challengers by win prob (excluding leader)
            challengers = (
                rows[rows["player_name"] != leader_name]
                .sort_values("mc_win_prob", ascending=False)
                .head(8)
            )
            chal_rows = []
            for _, r in challengers.iterrows():
                wp = r["mc_win_prob"]
                if wp < 0.002:
                    continue
                chal_rows.append({
                    "Player":      r["player_name"],
                    "Score":       score_str(r["current_score_to_par"])
                                   if "current_score_to_par" in r else "—",
                    "Win%":        pct_str(wp),
                    "Proj Total":  f"{int(r['mc_projected_total']):+d}"
                                   if pd.notna(r.get("mc_projected_total")) else "—",
                    "Range":       (f"{int(r['mc_proj_p25']):+d} / {int(r['mc_proj_p75']):+d}"
                                    if "mc_proj_p25" in r and pd.notna(r["mc_proj_p25"]) else "—"),
                })
            if chal_rows:
                st.dataframe(pd.DataFrame(chal_rows), use_container_width=True, hide_index=True)

            st.divider()

            # ── Section 2: What It Takes to Win ──────────────────────────────
            st.markdown("### 🎯 What It Takes to Win")
            st.caption(
                "For each player: their **projected pace** (skill/form estimate) vs the "
                "**pace they actually need** in the scenarios where they win. "
                "A wide gap = they need to go on a hot streak. A small gap = well-positioned."
            )

            win_rows = []
            all_players_rows = rows.sort_values("mc_win_prob", ascending=False).head(12)
            for _, r in all_players_rows.iterrows():
                wp  = r["mc_win_prob"]
                wss = r.get("mc_win_scenario_score")
                proj_rd = r.get("expected_score_per_round")
                need_rd = (
                    (leader_proj_total - 1 - r.get("current_score_to_par", 0)) / rounds_remaining
                    if leader_proj_total is not None and rounds_remaining > 0 else np.nan
                )
                if wp < 0.003 or pd.isna(wss):
                    continue
                win_rows.append({
                    "Player":           r["player_name"],
                    "Win%":             pct_str(wp),
                    "Proj pace/rd":     f"{proj_rd:+.2f}" if pd.notna(proj_rd) else "—",
                    "Need/rd (to win)": f"{need_rd:+.1f}" if pd.notna(need_rd) else "—",
                    "Actual pace in wins": f"{wss:+.2f}" if pd.notna(wss) else "—",
                    "Gap (need vs proj)": (
                        f"{wss - proj_rd:+.2f}" if pd.notna(wss) and pd.notna(proj_rd) else "—"
                    ),
                })
            if win_rows:
                st.dataframe(pd.DataFrame(win_rows), use_container_width=True, hide_index=True)
                st.caption(
                    "**Actual pace in wins** = avg remaining score/round in simulations where that "
                    "player wins. **Gap** = how far above their projected pace they need to run — "
                    "smaller gap = more realistic path to victory."
                )

            st.divider()

            # ── Section 3: Leader Collapse Scenarios ─────────────────────────
            st.markdown("### 💥 Leader Collapse Scenarios")
            st.caption(
                f"What does a {leader_name} collapse look like? "
                f"The distribution of their projected finish score across all simulations."
            )
            if "mc_proj_p25" in rows.columns:
                leader_r = rows.loc[leader_idx]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Best case (P10)", f"{int(leader_r.get('mc_proj_p25', 0) - 3):+d}" if pd.notna(leader_r.get("mc_proj_p25")) else "—")
                c2.metric("Optimistic (P25)", f"{int(leader_r['mc_proj_p25']):+d}" if pd.notna(leader_r.get("mc_proj_p25")) else "—")
                c3.metric("Median (P50)", f"{int(leader_r['mc_projected_total']):+d}" if pd.notna(leader_r.get("mc_projected_total")) else "—")
                c4.metric("Pessimistic (P75)", f"{int(leader_r['mc_proj_p75']):+d}" if pd.notna(leader_r.get("mc_proj_p75")) else "—")

                # How many players have P25 (optimistic) better than leader's P75 (pessimistic)?
                leader_p75 = leader_r.get("mc_proj_p75")
                if pd.notna(leader_p75) and "mc_proj_p25" in rows.columns:
                    threats = rows[
                        (rows["player_name"] != leader_name) &
                        (rows["mc_proj_p25"] <= leader_p75)
                    ].sort_values("mc_win_prob", ascending=False)
                    if not threats.empty:
                        st.markdown(
                            f"**{len(threats)} player(s) have an optimistic scenario that beats "
                            f"{leader_name}'s pessimistic finish ({int(leader_p75):+d}):**"
                        )
                        threat_tbl = threats.head(5)[["player_name", "current_score_to_par",
                                                       "mc_proj_p25", "mc_win_prob"]].copy()
                        threat_tbl.columns = ["Player", "Score", "Best proj", "Win%"]
                        threat_tbl["Score"]    = threat_tbl["Score"].apply(score_str)
                        threat_tbl["Best proj"] = threat_tbl["Best proj"].apply(
                            lambda x: f"{int(x):+d}" if pd.notna(x) else "—")
                        threat_tbl["Win%"]     = threat_tbl["Win%"].apply(pct_str)
                        st.dataframe(threat_tbl, use_container_width=True, hide_index=True)
        else:
            st.info("Run live inference to see scenario analysis.")

    # ── Movers section ──────────────────────────────────────────────────────
    if "rank_change" in display.columns:
        st.divider()
        st.subheader("📈 Biggest Movers")
        mc1, mc2 = st.columns(2)

        with mc1:
            st.markdown("**Gaining probability**")
            gainers = display.nlargest(5, "rank_change")[
                ["player_name", "live_rank", "rank_change", "blended_top10_prob",
                 "current_score_to_par"] if "current_score_to_par" in display.columns
                else ["player_name", "live_rank", "rank_change", "blended_top10_prob"]
            ].copy()
            gainers["Move"] = gainers["rank_change"].apply(movement_arrow)
            gainers["T10%"] = gainers["blended_top10_prob"].apply(pct_str)
            if "current_score_to_par" in gainers.columns:
                gainers["Score"] = gainers["current_score_to_par"].apply(score_str)
            st.dataframe(
                gainers[["player_name", "live_rank", "Move", "T10%"] +
                        (["Score"] if "Score" in gainers.columns else [])].rename(columns={"player_name": "Player", "live_rank": "Rank"}),
                use_container_width=True, hide_index=True
            )

        with mc2:
            st.markdown("**Losing probability**")
            losers = display.nsmallest(5, "rank_change")[
                ["player_name", "live_rank", "rank_change", "blended_top10_prob",
                 "current_score_to_par"] if "current_score_to_par" in display.columns
                else ["player_name", "live_rank", "rank_change", "blended_top10_prob"]
            ].copy()
            losers["Move"] = losers["rank_change"].apply(movement_arrow)
            losers["T10%"] = losers["blended_top10_prob"].apply(pct_str)
            if "current_score_to_par" in losers.columns:
                losers["Score"] = losers["current_score_to_par"].apply(score_str)
            st.dataframe(
                losers[["player_name", "live_rank", "Move", "T10%"] +
                       (["Score"] if "Score" in losers.columns else [])].rename(columns={"player_name": "Player", "live_rank": "Rank"}),
                use_container_width=True, hide_index=True
            )

elif pre_df is not None:
    # Show pre-tournament table as fallback
    st.subheader("Pre-Tournament Predictions (No Live Data)")
    disp = pre_df.sort_values("win_prob", ascending=False).head(top_n).copy()
    disp["Win%"] = disp["win_prob"].apply(pct_str)
    disp["T10%"] = disp["top10_prob"].apply(pct_str) if "top10_prob" in disp.columns else "—"
    disp["Rank"] = range(1, len(disp) + 1)
    cols = ["Rank", "player_name", "dg_rank", "Win%", "T10%"]
    cols = [c for c in cols if c in disp.columns]
    st.dataframe(
        disp[cols].rename(columns={"player_name": "Player", "dg_rank": "World Rank"}),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.error("No data available. Run `python3 run_production.py` first.")
    st.stop()

# ── Course Hole Difficulty Chart ─────────────────────────────────────────────

with st.expander("📊 Augusta Hole Difficulty (Historical)"):
    if course_stats is not None:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=course_stats["hole_number"],
            y=course_stats["avg_score_to_par"],
            marker_color=[
                "#28a745" if v < 0 else "#dc3545" if v > 0.2 else "#ffc107"
                for v in course_stats["avg_score_to_par"]
            ],
            name="Avg Score vs Par",
            hovertemplate=(
                "Hole %{x}<br>"
                "Avg: %{y:+.3f}<br>"
                "<extra></extra>"
            ),
        ))

        # Annotate Amen Corner
        for hole in [11, 12, 13]:
            fig.add_vline(x=hole, line_dash="dot", line_color="rgba(200,100,0,0.4)")

        fig.update_layout(
            title="Scoring difficulty per hole (+ = harder, − = easier)",
            xaxis_title="Hole",
            yaxis_title="Avg score vs par",
            xaxis=dict(tickmode="linear", dtick=1),
            height=300,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Orange dotted lines = Amen Corner (holes 11-13). Green = birdie opportunity, red = bogey-prone.")
    else:
        st.info("Course stats not available. Run training pipeline first.")

# ── Live Model Confidence Explanation ────────────────────────────────────────

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

# ── Auto-refresh ──────────────────────────────────────────────────────────────

if auto_refresh:
    st.caption(f"Auto-refreshing every {refresh_interval} seconds...")
    time.sleep(refresh_interval)
    st.rerun()
