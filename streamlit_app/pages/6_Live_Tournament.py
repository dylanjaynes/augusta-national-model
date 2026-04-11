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

_GITHUB_API_URL = (
    "https://api.github.com/repos/dylanjaynes/augusta-national-model"
    "/contents/data/live/live_predictions_latest.csv?ref=data-live"
)
_RAW_FALLBACK_URL = (
    "https://raw.githubusercontent.com/dylanjaynes/augusta-national-model"
    "/data-live/data/live/live_predictions_latest.csv"
)

@st.cache_data(ttl=55)   # slightly under 60s so two refreshes-per-cycle can't both hit stale cache
def load_live_predictions() -> pd.DataFrame | None:
    """
    Fetches live predictions via the GitHub Contents API (bypasses CDN — always latest).
    Falls back to raw URL, then local file for offline development.

    GitHub Actions pushes to the orphan 'data-live' branch every 5 minutes.
    This function is called on every page rerun; the page reruns every 90s
    unconditionally (see bottom of file), so data is at most ~6 min stale.
    """
    import base64, io, json, urllib.request

    # ── Method 1: GitHub Contents API (guaranteed fresh, no CDN cache) ────────
    try:
        req = urllib.request.Request(
            _GITHUB_API_URL,
            headers={"Accept": "application/vnd.github.v3+json",
                     "Cache-Control": "no-cache"},
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            meta = json.loads(r.read())
        content = base64.b64decode(meta["content"])
        return pd.read_csv(io.BytesIO(content))
    except Exception:
        pass

    # ── Method 2: Raw URL with aggressive cache-busting ───────────────────────
    try:
        bust = int(time.time())          # unique per second — guarantees CDN miss
        df = pd.read_csv(f"{_RAW_FALLBACK_URL}?t={bust}")
        return df
    except Exception:
        pass

    # ── Method 3: Local file (dev / offline) ──────────────────────────────────
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
    """
    Find the historical Masters R2 leaderboard most similar to the current one.
    Only uses 2021-2025 (SG era, to-par scores). Similarity = weighted distance
    on (leader score, lead margin, field compression).
    """
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
        h_leader_score   = float(yr["r2_cum"].iloc[0])
        h_margin         = float(yr["r2_cum"].iloc[1] - yr["r2_cum"].iloc[0])
        h_within_5       = int((yr["r2_cum"] <= h_leader_score + 5).sum())
        h_leader_name    = yr.iloc[0]["player_name"]
        h_leader_won     = bool(yr.iloc[0].get("finish_num", 999) == 1)
        winner_row       = yr[yr["finish_num"] == 1]
        h_winner_name    = winner_row.iloc[0]["player_name"] if not winner_row.empty else "Unknown"
        # Final winning margin
        h_final_margin   = None
        if not winner_row.empty:
            ru = yr[yr["finish_num"] == 2]
            if not ru.empty:
                try:
                    w_tot  = winner_row.iloc[0][["r1_score","r2_score","r3_score","r4_score"]].sum()
                    ru_tot = ru.iloc[0][["r1_score","r2_score","r3_score","r4_score"]].sum()
                    h_final_margin = int(round(ru_tot - w_tot))
                except Exception:
                    pass
        # Top-6 historical leaderboard
        top6 = yr.head(6)[["player_name","r2_cum","finish_pos","finish_num"]].copy()
        top6["Score"] = top6["r2_cum"].apply(lambda x: "E" if int(x)==0 else f"{int(x):+d}")
        top6["Finish"] = top6["finish_pos"].fillna(
            top6["finish_num"].apply(lambda x: f"T{int(x)}" if pd.notna(x) and x < 900 else "MC")
        )
        # Similarity score (lower = closer match)
        sim = (abs(h_leader_score - leader_score) * 1.0
               + abs(h_margin - margin_over_2nd) * 1.5
               + abs(h_within_5 - within_5) * 0.5)
        results.append(dict(year=int(year), leader_name=h_leader_name, leader_score=h_leader_score,
                            margin=h_margin, within_5=h_within_5, leader_won=h_leader_won,
                            winner_name=h_winner_name, final_margin=h_final_margin,
                            similarity=sim, top6=top6))
    if not results:
        return None
    return min(results, key=lambda x: x["similarity"])


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


def pct_num(v):
    """Return v * 100 as float for NumberColumn display (NaN → None)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return round(float(v) * 100, 1)
    except (TypeError, ValueError):
        return None


_PCT_CFG  = {"format": "%.1f%%"}
_EDGE_CFG = {"format": "%+.1f%%"}


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

    st.info("🔄 Auto-refreshes every 90 seconds")

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

    tbl["MC Win%"] = rows["mc_win_prob"].apply(pct_num).values \
                     if "mc_win_prob" in rows.columns else None
    tbl["MC T10%"] = rows["mc_top10_prob"].apply(pct_num).values \
                     if "mc_top10_prob" in rows.columns else None
    tbl["M Win%"]  = rows["blended_win_prob"].apply(pct_num).values
    tbl["M Odds"]  = rows["model_american_win"].apply(format_american).values \
                     if "model_american_win" in rows.columns else "—"
    tbl["Bk Odds"] = rows["book_american_win"].apply(format_american).values \
                     if "book_american_win" in rows.columns else "—"
    tbl["M T10%"]  = rows["blended_top10_prob"].apply(pct_num).values \
                     if "blended_top10_prob" in rows.columns else None
    tbl["DG T10%"] = rows["dg_top10_prob"].apply(pct_num).values \
                     if "dg_top10_prob" in rows.columns else None
    tbl["Edge"]    = rows["live_edge_vs_book"].apply(pct_num).values \
                     if "live_edge_vs_book" in rows.columns else None

    tbl_col_cfg = {c: st.column_config.NumberColumn(c, **_PCT_CFG)
                   for c in ["MC Win%", "MC T10%", "M Win%", "M T10%", "DG T10%"]
                   if c in tbl.columns}
    if "Edge" in tbl.columns:
        tbl_col_cfg["Edge"] = st.column_config.NumberColumn("Edge", **_EDGE_CFG)

    st.subheader(f"Live Leaderboard — Top {min(top_n, len(tbl))} Players")
    st.caption(
        "Sorted by Model Win% (descending). "
        "**Proj/Rd** = projected scoring rate per remaining round. "
        "**Need/Rd** = pace needed to beat the leader's projected finish. "
        "**Range (25/75)** = optimistic / pessimistic projected total. "
        "**MC Win%** / **MC T10%** = Monte Carlo win / top-10 probability (position-aware, 20k sims). "
        "Bk Odds = best of DK/FD/BetMGM. Edge = M Win% minus Book implied win%."
    )
    st.dataframe(tbl, use_container_width=True, hide_index=True, column_config=tbl_col_cfg)

    # ── Scenario Analysis (narrative, intuitive) ─────────────────────────────
    has_mc = "mc_win_prob" in rows.columns and rows["mc_win_prob"].notna().any()

    st.divider()
    st.subheader("📊 Scenario Analysis")

    if has_mc and "current_score_to_par" in rows.columns:
        # Core leader facts
        all_rows = display.copy()   # full field, not just top_n
        sc_sorted = all_rows.sort_values("current_score_to_par").reset_index(drop=True)
        leader_r      = sc_sorted.iloc[0]
        leader_name   = leader_r["player_name"]
        leader_score  = int(leader_r["current_score_to_par"])
        leader_win_pct = float(leader_r.get("mc_win_prob", 0))
        collapse_pct   = float(leader_r.get("mc_collapse_prob", 0) or 0)
        leader_p25     = leader_r.get("mc_proj_p25")
        leader_p75     = leader_r.get("mc_proj_p75")
        leader_p90     = leader_r.get("mc_proj_p90")
        leader_proj    = leader_r.get("mc_projected_total")
        second_r       = sc_sorted.iloc[1] if len(sc_sorted) > 1 else None
        shots_ahead    = int(second_r["current_score_to_par"]) - leader_score if second_r is not None else 0

        # ── Section 1: Leader status card ────────────────────────────────────
        if collapse_pct < 0.05:
            risk_color, risk_label, risk_emoji = "#28a745", "LOW RISK", "🟢"
        elif collapse_pct < 0.15:
            risk_color, risk_label, risk_emoji = "#e6a817", "MODERATE RISK", "🟡"
        else:
            risk_color, risk_label, risk_emoji = "#dc3545", "HIGH RISK", "🔴"

        # Floor narrative: compare leader's worst case vs challengers' best case
        second_best_case = second_r.get("mc_proj_p25") if second_r is not None else None
        if pd.notna(leader_p90) and pd.notna(second_best_case):
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
                f"</div>", unsafe_allow_html=True)
        with lc2:
            st.markdown(f"**Leads by {shots_ahead} shots** with {rounds_remaining:.0f} rounds to play.")
            if floor_txt:
                st.markdown(floor_txt)
            # Projected finish band for leader
            if pd.notna(leader_p25) and pd.notna(leader_p75):
                st.markdown(
                    f"Projected finish range: **{score_str(leader_p25)}** *(best case)* "
                    f"→ **{score_str(leader_proj)}** *(most likely)* "
                    f"→ **{score_str(leader_p75)}** *(bad day)*"
                )

        st.markdown("")

        # ── Section 2: Win probability bar ───────────────────────────────────
        import plotly.graph_objects as go
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
                marker_color=bar_colors[min(i, len(bar_colors)-1)],
                text=label, textposition="inside", insidetextanchor="middle",
                hovertemplate=f"{row['player_name']}: {p:.1%}<extra></extra>",
            ))
        fig_bar.update_layout(
            barmode="stack", height=70,
            margin=dict(l=0, r=0, t=4, b=4), showlegend=False,
            xaxis=dict(range=[0,1], visible=False),
            yaxis=dict(visible=False),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        # ── Section 3: What challengers need ─────────────────────────────────
        st.markdown(f"**In {1-leader_win_pct:.0%} of scenarios {leader_name} doesn't win — what would it take?**")

        chal_rows_out = []
        challengers = sc_sorted[sc_sorted["player_name"] != leader_name].head(9)
        for _, r in challengers.iterrows():
            wp        = float(r.get("mc_win_prob", 0))
            shots_bk  = int(r["current_score_to_par"]) - leader_score
            proj_rd   = r.get("expected_score_per_round")
            wss       = r.get("mc_win_scenario_score")          # pace in winning sims

            # What they need per round to beat leader's projection
            if pd.notna(leader_proj) and rounds_remaining > 0:
                need_total = (leader_proj - 1) - r["current_score_to_par"]
                need_rd    = need_total / rounds_remaining
            else:
                need_rd = np.nan

            # Human-readable difficulty
            if pd.notna(need_rd) and pd.notna(proj_rd):
                gap = need_rd - proj_rd   # how much better than projected they need
                if gap > -0.5:
                    difficulty = "🟢 Realistic"
                elif gap > -2.0:
                    difficulty = "🟡 Stretch"
                else:
                    difficulty = "🔴 Requires hot streak"
            else:
                difficulty = "—"

            # Plain english: "needs to gain X shots on Rory over 2 rounds"
            gain_needed = abs(shots_bk) + 1   # to beat, not tie
            gain_desc = (f"needs to gain {gain_needed} shots on {leader_name} "
                         f"over {rounds_remaining:.0f} round{'s' if rounds_remaining != 1 else ''}"
                         if rounds_remaining > 0 else "—")

            chal_rows_out.append({
                "Player":        r["player_name"],
                "Shots back":    abs(shots_bk),
                "Win chance":    pct_str(wp),
                "What they need": (f"{gain_desc} — "
                                   f"avg {need_rd:+.1f}/rd vs projecting {proj_rd:+.1f}/rd"
                                   if pd.notna(need_rd) and pd.notna(proj_rd) else gain_desc),
                "Realistic?":    difficulty,
            })

        if chal_rows_out:
            st.dataframe(pd.DataFrame(chal_rows_out), use_container_width=True, hide_index=True)

        # ── Section 4: Historical comparison ─────────────────────────────────
        st.divider()
        st.subheader("📖 Closest Historical Match")
        st.caption("Which past Masters did this leaderboard most resemble after Round 2?")

        within_5_count = int((sc_sorted["current_score_to_par"] <= leader_score + 5).sum())
        margin_over_2nd = float(second_r["current_score_to_par"] - leader_score) if second_r is not None else 0.0

        match = find_historical_match(float(leader_score), margin_over_2nd, within_5_count)

        if match is not None:
            won_icon = "✅" if match["leader_won"] else "❌"
            outcome  = (f"went on to **win by {match['final_margin']}**" if match["leader_won"]
                        else f"**did not win** — **{match['winner_name']}** came from behind")
            margin_s = (f"led by {int(match['margin'])} shot{'s' if int(match['margin'])!=1 else ''}"
                        if match["margin"] > 0 else "was tied for the lead")

            st.info(
                f"{won_icon} **{match['year']} Masters** — {match['leader_name']} {margin_s} "
                f"after R2 at **{score_str(match['leader_score'])}**, and {outcome}."
            )

            hcol1, hcol2 = st.columns(2)
            with hcol1:
                st.markdown(f"**{match['year']} R2 leaderboard** (historical)")
                hist_tbl = match["top6"][["player_name","Score","Finish"]].rename(
                    columns={"player_name":"Player","Score":"Score after R2","Finish":"Final finish"})
                st.dataframe(hist_tbl, use_container_width=True, hide_index=True)

            with hcol2:
                st.markdown("**Today's leaderboard**")
                today_tbl = sc_sorted.head(6)[["player_name","current_score_to_par","mc_win_prob"]].copy()
                today_tbl["Score"] = today_tbl["current_score_to_par"].apply(score_str)
                today_tbl["Win%"]  = today_tbl["mc_win_prob"].apply(pct_num)
                st.dataframe(today_tbl[["player_name","Score","Win%"]].rename(
                    columns={"player_name":"Player"}), use_container_width=True, hide_index=True,
                    column_config={"Win%": st.column_config.NumberColumn("Win%", **_PCT_CFG)})
        else:
            st.info("Historical data not available for comparison.")

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
            gainers["T10%"] = gainers["blended_top10_prob"].apply(pct_num)
            if "current_score_to_par" in gainers.columns:
                gainers["Score"] = gainers["current_score_to_par"].apply(score_str)
            st.dataframe(
                gainers[["player_name", "live_rank", "Move", "T10%"] +
                        (["Score"] if "Score" in gainers.columns else [])].rename(columns={"player_name": "Player", "live_rank": "Rank"}),
                use_container_width=True, hide_index=True,
                column_config={"T10%": st.column_config.NumberColumn("T10%", **_PCT_CFG)},
            )

        with mc2:
            st.markdown("**Losing probability**")
            losers = display.nsmallest(5, "rank_change")[
                ["player_name", "live_rank", "rank_change", "blended_top10_prob",
                 "current_score_to_par"] if "current_score_to_par" in display.columns
                else ["player_name", "live_rank", "rank_change", "blended_top10_prob"]
            ].copy()
            losers["Move"] = losers["rank_change"].apply(movement_arrow)
            losers["T10%"] = losers["blended_top10_prob"].apply(pct_num)
            if "current_score_to_par" in losers.columns:
                losers["Score"] = losers["current_score_to_par"].apply(score_str)
            st.dataframe(
                losers[["player_name", "live_rank", "Move", "T10%"] +
                       (["Score"] if "Score" in losers.columns else [])].rename(columns={"player_name": "Player", "live_rank": "Rank"}),
                use_container_width=True, hide_index=True,
                column_config={"T10%": st.column_config.NumberColumn("T10%", **_PCT_CFG)},
            )

elif pre_df is not None:
    # Show pre-tournament table as fallback
    st.subheader("Pre-Tournament Predictions (No Live Data)")
    disp = pre_df.sort_values("win_prob", ascending=False).head(top_n).copy()
    disp["Win%"] = disp["win_prob"].apply(pct_num)
    disp["T10%"] = disp["top10_prob"].apply(pct_num) if "top10_prob" in disp.columns else None
    disp["Rank"] = range(1, len(disp) + 1)
    cols = ["Rank", "player_name", "dg_rank", "Win%", "T10%"]
    cols = [c for c in cols if c in disp.columns]
    st.dataframe(
        disp[cols].rename(columns={"player_name": "Player", "dg_rank": "World Rank"}),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Win%": st.column_config.NumberColumn("Win%", **_PCT_CFG),
            "T10%": st.column_config.NumberColumn("T10%", **_PCT_CFG),
        },
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

# ── Unconditional auto-refresh ────────────────────────────────────────────────
# Page fully renders to the user above, then sleeps 90s, then reruns from top.
# Combined with ttl=55 cache on load_live_predictions(), data is always fresh.
# GitHub Actions pushes new data to 'data-live' branch every 5 minutes.

_REFRESH_SECS = 90
st.caption(f"🔄 Auto-refreshing every {_REFRESH_SECS}s — data updated every 5 min by GitHub Actions")
time.sleep(_REFRESH_SECS)
st.rerun()
