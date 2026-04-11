"""
Head-to-Head Comparison page — Augusta National Model (mobile-first redesign)
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(
    page_title="Head-to-Head — Augusta Model",
    layout="wide",
    page_icon="⚔️",
)

# ── CSS — same clean style as Player Profiles ─────────────────────────────────
st.markdown("""
<style>
.player-card {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 8px;
}
.player-card-a { border-top: 4px solid #1565c0; }
.player-card-b { border-top: 4px solid #c62828; }
.big-score {
    font-size: 3rem;
    font-weight: 700;
    line-height: 1.1;
    color: #1a1a1a;
}
.big-pos {
    font-size: 2.2rem;
    font-weight: 600;
    color: #444;
}
.stat-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888;
    margin-bottom: 2px;
}
.badge {
    display: inline-block;
    background: #e8f4fd;
    color: #1565c0;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.9rem;
    font-weight: 600;
    margin-right: 6px;
    margin-top: 4px;
}
.vs-block {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    font-weight: 800;
    color: #999;
    padding: 20px 0;
}
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #333;
    margin: 24px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 2px solid #e0e0e0;
}
.projection-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
    border-left: 4px solid #1565c0;
}
.projection-card-b {
    border-left-color: #c62828;
}
.projection-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #888;
    margin-bottom: 4px;
}
.projection-text {
    font-size: 1.05rem;
    font-weight: 600;
    color: #1a1a1a;
    line-height: 1.4;
}
.edge-card {
    background: #f0f7ff;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    border-left: 4px solid #1976d2;
    font-size: 0.95rem;
    color: #1a1a1a;
}
.bet-card {
    background: #fffde7;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    border-left: 4px solid #f9a825;
    font-size: 0.95rem;
    color: #1a1a1a;
}
.callout-text {
    background: #f8f9fa;
    border-left: 3px solid #4caf50;
    padding: 8px 14px;
    border-radius: 0 6px 6px 0;
    font-size: 0.9rem;
    margin-bottom: 6px;
}
.callout-text-bad {
    background: #fff5f5;
    border-left: 3px solid #ef5350;
    padding: 8px 14px;
    border-radius: 0 6px 6px 0;
    font-size: 0.9rem;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

CHART_CONFIG = {"displayModeBar": False}

ROOT = Path(__file__).parent.parent.parent
LIVE = ROOT / "data" / "live"
PROCESSED = ROOT / "data" / "processed"

# Per-hole benchmark lines (from Player Profiles — same constants)
CHAMPION_HOLE_AVG = {
    1: 0.045, 2: -0.500, 3: -0.386, 4: 0.136, 5: 0.114, 6: -0.136,
    7: -0.114, 8: -0.568, 9: -0.295, 10: 0.023, 11: 0.045, 12: -0.159,
    13: -0.591, 14: -0.205, 15: -0.523, 16: -0.091, 17: 0.023, 18: 0.087,
}
TOP10_HOLE_AVG = {
    1: 0.085, 2: -0.550, 3: -0.135, 4: 0.147, 5: 0.143, 6: -0.014,
    7: 0.028, 8: -0.486, 9: -0.103, 10: 0.036, 11: 0.153, 12: 0.020,
    13: -0.542, 14: -0.054, 15: -0.442, 16: -0.125, 17: 0.058, 18: 0.094,
}


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=120)
def load_live():
    p = LIVE / "live_predictions_latest.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(ttl=300)
def load_predictions():
    p = PROCESSED / "predictions_2026.parquet"
    if p.exists():
        return pd.read_parquet(p)
    p = PROCESSED / "predictions_2026.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data(ttl=3600)
def load_hole_by_hole():
    p = PROCESSED / "masters_hole_by_hole.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data(ttl=3600)
def load_masters_sg_rounds():
    p = PROCESSED / "masters_sg_rounds.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data(ttl=300)
def load_matchup_odds():
    """Load matchup odds from DataGolf if available."""
    p = LIVE / "matchup_odds.csv"
    return pd.read_csv(p) if p.exists() else None


live = load_live()
preds = load_predictions()
hbh = load_hole_by_hole()
sg_rounds = load_masters_sg_rounds()
matchup_odds = load_matchup_odds()

if live is None and preds is None:
    st.error("No prediction data found. Run the pipeline.")
    st.stop()

df = live if live is not None else preds
players = sorted(df["player_name"].dropna().tolist())


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_get(row, *cols):
    for col in cols:
        val = row.get(col, None)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            return val
    return None


def score_fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    v = int(round(float(v)))
    if v == 0:
        return "E"
    return f"{v:+d}"


def pct_fmt(v, decimals=1) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{float(v) * 100:.{decimals}f}%"


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def parse_finish(pos_str):
    if pd.isna(pos_str):
        return np.nan
    s = str(pos_str).upper().strip()
    if s in {"MC", "CUT", "WD", "DQ", "MDF", ""}:
        return np.nan
    s = s.lstrip("T")
    try:
        return int(float(s))
    except Exception:
        return np.nan


def american_to_prob(american: float) -> float:
    """American odds → implied probability (no vig adjustment)."""
    if american > 0:
        return 100 / (american + 100)
    else:
        return abs(american) / (abs(american) + 100)


def decimal_to_prob(decimal: float) -> float:
    return 1.0 / decimal


# ── Player selectors ──────────────────────────────────────────────────────────

st.title("Head-to-Head Comparison")
st.caption("Model probabilities, hole-by-hole Augusta history, and betting edge.")

col_a, col_vs, col_b = st.columns([5, 1, 5])
with col_a:
    player_a = st.selectbox("Player A", players, index=0, label_visibility="collapsed",
                            placeholder="Select Player A")
    st.markdown(f"<div style='font-size:1.1rem;font-weight:700;color:#1565c0;margin-top:4px'>{player_a}</div>",
                unsafe_allow_html=True)
with col_vs:
    st.markdown("<div class='vs-block'>vs</div>", unsafe_allow_html=True)
with col_b:
    default_b = next((i for i, p in enumerate(players) if p != player_a), 1)
    player_b = st.selectbox("Player B", players, index=default_b, label_visibility="collapsed",
                            placeholder="Select Player B")
    st.markdown(f"<div style='font-size:1.1rem;font-weight:700;color:#c62828;margin-top:4px'>{player_b}</div>",
                unsafe_allow_html=True)

if player_a == player_b:
    st.warning("Select two different players.")
    st.stop()

a = df[df["player_name"] == player_a].iloc[0]
b = df[df["player_name"] == player_b].iloc[0]


# ── Pull probabilities ────────────────────────────────────────────────────────

def get_probs(row):
    win   = float(safe_get(row, "mc_win_prob", "win_prob") or 0)
    top5  = float(safe_get(row, "mc_top5_prob") or 0)
    top10 = float(safe_get(row, "mc_top10_prob", "top10_prob") or 0)
    top20 = float(safe_get(row, "mc_top20_prob", "top20_prob") or 0)
    return win, top5, top10, top20


win_a, top5_a, top10_a, top20_a = get_probs(a)
win_b, top5_b, top10_b, top20_b = get_probs(b)

score_a = safe_get(a, "current_score_to_par", "cumulative_score_to_par", "current_score")
score_b = safe_get(b, "current_score_to_par", "cumulative_score_to_par", "current_score")
pos_a   = safe_get(a, "current_pos") or "—"
pos_b   = safe_get(b, "current_pos") or "—"
thru_a  = safe_get(a, "thru", "holes_completed")
thru_b  = safe_get(b, "thru", "holes_completed")

# Score color
def score_color(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "#1a1a1a"
    return "#1a7a45" if float(v) < 0 else ("#c62828" if float(v) > 0 else "#1a1a1a")


# ── Section 1: Player Summary Cards ──────────────────────────────────────────

st.markdown("---")
col_card_a, col_spacer, col_card_b = st.columns([10, 1, 10])

for col, player, row, win, top10, top20, score, pos, thru, card_cls in [
    (col_card_a, player_a, a, win_a, top10_a, top20_a, score_a, pos_a, thru_a, "player-card-a"),
    (col_card_b, player_b, b, win_b, top10_b, top20_b, score_b, pos_b, thru_b, "player-card-b"),
]:
    thru_label = ""
    if thru is not None and not (isinstance(thru, float) and np.isnan(thru)) and int(float(thru)) > 0:
        thru_label = f" · Thru {int(float(thru))}"

    pos_s = str(pos)
    if pos_s.replace(".", "").isdigit():
        pos_s = f"T{int(float(pos_s))}"

    with col:
        st.markdown(f"""
<div class="player-card {card_cls}">
  <div style="font-size:1.3rem;font-weight:700;color:#1a1a1a;margin-bottom:14px">{player}</div>
  <div style="display:flex;gap:28px;align-items:flex-end;margin-bottom:14px">
    <div>
      <div class="stat-label">Score to Par{thru_label}</div>
      <div class="big-score" style="color:{score_color(score)}">{score_fmt(score)}</div>
    </div>
    <div>
      <div class="stat-label">Position</div>
      <div class="big-pos">{pos_s}</div>
    </div>
  </div>
  <div>
    <span class="badge">Win {pct_fmt(win)}</span>
    <span class="badge">Top 10 {pct_fmt(top10)}</span>
    <span class="badge">Top 20 {pct_fmt(top20)}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Section 2: Clear Projections ──────────────────────────────────────────────

st.markdown('<div class="section-header">Model Projections</div>', unsafe_allow_html=True)

ROUND_STD = 3.5  # typical PGA Tour round-to-round scoring std

# ── 2a: Tournament winner — remaining-rounds variance, not IQR collapse ───────
# IQR-based sigma collapses mid-tournament (tight MC distribution around actual scores),
# producing artificially extreme probs. Instead: use current score gap + variance
# from the REMAINING rounds yet to be played.

curr_score_a = float(safe_get(a, "current_score_to_par", "cumulative_score_to_par", "current_score") or 0)
curr_score_b = float(safe_get(b, "current_score_to_par", "cumulative_score_to_par", "current_score") or 0)

# Fractional remaining rounds: (current_round - 1) + thru_holes/18 rounds are done.
# e.g. R3 with thru=8 → 2 + 8/18 = 2.44 done → 1.56 remaining
_round_num = safe_get(a, "current_round", "round_num")
_thru = safe_get(a, "thru", "holes_completed")
thru_holes = float(_thru) if (_thru is not None and not (isinstance(_thru, float) and np.isnan(_thru))) else 0.0

if _round_num is not None:
    current_round_int = int(float(_round_num))
    rounds_done_float = (current_round_int - 1) + min(thru_holes, 18) / 18
else:
    rounds_done_float = 2.0  # conservative fallback

remaining_rounds = max(4.0 - rounds_done_float, 0.5)

# Compute player B's remaining rounds separately for a correct joint variance
_round_num_b = safe_get(b, "current_round", "round_num")
_thru_b = safe_get(b, "thru", "holes_completed")
thru_holes_b = float(_thru_b) if (_thru_b is not None and not (isinstance(_thru_b, float) and np.isnan(_thru_b))) else 0.0
if _round_num_b is not None:
    current_round_b = int(float(_round_num_b))
    rounds_done_b = (current_round_b - 1) + min(thru_holes_b, 18) / 18
else:
    rounds_done_b = rounds_done_float
remaining_b = max(4.0 - rounds_done_b, 0.5)

# diff_std = ROUND_STD * sqrt(remaining_a + remaining_b) — correct joint variance
remaining_diff_std = ROUND_STD * math.sqrt(remaining_rounds + remaining_b)

# Use projected total if available and non-trivial, else fall back to current score
mu_a_raw = safe_get(a, "mc_projected_total")
mu_b_raw = safe_get(b, "mc_projected_total")
if mu_a_raw is not None and mu_b_raw is not None and (abs(float(mu_a_raw)) > 0.1 or abs(float(mu_b_raw)) > 0.1):
    effective_diff = float(mu_a_raw) - float(mu_b_raw)
    basis_note = "MC projected total"
else:
    effective_diff = curr_score_a - curr_score_b
    basis_note = "current score to par"

h2h_a_prob = min(0.99, max(0.01, norm_cdf(-effective_diff / remaining_diff_std))) if remaining_diff_std > 0 else 0.5
h2h_b_prob = 1.0 - h2h_a_prob

if h2h_a_prob >= h2h_b_prob:
    tourn_leader, tourn_trailer, tourn_prob = player_a, player_b, h2h_a_prob
else:
    tourn_leader, tourn_trailer, tourn_prob = player_b, player_a, h2h_b_prob

leader_color = "#1565c0" if tourn_leader == player_a else "#c62828"
score_gap = abs(curr_score_a - curr_score_b)
st.markdown(f"""
<div class="edge-card">
  <div class="projection-label">Tournament Winner — Better Final Position</div>
  <div class="projection-text">
    Model gives <strong style="color:{leader_color}">{tourn_leader}</strong> a
    <strong>{tourn_prob:.0%}</strong> chance of finishing ahead of
    <strong>{tourn_trailer}</strong> over the full tournament.
    ({score_gap:.0f}-shot gap, ~{(remaining_rounds + remaining_b) / 2:.1f} rounds remaining avg,
    based on {basis_note}.)
  </div>
</div>
""", unsafe_allow_html=True)

# ── 2b: Round winner — Augusta career SG/round (not cumulative tournament SG) ──
# live sg_total = cumulative tournament SG (+15.50 after 2 hot rounds = wrong).
# preds file has no sg_total column.
# Best available signal: career SG/round at Augusta from masters_sg_rounds.parquet
# (per-round data 2021-2025, Augusta-specific, much more predictive than season-wide).

def augusta_career_sg(player_name):
    if sg_rounds is None or sg_rounds.empty:
        return None, 0
    p = sg_rounds[sg_rounds["player_name"] == player_name]
    if p.empty:
        return None, 0
    sg_vals = p["sg_total"].dropna()
    return (float(sg_vals.mean()), len(sg_vals)) if len(sg_vals) > 0 else (None, 0)

sg_a_aug, rounds_a = augusta_career_sg(player_a)
sg_b_aug, rounds_b = augusta_career_sg(player_b)

# Fall back to 0 (field average) if no Augusta history
sg_a_season = sg_a_aug if sg_a_aug is not None else 0.0
sg_b_season = sg_b_aug if sg_b_aug is not None else 0.0

sg_a_label = f"{sg_a_season:+.2f} ({rounds_a} Augusta rounds)" if sg_a_aug is not None else "no Augusta history"
sg_b_label = f"{sg_b_season:+.2f} ({rounds_b} Augusta rounds)" if sg_b_aug is not None else "no Augusta history"

# P(A lower score than B in one round) = Φ((sg_a - sg_b) / round_std)
sg_diff_season = sg_a_season - sg_b_season
round_h2h_a = min(0.99, max(0.01, norm_cdf(sg_diff_season / ROUND_STD)))
round_h2h_b = 1.0 - round_h2h_a

# Round label: current_round IS the round in progress (no +1)
if _round_num is not None:
    round_label = f"Round {current_round_int}"
else:
    round_label = "Current Round"

if round_h2h_a >= round_h2h_b:
    rnd_leader, rnd_trailer, rnd_prob = player_a, player_b, round_h2h_a
else:
    rnd_leader, rnd_trailer, rnd_prob = player_b, player_a, round_h2h_b

rnd_color = "#1565c0" if rnd_leader == player_a else "#c62828"
st.markdown(f"""
<div class="edge-card">
  <div class="projection-label">{round_label} Winner — Lower Score That Round</div>
  <div class="projection-text">
    Model expects <strong style="color:{rnd_color}">{rnd_leader}</strong> to
    post a lower score than <strong>{rnd_trailer}</strong> in {round_label.lower()}
    <strong>{rnd_prob:.0%}</strong> of the time.
    (Augusta career SG/round: {player_a} {sg_a_label} vs {player_b} {sg_b_label}.)
  </div>
</div>
""", unsafe_allow_html=True)


# ── Section 3: Comparison Table ───────────────────────────────────────────────

st.markdown('<div class="section-header">Side-by-Side Statistics</div>', unsafe_allow_html=True)

# Augusta history
def get_masters_history(player_name):
    if hbh is None or hbh.empty:
        return {"times_played": "—", "avg_finish": "—", "best_finish": "—"}
    pdata = hbh[hbh["player_name"] == player_name]
    if pdata.empty:
        return {"times_played": 0, "avg_finish": "—", "best_finish": "—"}
    yearly = pdata.drop_duplicates("year")[["year", "finish_pos"]]
    times = len(yearly)
    finishes = yearly["finish_pos"].apply(parse_finish).dropna()
    avg_f = f"{finishes.mean():.1f}" if len(finishes) > 0 else "—"
    best_v = int(finishes.min()) if len(finishes) > 0 else None
    best_f = f"T{best_v}" if best_v and best_v > 1 else ("Won" if best_v == 1 else "—")
    return {"times_played": times, "avg_finish": avg_f, "best_finish": best_f}


hist_a = get_masters_history(player_a)
hist_b = get_masters_history(player_b)


def fmt_sg(val):
    try:
        return f"{float(val):+.2f}"
    except Exception:
        return "—"


# Build rows: (section, label, val_a, val_b, lower_is_better)
sections = [
    ("THIS TOURNAMENT", [
        ("Current Score", score_fmt(score_a), score_fmt(score_b), True),
        ("Position", str(pos_a), str(pos_b), True),
        ("Win %", pct_fmt(win_a), pct_fmt(win_b), False),
        ("Top-5 %", pct_fmt(top5_a), pct_fmt(top5_b), False),
        ("Top-10 %", pct_fmt(top10_a), pct_fmt(top10_b), False),
        ("Top-20 %", pct_fmt(top20_a), pct_fmt(top20_b), False),
    ]),
    ("MASTERS HISTORY", [
        ("Times at Augusta", str(hist_a["times_played"]), str(hist_b["times_played"]), False),
        ("Avg Finish", hist_a["avg_finish"], hist_b["avg_finish"], True),
        ("Best Finish", hist_a["best_finish"], hist_b["best_finish"], True),
    ]),
    ("SEASON FORM (SG)", [
        ("SG Total", fmt_sg(safe_get(a, "sg_total")), fmt_sg(safe_get(b, "sg_total")), False),
        ("SG Off-Tee", fmt_sg(safe_get(a, "sg_ott")), fmt_sg(safe_get(b, "sg_ott")), False),
        ("SG Approach", fmt_sg(safe_get(a, "sg_app")), fmt_sg(safe_get(b, "sg_app")), False),
        ("SG Around Green", fmt_sg(safe_get(a, "sg_arg")), fmt_sg(safe_get(b, "sg_arg")), False),
        ("SG Putting", fmt_sg(safe_get(a, "sg_putt")), fmt_sg(safe_get(b, "sg_putt")), False),
    ]),
]

def numeric_val(s: str):
    """Try to extract a float from a formatted string like '+3.21', '14.2%', 'T4', '—'."""
    if s in ("—", "", "Won"):
        return None
    s2 = s.replace("%", "").replace("+", "").replace("E", "0").lstrip("T")
    try:
        return float(s2)
    except Exception:
        return None


GREEN = "#e8f5e9"
RED   = "#ffebee"

for section_name, rows in sections:
    # Section header row
    header_html = f"""
<div style="background:#f0f0f0;padding:8px 12px;font-size:0.78rem;font-weight:700;
            letter-spacing:0.07em;color:#555;margin-top:8px;border-radius:6px 6px 0 0">
  {section_name}
</div>"""
    st.markdown(header_html, unsafe_allow_html=True)

    table_html = """<table style="width:100%;border-collapse:collapse;font-size:0.95rem">"""
    for i, (label, va, vb, lower_better) in enumerate(rows):
        bg = "#fafafa" if i % 2 == 0 else "white"

        # Determine color coding
        na = numeric_val(va)
        nb = numeric_val(vb)
        bg_a = bg
        bg_b = bg
        if na is not None and nb is not None and na != nb:
            if lower_better:
                a_wins = na < nb
            else:
                a_wins = na > nb
            bg_a = GREEN if a_wins else RED
            bg_b = GREEN if not a_wins else RED

        table_html += f"""
<tr style="background:{bg}">
  <td style="padding:8px 12px;color:#555;width:40%">{label}</td>
  <td style="padding:8px 12px;text-align:center;font-weight:600;background:{bg_a};
             color:#1565c0;width:30%">{va}</td>
  <td style="padding:8px 12px;text-align:center;font-weight:600;background:{bg_b};
             color:#c62828;width:30%">{vb}</td>
</tr>"""

    table_html += "</table>"
    st.markdown(table_html, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)


# ── Section 4: Hole-by-Hole Overlay ──────────────────────────────────────────

st.markdown('<div class="section-header">Hole-by-Hole: Historical Scoring at Augusta</div>',
            unsafe_allow_html=True)
st.caption("Average score to par per hole — lower (more negative) is better. Champion and Top-10 lines shown for reference.")

if hbh is None or hbh.empty:
    st.info("Hole-by-hole data not available.")
else:
    def player_hole_avg(player_name):
        pdata = hbh[hbh["player_name"] == player_name]
        if pdata.empty:
            return None, 0
        avg = pdata.groupby("hole_number")["score_to_par"].mean().reindex(range(1, 19))
        years = pdata["year"].nunique()
        return avg, years

    avg_a, years_a = player_hole_avg(player_a)
    avg_b, years_b = player_hole_avg(player_b)

    if avg_a is None and avg_b is None:
        st.info("No Augusta hole-by-hole data for either player.")
    else:
        x_holes = [str(h) for h in range(1, 19)]
        champ_y = [CHAMPION_HOLE_AVG.get(h, 0) for h in range(1, 19)]
        top10_y = [TOP10_HOLE_AVG.get(h, 0) for h in range(1, 19)]

        fig = go.Figure()

        # Reference lines first (behind player lines)
        fig.add_trace(go.Scatter(
            x=x_holes, y=top10_y,
            mode="lines+markers",
            name="Avg Top 10",
            line=dict(color="#2e7d32", width=1.5, dash="dot"),
            marker=dict(size=4),
            hovertemplate="Hole %{x} — Top-10 avg: %{y:+.2f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=x_holes, y=champ_y,
            mode="lines+markers",
            name="Avg Champion",
            line=dict(color="#f9a825", width=2, dash="dash"),
            marker=dict(size=4),
            hovertemplate="Hole %{x} — Champion avg: %{y:+.2f}<extra></extra>",
        ))

        # Par baseline
        fig.add_hline(y=0, line_color="#999", line_width=1, opacity=0.5)

        if avg_a is not None:
            fig.add_trace(go.Scatter(
                x=x_holes, y=[round(v, 3) if not np.isnan(v) else None for v in avg_a.values],
                mode="lines+markers",
                name=f"{player_a} ({years_a} yr{'s' if years_a != 1 else ''})",
                line=dict(color="#1565c0", width=2.5),
                marker=dict(size=7),
                hovertemplate="Hole %{x} — " + player_a + ": %{y:+.2f}<extra></extra>",
            ))

        if avg_b is not None:
            fig.add_trace(go.Scatter(
                x=x_holes, y=[round(v, 3) if not np.isnan(v) else None for v in avg_b.values],
                mode="lines+markers",
                name=f"{player_b} ({years_b} yr{'s' if years_b != 1 else ''})",
                line=dict(color="#c62828", width=2.5),
                marker=dict(size=7),
                hovertemplate="Hole %{x} — " + player_b + ": %{y:+.2f}<extra></extra>",
            ))

        # Amen Corner shading
        fig.add_vrect(
            x0="10.5", x1="13.5",
            fillcolor="rgba(255,165,0,0.10)",
            line_width=0,
            annotation_text="Amen Corner",
            annotation_position="top right",
            annotation_font_size=11,
            annotation_font_color="#f57c00",
        )

        fig.update_layout(
            template="plotly_white",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(
                title=None,
                tickvals=x_holes,
                ticktext=x_holes,
                tickfont=dict(size=14),
            ),
            yaxis=dict(
                title=None,
                tickformat="+.2f",
                tickfont=dict(size=13),
                zeroline=False,
            ),
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom", y=-0.22,
                xanchor="center", x=0.5,
                font=dict(size=11),
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

        # Hole advantage callouts
        if avg_a is not None and avg_b is not None:
            diff = avg_a - avg_b  # negative → A scores lower (better) on that hole
            a_better_holes = [str(h) for h in range(1, 19)
                              if not np.isnan(diff.iloc[h - 1]) and diff.iloc[h - 1] < -0.05]
            b_better_holes = [str(h) for h in range(1, 19)
                              if not np.isnan(diff.iloc[h - 1]) and diff.iloc[h - 1] > 0.05]

            amen_a = [h for h in ["11", "12", "13"] if h in a_better_holes]
            amen_b = [h for h in ["11", "12", "13"] if h in b_better_holes]

            if a_better_holes:
                amen_note = f" (incl. Amen Corner holes {', '.join(amen_a)})" if amen_a else ""
                st.markdown(
                    f'<div class="callout-text">'
                    f'<strong style="color:#1565c0">{player_a}</strong> advantage holes: '
                    f'{", ".join(a_better_holes[:9])}{"..." if len(a_better_holes) > 9 else ""}'
                    f'{amen_note}</div>',
                    unsafe_allow_html=True,
                )
            if b_better_holes:
                amen_note = f" (incl. Amen Corner holes {', '.join(amen_b)})" if amen_b else ""
                st.markdown(
                    f'<div class="callout-text-bad">'
                    f'<strong style="color:#c62828">{player_b}</strong> advantage holes: '
                    f'{", ".join(b_better_holes[:9])}{"..." if len(b_better_holes) > 9 else ""}'
                    f'{amen_note}</div>',
                    unsafe_allow_html=True,
                )


# ── Section 5: Betting Context ────────────────────────────────────────────────

st.markdown('<div class="section-header">Betting Context</div>', unsafe_allow_html=True)

# Tournament edge expressed as fair odds
model_a_prob = h2h_a_prob
model_b_prob = h2h_b_prob
fair_a_american = round((100 / model_a_prob) - 100) if model_a_prob < 0.5 else round(-100 * model_a_prob / (1 - model_a_prob))
fair_b_american = round((100 / model_b_prob) - 100) if model_b_prob < 0.5 else round(-100 * model_b_prob / (1 - model_b_prob))

def american_fmt(v):
    return f"+{int(v)}" if v >= 0 else str(int(v))

st.markdown(f"""
<div class="bet-card">
  <strong>Model Fair Odds — Tournament Matchup</strong><br>
  <span style="color:#1565c0">{player_a}:</span> {american_fmt(fair_a_american)} ({model_a_prob:.1%})&nbsp;&nbsp;|&nbsp;&nbsp;
  <span style="color:#c62828">{player_b}:</span> {american_fmt(fair_b_american)} ({model_b_prob:.1%})
</div>
""", unsafe_allow_html=True)

# Live matchup odds if available
if matchup_odds is not None and not matchup_odds.empty:
    for _, mrow in matchup_odds.iterrows():
        p1 = str(mrow.get("player1", "")).strip()
        p2 = str(mrow.get("player2", "")).strip()
        if set([p1, p2]) == set([player_a, player_b]):
            try:
                odds1 = float(mrow.get("odds1", np.nan))
                odds2 = float(mrow.get("odds2", np.nan))
                if np.isnan(odds1) or np.isnan(odds2):
                    break
                # Determine which col maps to which player
                if p1 == player_a:
                    mkt_a, mkt_b = odds1, odds2
                else:
                    mkt_a, mkt_b = odds2, odds1
                # Convert American to implied prob
                mkt_prob_a = american_to_prob(mkt_a)
                mkt_prob_b = american_to_prob(mkt_b)
                edge_a = model_a_prob - mkt_prob_a

                edge_color = "#2e7d32" if abs(edge_a) >= 0.05 else "#555"
                edge_label = player_a if edge_a > 0 else player_b
                edge_val = edge_a if edge_a > 0 else -edge_a

                st.markdown(f"""
<div class="bet-card">
  <strong>Matchup Bet (Live Market)</strong><br>
  {player_a}: Model {model_a_prob:.1%} | Market {american_fmt(int(mkt_a))} ({mkt_prob_a:.1%})<br>
  {player_b}: Model {model_b_prob:.1%} | Market {american_fmt(int(mkt_b))} ({mkt_prob_b:.1%})<br>
  <span style="color:{edge_color};font-weight:700">Edge: {edge_label} +{edge_val:.1%}</span>
</div>
""", unsafe_allow_html=True)
            except Exception:
                pass
            break
else:
    st.markdown("""
<div class="bet-card">
  <em>Live matchup odds not available. Place <code>data/live/matchup_odds.csv</code>
  with columns: player1, player2, odds1, odds2 (American) to show betting edge.</em>
</div>
""", unsafe_allow_html=True)
