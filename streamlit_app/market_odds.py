"""Live market odds fetcher.

Win market:  The Odds API (live sportsbook lines, stays open during tournament)
T5/T10/T20:  DataGolf betting-tools/outrights (pre-event only) →
             DataGolf preds/pre-tournament (fallback once markets close)

Returns a DataFrame with columns:
  player_name, market_win, market_top5, market_top10, market_top20, _source
"""
import os
import unicodedata
import requests
import pandas as pd
import streamlit as st

DG_BASE    = "https://feeds.datagolf.com"
ODDS_BASE  = "https://api.the-odds-api.com/v4"
MASTERS_SPORT = "golf_masters_tournament_winner"


def _get_dg_key():
    try:
        return st.secrets["DATAGOLF_API_KEY"]
    except Exception:
        return os.getenv("DATAGOLF_API_KEY", "")


def _get_odds_key():
    try:
        return st.secrets["ODDS_API_KEY"]
    except Exception:
        return os.getenv("ODDS_API_KEY", "")


def _normalize(name: str) -> str:
    """Flip 'Last, First' → 'First Last'. Keep original casing/accents."""
    name = str(name).strip()
    if "," in name:
        parts = name.split(",", 1)
        return f"{parts[1].strip()} {parts[0].strip()}"
    return name


def _ascii_lower(name: str) -> str:
    """Strip accents and lowercase for fuzzy matching only."""
    return unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode().lower()


def _american_to_prob(american: int) -> float:
    """Convert American odds to no-vig implied probability (raw, not vig-removed)."""
    if american > 0:
        return 100 / (american + 100)
    else:
        return abs(american) / (abs(american) + 100)


def _fetch_odds_api_win():
    """Fetch live Masters win odds from The Odds API.

    Returns dict {player_name: implied_prob} or {} on failure.
    Uses DraftKings consensus; falls back to any bookmaker.
    """
    key = _get_odds_key()
    if not key:
        return {}
    try:
        r = requests.get(
            f"{ODDS_BASE}/sports/{MASTERS_SPORT}/odds/",
            params={"apiKey": key, "regions": "us", "markets": "outrights",
                    "oddsFormat": "american", "bookmakers": "draftkings"},
            timeout=15,
        )
        if r.status_code != 200:
            # retry without bookmakers filter (any book)
            r = requests.get(
                f"{ODDS_BASE}/sports/{MASTERS_SPORT}/odds/",
                params={"apiKey": key, "regions": "us", "markets": "outrights",
                        "oddsFormat": "american"},
                timeout=15,
            )
        if r.status_code != 200:
            return {}

        events = r.json()
        if not events:
            return {}

        # Take first event (Masters), first bookmaker
        bookmakers = events[0].get("bookmakers", [])
        if not bookmakers:
            return {}

        outcomes = bookmakers[0]["markets"][0]["outcomes"]
        probs = {}
        for o in outcomes:
            name = _normalize(o["name"])
            probs[name] = _american_to_prob(o["price"])

        return probs
    except Exception:
        return {}


def _fetch_dg_position_markets(dg_key: str):
    """Fetch T5/T10/T20 from DG betting-tools/outrights.

    Returns (dict {player: {market_top5, market_top10, market_top20}}, n_markets_found).
    """
    market_data: dict = {}
    found = 0
    for mkt_key, col in [
        ("top_5",  "market_top5"),
        ("top_10", "market_top10"),
        ("top_20", "market_top20"),
    ]:
        try:
            r = requests.get(
                f"{DG_BASE}/betting-tools/outrights",
                params={"tour": "pga", "market": mkt_key, "odds_format": "percent",
                        "file_format": "json", "key": dg_key},
                timeout=15,
            )
            if r.status_code == 200:
                players = r.json().get("results", [])
                if players:
                    found += 1
                    for p in players:
                        name = _normalize(p.get("player_name", ""))
                        market_data.setdefault(name, {})[col] = p.get("baseline")
        except Exception:
            pass
    return market_data, found


def _fetch_dg_pretournament(dg_key: str):
    """Fetch DG pre-tournament snapshot for T5/T10/T20 (and win fallback).

    Returns (list of player dicts, source_label) or ([], "").
    """
    try:
        r = requests.get(
            f"{DG_BASE}/preds/pre-tournament",
            params={"tour": "pga", "add_position": "1,5,10,20",
                    "file_format": "json", "key": dg_key},
            timeout=15,
        )
        if r.status_code == 200:
            data = r.json()
            players = data.get("baseline", [])
            label = f"DG pre-tournament ({data.get('last_updated', '?')})"
            return players, label
    except Exception:
        pass
    return [], ""


@st.cache_data(ttl=1800, show_spinner="Fetching live market odds…")
def fetch_market_odds():
    """Return DataFrame: player_name, market_win, market_top5, market_top10, market_top20, _source.

    Win odds:   The Odds API (live sportsbook, open during tournament)
    T5/T10/T20: DG betting-tools/outrights → DG pre-tournament fallback
    Cached 30 min. Returns None if all sources fail.
    """
    dg_key = _get_dg_key()

    # ── Win: The Odds API ─────────────────────────────────────────────────────
    win_probs = _fetch_odds_api_win()
    win_source = "The Odds API (DraftKings)" if win_probs else None

    # ── T5/T10/T20: DG outrights ─────────────────────────────────────────────
    pos_data: dict = {}
    pos_source = None

    if dg_key:
        pos_data, n_found = _fetch_dg_position_markets(dg_key)
        if n_found > 0:
            pos_source = "DG live outrights"

    # Fallback: DG pre-tournament (also fills win if Odds API failed)
    dg_pt_players, dg_pt_label = [], ""
    if not pos_source and dg_key:
        dg_pt_players, dg_pt_label = _fetch_dg_pretournament(dg_key)
        if dg_pt_players:
            for p in dg_pt_players:
                name = _normalize(p["player_name"])
                pos_data.setdefault(name, {}).update({
                    "market_top5":  p.get("top_5"),
                    "market_top10": p.get("top_10"),
                    "market_top20": p.get("top_20"),
                })
            pos_source = dg_pt_label

    # Also use DG pre-tournament for win if Odds API failed
    if not win_probs and dg_pt_players:
        for p in dg_pt_players:
            name = _normalize(p["player_name"])
            win_probs[name] = p.get("win")
        win_source = dg_pt_label

    # If nothing at all, try DG pre-tournament as last resort for everything
    if not win_probs and not pos_data and dg_key and not dg_pt_players:
        dg_pt_players, dg_pt_label = _fetch_dg_pretournament(dg_key)
        if dg_pt_players:
            for p in dg_pt_players:
                name = _normalize(p["player_name"])
                win_probs[name] = p.get("win")
                pos_data[name] = {
                    "market_top5":  p.get("top_5"),
                    "market_top10": p.get("top_10"),
                    "market_top20": p.get("top_20"),
                }
            win_source = pos_source = dg_pt_label

    if not win_probs and not pos_data:
        return None

    # ── Merge sources ─────────────────────────────────────────────────────────
    all_names = set(win_probs.keys()) | set(pos_data.keys())
    rows = []
    for name in all_names:
        row = {"player_name": name}
        w = win_probs.get(name)
        row["market_win"] = float(w) if w is not None else float("nan")
        pos = pos_data.get(name, {})
        row["market_top5"]  = float(pos["market_top5"])  if pos.get("market_top5")  is not None else float("nan")
        row["market_top10"] = float(pos["market_top10"]) if pos.get("market_top10") is not None else float("nan")
        row["market_top20"] = float(pos["market_top20"]) if pos.get("market_top20") is not None else float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)

    # Build source label
    if win_source and pos_source and win_source != pos_source:
        df["_source"] = f"Win: {win_source} | T5/T10/T20: {pos_source}"
    else:
        df["_source"] = win_source or pos_source or "DataGolf"

    return df


def merge_with_model(model_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """Fuzzy-merge market odds onto model predictions, compute Kelly edges.

    Returns model_df with added columns:
      market_win, market_top5, market_top10, market_top20,
      kelly_edge_win, kelly_edge_top10, kelly_edge_top20
    """
    try:
        from rapidfuzz import fuzz, process as rfprocess
    except ImportError:
        merged = model_df.merge(market_df, on="player_name", how="left")
        _add_edges(merged)
        return merged

    result = model_df.copy()

    # Build ASCII-lowercased lookup for robust matching
    mkt_ascii = {_ascii_lower(n): n for n in market_df["player_name"].tolist()}
    mkt_ascii_keys = list(mkt_ascii.keys())

    for col in ["market_win", "market_top5", "market_top10", "market_top20"]:
        result[col] = float("nan")

    for i, row in result.iterrows():
        query = _ascii_lower(row["player_name"])
        # token_set_ratio handles "Matt" vs "Matthew", "Alex" vs "Alexander", etc.
        best = rfprocess.extractOne(query, mkt_ascii_keys, scorer=fuzz.token_set_ratio)
        if best and best[1] >= 80:
            original_name = mkt_ascii[best[0]]
            mrow = market_df[market_df["player_name"] == original_name].iloc[0]
            for col in ["market_win", "market_top5", "market_top10", "market_top20"]:
                if col in mrow and pd.notna(mrow[col]):
                    result.at[i, col] = mrow[col]

    _add_edges(result)
    return result


def _add_edges(df: pd.DataFrame) -> None:
    """Add Kelly edge columns in-place."""
    for model_col, mkt_col, edge_col in [
        ("win_prob",    "market_win",   "kelly_edge_win"),
        ("top10_prob",  "market_top10", "kelly_edge_top10"),
        ("top20_prob",  "market_top20", "kelly_edge_top20"),
    ]:
        if model_col in df.columns and mkt_col in df.columns:
            mkt = df[mkt_col].fillna(0)
            df[edge_col] = (df[model_col] - mkt).where(mkt > 0) / mkt.where(mkt > 0)
