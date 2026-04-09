"""Live market odds fetcher — DataGolf API.

Tries betting-tools/outrights first (live pre-event lines from 11 sportsbooks).
Falls back to preds/pre-tournament snapshot if the event is in progress and
outright markets are closed.

Returns a DataFrame with columns:
  player_name, market_win, market_top5, market_top10, market_top20
"""
import os
import requests
import pandas as pd
import streamlit as st

DG_BASE = "https://feeds.datagolf.com"


def _get_api_key():
    # Streamlit Cloud: use st.secrets; local: use env var
    try:
        return st.secrets["DATAGOLF_API_KEY"]
    except Exception:
        return os.getenv("DATAGOLF_API_KEY", "")


def _normalize(name: str) -> str:
    name = str(name).strip()
    if "," in name:
        parts = name.split(",", 1)
        return f"{parts[1].strip()} {parts[0].strip()}"
    return name


@st.cache_data(ttl=1800, show_spinner="Fetching live market odds…")
def fetch_market_odds():
    """Return DataFrame: player_name, market_win, market_top5, market_top10, market_top20.

    Cached for 30 minutes to avoid hammering the API.
    Returns None if both sources fail.
    """
    key = _get_api_key()
    if not key:
        return None

    # ── Attempt 1: live outright markets ─────────────────────────────────────
    market_data: dict[str, dict] = {}
    markets_found = 0
    for mkt_key, col in [
        ("win",    "market_win"),
        ("top_5",  "market_top5"),
        ("top_10", "market_top10"),
        ("top_20", "market_top20"),
    ]:
        try:
            r = requests.get(
                f"{DG_BASE}/betting-tools/outrights",
                params={"tour": "pga", "market": mkt_key, "odds_format": "percent",
                        "file_format": "json", "key": key},
                timeout=15,
            )
            if r.status_code == 200:
                players = r.json().get("results", [])
                if players:
                    markets_found += 1
                    for p in players:
                        name = _normalize(p.get("player_name", ""))
                        market_data.setdefault(name, {})[col] = p.get("baseline")
        except Exception:
            pass

    if markets_found > 0:
        rows = [{"player_name": k, **v} for k, v in market_data.items()]
        return pd.DataFrame(rows)

    # ── Attempt 2: pre-tournament snapshot (market-calibrated) ───────────────
    try:
        r = requests.get(
            f"{DG_BASE}/preds/pre-tournament",
            params={"tour": "pga", "add_position": "1,5,10,20",
                    "file_format": "json", "key": key},
            timeout=15,
        )
        if r.status_code == 200:
            data = r.json()
            players = data.get("baseline", [])
            if players:
                rows = [{
                    "player_name":  _normalize(p["player_name"]),
                    "market_win":   p.get("win"),
                    "market_top5":  p.get("top_5"),
                    "market_top10": p.get("top_10"),
                    "market_top20": p.get("top_20"),
                    "_source":      f"DG pre-tournament ({data.get('last_updated', '?')})",
                } for p in players]
                return pd.DataFrame(rows)
    except Exception:
        pass

    return None


def merge_with_model(model_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """Fuzzy-merge market odds onto model predictions, compute Kelly edges.

    Returns model_df with added columns:
      market_win, market_top5, market_top10, market_top20,
      kelly_edge_win, kelly_edge_top10, kelly_edge_top20
    """
    try:
        from rapidfuzz import fuzz, process as rfprocess
    except ImportError:
        # Plain merge fallback
        merged = model_df.merge(market_df, on="player_name", how="left")
        _add_edges(merged)
        return merged

    result = model_df.copy()
    mkt_names = market_df["player_name"].tolist()

    for col in ["market_win", "market_top5", "market_top10", "market_top20"]:
        result[col] = float("nan")

    for i, row in result.iterrows():
        best = rfprocess.extractOne(row["player_name"], mkt_names, scorer=fuzz.ratio)
        if best and best[1] >= 80:
            mrow = market_df[market_df["player_name"] == best[0]].iloc[0]
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
