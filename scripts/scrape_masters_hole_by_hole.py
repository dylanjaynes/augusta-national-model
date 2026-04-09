"""
Scrape hole-by-hole Masters Tournament scorecards from ESPN's undocumented API.

Output: data/processed/masters_hole_by_hole.parquet + .csv

Columns:
  year, player_name, espn_id, round, hole_number, par,
  score, score_to_par, score_type,
  tee_time, starting_hole,
  made_cut, finish_pos

Augusta National hole pars (static):
  1-4  2-5  3-4  4-3  5-4  6-3  7-4  8-5  9-4
  10-4 11-4 12-3 13-5 14-4 15-5 16-3 17-4 18-4
  Total: 36 out + 36 in = 72

Usage:
  python3 scripts/scrape_masters_hole_by_hole.py

Rate-limited to ~2 req/sec; expect ~30-40 min for full 2015-2025 run.
"""

from __future__ import annotations

import os
import sys
import time
import json
import logging
import requests
import pandas as pd
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

MASTERS_EVENT_IDS = {
    2015: "2241",
    2016: "2493",
    2017: "2700",
    2018: "401025221",
    2019: "401056527",
    2020: "401219478",
    2021: "401243010",
    2022: "401353232",
    2023: "401465508",
    2024: "401580344",
    2025: "401703504",
}

AUGUSTA_PARS = {
    1: 4, 2: 5, 3: 4, 4: 3, 5: 4, 6: 3, 7: 4, 8: 5, 9: 4,
    10: 4, 11: 4, 12: 3, 13: 5, 14: 4, 15: 5, 16: 3, 17: 4, 18: 4,
}

AUGUSTA_YARDS = {
    1: 445, 2: 575, 3: 350, 4: 240, 5: 495, 6: 180, 7: 450, 8: 570, 9: 460,
    10: 495, 11: 520, 12: 155, 13: 510, 14: 440, 15: 530, 16: 170, 17: 440, 18: 465,
}

AUGUSTA_NAMES = {
    1: "Tea Olive", 2: "Pink Dogwood", 3: "Flowering Peach",
    4: "Flowering Crab Apple", 5: "Magnolia", 6: "Juniper",
    7: "Pampas", 8: "Yellow Jasmine", 9: "Carolina Cherry",
    10: "Camellia", 11: "White Dogwood", 12: "Golden Bell",
    13: "Azalea", 14: "Chinese Fir", 15: "Firethorn",
    16: "Redbud", 17: "Nandina", 18: "Holly",
}

BASE_URL = "https://sports.core.api.espn.com/v2/sports/golf/leagues/pga"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
REQUEST_DELAY = 0.5  # seconds between requests

# ── HTTP helpers ─────────────────────────────────────────────────────────────

session = requests.Session()
session.headers.update(HEADERS)
_last_request = 0.0


def fetch(url: str, retries: int = 3) -> Optional[dict]:
    global _last_request
    # Rate-limit
    elapsed = time.time() - _last_request
    if elapsed < REQUEST_DELAY:
        time.sleep(REQUEST_DELAY - elapsed)
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=20)
            _last_request = time.time()
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 404:
                return None
            else:
                log.warning(f"HTTP {r.status_code} for {url[:80]} (attempt {attempt+1})")
                time.sleep(2 ** attempt)
        except Exception as e:
            log.warning(f"Request error: {e} (attempt {attempt+1})")
            time.sleep(2 ** attempt)
    return None


# ── Scraping functions ────────────────────────────────────────────────────────

def get_competitors(event_id: str) -> list[dict]:
    """Return list of competitor dicts from ESPN competition endpoint."""
    url = f"{BASE_URL}/events/{event_id}/competitions/{event_id}?lang=en"
    data = fetch(url)
    if data is None:
        log.error(f"Could not fetch competition for event {event_id}")
        return []
    return data.get("competitors", [])


def get_athlete_name(athlete_ref: str) -> tuple:
    """Fetch athlete full name from $ref. Returns (fullName, displayName)."""
    data = fetch(athlete_ref)
    if data:
        return data.get("fullName", ""), data.get("displayName", "")
    return "", ""


def get_linescores(event_id: str, competitor_id: str) -> list[dict]:
    """
    Fetch hole-by-hole linescores for one competitor.
    Returns list of round dicts, each with 'period' (round) and 'linescores' (18 holes).
    """
    url = f"{BASE_URL}/events/{event_id}/competitions/{event_id}/competitors/{competitor_id}/linescores?lang=en"
    data = fetch(url)
    if data is None:
        return []
    return data.get("items", [])


def get_competitor_status(event_id: str, competitor_id: str) -> dict:
    """Fetch competitor status for tee time, position, starting hole."""
    url = f"{BASE_URL}/events/{event_id}/competitions/{event_id}/competitors/{competitor_id}/status?lang=en"
    data = fetch(url)
    return data or {}


def parse_finish_pos(position_data: dict | None) -> str | None:
    if not position_data:
        return None
    pos = position_data.get("displayName", "")
    is_tie = position_data.get("isTie", False)
    if pos and is_tie:
        return f"T{pos}"
    return pos or None


# ── Main scraper ──────────────────────────────────────────────────────────────

def scrape_year(year: int) -> list[dict]:
    event_id = MASTERS_EVENT_IDS[year]
    log.info(f"{'='*60}")
    log.info(f"Scraping {year} Masters (event_id={event_id})")

    competitors = get_competitors(event_id)
    log.info(f"  {len(competitors)} competitors found")

    rows = []
    athlete_cache: dict[str, tuple[str, str]] = {}

    for idx, comp in enumerate(competitors):
        competitor_id = comp.get("id", "")
        if not competitor_id:
            continue

        # ── Athlete name ──
        athlete = comp.get("athlete", {})
        athlete_ref = athlete.get("$ref", "")
        if athlete_ref in athlete_cache:
            full_name, display_name = athlete_cache[athlete_ref]
        else:
            full_name, display_name = get_athlete_name(athlete_ref)
            athlete_cache[athlete_ref] = (full_name, display_name)
            time.sleep(0.1)  # small extra pause for athlete fetch

        player_name = full_name or display_name or f"Unknown_{competitor_id}"

        if idx % 10 == 0:
            log.info(f"  [{idx+1}/{len(competitors)}] {player_name}")

        # ── Status (tee time, position, starting hole) ──
        status = get_competitor_status(event_id, competitor_id)
        tee_time_raw = status.get("teeTime")  # ISO format, last round's tee time
        starting_hole = status.get("startHole", 1)
        position = parse_finish_pos(status.get("position"))
        made_cut = status.get("type", {}).get("name") not in ("STATUS_MISSED_CUT",) if status else None

        # Determine made_cut from rounds played (more reliable)
        # We'll update after getting linescores

        # ── Linescores (hole-by-hole) ──
        linescores = get_linescores(event_id, competitor_id)
        rounds_played = len(linescores)

        # Determine made_cut: if > 2 rounds played, made cut
        if rounds_played > 2:
            made_cut = True
        elif rounds_played <= 2 and rounds_played > 0:
            made_cut = False
        else:
            made_cut = None

        for round_data in linescores:
            round_num = round_data.get("period")  # 1-4
            hole_scores = round_data.get("linescores", [])

            # tee_time from status is for last round only;
            # we'll attach it to that specific round when possible
            round_tee_time = tee_time_raw if round_num == rounds_played else None

            for hole_data in hole_scores:
                hole_num = hole_data.get("period")  # 1-18
                score_val = hole_data.get("value")  # strokes taken
                par = hole_data.get("par", AUGUSTA_PARS.get(hole_num))
                score_type = hole_data.get("scoreType", {}).get("name", "")

                if score_val is not None and hole_num:
                    score_to_par = int(score_val) - par if par else None
                    rows.append({
                        "year": year,
                        "player_name": player_name,
                        "espn_id": competitor_id,
                        "round": round_num,
                        "hole_number": hole_num,
                        "par": par,
                        "score": int(score_val),
                        "score_to_par": score_to_par,
                        "score_type": score_type,
                        "tee_time": round_tee_time,
                        "starting_hole": starting_hole if round_num == rounds_played else None,
                        "made_cut": made_cut,
                        "finish_pos": position,
                    })

    log.info(f"  {year}: {len(rows)} hole-rows from {len(competitors)} players")
    return rows


# ── DG ID join ────────────────────────────────────────────────────────────────

def add_dg_ids(df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """
    Try to join dg_id from masters_unified.parquet by player_name + year.
    Uses fuzzy matching as fallback.
    """
    try:
        unified = pd.read_parquet(base_dir / "data/processed/masters_unified.parquet")
        unified = unified[["dg_id", "player_name", "season"]].dropna(subset=["player_name"])
        unified = unified.rename(columns={"season": "year"})
        unified["year"] = unified["year"].astype(int)

        df_merged = df.merge(
            unified[["dg_id", "player_name", "year"]].drop_duplicates(),
            on=["player_name", "year"],
            how="left",
        )
        match_pct = df_merged["dg_id"].notna().mean() * 100
        log.info(f"DG ID exact match rate: {match_pct:.1f}%")

        # Fuzzy fallback for unmatched rows
        unmatched = df_merged[df_merged["dg_id"].isna()]["player_name"].unique()
        if len(unmatched) > 0:
            log.info(f"Trying fuzzy match for {len(unmatched)} unmatched players")
            try:
                from rapidfuzz import process, fuzz
                dg_names = unified["player_name"].unique().tolist()
                name_map = {}
                for name in unmatched:
                    result = process.extractOne(name, dg_names, scorer=fuzz.token_sort_ratio, score_cutoff=85)
                    if result:
                        name_map[name] = result[0]
                        log.debug(f"  Fuzzy: '{name}' → '{result[0]}' ({result[1]})")
                if name_map:
                    # Get dg_ids for fuzzy-matched names
                    fuzzy_df = pd.DataFrame({"player_name": list(name_map.keys()),
                                              "player_name_dg": list(name_map.values())})
                    fuzzy_df = fuzzy_df.merge(
                        unified[["dg_id", "player_name"]].drop_duplicates().rename(
                            columns={"player_name": "player_name_dg"}),
                        on="player_name_dg", how="left"
                    )
                    fuzzy_map = fuzzy_df.set_index("player_name")["dg_id"].to_dict()
                    mask = df_merged["dg_id"].isna()
                    df_merged.loc[mask, "dg_id"] = df_merged.loc[mask, "player_name"].map(fuzzy_map)
                    new_pct = df_merged["dg_id"].notna().mean() * 100
                    log.info(f"DG ID match after fuzzy: {new_pct:.1f}%")
            except ImportError:
                log.info("rapidfuzz not available; skipping fuzzy match")

        return df_merged
    except Exception as e:
        log.warning(f"Could not join DG IDs: {e}")
        df["dg_id"] = None
        return df


# ── Course info table ─────────────────────────────────────────────────────────

def build_course_info() -> pd.DataFrame:
    rows = []
    for hole in range(1, 19):
        rows.append({
            "hole_number": hole,
            "par": AUGUSTA_PARS[hole],
            "yards": AUGUSTA_YARDS[hole],
            "hole_name": AUGUSTA_NAMES[hole],
        })
    return pd.DataFrame(rows)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    base_dir = Path(__file__).parent.parent
    out_dir = base_dir / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which years to scrape
    years_arg = sys.argv[1:]  # e.g. "2024 2025" or empty for all
    if years_arg:
        years = [int(y) for y in years_arg]
    else:
        years = list(MASTERS_EVENT_IDS.keys())

    all_rows = []

    # Check for existing checkpoint
    checkpoint_path = out_dir / "masters_hole_by_hole_checkpoint.json"
    completed_years = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            cp = json.load(f)
        completed_years = set(cp.get("completed_years", []))
        all_rows = cp.get("rows", [])
        log.info(f"Resuming from checkpoint: {sorted(completed_years)} already done")

    for year in years:
        if year in completed_years:
            log.info(f"Skipping {year} (already in checkpoint)")
            continue
        try:
            rows = scrape_year(year)
            all_rows.extend(rows)
            completed_years.add(year)
            # Save checkpoint after each year
            with open(checkpoint_path, "w") as f:
                json.dump({"completed_years": list(completed_years), "rows": all_rows}, f)
            log.info(f"Checkpoint saved after {year}")
        except Exception as e:
            log.error(f"Error scraping {year}: {e}")
            continue

    if not all_rows:
        log.error("No data scraped!")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    log.info(f"\nTotal rows before joining: {len(df)}")
    log.info(f"Years covered: {sorted(df['year'].unique())}")
    log.info(f"Players: {df['player_name'].nunique()}")

    # Add course info columns
    course_df = build_course_info()
    df = df.merge(course_df[["hole_number", "hole_name", "yards"]], on="hole_number", how="left")

    # Add DG IDs
    df = add_dg_ids(df, base_dir)

    # Clean up dtypes
    df["year"] = df["year"].astype(int)
    df["round"] = df["round"].astype("Int64")
    df["hole_number"] = df["hole_number"].astype("Int64")
    df["par"] = df["par"].astype("Int64")
    df["score"] = df["score"].astype("Int64")
    df["score_to_par"] = df["score_to_par"].astype("Int64")

    # Sort
    df = df.sort_values(["year", "player_name", "round", "hole_number"]).reset_index(drop=True)

    # Save
    parquet_path = out_dir / "masters_hole_by_hole.parquet"
    csv_path = out_dir / "masters_hole_by_hole.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    log.info(f"\nSaved {len(df):,} rows to:")
    log.info(f"  {parquet_path}")
    log.info(f"  {csv_path}")

    # Summary stats
    log.info("\n── Summary ──────────────────────────────")
    summary = df.groupby("year").agg(
        players=("player_name", "nunique"),
        rounds=("round", "max"),
        holes=("hole_number", "count"),
        made_cut=("made_cut", lambda x: x.iloc[0] if len(x) else None),
    )
    log.info(f"\n{df.groupby('year').agg(players=('player_name','nunique'), hole_rows=('hole_number','count'))}")

    # Save course info separately
    course_path = out_dir / "augusta_course_info.parquet"
    course_df.to_parquet(course_path, index=False)
    course_df.to_csv(out_dir / "augusta_course_info.csv", index=False)
    log.info(f"Course info saved to {course_path}")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        log.info("Checkpoint deleted (run complete)")

    return df


if __name__ == "__main__":
    main()
