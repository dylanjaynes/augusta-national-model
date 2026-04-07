"""
Task 1 — Pull DataGolf Masters SG data (2019–2025)
Task 2 — Scrape historical Masters scores (2004–2020)
Task 3 — Build unified master dataset
"""
import os
import re
import time
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DG_BASE = "https://feeds.datagolf.com"
API_KEY = os.getenv("DATAGOLF_API_KEY")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Path to existing historical rounds from golf_model
GOLF_MODEL_HISTORY = Path("/Users/dylanjaynes/golf_model/golf_model_data/historical_rounds.csv")
# Path to local Masters round-level CSVs (2021-2025)
MASTERS_CSV_DIR = Path("masters 2021-2025 csvs")


class SubscriptionError(Exception):
    pass


def dg_get(endpoint: str, params: dict = {}) -> dict:
    url = f"{DG_BASE}/{endpoint}"
    params = {"key": API_KEY, "file_format": "json", **params}
    if "tour" not in params:
        params["tour"] = "pga"
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 403:
                raise SubscriptionError(f"403 Forbidden on {endpoint}")
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            if r.status_code == 429:
                print("  Rate limited, waiting 10s...")
                time.sleep(10)
            else:
                raise
    raise RuntimeError(f"Failed after 3 attempts: {endpoint}")


# ═══════════════════════════════════════════════════════════
# TASK 1 — PULL DATAGOLF MASTERS SG DATA
# ═══════════════════════════════════════════════════════════

def _normalize_player_name(name):
    """Convert 'Last, First' to 'First Last' format."""
    name = str(name).strip().strip('"')
    if "," in name:
        parts = name.split(",", 1)
        return f"{parts[1].strip()} {parts[0].strip()}"
    return name


def _load_masters_round_csvs():
    """Load round-level Masters CSVs from local directory.

    Files: {year}_masters_tournament_r{N}_values.csv
    Columns: position, player_name, total_score, rN_score,
             sg_putt, sg_arg, sg_app, sg_ott, sg_t2g, sg_total
    """
    csv_dir = MASTERS_CSV_DIR
    if not csv_dir.exists():
        return pd.DataFrame(), pd.DataFrame()

    round_rows = []
    # Group files by year
    import glob
    files = sorted(glob.glob(str(csv_dir / "*_masters_tournament_r*_values.csv")))

    years_found = set()
    for fpath in files:
        fname = Path(fpath).name
        # Parse year and round from filename
        match = re.match(r"(\d{4})_masters_tournament_r(\d)_values\.csv", fname)
        if not match:
            continue
        year = int(match.group(1))
        round_num = int(match.group(2))
        years_found.add(year)

        df = pd.read_csv(fpath, na_values=["null", ""])
        # Determine score column name (r1_score, r2_score, etc.)
        score_col = f"r{round_num}_score"
        if score_col not in df.columns:
            # Might be named differently
            for c in df.columns:
                if c.startswith("r") and c.endswith("_score"):
                    score_col = c
                    break

        for _, row in df.iterrows():
            round_rows.append({
                "player_name": _normalize_player_name(row["player_name"]),
                "season": year,
                "round_num": round_num,
                "finish_pos": row.get("position"),
                "total_score": row.get("total_score"),
                "score": row.get(score_col),
                "sg_ott": row.get("sg_ott"),
                "sg_app": row.get("sg_app"),
                "sg_arg": row.get("sg_arg"),
                "sg_putt": row.get("sg_putt"),
                "sg_t2g": row.get("sg_t2g"),
                "sg_total": row.get("sg_total"),
            })

    if not round_rows:
        return pd.DataFrame(), pd.DataFrame()

    rounds_df = pd.DataFrame(round_rows)
    rounds_df["event_name"] = "Masters Tournament"
    rounds_df["course"] = "Augusta National Golf Club"

    # Build tournament-level: average SG across played rounds per player-season
    sg_cols = ["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_t2g", "sg_total"]
    tourn_groups = rounds_df.groupby(["player_name", "season"])

    tourn_rows = []
    for (player, season), grp in tourn_groups:
        # Use finish_pos from the latest round the player has data for
        latest = grp.sort_values("round_num").iloc[-1]
        field_size = len(rounds_df[(rounds_df["season"] == season) & (rounds_df["round_num"] == 1)])

        sg_avgs = {}
        for col in sg_cols:
            vals = grp[col].dropna()
            sg_avgs[col] = vals.mean() if len(vals) > 0 else None

        # Get round scores
        rd_scores = {}
        for _, r in grp.iterrows():
            rd_scores[r["round_num"]] = r.get("score")

        tourn_rows.append({
            "player_name": player,
            "season": season,
            "event_name": "Masters Tournament",
            "course": "Augusta National Golf Club",
            "date": "",
            "field_size": field_size,
            "finish_pos": latest["finish_pos"],
            "total_score_to_par": latest.get("total_score"),
            "r1_score": rd_scores.get(1),
            "r2_score": rd_scores.get(2),
            "r3_score": rd_scores.get(3),
            "r4_score": rd_scores.get(4),
            **sg_avgs,
        })

    tourn_df = pd.DataFrame(tourn_rows)
    print(f"  Loaded {len(rounds_df)} round rows, {len(tourn_df)} tournament rows from local CSVs")
    print(f"  Years: {sorted(years_found)}")

    return tourn_df, rounds_df


def task1_pull_masters_sg():
    print("\n" + "=" * 60)
    print("TASK 1 — PULL DATAGOLF MASTERS SG DATA (2019–2025)")
    print("=" * 60)

    round_rows_all = []
    tournament_rows_all = []
    year_stats = {}

    # ── Source 1: Local round-level CSVs (2021-2025) ──
    print("\n  Loading local Masters round CSVs...")
    csv_tourn_df, csv_rounds_df = _load_masters_round_csvs()

    if len(csv_tourn_df) > 0:
        for year in sorted(csv_tourn_df["season"].unique()):
            yr_data = csv_tourn_df[csv_tourn_df["season"] == year]
            year_stats[year] = len(yr_data)
            print(f"  {year} (local CSV): {len(yr_data)} players")

    # ── Source 2: Existing golf_model historical_rounds.csv ──
    if GOLF_MODEL_HISTORY.exists():
        print(f"\n  Checking golf_model historical_rounds.csv...")
        hist_df = pd.read_csv(GOLF_MODEL_HISTORY)
        masters_hist = hist_df[
            hist_df["event_name"].str.contains("Masters|Augusta", case=False, na=False)
        ].copy()

        if len(masters_hist) > 0:
            for year in sorted(masters_hist["season"].unique()):
                if year in year_stats and year_stats[year] > 0:
                    continue  # already have from local CSVs
                yr_data = masters_hist[masters_hist["season"] == year]
                field_size = yr_data["field_size"].iloc[0] if "field_size" in yr_data.columns else len(yr_data)

                for _, row in yr_data.iterrows():
                    tournament_rows_all.append({
                        "player_name": row["player_name"],
                        "season": year,
                        "event_name": row.get("event_name", "Masters Tournament"),
                        "course": row.get("course", "Augusta National Golf Club"),
                        "date": row.get("date", ""),
                        "field_size": field_size,
                        "finish_pos": row.get("finish_pos"),
                        "sg_ott": row.get("sg_ott"),
                        "sg_app": row.get("sg_app"),
                        "sg_arg": row.get("sg_arg"),
                        "sg_putt": row.get("sg_putt"),
                        "sg_t2g": row.get("sg_t2g"),
                        "sg_total": row.get("sg_total"),
                    })
                year_stats[year] = len(yr_data)
                print(f"  {year} (golf_model CSV): {len(yr_data)} players")

    # Merge local CSV tournament rows + any additional from golf_model CSV
    if len(csv_tourn_df) > 0:
        # Convert csv_tourn_df rows into the same dict format
        for _, row in csv_tourn_df.iterrows():
            tournament_rows_all.append(row.to_dict())

    tourn_df = pd.DataFrame(tournament_rows_all)

    # Build rounds_df from CSV rounds (the golf_model CSV doesn't have round-level)
    rounds_df = csv_rounds_df if len(csv_rounds_df) > 0 else pd.DataFrame()

    # Add field_size to rounds_df for consistency
    if len(rounds_df) > 0 and "field_size" not in rounds_df.columns:
        fs_map = rounds_df[rounds_df["round_num"] == 1].groupby("season").size().to_dict()
        rounds_df["field_size"] = rounds_df["season"].map(fs_map)

    # Save
    if len(tourn_df) > 0:
        tourn_df.to_parquet(PROCESSED_DIR / "masters_sg_history.parquet", index=False)
        print(f"\n  Saved masters_sg_history.parquet: {len(tourn_df)} rows")
    if len(rounds_df) > 0:
        rounds_df.to_parquet(PROCESSED_DIR / "masters_sg_rounds.parquet", index=False)
        print(f"  Saved masters_sg_rounds.parquet: {len(rounds_df)} rows")

    print(f"\n  SUMMARY:")
    years_with_data = [y for y, c in year_stats.items() if c > 0]
    print(f"  Years with data: {sorted(years_with_data)}")
    print(f"  Rows per year: {dict(sorted(year_stats.items()))}")
    print(f"  Total tournament rows: {len(tourn_df)}")
    print(f"  Total round rows: {len(rounds_df)}")
    if len(tourn_df) > 0:
        print(f"  Unique players: {tourn_df['player_name'].nunique()}")

    for year in years_with_data:
        subset = tourn_df[tourn_df["season"] == year]
        sg_null_pct = subset[["sg_ott", "sg_app", "sg_arg", "sg_putt"]].isnull().mean().mean()
        if sg_null_pct > 0.5:
            print(f"  WARNING: {year} has {sg_null_pct:.0%} null SG columns")

    print("\n  Task 1 complete.")
    return tourn_df, rounds_df


# ═══════════════════════════════════════════════════════════
# TASK 2 — SCRAPE MASTERS HISTORICAL SCORES
# ═══════════════════════════════════════════════════════════

def _scrape_wikipedia_masters(year):
    """Scrape a single year's Masters results from Wikipedia.

    The full leaderboard is typically the LAST wikitable on the page.
    Format: Place | Player | Score (e.g. '70-68-67-70=275') | To par | Money
    """
    from bs4 import BeautifulSoup

    url = f"https://en.wikipedia.org/wiki/{year}_Masters_Tournament"
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "AugustaModel/1.0"})
        r.raise_for_status()
    except Exception as e:
        return None, str(e)

    soup = BeautifulSoup(r.text, "html.parser")
    tables = soup.find_all("table", class_="wikitable")

    if not tables:
        return None, "No wikitable found"

    # Find the full leaderboard — it's the table with the most rows and has Place + Player headers
    best_table = None
    best_row_count = 0
    for table in tables:
        row_count = len(table.find_all("tr"))
        headers_text = " ".join(th.get_text(strip=True).lower() for th in table.find_all("th", limit=10))
        if ("place" in headers_text or "pos" in headers_text) and "player" in headers_text:
            if row_count > best_row_count:
                best_table = table
                best_row_count = row_count

    if best_table is None:
        # Fallback: take the last large table
        for table in reversed(tables):
            if len(table.find_all("tr")) > 20:
                best_table = table
                break

    if best_table is None:
        return None, "No leaderboard table found"

    # Parse header row
    header_row = best_table.find("tr")
    headers = [th.get_text(strip=True).lower() for th in header_row.find_all("th")]

    col_map = {}
    for i, h in enumerate(headers):
        h_clean = h.strip()
        if h_clean in ("place", "pos", "finish"):
            col_map["place"] = i
        elif h_clean in ("player", "golfer"):
            col_map["player"] = i
        elif h_clean == "score":
            col_map["score"] = i
        elif h_clean in ("to par", "topar"):
            col_map["to_par"] = i
        elif h_clean in ("money", "money (us$)", "money(us$)"):
            col_map["money"] = i

    if "player" not in col_map:
        # Try to detect from content
        col_map = {"place": 0, "player": 1, "score": 2, "to_par": 3}

    rows = []
    current_place = None
    expected_cols = len(headers)

    for tr in best_table.find_all("tr")[1:]:
        cells = tr.find_all(["td", "th"])
        if len(cells) < 2:
            continue

        cell_texts = [c.get_text(strip=True) for c in cells]

        # Handle rowspan: if first cell has rowspan, it sets the place for multiple rows
        first_cell = cells[0]
        if first_cell.get("rowspan"):
            current_place = cell_texts[0]

        # Detect if place column was consumed by rowspan (fewer cells than headers)
        if len(cell_texts) < expected_cols:
            cell_texts = [current_place or ""] + cell_texts
        else:
            # Update current_place from first cell
            place_text = cell_texts[col_map.get("place", 0)]
            if place_text and (place_text[0].isdigit() or place_text.startswith("T") or place_text == "CUT"):
                current_place = place_text

        place_idx = col_map.get("place", 0)
        player_idx = col_map.get("player", 1)
        score_idx = col_map.get("score", 2)

        place = cell_texts[place_idx] if place_idx < len(cell_texts) else current_place
        player = cell_texts[player_idx] if player_idx < len(cell_texts) else ""
        score_text = cell_texts[score_idx] if score_idx < len(cell_texts) else ""

        # Clean player name
        player = player.replace("\xa0", " ").strip()
        # Remove annotations like (c), (a), etc.
        player = re.sub(r'\s*\([a-z]\)\s*$', '', player).strip()

        if not player or player.lower() in ("player", "golfer", ""):
            continue

        # Parse score: format is "70-68-67-70=275" or "76-72=148"
        r1 = r2 = r3 = r4 = None
        total = None

        if "=" in score_text:
            parts = score_text.split("=")
            try:
                total = int(parts[-1].strip())
            except (ValueError, TypeError):
                pass
            round_part = parts[0].strip()
            # Handle en-dash and regular dash
            round_scores = re.split(r'[-–]', round_part)
            for i, s in enumerate(round_scores):
                try:
                    val = int(s.strip())
                    if i == 0: r1 = val
                    elif i == 1: r2 = val
                    elif i == 2: r3 = val
                    elif i == 3: r4 = val
                except (ValueError, TypeError):
                    pass
        elif "-" in score_text or "–" in score_text:
            round_scores = re.split(r'[-–]', score_text)
            for i, s in enumerate(round_scores):
                try:
                    val = int(s.strip())
                    if i == 0: r1 = val
                    elif i == 1: r2 = val
                    elif i == 2: r3 = val
                    elif i == 3: r4 = val
                except (ValueError, TypeError):
                    pass
            rd_list = [s for s in [r1, r2, r3, r4] if s is not None]
            if rd_list:
                total = sum(rd_list)

        if total is None and any(s is not None for s in [r1, r2, r3, r4]):
            total = sum(s for s in [r1, r2, r3, r4] if s is not None)

        rows.append({
            "player_name": player,
            "season": year,
            "finish_pos": place or current_place,
            "round_1_score": r1,
            "round_2_score": r2,
            "round_3_score": r3,
            "round_4_score": r4,
            "total_score": total,
        })

    if not rows:
        return None, "No rows parsed from table"
    return pd.DataFrame(rows), None


def task2_scrape_historical_scores():
    print("\n" + "=" * 60)
    print("TASK 2 — SCRAPE MASTERS HISTORICAL SCORES (2004–2020)")
    print("=" * 60)

    print("\n  Masters.com is JS-rendered — using Wikipedia as data source...")

    all_dfs = []
    failed_years = []
    year_counts = {}

    for year in range(2004, 2021):
        print(f"  Scraping {year}...", end=" ")
        df, err = _scrape_wikipedia_masters(year)
        if df is not None and len(df) > 0:
            all_dfs.append(df)
            year_counts[year] = len(df)
            print(f"{len(df)} players")
        else:
            failed_years.append((year, err))
            print(f"FAILED: {err}")
        time.sleep(2)

    if not all_dfs:
        print("  ERROR: No data scraped from any year!")
        return pd.DataFrame()

    scores_df = pd.concat(all_dfs, ignore_index=True)

    # Compute derived fields
    scores_df["made_cut"] = scores_df.apply(
        lambda r: 1 if pd.notna(r["round_3_score"]) and pd.notna(r["round_4_score"]) else 0,
        axis=1
    )
    scores_df["rounds_played"] = scores_df[
        ["round_1_score", "round_2_score", "round_3_score", "round_4_score"]
    ].notna().sum(axis=1)

    round_cols = ["round_1_score", "round_2_score", "round_3_score", "round_4_score"]
    scores_df["scoring_avg"] = scores_df[round_cols].mean(axis=1)

    for season in scores_df["season"].unique():
        mask = (scores_df["season"] == season) & (scores_df["made_cut"] == 1)
        median_total = scores_df.loc[mask, "total_score"].median()
        scores_df.loc[scores_df["season"] == season, "score_vs_field"] = (
            scores_df.loc[scores_df["season"] == season, "total_score"] - median_total
        )

    scores_df["r3r4_scoring"] = scores_df[["round_3_score", "round_4_score"]].mean(axis=1)

    scores_df.to_parquet(PROCESSED_DIR / "masters_scores_historical.parquet", index=False)

    print(f"\n  SUMMARY:")
    print(f"  Years scraped: {sorted(year_counts.keys())}")
    print(f"  Total rows: {len(scores_df)}")
    for y in sorted(year_counts.keys()):
        print(f"    {y}: {year_counts[y]} players")
    if failed_years:
        print(f"  FAILED YEARS: {failed_years}")
    else:
        print(f"  All years successful")

    print("\n  Task 2 complete.")
    return scores_df


# ═══════════════════════════════════════════════════════════
# TASK 3 — BUILD UNIFIED MASTER DATASET
# ═══════════════════════════════════════════════════════════

def _parse_finish_num(pos):
    if pos is None or pd.isna(pos):
        return None
    pos = str(pos).strip().upper()
    if pos in ("CUT", "MC", "WD", "DQ", "MDF"):
        return 999
    pos = pos.replace("T", "").replace("=", "")
    try:
        return int(pos)
    except (ValueError, TypeError):
        return None


def task3_build_unified(sg_df=None, scores_df=None):
    print("\n" + "=" * 60)
    print("TASK 3 — BUILD UNIFIED MASTER DATASET")
    print("=" * 60)

    if sg_df is None:
        sg_path = PROCESSED_DIR / "masters_sg_history.parquet"
        sg_df = pd.read_parquet(sg_path) if sg_path.exists() else pd.DataFrame()

    if scores_df is None:
        scores_path = PROCESSED_DIR / "masters_scores_historical.parquet"
        scores_df = pd.read_parquet(scores_path) if scores_path.exists() else pd.DataFrame()

    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "rapidfuzz"])
        from rapidfuzz import fuzz, process

    rows = []

    # Part A: SG data rows
    rounds_path = PROCESSED_DIR / "masters_sg_rounds.parquet"
    rounds_df = pd.read_parquet(rounds_path) if rounds_path.exists() else pd.DataFrame()

    if len(sg_df) > 0:
        for _, row in sg_df.iterrows():
            # Check if round scores are already in the tournament-level data (from local CSVs)
            r1 = row.get("r1_score")
            r2 = row.get("r2_score")
            r3 = row.get("r3_score")
            r4 = row.get("r4_score")

            # If not, try to get from rounds parquet
            if r1 is None and len(rounds_df) > 0:
                player_rounds = rounds_df[
                    (rounds_df["player_name"] == row["player_name"]) &
                    (rounds_df["season"] == row["season"])
                ].sort_values("round_num")
                for _, rnd in player_rounds.iterrows():
                    rn = rnd.get("round_num")
                    sc = rnd.get("score")
                    if rn == 1: r1 = sc
                    elif rn == 2: r2 = sc
                    elif rn == 3: r3 = sc
                    elif rn == 4: r4 = sc

            # Convert pandas NA to None
            r1 = None if pd.isna(r1) else r1
            r2 = None if pd.isna(r2) else r2
            r3 = None if pd.isna(r3) else r3
            r4 = None if pd.isna(r4) else r4

            finish_num = _parse_finish_num(row["finish_pos"])
            total = sum(s for s in [r1, r2, r3, r4] if s is not None) if any(s is not None for s in [r1, r2, r3, r4]) else None
            made_cut = 1 if r3 is not None and r4 is not None else (0 if finish_num == 999 else None)

            # Check if SG data is actually present (not all null)
            sg_vals = [row.get("sg_ott"), row.get("sg_app"), row.get("sg_arg"), row.get("sg_putt")]
            has_sg = any(v is not None and not (isinstance(v, float) and np.isnan(v)) for v in sg_vals)

            rows.append({
                "dg_id": row.get("dg_id"),
                "player_name": row["player_name"],
                "season": row["season"],
                "finish_pos": row["finish_pos"],
                "finish_num": finish_num,
                "made_cut": made_cut,
                "field_size": row.get("field_size"),
                "sg_ott": row.get("sg_ott"),
                "sg_app": row.get("sg_app"),
                "sg_arg": row.get("sg_arg"),
                "sg_putt": row.get("sg_putt"),
                "sg_t2g": row.get("sg_t2g"),
                "sg_total": row.get("sg_total"),
                "has_sg_data": has_sg,
                "r1_score": r1, "r2_score": r2, "r3_score": r3, "r4_score": r4,
                "total_score": total,
                "score_vs_field": None,
                "r3r4_scoring": np.mean([s for s in [r3, r4] if s is not None]) if any(s is not None for s in [r3, r4]) else None,
                "data_source": "dg_sg",
            })

    # Part B: Scores-only data
    sg_years = set(sg_df["season"].unique()) if len(sg_df) > 0 else set()
    sg_names_by_year = {}
    if len(sg_df) > 0:
        for yr in sg_years:
            sg_names_by_year[yr] = set(sg_df[sg_df["season"] == yr]["player_name"].str.lower().tolist())

    if len(scores_df) > 0:
        low_confidence_matches = []
        for _, row in scores_df.iterrows():
            season = row["season"]
            if season in sg_years:
                player_lower = str(row["player_name"]).lower()
                if player_lower in sg_names_by_year.get(season, set()):
                    continue
                sg_names_list = list(sg_names_by_year.get(season, []))
                if sg_names_list:
                    best = process.extractOne(player_lower, sg_names_list, scorer=fuzz.ratio)
                    if best and best[1] >= 85:
                        if best[1] < 95:
                            low_confidence_matches.append((row["player_name"], best[0], best[1]))
                        continue

            finish_num = _parse_finish_num(row["finish_pos"])
            # Estimate field_size from total players in this year's scores
            yr_count = len(scores_df[scores_df["season"] == season])
            rows.append({
                "dg_id": None,
                "player_name": row["player_name"],
                "season": season,
                "finish_pos": row["finish_pos"],
                "finish_num": finish_num,
                "made_cut": row.get("made_cut"),
                "field_size": yr_count,
                "sg_ott": None, "sg_app": None, "sg_arg": None,
                "sg_putt": None, "sg_t2g": None, "sg_total": None,
                "has_sg_data": False,
                "r1_score": row.get("round_1_score"),
                "r2_score": row.get("round_2_score"),
                "r3_score": row.get("round_3_score"),
                "r4_score": row.get("round_4_score"),
                "total_score": row.get("total_score"),
                "score_vs_field": row.get("score_vs_field"),
                "r3r4_scoring": row.get("r3r4_scoring"),
                "data_source": "scores_only",
            })

        if low_confidence_matches:
            print(f"\n  Low-confidence name matches (85-95):")
            for name1, name2, score in low_confidence_matches[:20]:
                print(f"    '{name1}' -> '{name2}' ({score})")

    unified = pd.DataFrame(rows)

    if len(unified) > 0:
        # Compute score_vs_field for SG rows
        for season in unified["season"].unique():
            mask_cut = (unified["season"] == season) & (unified["made_cut"] == 1)
            median_total = unified.loc[mask_cut, "total_score"].median()
            if pd.notna(median_total):
                mask_sg = (unified["season"] == season) & (unified["data_source"] == "dg_sg")
                unified.loc[mask_sg, "score_vs_field"] = (
                    unified.loc[mask_sg, "total_score"] - median_total
                )

        unified.to_parquet(PROCESSED_DIR / "masters_unified.parquet", index=False)

    print(f"\n  SCHEMA: {list(unified.columns)}")
    print(f"  Shape: {unified.shape}")
    print(f"  Rows with full SG data: {(unified['has_sg_data'] == True).sum()}")
    print(f"  Rows with scores only: {(unified['data_source'] == 'scores_only').sum()}")
    print(f"  Years covered: {sorted(unified['season'].unique())}")
    print(f"  Unique players: {unified['player_name'].nunique()}")

    print("\n  Task 3 complete.")
    return unified
