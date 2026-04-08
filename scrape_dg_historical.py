#!/usr/bin/env python3
"""
Scrape 2023-2025 PGA Tour SG data from datagolf.com/historical-tournament-stats.
Requires logged-in session cookies in .env.
"""
import os, re, json, time, sys
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

API_KEY = os.getenv("DATAGOLF_API_KEY")
DG_SESSION = os.getenv("DG_SESSION_COOKIE", "")
DG_REMEMBER = os.getenv("DG_REMEMBER_COOKIE", "")
RAW_DIR = Path("data/raw"); RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = Path("data/processed")
HIST_PATH = Path("/Users/dylanjaynes/golf_model/golf_model_data/historical_rounds.csv")

# Known PGA Tour event IDs for 2023-2025 (from DG's URL patterns)
# These are the event_id values used in datagolf.com URLs
# We'll discover them dynamically from the schedule page


def get_session():
    """Build a requests session with DG auth cookies."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    cookies = {}
    if DG_SESSION:
        cookies["session"] = DG_SESSION
    if DG_REMEMBER:
        cookies["remember_token"] = DG_REMEMBER

    # Also try loading all cookies from .env that start with DG_COOKIE_
    for key, val in os.environ.items():
        if key.startswith("DG_COOKIE_"):
            cookie_name = key[len("DG_COOKIE_"):].lower()
            cookies[cookie_name] = val

    for name, val in cookies.items():
        s.cookies.set(name, val, domain="datagolf.com")

    return s


def discover_events(session, year):
    """Scrape the event list for a given year from DG's historical stats page."""
    url = f"https://datagolf.com/historical-tournament-stats?year={year}"
    r = session.get(url, timeout=30)
    if r.status_code != 200:
        print(f"  Failed to load page for {year}: {r.status_code}")
        return []

    # Look for event dropdown or event list in the page
    # DG uses JavaScript to populate — look for event data in JS
    patterns = [
        r'var\s+event_list\s*=\s*(\[.*?\]);',
        r'var\s+events\s*=\s*(\[.*?\]);',
        r'"events"\s*:\s*(\[.*?\])',
        r'event_id.*?(\d+).*?event_name.*?"(.*?)"',
    ]

    events = []
    for pat in patterns[:3]:
        match = re.search(pat, r.text, re.DOTALL)
        if match:
            try:
                events = json.loads(match.group(1))
                break
            except json.JSONDecodeError:
                continue

    if not events:
        # Try to find all event_id references
        event_ids = set(re.findall(r'event_id["\s:=]+(\d+)', r.text))
        event_names = re.findall(r'"event_name"\s*:\s*"(.*?)"', r.text)
        if event_ids:
            print(f"  Found event_ids: {event_ids}")
            events = [{"event_id": eid} for eid in event_ids]

    # Also look for the select/dropdown options
    option_matches = re.findall(r'value="(\d+)"[^>]*>(.*?)</option>', r.text)
    if option_matches and not events:
        events = [{"event_id": eid, "event_name": name.strip()} for eid, name in option_matches]

    return events


def scrape_event(session, year, event_id):
    """Scrape SG data for a single event from the historical stats page."""
    url = f"https://datagolf.com/historical-tournament-stats?year={year}&event_id={event_id}"
    r = session.get(url, timeout=30)
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"

    html = r.text

    # Extract window.current_data or current_data
    patterns = [
        r'var\s+current_data\s*=\s*({.*?});\s*(?:var\s|//|$)',
        r'current_data\s*=\s*({.*?});\s*(?:var\s|//|$)',
        r'window\.current_data\s*=\s*({.*?});\s*(?:var\s|//|$)',
    ]

    data = None
    for pat in patterns:
        match = re.search(pat, html, re.DOTALL)
        if match:
            try:
                raw = match.group(1)
                # Fix common JS → JSON issues
                raw = re.sub(r'(\w+)\s*:', r'"\1":', raw)  # unquoted keys
                raw = raw.replace("'", '"')
                data = json.loads(raw)
                break
            except json.JSONDecodeError:
                continue

    if data is None:
        # Try a broader regex — just find the biggest JSON-like block
        all_json = re.findall(r'(?:var\s+\w+\s*=\s*)({[^;]{500,}});', html, re.DOTALL)
        for candidate in all_json:
            try:
                candidate = re.sub(r"(?<=[{,])\s*(\w+)\s*:", r'"\1":', candidate)
                candidate = candidate.replace("'", '"')
                data = json.loads(candidate)
                if "lb" in data or "stats" in data or "leaderboard" in data:
                    break
                data = None
            except:
                continue

    if data is None:
        # Check if page requires login
        if "login" in html.lower() or "sign in" in html.lower():
            return None, "Requires login"
        return None, "Could not find current_data in page"

    return data, None


def parse_event_data(data, year, event_id):
    """Parse the scraped current_data into rows matching historical_rounds schema."""
    rows = []

    # Get event metadata
    event_name = ""
    course = ""
    date = ""

    if isinstance(data.get("course_info"), dict):
        course = data["course_info"].get("course_name", "")
        event_name = data["course_info"].get("event_name", "")
    elif isinstance(data.get("event_info"), dict):
        event_name = data["event_info"].get("event_name", "")
        course = data["event_info"].get("course_name", "")
        date = data["event_info"].get("date", "")

    if not event_name:
        event_name = data.get("event_name", f"Event {event_id}")

    # Get leaderboard
    lb = data.get("lb", data.get("leaderboard", []))
    if not lb:
        return [], event_name

    # Build player lookup from leaderboard
    players = {}
    for p in lb:
        pid = p.get("player_num", p.get("dg_id", p.get("player_id")))
        if pid is None:
            continue
        players[pid] = {
            "dg_id": p.get("dg_id", pid),
            "player_name": p.get("player_name", ""),
            "finish_pos": p.get("fin_text", p.get("finish_pos", "")),
        }

    field_size = len(players)

    # Get SG stats
    sg_stats = data.get("stats", {})
    sg_by_player = {}  # pid → {sg_ott: ..., sg_app: ..., ...}

    for sg_cat in ["sg_putt", "sg_arg", "sg_app", "sg_ott", "sg_t2g", "sg_total"]:
        cat_data = sg_stats.get(sg_cat, [])
        if isinstance(cat_data, list):
            for entry in cat_data:
                pid = entry.get("player_num", entry.get("player_id"))
                if pid is None:
                    continue
                if pid not in sg_by_player:
                    sg_by_player[pid] = {}
                # Event-level average
                sg_by_player[pid][sg_cat] = entry.get("event", entry.get("value"))

    # Build rows
    for pid, pinfo in players.items():
        sg = sg_by_player.get(pid, {})
        rows.append({
            "dg_id": pinfo["dg_id"],
            "player_name": pinfo["player_name"],
            "season": year,
            "event_name": event_name,
            "course": course,
            "date": date,
            "field_size": field_size,
            "finish_pos": pinfo["finish_pos"],
            "sg_ott": sg.get("sg_ott"),
            "sg_app": sg.get("sg_app"),
            "sg_arg": sg.get("sg_arg"),
            "sg_putt": sg.get("sg_putt"),
            "sg_t2g": sg.get("sg_t2g"),
            "sg_total": sg.get("sg_total"),
        })

    return rows, event_name


def main():
    print("=" * 60)
    print("  SCRAPE DG HISTORICAL TOURNAMENT STATS (2023-2025)")
    print("=" * 60)

    if not DG_SESSION and not DG_REMEMBER and not any(k.startswith("DG_COOKIE_") for k in os.environ):
        print("\n  ERROR: No DG session cookies found in .env")
        print("  Add your datagolf.com cookies to .env:")
        print("    DG_SESSION_COOKIE=<your session cookie value>")
        print("    DG_REMEMBER_COOKIE=<your remember_token value>")
        print("  Or use DG_COOKIE_<name>=<value> for other cookie names.")
        print("\n  To get cookies:")
        print("  1. Open Chrome, go to datagolf.com (logged in)")
        print("  2. F12 → Application → Cookies → datagolf.com")
        print("  3. Copy cookie name/value pairs")
        return

    session = get_session()

    # Test authentication
    print("\n  Testing authentication...")
    r = session.get("https://datagolf.com/historical-tournament-stats", timeout=15)
    if "login" in r.text.lower()[:2000] and "historical" not in r.text.lower()[:2000]:
        print("  NOT LOGGED IN — cookies may be expired.")
        print("  Please refresh your cookies from Chrome DevTools.")
        return
    print(f"  Page loaded ({len(r.text)} bytes)")

    # Discover available years and events
    all_rows = []
    checkpoint_path = RAW_DIR / "dg_scrape_progress.parquet"

    # Load checkpoint if exists
    if checkpoint_path.exists():
        existing = pd.read_parquet(checkpoint_path)
        all_rows = existing.to_dict("records")
        scraped_keys = set(zip(existing["season"], existing["event_name"]))
        print(f"  Loaded checkpoint: {len(existing)} rows from {len(scraped_keys)} events")
    else:
        scraped_keys = set()

    for year in [2023, 2024, 2025]:
        print(f"\n  {'─'*50}")
        print(f"  YEAR: {year}")
        print(f"  {'─'*50}")

        # Discover events from the page
        print(f"  Discovering events...")
        events = discover_events(session, year)
        print(f"  Found {len(events)} events")

        if not events:
            # Try known event_id range (DG uses sequential IDs, ~1-50 per year)
            print(f"  Trying sequential event IDs...")
            for eid in range(1, 60):
                if (year, f"event_{eid}") in scraped_keys:
                    continue
                data, err = scrape_event(session, year, eid)
                if data is None:
                    if err == "Requires login":
                        print(f"  Login required — stopping")
                        break
                    continue

                rows, ename = parse_event_data(data, year, eid)
                if rows:
                    all_rows.extend(rows)
                    scraped_keys.add((year, ename))
                    print(f"    Event {eid}: {ename} — {len(rows)} players")
                time.sleep(1)

                # Save progress every 10 events
                if len(all_rows) > 0 and eid % 10 == 0:
                    pd.DataFrame(all_rows).to_parquet(checkpoint_path, index=False)
        else:
            for evt in events:
                eid = evt.get("event_id", evt.get("id"))
                ename = evt.get("event_name", f"event_{eid}")

                if (year, ename) in scraped_keys:
                    print(f"    Skipping {ename} (already scraped)")
                    continue

                data, err = scrape_event(session, year, eid)
                if data is None:
                    print(f"    FAILED {ename}: {err}")
                    if err == "Requires login":
                        break
                    continue

                rows, parsed_name = parse_event_data(data, year, eid)
                if rows:
                    all_rows.extend(rows)
                    scraped_keys.add((year, parsed_name or ename))
                    print(f"    {parsed_name or ename}: {len(rows)} players")
                else:
                    print(f"    {ename}: no rows parsed")

                time.sleep(1)

                if len(all_rows) % 500 == 0 and len(all_rows) > 0:
                    pd.DataFrame(all_rows).to_parquet(checkpoint_path, index=False)

    if not all_rows:
        print("\n  No data scraped. Check cookies and try again.")
        return

    # Save scraped data
    scraped_df = pd.DataFrame(all_rows)
    scraped_df.to_parquet(checkpoint_path, index=False)
    print(f"\n  Total scraped: {len(scraped_df)} rows")
    print(f"  Seasons: {sorted(scraped_df['season'].unique())}")
    print(f"  Events: {scraped_df['event_name'].nunique()}")
    print(f"  Players: {scraped_df['player_name'].nunique()}")

    # Normalize player names (Last, First → First Last)
    def norm(name):
        name = str(name).strip()
        if "," in name:
            parts = name.split(",", 1)
            return f"{parts[1].strip()} {parts[0].strip()}"
        return name
    scraped_df["player_name"] = scraped_df["player_name"].map(norm)

    # Merge with existing historical_rounds.csv
    print(f"\n  Merging with existing historical_rounds.csv...")
    existing = pd.read_csv(HIST_PATH)
    print(f"  Existing: {len(existing)} rows ({sorted(existing['season'].unique())})")

    combined = pd.concat([existing, scraped_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["player_name", "season", "event_name"], keep="last")

    out_path = PROCESSED_DIR / "historical_rounds_extended.parquet"
    combined.to_parquet(out_path, index=False)
    # Also save CSV for compatibility
    csv_path = RAW_DIR / "historical_rounds_extended.csv"
    combined.to_csv(csv_path, index=False)

    print(f"  Combined: {len(combined)} rows")
    print(f"  Seasons: {sorted(combined['season'].unique())}")
    print(f"  Saved to: {out_path}")
    print(f"  Saved CSV: {csv_path}")

    # Summary by year
    print(f"\n  Rows per season:")
    for yr in sorted(combined["season"].unique()):
        n = len(combined[combined["season"] == yr])
        ne = combined[combined["season"] == yr]["event_name"].nunique()
        print(f"    {yr}: {n:,} rows, {ne} events")

    print("\n  DONE. Now update HIST_PATH in pipeline scripts to use historical_rounds_extended.")


if __name__ == "__main__":
    main()
