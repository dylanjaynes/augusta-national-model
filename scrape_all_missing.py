#!/usr/bin/env python3
"""
Scrape all missing data: 2023 PGA (from stats pages) + LIV 2022-2026 (from past-results).
Then merge, retrain, and regenerate 2026 predictions.
"""
import os, re, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import requests
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

os.environ.update({k.split("=",1)[0]:k.split("=",1)[1]
                   for k in open(".env").read().strip().split("\n") if "=" in k})

API_KEY = os.environ["DATAGOLF_API_KEY"]
DG_SESSION = os.environ.get("DG_SESSION_COOKIE", "")
PROCESSED = Path("data/processed")
RAW = Path("data/raw"); RAW.mkdir(parents=True, exist_ok=True)


def get_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"})
    if DG_SESSION:
        s.cookies.set("session", DG_SESSION, domain="datagolf.com")
    return s


def norm_name(n):
    n = str(n).strip()
    if "," in n:
        parts = n.split(",", 1)
        return f"{parts[1].strip()} {parts[0].strip()}"
    return n


def pfn(p):
    if p is None or pd.isna(p): return None
    p = str(p).strip().upper()
    if p in ("CUT","MC","WD","DQ","MDF"): return 999
    p = p.replace("T","").replace("=","")
    try: return int(p)
    except: return None


# ═══════════════════════════════════════════════════════════
# TASK 1 — SCRAPE 2023 PGA FROM HISTORICAL-TOURNAMENT-STATS
# ═══════════════════════════════════════════════════════════

def scrape_2023_pga(session):
    print("\n" + "=" * 60)
    print("TASK 1 — SCRAPE 2023 PGA TOUR DATA")
    print("=" * 60)

    # Get 2023 PGA event list
    r = requests.get("https://feeds.datagolf.com/get-event-list",
                     params={"tour": "pga", "file_format": "json", "key": API_KEY}, timeout=15)
    all_events = r.json()
    events_2023 = [e for e in all_events if e.get("calendar_year") == 2023]
    print(f"  2023 PGA events from API: {len(events_2023)}")

    # We know the stats page loads one event at a time and uses t_num
    # We already have t_num -> API event_id mappings for some events
    # But for 2023, the stats page should load 2023 data since that's the latest
    # year for events that didn't run in 2024+

    # Strategy: for each t_num (1-55), load the stats page.
    # If start_year >= 2023, extract the data.
    # Skip if we already have this event-year in our extended dataset.

    existing = pd.read_parquet(PROCESSED / "historical_rounds_extended.parquet")
    existing_keys = set(zip(existing["event_name"], existing["season"].astype(int)))

    all_rows = []
    events_scraped = {}

    for t_num in range(1, 56):
        url = f"https://datagolf.com/historical-tournament-stats?event_id={t_num}"
        try:
            r = session.get(url, timeout=15)
        except Exception as e:
            continue

        for jstr in re.findall(r"JSON\.parse\('(.*?)'\)", r.text, re.DOTALL):
            jstr = jstr.replace("\\'", "'")
            if len(jstr) < 1000: continue

            d = json.loads(jstr)
            actual_tnum = d.get("t_num")
            event_name = d.get("event_name", "?")
            start_year = d.get("start_year")

            if actual_tnum in events_scraped:
                break

            # Only want 2023 data (or 2024 if we're missing it)
            yr_data = d.get(str(start_year), {})
            lb = yr_data.get("lb", [])
            stats = yr_data.get("stats", {})
            ci = yr_data.get("course_info", {})
            course = ci.get("course_name", "")

            if not lb:
                break

            # Skip if already in extended dataset
            if (event_name, start_year) in existing_keys:
                events_scraped[actual_tnum] = {"name": event_name, "year": start_year, "status": "exists"}
                break

            # Only want 2023-2024 data we're missing
            if start_year not in (2023, 2024):
                break

            # Build SG lookup
            sg_by_pnum = {}
            for sg_cat in ["sg_putt","sg_arg","sg_app","sg_ott","sg_t2g","sg_total"]:
                for entry in stats.get(sg_cat, []):
                    pnum = str(entry.get("player_num", ""))
                    if pnum not in sg_by_pnum: sg_by_pnum[pnum] = {}
                    sg_by_pnum[pnum][sg_cat] = entry.get("event")

            for p in lb:
                pnum = str(p.get("player_num", ""))
                sg = sg_by_pnum.get(pnum, {})
                all_rows.append({
                    "dg_id": p.get("dg_id"),
                    "player_name": norm_name(p.get("player_name", "")),
                    "season": start_year,
                    "event_name": event_name,
                    "course": course,
                    "date": "",
                    "field_size": len(lb),
                    "finish_pos": p.get("fin_text", ""),
                    "sg_ott": sg.get("sg_ott"),
                    "sg_app": sg.get("sg_app"),
                    "sg_arg": sg.get("sg_arg"),
                    "sg_putt": sg.get("sg_putt"),
                    "sg_t2g": sg.get("sg_t2g"),
                    "sg_total": sg.get("sg_total"),
                })

            events_scraped[actual_tnum] = {"name": event_name, "year": start_year, "players": len(lb)}
            print(f"  t_num={t_num:>2} -> {event_name:<40} {start_year} {len(lb)} players")
            break

        time.sleep(0.3)

    if all_rows:
        df = pd.DataFrame(all_rows)
        # Clean sentinels
        for c in ["sg_ott","sg_app","sg_arg","sg_putt","sg_t2g","sg_total"]:
            df.loc[df[c] == -9999, c] = np.nan
        df.to_parquet(RAW / "scraped_sg_2023_pga.parquet", index=False)
        print(f"\n  Scraped: {len(df)} rows from {len([e for e in events_scraped.values() if 'players' in e])} new events")
        print(f"  Skipped (already have): {len([e for e in events_scraped.values() if e.get('status')=='exists'])}")
    else:
        df = pd.DataFrame()
        print("\n  No new 2023 PGA data found")

    return df


# ═══════════════════════════════════════════════════════════
# TASK 2-4 — SCRAPE LIV FROM PAST-RESULTS PAGES
# ═══════════════════════════════════════════════════════════

def scrape_liv(session):
    print("\n" + "=" * 60)
    print("TASKS 2-4 — SCRAPE LIV DATA FROM PAST-RESULTS")
    print("=" * 60)

    # Get LIV event list from API
    r = requests.get("https://feeds.datagolf.com/get-event-list",
                     params={"tour": "alt", "file_format": "json", "key": API_KEY}, timeout=15)
    liv_events = r.json()
    print(f"  LIV events from API: {len(liv_events)}")

    all_rows = []
    events_scraped = {}

    for evt in liv_events:
        eid = evt["event_id"]
        yr = evt["calendar_year"]
        ename = evt["event_name"]

        # Past-results URL pattern: /past-results/liv-golf/{event_id}/{year}
        # But the event_id in the URL might differ from the API event_id
        # The API event_id increments across years (24=Mayakoba 2024, 9=Mayakoba 2023)
        # Try the direct past-results page
        url = f"https://datagolf.com/past-results/liv-golf/{eid}/{yr}"
        try:
            r2 = session.get(url, timeout=15)
        except:
            continue

        if r2.status_code != 200 or len(r2.text) < 5000:
            # Try alternative URL patterns
            continue

        # Extract JSON
        for jstr in re.findall(r"JSON\.parse\('(.*?)'\)", r2.text, re.DOTALL):
            jstr = jstr.replace("\\'", "'")
            if len(jstr) < 2000: continue

            d = json.loads(jstr)
            if not isinstance(d, dict) or "lb" not in d:
                continue

            lb = d.get("lb", [])
            info = d.get("info", {})
            actual_name = info.get("event_name", ename)
            actual_year = info.get("calendar_year", yr)
            num_rounds = info.get("num_rounds", 3)

            if not lb:
                continue

            for p in lb:
                name = norm_name(p.get("player_name", ""))
                dg_id = p.get("dg_id")
                fin = p.get("fin", "")

                # Compute tournament sg_total from per-round SG
                round_sgs = []
                for rn in range(1, num_rounds + 1):
                    rsg = p.get(f"R{rn}_sg")
                    if rsg is not None and rsg != 0:
                        round_sgs.append(rsg)

                sg_total = np.mean(round_sgs) if round_sgs else None

                all_rows.append({
                    "dg_id": dg_id,
                    "player_name": name,
                    "season": actual_year,
                    "event_name": f"LIV {actual_name}",
                    "course": d.get("course", ""),
                    "date": info.get("date", ""),
                    "field_size": len(lb),
                    "finish_pos": fin,
                    "sg_ott": None,  # LIV past-results only has total SG
                    "sg_app": None,
                    "sg_arg": None,
                    "sg_putt": None,
                    "sg_t2g": None,
                    "sg_total": sg_total,
                })

            events_scraped[f"{actual_name}_{actual_year}"] = len(lb)
            print(f"  LIV {actual_name} {actual_year}: {len(lb)} players, {len(round_sgs) if lb else 0} with SG")
            break

        time.sleep(0.5)

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_parquet(RAW / "scraped_sg_liv.parquet", index=False)
        print(f"\n  LIV total: {len(df)} rows from {len(events_scraped)} events")
        print(f"  Years: {sorted(df['season'].unique())}")
        print(f"  Unique players: {df['player_name'].nunique()}")

        # Check key players
        for pn in ["Jon Rahm","Bryson DeChambeau","Cameron Smith","Brooks Koepka","Dustin Johnson"]:
            pt = df[df["player_name"]==pn]
            if len(pt) > 0:
                sg_mean = pt["sg_total"].dropna().mean()
                print(f"    {pn}: {len(pt)} events, sg_total avg={sg_mean:.3f}")
            else:
                print(f"    {pn}: NOT FOUND")

        # Check SG availability
        sg_avail = df["sg_total"].notna().sum()
        print(f"  SG total non-null: {sg_avail}/{len(df)} ({sg_avail/len(df):.0%})")
    else:
        df = pd.DataFrame()
        print("\n  No LIV data scraped")

    return df


# ═══════════════════════════════════════════════════════════
# TASK 5 — MERGE AND REBUILD
# ═══════════════════════════════════════════════════════════

def merge_and_rebuild(pga_2023_df, liv_df):
    print("\n" + "=" * 60)
    print("TASK 5 — MERGE AND REBUILD")
    print("=" * 60)

    existing = pd.read_parquet(PROCESSED / "historical_rounds_extended.parquet")
    print(f"  Existing: {len(existing):,} rows")

    new_dfs = [existing]
    if len(pga_2023_df) > 0:
        print(f"  Adding 2023 PGA: {len(pga_2023_df)} rows")
        new_dfs.append(pga_2023_df)
    if len(liv_df) > 0:
        print(f"  Adding LIV: {len(liv_df)} rows")
        new_dfs.append(liv_df)

    combined = pd.concat(new_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["player_name","season","event_name"], keep="last")

    # Clean sentinels
    for c in ["sg_ott","sg_app","sg_arg","sg_putt","sg_t2g","sg_total"]:
        if c in combined.columns:
            combined.loc[combined[c] == -9999, c] = np.nan

    combined.to_parquet(PROCESSED / "historical_rounds_extended.parquet", index=False)

    print(f"  Combined: {len(combined):,} rows")
    for yr in sorted(combined["season"].unique()):
        n = len(combined[combined["season"]==yr])
        ne = combined[combined["season"]==yr]["event_name"].nunique()
        np_ = combined[combined["season"]==yr]["player_name"].nunique()
        print(f"    {yr}: {n:,} rows, {ne} events, {np_} players")

    # Check 2026 field coverage
    preds = pd.read_parquet(PROCESSED / "predictions_2026.parquet")
    field_names = set(preds["player_name"])
    combined_names = set(combined["player_name"])
    print(f"\n  2026 field: {len(field_names & combined_names)}/{len(field_names)} have tour data")

    # Key player recency check
    print(f"\n  Key player latest data:")
    for pn in ["Scottie Scheffler","Jon Rahm","Rory McIlroy","Bryson DeChambeau",
               "Cameron Smith","Cameron Young","Viktor Hovland","Wyndham Clark","Ludvig Aberg"]:
        pt = combined[combined["player_name"]==pn]
        if len(pt) > 0:
            latest = pt["season"].max()
            n_events = len(pt)
            liv_events = len(pt[pt["event_name"].str.contains("LIV", case=False, na=False)])
            print(f"    {pn:<25} {n_events:>3} events (latest={latest}, {liv_events} LIV)")
        else:
            print(f"    {pn:<25} MISSING")

    return combined


def main():
    print("=" * 60)
    print("  SCRAPE ALL MISSING DATA")
    print("=" * 60)

    session = get_session()

    # Test auth
    r = session.get("https://datagolf.com/historical-tournament-stats", timeout=15)
    if "login" in r.text[:3000].lower() and len(r.text) < 10000:
        print("  SESSION EXPIRED — need fresh cookies")
        return
    print("  Session authenticated")

    pga_2023 = scrape_2023_pga(session)
    liv = scrape_liv(session)
    combined = merge_and_rebuild(pga_2023, liv)

    print("\n" + "=" * 60)
    print("  SCRAPING COMPLETE — RUN run_field_strength.py TO RETRAIN")
    print("=" * 60)


if __name__ == "__main__":
    main()
