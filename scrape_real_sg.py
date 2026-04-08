#!/usr/bin/env python3
"""
Scrape REAL SG breakdown (ott/app/arg/putt) for all missing event-years
using PUT /get-historical-stat-file endpoint.
"""
import os, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import requests

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ.update({k.split("=",1)[0]:k.split("=",1)[1]
                   for k in open(".env").read().strip().split("\n") if "=" in k})

PROCESSED = Path("data/processed")
RAW = Path("data/raw"); RAW.mkdir(parents=True, exist_ok=True)
API_KEY = os.environ["DATAGOLF_API_KEY"]

# All known PGA t_nums
PGA_TNUMS = [2,3,4,5,6,7,9,10,11,12,13,14,16,19,20,21,23,26,27,28,30,32,33,34,41,47,54]
# Years we need real SG for
YEARS_NEEDED = [2023, 2024]

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Content-Type": "application/json",
    "X-Requested-With": "XMLHttpRequest",
    "Origin": "https://datagolf.com",
    "Referer": "https://datagolf.com/historical-tournament-stats",
})
session.cookies.set("session", os.environ.get("DG_SESSION_COOKIE",""), domain="datagolf.com")


def norm_name(n):
    n = str(n).strip()
    if "," in n:
        parts = n.split(",", 1)
        return f"{parts[1].strip()} {parts[0].strip()}"
    return n


def fetch_event(t_num, year):
    """Fetch full SG data for one event-year via the golden endpoint."""
    r = session.put("https://datagolf.com/get-historical-stat-file",
                     json={"base": "no", "t_num": t_num, "year": year},
                     timeout=30)
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"

    data = r.json()
    lb = data.get("lb", [])
    stats = data.get("stats", {})
    course_info = data.get("course_info", {})

    if not lb:
        return None, "empty leaderboard"

    # Build SG lookup by player_num
    sg_by_pnum = {}
    for sg_cat in ["sg_ott","sg_app","sg_arg","sg_putt","sg_t2g","sg_total"]:
        for entry in stats.get(sg_cat, []):
            pnum = str(entry.get("player_num",""))
            if pnum not in sg_by_pnum:
                sg_by_pnum[pnum] = {}
            val = entry.get("event")
            # Clean -9999 sentinel
            if val is not None and val == -9999:
                val = None
            sg_by_pnum[pnum][sg_cat] = val

    rows = []
    for p in lb:
        pnum = str(p.get("player_num",""))
        sg = sg_by_pnum.get(pnum, {})
        rows.append({
            "dg_id": p.get("dg_id"),
            "player_name": norm_name(p.get("player_name","")),
            "season": year,
            "event_name": course_info.get("event_name", f"t_num_{t_num}"),
            "course": course_info.get("course_name",""),
            "date": course_info.get("date",""),
            "field_size": len(lb),
            "finish_pos": p.get("fin_text",""),
            "sg_ott": sg.get("sg_ott"),
            "sg_app": sg.get("sg_app"),
            "sg_arg": sg.get("sg_arg"),
            "sg_putt": sg.get("sg_putt"),
            "sg_t2g": sg.get("sg_t2g"),
            "sg_total": sg.get("sg_total"),
        })

    return rows, None


def main():
    print("=" * 60)
    print("  SCRAPE REAL SG BREAKDOWN — /get-historical-stat-file")
    print("=" * 60)

    # Load existing data to check what we already have with real SG
    ext = pd.read_parquet(PROCESSED / "historical_rounds_extended.parquet")

    all_rows = []
    events_scraped = {}
    events_failed = []

    for year in YEARS_NEEDED:
        print(f"\n  {'─'*50}")
        print(f"  YEAR: {year}")
        print(f"  {'─'*50}")

        for t_num in PGA_TNUMS:
            rows, err = fetch_event(t_num, year)

            if rows is None:
                if err != "empty leaderboard":
                    events_failed.append((t_num, year, err))
                continue

            ename = rows[0]["event_name"] if rows else f"t_num_{t_num}"
            has_ott = sum(1 for r in rows if r["sg_ott"] is not None)

            all_rows.extend(rows)
            events_scraped[(t_num, year)] = {"name": ename, "players": len(rows), "has_ott": has_ott}
            print(f"    t_num={t_num:>2} {ename:<40} {len(rows):>3} players, sg_ott={has_ott:>3}/{len(rows)}")

            time.sleep(1.2)

    print(f"\n  Total scraped: {len(all_rows):,} rows from {len(events_scraped)} events")
    if events_failed:
        print(f"  Failed: {len(events_failed)}")
        for t, y, e in events_failed:
            print(f"    t_num={t} year={y}: {e}")

    if not all_rows:
        print("  No data scraped!")
        return

    scraped_df = pd.DataFrame(all_rows)
    scraped_df.to_parquet(RAW / "scraped_real_sg_2023_2024.parquet", index=False)

    # Verify SG coverage
    print(f"\n  SG COVERAGE:")
    for col in ["sg_ott","sg_app","sg_arg","sg_putt","sg_t2g","sg_total"]:
        non_null = scraped_df[col].notna().sum()
        print(f"    {col}: {non_null}/{len(scraped_df)} ({non_null/len(scraped_df):.0%})")

    # Now replace the bad imputed rows in the extended dataset
    print(f"\n  REPLACING IMPUTED ROWS IN EXTENDED DATASET...")
    print(f"  Before: {len(ext):,} rows")

    # Drop existing 2023-2024 PGA rows (which have imputed SG)
    # Keep LIV rows (those are fine with sg_total only for now)
    mask_drop = (
        ext["season"].isin(YEARS_NEEDED) &
        ~ext["event_name"].str.contains("LIV", case=False, na=False)
    )
    print(f"  Dropping {mask_drop.sum():,} imputed 2023-2024 PGA rows")
    ext_clean = ext[~mask_drop].copy()

    # Add real scraped data
    combined = pd.concat([ext_clean, scraped_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["player_name","season","event_name"], keep="last")

    # Clean any remaining -9999 sentinels
    for c in ["sg_ott","sg_app","sg_arg","sg_putt","sg_t2g","sg_total"]:
        combined.loc[combined[c] == -9999, c] = np.nan

    combined.to_parquet(PROCESSED / "historical_rounds_extended.parquet", index=False)
    print(f"  After: {len(combined):,} rows")

    # Verify fix
    print(f"\n  SG BREAKDOWN VERIFICATION:")
    for yr in sorted(combined["season"].unique()):
        yrdf = combined[combined["season"]==yr]
        has_ott = yrdf["sg_ott"].notna().sum()
        has_total = yrdf["sg_total"].notna().sum()
        total = len(yrdf)
        print(f"    {yr}: {total:>5} rows | sg_ott: {has_ott:>5} ({has_ott/total:.0%}) | sg_total: {has_total:>5} ({has_total/total:.0%})")

    # Key player check
    print(f"\n  KEY PLAYERS (last 3 events):")
    for pn in ["Ludvig Aberg","Scottie Scheffler","Jon Rahm","Rory McIlroy","Collin Morikawa"]:
        pt = combined[combined["player_name"]==pn].tail(3)
        if len(pt)==0: continue
        print(f"    {pn}:")
        for _,r in pt.iterrows():
            ott = f"{r['sg_ott']:.3f}" if pd.notna(r['sg_ott']) else "NULL"
            tot = f"{r['sg_total']:.3f}" if pd.notna(r['sg_total']) else "NULL"
            print(f"      {r['season']} {r['event_name'][:30]:<30} ott={ott:>7} total={tot:>7}")

    print(f"\n{'='*60}")
    print("  DONE — run: python3 run_final_v7.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
