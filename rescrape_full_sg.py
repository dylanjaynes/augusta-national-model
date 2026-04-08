"""
rescrape_full_sg.py — Fix the sg_ott/app/arg/putt zeroing bug.

The /past-results/ pages only have sg_total (no breakdown).
The /historical-tournament-stats pages have full breakdown but
only serve the latest year per event (year-switching is client-side JS).

Strategy: For 2023/2024 rows where we have sg_total but null sg_ott,
impute the category breakdown using each player's known category
proportions from years where we DO have full data (2015-2022, 2025-2026).

This is more accurate than zeroing out 12 of 16 features.
"""
import os, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import requests
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

DATA_DIR = Path("data/processed")
API_KEY = os.getenv("DATAGOLF_API_KEY")


def main():
    print("=" * 60)
    print("FIX SG CATEGORY BREAKDOWN — IMPUTATION")
    print("=" * 60)

    ext = pd.read_parquet(DATA_DIR / "historical_rounds_extended.parquet")
    print(f"\nDataset: {len(ext):,} rows")

    # Identify problem rows: sg_total present, sg_ott null
    bad = ext["sg_total"].notna() & ext["sg_ott"].isna()
    print(f"Rows with sg_total but no sg_ott: {bad.sum():,} ({bad.mean():.0%})")
    print(f"By season:")
    for yr in sorted(ext[bad]["season"].unique()):
        n = (bad & (ext["season"]==yr)).sum()
        print(f"  {yr}: {n:,}")

    # Rows WITH full breakdown (our reference)
    good = ext["sg_ott"].notna() & ext["sg_total"].notna() & (ext["sg_total"].abs() > 0.01)
    good_df = ext[good].copy()
    print(f"\nRows with full breakdown (reference): {good.sum():,}")

    # Step 1: Compute per-player SG category proportions from reference data
    # For each player, what fraction of their sg_total is ott/app/arg/putt?
    print("\nComputing per-player SG category proportions...")

    proportions = {}
    for player, grp in good_df.groupby("player_name"):
        # Use absolute values for proportion calculation to handle negative SG
        vals = {
            "sg_ott": grp["sg_ott"].mean(),
            "sg_app": grp["sg_app"].mean(),
            "sg_arg": grp["sg_arg"].mean(),
            "sg_putt": grp["sg_putt"].mean(),
        }
        total = grp["sg_total"].mean()
        # t2g is ott + app + arg
        vals["sg_t2g"] = vals["sg_ott"] + vals["sg_app"] + vals["sg_arg"]

        if abs(total) > 0.01:
            proportions[player] = {k: v / total for k, v in vals.items()}
        else:
            proportions[player] = {"sg_ott": 0.25, "sg_app": 0.25, "sg_arg": 0.25, "sg_putt": 0.25, "sg_t2g": 0.75}

    print(f"  Computed proportions for {len(proportions)} players")

    # Step 2: Also get DG current skill ratings as fallback for players with no reference data
    print("  Fetching DG skill ratings as fallback...")
    sg_ratings = {}
    try:
        r = requests.get("https://feeds.datagolf.com/preds/skill-ratings",
                         params={"key": API_KEY, "file_format": "json"}, timeout=15)
        if r.status_code == 200:
            for p in r.json().get("players", []):
                name = p.get("player_name", "")
                if "," in name:
                    parts = name.split(",", 1)
                    name = f"{parts[1].strip()} {parts[0].strip()}"
                total = p.get("sg_total", 0) or 0
                if abs(total) > 0.01:
                    sg_ratings[name] = {
                        "sg_ott": (p.get("sg_ott") or 0) / total,
                        "sg_app": (p.get("sg_app") or 0) / total,
                        "sg_arg": (p.get("sg_arg") or 0) / total,
                        "sg_putt": (p.get("sg_putt") or 0) / total,
                        "sg_t2g": ((p.get("sg_ott") or 0) + (p.get("sg_app") or 0) + (p.get("sg_arg") or 0)) / total,
                    }
            print(f"  DG skill ratings: {len(sg_ratings)} players")
    except Exception as e:
        print(f"  DG skill ratings failed: {e}")

    # Tour-average proportions as final fallback
    tour_avg = {
        "sg_ott": good_df["sg_ott"].mean() / good_df["sg_total"].mean() if abs(good_df["sg_total"].mean()) > 0.01 else 0.25,
        "sg_app": good_df["sg_app"].mean() / good_df["sg_total"].mean() if abs(good_df["sg_total"].mean()) > 0.01 else 0.25,
        "sg_arg": good_df["sg_arg"].mean() / good_df["sg_total"].mean() if abs(good_df["sg_total"].mean()) > 0.01 else 0.25,
        "sg_putt": good_df["sg_putt"].mean() / good_df["sg_total"].mean() if abs(good_df["sg_total"].mean()) > 0.01 else 0.25,
    }
    tour_avg["sg_t2g"] = tour_avg["sg_ott"] + tour_avg["sg_app"] + tour_avg["sg_arg"]
    print(f"  Tour average proportions: ott={tour_avg['sg_ott']:.2f} app={tour_avg['sg_app']:.2f} arg={tour_avg['sg_arg']:.2f} putt={tour_avg['sg_putt']:.2f}")

    # Step 3: Impute
    print(f"\nImputing {bad.sum():,} rows...")
    imputed = 0
    used_player_props = 0
    used_dg_ratings = 0
    used_tour_avg = 0

    for idx in ext.index[bad]:
        player = ext.at[idx, "player_name"]
        sg_total = ext.at[idx, "sg_total"]

        if player in proportions:
            props = proportions[player]
            used_player_props += 1
        elif player in sg_ratings:
            props = sg_ratings[player]
            used_dg_ratings += 1
        else:
            props = tour_avg
            used_tour_avg += 1

        ext.at[idx, "sg_ott"] = sg_total * props["sg_ott"]
        ext.at[idx, "sg_app"] = sg_total * props["sg_app"]
        ext.at[idx, "sg_arg"] = sg_total * props["sg_arg"]
        ext.at[idx, "sg_putt"] = sg_total * props["sg_putt"]
        ext.at[idx, "sg_t2g"] = sg_total * props["sg_t2g"]
        imputed += 1

    print(f"  Imputed: {imputed:,}")
    print(f"  Using player's own proportions: {used_player_props:,}")
    print(f"  Using DG skill ratings: {used_dg_ratings:,}")
    print(f"  Using tour average: {used_tour_avg:,}")

    # Verify
    remaining_bad = ext["sg_total"].notna() & ext["sg_ott"].isna()
    print(f"\n  Remaining null sg_ott with non-null sg_total: {remaining_bad.sum()}")

    # Save
    ext.to_parquet(DATA_DIR / "historical_rounds_extended.parquet", index=False)
    print(f"  Saved: {len(ext):,} rows")

    # Spot check
    print(f"\n{'='*60}")
    print("SPOT CHECK — SG breakdown availability")
    print(f"{'='*60}")
    for yr in sorted(ext["season"].unique()):
        yrdf = ext[ext["season"]==yr]
        has_ott = yrdf["sg_ott"].notna().sum()
        has_total = yrdf["sg_total"].notna().sum()
        total = len(yrdf)
        print(f"  {yr}: {total:>5} rows | sg_ott: {has_ott:>5} ({has_ott/total:.0%}) | sg_total: {has_total:>5}")

    print(f"\nKey players:")
    for pn in ["Ludvig Aberg","Rory McIlroy","Collin Morikawa","Jon Rahm","Scottie Scheffler","Cameron Young","Viktor Hovland"]:
        pt = ext[ext["player_name"]==pn].tail(5)
        if len(pt)==0: continue
        has_ott = pt["sg_ott"].notna().sum()
        print(f"  {pn:<25} last 5 events: sg_ott non-null={has_ott}/5")
        for _, r in pt.iterrows():
            ott = f"{r['sg_ott']:.3f}" if pd.notna(r['sg_ott']) else "NULL"
            tot = f"{r['sg_total']:.3f}" if pd.notna(r['sg_total']) else "NULL"
            print(f"    {r['season']} {r['event_name'][:30]:<30} ott={ott:>7} total={tot:>7}")

    print(f"\n{'='*60}")
    print("DONE — run: python3 run_field_strength.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
