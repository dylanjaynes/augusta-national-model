#!/usr/bin/env python3
"""
Fix placeholder dates for Arnold Palmer Invitational and Memorial Tournament.

Both events consistently have YYYY-06-15 placeholder dates in the parquet,
when they actually occur in early March and late May respectively.

Arnold Palmer Invitational (held early March, before Masters):
  - 2019: Mar 7  | 2020: Mar 5  | 2021: Mar 4  | 2022: Mar 3
  - 2023: Mar 2  | 2024: Mar 7  | 2025: Mar 6  | 2026: Mar 6

Memorial Tournament presented by Workday (held late May/early June):
  - 2019: May 30 | 2020: (not played) | 2021: Jun 3 | 2022: Jun 2
  - 2023: Jun 1  | 2024: May 30       | 2025: May 29| 2026: May 28
"""

import os
import pandas as pd
from pathlib import Path

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PARQUET = Path("data/processed/historical_rounds_extended.parquet")

ARNOLD_PALMER_DATES = {
    2019: "2019-03-07",
    2020: "2020-03-05",
    2021: "2021-03-04",
    2022: "2022-03-03",
    2023: "2023-03-02",
    2024: "2024-03-07",
    2025: "2025-03-06",
    2026: "2026-03-06",
}

MEMORIAL_DATES = {
    2019: "2019-05-30",
    # 2020: not played (COVID)
    2021: "2021-06-03",
    2022: "2022-06-02",
    2023: "2023-06-01",
    2024: "2024-05-30",
    2025: "2025-05-29",
    2026: "2026-05-28",
}

def fix_event_dates(df, event_name_fragment, date_map, description):
    mask = df["event_name"].str.contains(event_name_fragment, case=False, na=False)
    total_fixed = 0
    for season, real_date in date_map.items():
        event_mask = mask & (df["season"] == season)
        count = event_mask.sum()
        if count > 0:
            df.loc[event_mask, "date"] = real_date
            total_fixed += count
    print(f"  {description}: fixed {total_fixed} rows")
    return df


def main():
    print(f"Loading {PARQUET}...")
    df = pd.read_parquet(PARQUET)
    print(f"  {len(df)} rows loaded")

    # Check before state
    api_jun15 = df[df["event_name"].str.contains("Arnold Palmer", case=False, na=False)
                   & df["date"].str.endswith("-06-15")]
    mem_jun15 = df[df["event_name"].str.contains("Memorial", case=False, na=False)
                   & df["date"].str.endswith("-06-15")]
    print(f"\nBefore: Arnold Palmer Jun-15 rows: {len(api_jun15)}")
    print(f"Before: Memorial Jun-15 rows: {len(mem_jun15)}")

    print("\nApplying fixes...")
    df = fix_event_dates(df, "Arnold Palmer", ARNOLD_PALMER_DATES,
                         "Arnold Palmer Invitational")
    df = fix_event_dates(df, "Memorial Tournament", MEMORIAL_DATES,
                         "Memorial Tournament")

    # Verify after state
    api_after = df[df["event_name"].str.contains("Arnold Palmer", case=False, na=False)
                   & df["date"].str.endswith("-06-15")]
    mem_after = df[df["event_name"].str.contains("Memorial", case=False, na=False)
                   & df["date"].str.endswith("-06-15")]
    print(f"\nAfter: Arnold Palmer Jun-15 rows: {len(api_after)}")
    print(f"After: Memorial Jun-15 rows: {len(mem_after)}")

    # Show corrected dates
    print("\nArnold Palmer dates after fix (by season):")
    api_fixed = df[df["event_name"].str.contains("Arnold Palmer", case=False, na=False)]
    print(api_fixed.groupby("season")["date"].first().to_string())

    print("\nSaving updated parquet...")
    df.to_parquet(PARQUET, index=False)
    print(f"Saved: {PARQUET}")
    print("\nIMPORTANT: Run `python3 run_production.py` to rebuild features and predictions.")


if __name__ == "__main__":
    main()
