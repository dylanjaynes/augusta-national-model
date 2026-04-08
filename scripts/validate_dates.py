"""
Validate dates in historical_rounds_extended.parquet.

Checks:
1. Date coverage by season (% non-null, % non-empty)
2. For every player, sort_values('date') produces monotonic season ordering
3. Flags players where date-sorted season order is NOT monotonic
4. Reports overall stats

Run AFTER Session B's date fix to catch errors:
    python3 scripts/validate_dates.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "historical_rounds_extended.parquet"


def load_data():
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df):,} rows from {DATA_PATH.name}")
    return df


def check_date_coverage(df):
    """Report date coverage by season."""
    print("\n" + "=" * 60)
    print("DATE COVERAGE BY SEASON")
    print("=" * 60)

    # Coerce dates
    df["_date"] = pd.to_datetime(df["date"], errors="coerce")

    total_missing = 0
    total_rows = 0
    for season in sorted(df["season"].unique()):
        subset = df[df["season"] == season]
        n = len(subset)
        total_rows += n

        null_count = subset["_date"].isna().sum()
        empty_str = 0
        if df["date"].dtype == object:
            empty_str = ((subset["date"].isna()) | (subset["date"] == "") | (subset["date"] == "None")).sum()

        valid = n - null_count
        pct = valid / n * 100 if n > 0 else 0
        total_missing += null_count

        status = "OK" if null_count == 0 else "MISSING"
        print(f"  {season}: {valid:>5}/{n:>5} ({pct:5.1f}%) valid dates  [{status}]"
              + (f" — {null_count} missing" if null_count > 0 else ""))

    print(f"\n  TOTAL: {total_rows - total_missing:,}/{total_rows:,} "
          f"({(total_rows - total_missing) / total_rows * 100:.1f}%) valid dates")
    if total_missing > 0:
        print(f"  *** {total_missing:,} rows with missing dates ***")
    else:
        print("  ALL DATES PRESENT")

    return total_missing


def check_monotonic_seasons(df):
    """For each player, verify date-sorted rows have monotonically non-decreasing seasons.

    PGA Tour uses wrap-around seasons (fall events in Oct/Nov count as next season).
    A season going from S+1 → S where dates are in Oct-Dec is EXPECTED. We only flag
    jumps that aren't explainable by wrap-around (e.g., season 2024 → 2022).
    """
    print("\n" + "=" * 60)
    print("MONOTONIC SEASON CHECK (date sort should preserve season order)")
    print("=" * 60)

    df["_date"] = pd.to_datetime(df["date"], errors="coerce")

    valid = df.dropna(subset=["_date"]).copy()
    if len(valid) == 0:
        print("  No valid dates to check!")
        return []

    real_violations = []
    wraparound_count = 0
    players = valid["player_name"].unique()

    for player in players:
        pdata = valid[valid["player_name"] == player].sort_values("_date")
        seasons = pdata["season"].values
        dates = pdata["_date"].values

        if len(seasons) < 2:
            continue

        diffs = np.diff(seasons)
        bad_idx = np.where(diffs < 0)[0]

        real_bad = []
        for i in bad_idx:
            # Wrap-around: season goes S+1 → S where the S+1 event is in Oct-Dec
            month = pd.Timestamp(dates[i]).month
            jump = int(seasons[i] - seasons[i + 1])
            if jump == 1 and month >= 9:
                wraparound_count += 1
            else:
                real_bad.append(i)

        if real_bad:
            i = real_bad[0]
            real_violations.append({
                "player": player,
                "n_violations": len(real_bad),
                "example": f"season {seasons[i]}→{seasons[i+1]} "
                           f"(dates {pd.Timestamp(dates[i]).date()}→{pd.Timestamp(dates[i+1]).date()})",
            })

    print(f"\n  Wrap-around season overlaps (expected, PGA fall events): {wraparound_count}")

    if real_violations:
        print(f"  REAL VIOLATIONS: {len(real_violations)} players have unexpected season jumps")
        print()
        for v in sorted(real_violations, key=lambda x: -x["n_violations"])[:20]:
            print(f"    {v['player']}: {v['n_violations']} backward jump(s) — {v['example']}")
        if len(real_violations) > 20:
            print(f"    ... and {len(real_violations) - 20} more")
    else:
        print(f"  ALL {len(players):,} players have valid season ordering (after accounting for wrap-around)")

    return real_violations


def check_date_range_sanity(df):
    """Check for dates outside expected range (2014-2027)."""
    print("\n" + "=" * 60)
    print("DATE RANGE SANITY CHECK")
    print("=" * 60)

    df["_date"] = pd.to_datetime(df["date"], errors="coerce")
    valid = df.dropna(subset=["_date"])

    if len(valid) == 0:
        print("  No valid dates to check!")
        return

    min_date = valid["_date"].min()
    max_date = valid["_date"].max()
    print(f"  Date range: {min_date.date()} to {max_date.date()}")

    # Flag anything before 2014 or after 2027
    early = valid[valid["_date"] < pd.Timestamp("2014-01-01")]
    late = valid[valid["_date"] > pd.Timestamp("2027-12-31")]

    if len(early) > 0:
        print(f"  WARNING: {len(early)} rows before 2014")
        print(f"    Earliest: {early['_date'].min().date()} — {early.iloc[0].get('event_name', '?')}")
    if len(late) > 0:
        print(f"  WARNING: {len(late)} rows after 2027")

    if len(early) == 0 and len(late) == 0:
        print("  All dates within expected range (2014-2027)")


def check_within_season_ordering(df):
    """Check that dates within a season are plausible (Jan-Dec of that year or Dec of prior year)."""
    print("\n" + "=" * 60)
    print("WITHIN-SEASON DATE CONSISTENCY")
    print("=" * 60)

    df["_date"] = pd.to_datetime(df["date"], errors="coerce")
    valid = df.dropna(subset=["_date"]).copy()

    if len(valid) == 0:
        print("  No valid dates to check!")
        return

    # PGA Tour seasons can start in Sept/Oct of prior year (fall events)
    # So season 2024 events can have dates in late 2023
    mismatches = 0
    for season in sorted(valid["season"].unique()):
        subset = valid[valid["season"] == season]
        date_years = subset["_date"].dt.year
        # Allow season year or season-1 year (fall events)
        bad = subset[(date_years != season) & (date_years != season - 1)]
        if len(bad) > 0:
            mismatches += len(bad)
            events = bad["event_name"].unique()[:3]
            print(f"  Season {season}: {len(bad)} rows with date year ≠ {season} or {season-1}")
            for e in events:
                row = bad[bad["event_name"] == e].iloc[0]
                print(f"    {e}: date={row['_date'].date()}")

    if mismatches == 0:
        print("  All dates consistent with their season (year or year-1 for fall events)")
    else:
        print(f"\n  TOTAL: {mismatches} rows with date/season mismatch")


def check_date_clustering(df):
    """Flag seasons where many events share the same date (likely placeholder dates)."""
    print("\n" + "=" * 60)
    print("DATE CLUSTERING CHECK (placeholder date detection)")
    print("=" * 60)

    df["_date"] = pd.to_datetime(df["date"], errors="coerce")
    valid = df.dropna(subset=["_date"]).copy()

    for season in sorted(valid["season"].unique()):
        subset = valid[valid["season"] == season]
        # Count unique events per date
        event_dates = subset.groupby("_date")["event_name"].nunique()
        clustered = event_dates[event_dates >= 3]  # 3+ events on same date is suspicious
        if len(clustered) > 0:
            for date, n_events in clustered.items():
                events = subset[subset["_date"] == date]["event_name"].unique()
                print(f"  Season {season}, {date.date()}: {n_events} different events on same date")
                for e in events[:5]:
                    print(f"    - {e}")
                if len(events) > 5:
                    print(f"    ... and {len(events) - 5} more")


def check_duplicate_events(df):
    """Flag potential duplicate rows (same player + same event appearing twice)."""
    print("\n" + "=" * 60)
    print("DUPLICATE EVENT CHECK")
    print("=" * 60)

    # Look for same player appearing in very similar event names on same date
    df["_date"] = pd.to_datetime(df["date"], errors="coerce")
    dupes = df.groupby(["player_name", "_date", "season"]).size()
    multi = dupes[dupes > 1]

    if len(multi) > 0:
        n_affected = len(multi)
        total_extra = multi.sum() - len(multi)
        print(f"  {n_affected} player+date combos with multiple rows ({total_extra} extra rows)")
        # Show examples
        for (player, date, season), count in multi.head(10).items():
            rows = df[(df["player_name"] == player) & (df["_date"] == date) & (df["season"] == season)]
            events = rows["event_name"].unique()
            print(f"    {player} on {date.date()} (S{season}): {count} rows — {list(events)}")
    else:
        print("  No duplicate player+date rows found")


def spot_check_players(df, players=None):
    """Spot-check specific high-profile players."""
    print("\n" + "=" * 60)
    print("SPOT CHECK — KEY PLAYERS")
    print("=" * 60)

    df["_date"] = pd.to_datetime(df["date"], errors="coerce")

    if players is None:
        players = ["Rory McIlroy", "Scottie Scheffler", "Xander Schauffele",
                    "Jon Rahm", "Ludvig Aberg", "Brooks Koepka"]

    for player in players:
        pdata = df[df["player_name"] == player].copy()
        if len(pdata) == 0:
            # Try partial match
            matches = df[df["player_name"].str.contains(player.split()[-1], case=False, na=False)]
            if len(matches) > 0:
                player = matches["player_name"].iloc[0]
                pdata = df[df["player_name"] == player].copy()

        if len(pdata) == 0:
            print(f"\n  {player}: NOT FOUND")
            continue

        valid_dates = pdata["_date"].notna().sum()
        seasons = sorted(pdata["season"].unique())
        print(f"\n  {player}: {len(pdata)} rows, {valid_dates}/{len(pdata)} with dates, "
              f"seasons {seasons[0]}-{seasons[-1]}")

        # Show last 5 events by date
        dated = pdata.dropna(subset=["_date"]).sort_values("_date").tail(5)
        if len(dated) > 0:
            print("    Last 5 events:")
            for _, row in dated.iterrows():
                event = row.get("event_name", "?")
                sg = row.get("sg_total", "?")
                print(f"      {row['_date'].date()} | S{row['season']} | {event} | sg_total={sg}")


def main():
    df = load_data()

    missing = check_date_coverage(df)
    violations = check_monotonic_seasons(df)
    check_date_range_sanity(df)
    check_within_season_ordering(df)
    check_date_clustering(df)
    check_duplicate_events(df)
    spot_check_players(df)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if missing == 0 and len(violations) == 0:
        print("  ALL CHECKS PASSED — dates are clean and monotonic")
    else:
        if missing > 0:
            print(f"  FAIL: {missing:,} rows missing dates")
        if len(violations) > 0:
            print(f"  FAIL: {len(violations)} players with non-monotonic seasons")
        print("  Data is NOT ready for rolling feature computation")


if __name__ == "__main__":
    main()
