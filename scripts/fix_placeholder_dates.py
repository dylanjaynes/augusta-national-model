#!/usr/bin/env python3
"""
Fix placeholder dates in historical_rounds_extended.parquet.

Problem: ALL 2015-2023 events and a handful of 2024-2026 events use
'YYYY-06-15' as a placeholder date. This makes within-season chronological
ordering wrong (events sort alphabetically, not chronologically), which
corrupts rolling feature calculations.

Fix: Patch known events with their real start dates using a hardcoded
PGA Tour schedule. For events not in the lookup, the placeholder is left
in place (which is better than a wrong date).

Run with:
    python3 scripts/fix_placeholder_dates.py
    python3 scripts/fix_placeholder_dates.py --dry-run  # check without saving
"""
import argparse
import re
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "historical_rounds_extended.parquet"

# ─────────────────────────────────────────────────────────────────────────────
# PGA TOUR EVENT SCHEDULE
# Format: {normalized_event_name_fragment: {season: "YYYY-MM-DD", ...}}
# Using the Thursday (first round) start date for each event.
# Sources: PGA Tour schedule archives, Wikipedia, PGA.com
# ─────────────────────────────────────────────────────────────────────────────

# Normalisation helper applied to event_name before lookup
def _norm(name):
    """Lower-case, strip sponsor suffixes, collapse whitespace."""
    name = name.lower().strip()
    # remove common sponsor suffixes
    for suffix in [
        "presented by mastercard", "presented by workday", "presented by transamerica",
        "presented by", "powered by", "in partnership with", "sponsoring", "sponsored by",
        "bmo", "at&t", "the ", " championship", " tournament", " classic",
        " invitational", " open", " pro-am",
    ]:
        name = name.replace(suffix, " ")
    # collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name


# ─────────────────────────────────────────────────────────────────────────────
# Full schedule: (key_fragment, season) → start_date
# key_fragment is matched with `key_fragment in _norm(event_name)`.
# Keep fragments specific enough to avoid false matches.
# ─────────────────────────────────────────────────────────────────────────────
SCHEDULE = {
    # ── The Sentry / Tournament of Champions (Kapalua, Jan week 1) ──────────
    "sentry tournament of champions": {
        2015: "2015-01-08", 2016: "2016-01-07", 2017: "2017-01-05",
        2018: "2018-01-04", 2019: "2019-01-03", 2020: "2020-01-02",
        2021: "2021-01-07", 2022: "2022-01-06", 2023: "2023-01-05",
    },
    "the sentry": {
        2024: "2024-01-04", 2025: "2025-01-02",
    },
    # ── Sony Open in Hawaii (Honolulu, Jan week 2) ───────────────────────────
    "sony open": {
        2015: "2015-01-15", 2016: "2016-01-14", 2017: "2017-01-12",
        2018: "2018-01-11", 2019: "2019-01-10", 2020: "2020-01-09",
        2021: "2021-01-14", 2022: "2022-01-13", 2023: "2023-01-12",
        2024: "2024-01-11", 2025: "2025-01-09", 2026: "2026-01-09",
    },
    # ── The American Express (Palm Springs, late Jan) ───────────────────────
    "american express": {
        2015: "2015-01-22", 2016: "2016-01-21", 2017: "2017-01-19",
        2018: "2018-01-18", 2019: "2019-01-17", 2020: "2020-01-16",
        2021: "2021-01-21", 2022: "2022-01-20", 2023: "2023-01-19",
        2024: "2024-01-18", 2025: "2025-01-16", 2026: "2026-01-16",
    },
    # ── Farmers Insurance Open (Torrey Pines, late Jan/early Feb) ───────────
    "farmers insurance": {
        2015: "2015-01-29", 2016: "2016-01-28", 2017: "2017-01-26",
        2018: "2018-01-25", 2019: "2019-01-24", 2020: "2020-01-23",
        2021: "2021-01-28", 2022: "2022-01-27", 2023: "2023-01-26",
        2024: "2024-01-25", 2025: "2025-01-23", 2026: "2026-01-23",
    },
    # ── AT&T Pebble Beach Pro-Am (Pebble Beach, early Feb) ──────────────────
    "pebble beach": {
        2015: "2015-02-05", 2016: "2016-02-11", 2017: "2017-02-09",
        2018: "2018-02-08", 2019: "2019-02-07", 2020: "2020-02-06",
        2021: "2021-02-11", 2022: "2022-02-03", 2023: "2023-02-02",
        2024: "2024-02-01", 2025: "2025-01-30", 2026: "2026-02-06",
    },
    # ── WM Phoenix Open (TPC Scottsdale, early Feb) ──────────────────────────
    "phoenix open": {
        2015: "2015-01-29", 2016: "2016-02-04", 2017: "2017-02-02",
        2018: "2018-02-01", 2019: "2019-01-31", 2020: "2020-01-30",
        2021: "2021-02-04", 2022: "2022-02-10", 2023: "2023-02-09",
        2024: "2024-02-08", 2025: "2025-02-06", 2026: "2026-02-06",
    },
    # ── The Genesis Invitational (Riviera, mid Feb) ──────────────────────────
    "genesis": {
        2015: "2015-02-19", 2016: "2016-02-18", 2017: "2017-02-16",
        2018: "2018-02-15", 2019: "2019-02-14", 2020: "2020-02-13",
        2021: "2021-02-18", 2022: "2022-02-17", 2023: "2023-02-16",
        2024: "2024-02-15", 2025: "2025-02-13", 2026: "2026-02-19",
    },
    # ── Cognizant Classic / Honda Classic (Palm Beach, late Feb) ────────────
    "cognizant": {
        2023: "2023-02-23", 2024: "2024-02-22", 2025: "2025-02-27", 2026: "2026-02-26",
    },
    "honda classic": {
        2015: "2015-02-26", 2016: "2016-02-25", 2017: "2017-02-23",
        2018: "2018-02-22", 2019: "2019-02-21", 2020: "2020-03-05",
        2021: "2021-02-25", 2022: "2022-02-24", 2023: "2023-02-23",
    },
    # ── THE PLAYERS Championship (TPC Sawgrass) ──────────────────────────────
    # NOTE: In 2019 moved from March to May; moved back to March in 2022+
    "players championship": {
        2015: "2015-05-07", 2016: "2016-05-12", 2017: "2017-05-11",
        2018: "2018-05-10", 2019: "2019-05-09", 2020: "2020-03-12",  # cancelled after R1
        2021: "2021-03-11", 2022: "2022-03-10", 2023: "2023-03-09",
        2024: "2024-03-14", 2025: "2025-03-13", 2026: "2026-03-12",
    },
    # ── Valspar Championship (Innisbrook, mid March) ─────────────────────────
    "valspar": {
        2016: "2016-03-10", 2017: "2017-03-09", 2018: "2018-03-08",
        2019: "2019-03-21", 2021: "2021-03-18", 2022: "2022-03-17",
        2023: "2023-03-16", 2024: "2024-03-21",
    },
    # ── Arnold Palmer Invitational (Bay Hill, early March) ──────────────────
    "arnold palmer": {
        2015: "2015-03-19", 2016: "2016-03-17", 2017: "2017-03-16",
        2018: "2018-03-15", 2019: "2019-03-07", 2020: "2020-03-05",
        2021: "2021-03-04", 2022: "2022-03-03", 2023: "2023-03-02",
        2024: "2024-03-07", 2025: "2025-03-06", 2026: "2026-03-06",
    },
    # ── Texas Children's / Houston Open (Memorial Park, late March) ──────────
    "houston open": {
        2015: "2015-03-26", 2016: "2016-03-31", 2017: "2017-03-30",
        2018: "2018-03-29", 2019: "2019-03-28",
    },
    "texas children": {
        2020: "2020-11-05",  # moved to fall 2020
        2021: "2021-11-11", 2022: "2022-11-10", 2023: "2023-11-09",
        2024: "2024-03-28", 2025: "2025-03-27", 2026: "2026-03-26",
    },
    # ── Valero Texas Open (TPC San Antonio, late March/early April) ──────────
    "valero texas": {
        2015: "2015-04-02", 2016: "2016-03-24", 2017: "2017-03-23",
        2018: "2018-03-22", 2019: "2019-03-28",
        2021: "2021-04-01", 2022: "2022-03-31", 2023: "2023-03-30",
        2024: "2024-04-04", 2025: "2025-04-03", 2026: "2026-04-03",
    },
    # ── Masters Tournament (Augusta National, early April) ───────────────────
    "masters tournament": {
        2015: "2015-04-09", 2016: "2016-04-07", 2017: "2017-04-06",
        2018: "2018-04-05", 2019: "2019-04-11", 2020: "2020-11-12",
        2021: "2021-04-08", 2022: "2022-04-07", 2023: "2023-04-06",
        2024: "2024-04-11", 2025: "2025-04-10",
    },
    # ── RBC Heritage (Harbour Town, week after Masters) ──────────────────────
    "rbc heritage": {
        2015: "2015-04-16", 2016: "2016-04-14", 2017: "2017-04-13",
        2018: "2018-04-19", 2019: "2019-04-18", 2020: "2020-06-18",
        2021: "2021-04-15", 2022: "2022-04-14", 2023: "2023-04-13",
        2024: "2024-04-18", 2025: "2025-04-17",
    },
    # ── Zurich Classic (TPC Louisiana, late April) ───────────────────────────
    "zurich classic": {
        2016: "2016-04-28", 2017: "2017-04-27", 2018: "2018-04-26",
        2019: "2019-04-25", 2020: "2020-04-23", 2021: "2021-04-22",
        2022: "2022-04-28", 2023: "2023-04-27", 2024: "2024-04-25",
        2025: "2025-04-24",
    },
    # ── CJ Cup Byron Nelson / AT&T Byron Nelson (late April/early May) ───────
    "byron nelson": {
        2015: "2015-04-30", 2016: "2016-05-05", 2017: "2017-05-18",
        2018: "2018-05-17", 2019: "2019-05-09",
        2021: "2021-05-13", 2022: "2022-05-12", 2023: "2023-05-11",
        2024: "2024-05-02", 2025: "2025-05-01",
    },
    # ── PGA Championship (varies, mid May) ────────────────────────────────────
    "pga championship": {
        2015: "2015-08-13", 2016: "2016-07-28", 2017: "2017-08-10",
        2018: "2018-08-09", 2019: "2019-05-16", 2020: "2020-08-06",
        2021: "2021-05-20", 2022: "2022-05-19", 2023: "2023-05-18",
        2024: "2024-05-16", 2025: "2025-05-15",
    },
    # ── Charles Schwab Challenge (Colonial, late May) ────────────────────────
    "charles schwab": {
        2015: "2015-05-21", 2016: "2016-05-26", 2017: "2017-05-25",
        2018: "2018-05-24", 2019: "2019-05-23",
        2021: "2021-05-27", 2022: "2022-05-26", 2023: "2023-05-25",
        2024: "2024-05-23", 2025: "2025-05-22",
    },
    # ── Memorial Tournament (Muirfield Village, early June) ──────────────────
    "memorial": {
        2015: "2015-06-04", 2016: "2016-06-02", 2017: "2017-06-01",
        2018: "2018-05-31", 2019: "2019-05-30",
        2020: "2020-07-16",  # moved to July due to COVID-19
        2021: "2021-06-03", 2022: "2022-06-02", 2023: "2023-06-01",
        2024: "2024-06-06", 2025: "2025-06-05",
    },
    # ── RBC Canadian Open (varies, early June) ────────────────────────────────
    "rbc canadian": {
        2015: "2015-07-23", 2016: "2016-07-21", 2017: "2017-07-20",
        2018: "2018-07-26", 2019: "2019-06-06",
        2021: "2021-06-10", 2022: "2022-06-09", 2023: "2023-06-08",
        2024: "2024-05-30", 2025: "2025-06-05",
    },
    # ── U.S. Open (mid June) ─────────────────────────────────────────────────
    "u.s. open": {
        2015: "2015-06-18", 2016: "2016-06-16", 2017: "2017-06-15",
        2018: "2018-06-14", 2019: "2019-06-13", 2020: "2020-09-17",
        2021: "2021-06-17", 2022: "2022-06-16", 2023: "2023-06-15",
        2024: "2024-06-13", 2025: "2025-06-12",
    },
    # ── Travelers Championship (TPC River Highlands, late June) ──────────────
    "travelers": {
        2015: "2015-06-25", 2016: "2016-06-23", 2017: "2017-06-22",
        2018: "2018-06-21", 2019: "2019-06-20",
        2021: "2021-06-24", 2022: "2022-06-23", 2023: "2023-06-22",
        2024: "2024-06-20", 2025: "2025-06-19",
    },
    # ── Rocket Mortgage Classic (Detroit, early July) ─────────────────────────
    "rocket mortgage": {
        2019: "2019-07-04", 2020: "2020-07-02",
        2021: "2021-07-01", 2022: "2022-06-30", 2023: "2023-06-29",
        2024: "2024-06-27", 2025: "2025-06-26",
    },
    # ── 3M Open (TPC Twin Cities, mid July) ──────────────────────────────────
    "3m open": {
        2019: "2019-07-18", 2020: "2020-07-23",
        2021: "2021-07-22", 2022: "2022-07-21", 2023: "2023-07-20",
        2024: "2024-07-25", 2025: "2025-07-24",
    },
    # ── John Deere Classic (TPC Deere Run, early July) ────────────────────────
    "john deere": {
        2015: "2015-07-09", 2016: "2016-07-14", 2017: "2017-07-13",
        2018: "2018-07-12", 2019: "2019-07-11",
        2021: "2021-07-08", 2022: "2022-07-07", 2023: "2023-07-06",
        2024: "2024-07-04", 2025: "2025-07-03",
    },
    # ── The Open Championship (The Open/British Open, mid-late July) ──────────
    "the open championship": {
        2015: "2015-07-16", 2016: "2016-07-14", 2017: "2017-07-20",
        2018: "2018-07-19", 2019: "2019-07-18", 2020: "2020-07-16",  # cancelled
        2021: "2021-07-15", 2022: "2022-07-14", 2023: "2023-07-20",
        2024: "2024-07-18", 2025: "2025-07-17",
    },
    # ── FedEx St. Jude Championship (TPC Southwind, Aug week 1) ──────────────
    "fedex st. jude": {
        2015: "2015-06-11", 2016: "2016-06-09", 2017: "2017-06-08",
        2018: "2018-06-07", 2019: "2019-08-01",
        2020: "2020-08-06",
        2021: "2021-08-05", 2022: "2022-08-11", 2023: "2023-08-10",
        2024: "2024-08-15", 2025: "2025-08-07",
    },
    # ── BMW Championship (Aug week 2) ─────────────────────────────────────────
    "bmw championship": {
        2015: "2015-08-13", 2016: "2016-08-18", 2017: "2017-08-17",
        2018: "2018-08-23", 2019: "2019-08-15",
        2020: "2020-08-20",
        2021: "2021-08-26", 2022: "2022-08-25", 2023: "2023-08-17",
        2024: "2024-08-22", 2025: "2025-08-14",
    },
    # ── TOUR Championship (East Lake, Aug/Sep) ────────────────────────────────
    "tour championship": {
        2015: "2015-09-24", 2016: "2016-09-22", 2017: "2017-09-21",
        2018: "2018-09-20", 2019: "2019-08-22",
        2020: "2020-09-04",
        2021: "2021-09-02", 2022: "2022-08-25", 2023: "2023-08-24",
        2024: "2024-08-29", 2025: "2025-08-28",
    },
    # ── Fall events (season year = following year on PGA Tour) ────────────────
    # ── Sanderson Farms Championship (early Oct) ─────────────────────────────
    "sanderson farms": {
        2015: "2015-10-01", 2016: "2016-10-06", 2017: "2017-10-05",
        2018: "2018-09-27", 2019: "2019-09-19",
        2021: "2021-09-30", 2022: "2022-09-29", 2023: "2023-10-05",
        2024: "2024-10-03",
    },
    # ── Shriners Children's Open (TPC Summerlin, early Oct) ──────────────────
    "shriners": {
        2015: "2015-10-01", 2016: "2016-10-06", 2017: "2017-10-05",
        2018: "2018-10-04", 2019: "2019-10-03",
        2021: "2021-10-07", 2022: "2022-10-06", 2023: "2023-10-12",
        2024: "2024-10-10",
    },
    # ── CJ Cup (varies by year/location) ──────────────────────────────────────
    "cj cup": {
        2017: "2017-10-19", 2018: "2018-10-18", 2019: "2019-10-17",
        2020: "2020-10-15",
        2021: "2021-10-14", 2022: "2022-10-20", 2023: "2023-10-19",
        2024: "2024-10-17",
    },
    # ── Zozo Championship (Japan, late Oct) ───────────────────────────────────
    "zozo": {
        2019: "2019-10-24",
        2020: "2020-10-22",
        2021: "2021-10-21", 2022: "2022-10-13", 2023: "2023-10-26",
        2024: "2024-10-24",
    },
    # ── WGC events ────────────────────────────────────────────────────────────
    "hsbc champions": {
        2015: "2015-11-05", 2016: "2016-11-03", 2017: "2017-11-02",
        2018: "2018-11-01", 2019: "2019-10-31",
    },
    # ── RSM Classic (Sea Island, Nov) ─────────────────────────────────────────
    "rsm classic": {
        2015: "2015-11-12", 2016: "2016-11-17", 2017: "2017-11-16",
        2018: "2018-11-15", 2019: "2019-11-21",
        2020: "2020-11-19",
        2021: "2021-11-18", 2022: "2022-11-17", 2023: "2023-11-16",
        2024: "2024-11-21",
    },
    # ── Hero World Challenge (Albany, Bahamas, late Nov/early Dec) ───────────
    "hero world challenge": {
        2015: "2015-12-03", 2016: "2016-11-30", 2017: "2017-11-30",
        2018: "2018-11-29", 2019: "2019-12-05",
        2021: "2021-12-02", 2022: "2022-12-01", 2023: "2023-11-30",
        2024: "2024-11-28",
    },
    # ── Barbasol Championship (parallel event to The Open, July) ─────────────
    "barbasol": {
        2015: "2015-07-16", 2016: "2016-07-21", 2017: "2017-07-20",
        2018: "2018-07-19", 2019: "2019-07-18", 2020: "2020-07-09",
        2021: "2021-07-15", 2022: "2022-07-14", 2023: "2023-07-13",
        2024: "2024-07-11",
    },
    # ── Barracuda Championship ────────────────────────────────────────────────
    "barracuda": {
        2017: "2017-07-20", 2018: "2018-07-19", 2019: "2019-07-18",
        2020: "2020-07-09",
        2021: "2021-07-15", 2022: "2022-07-14", 2023: "2023-07-13",
        2024: "2024-07-11",
    },
    # ── Puerto Rico Open / Corales Puntacana — merged below ───────────────────
    # ── World Golf Championships (various) ───────────────────────────────────
    "wgc-workday": {
        2021: "2021-03-04",
    },
    "wgc-dell technologies match play": {
        2015: "2015-10-07", 2016: "2016-03-23", 2017: "2017-03-22",
        2018: "2018-03-21", 2019: "2019-03-27",
        2021: "2021-03-24", 2022: "2022-03-23", 2023: "2023-03-22",
        2024: "2024-03-20",
    },
    "wgc-fedex st. jude": {
        2019: "2019-07-25",
    },
    # ── The Northern Trust / Northern Trust (FedEx playoffs, Aug) ───────────
    "northern trust": {
        2015: "2015-08-20", 2016: "2016-08-25", 2017: "2017-08-24",
        2018: "2018-08-23",
        2019: "2019-08-08", 2020: "2020-08-20",
        2021: "2021-08-19", 2022: "2022-08-18",
    },
    # ── Fortinet Championship / Safeway Open (Silverado, fall) ───────────────
    "fortinet championship": {
        2021: "2021-09-16", 2022: "2022-09-15", 2023: "2023-09-14",
        2024: "2024-09-12",
    },
    "safeway open": {
        2019: "2019-10-10", 2020: "2020-09-10",
    },
    # ── Bermuda Championship (fall) ───────────────────────────────────────────
    "bermuda championship": {
        2019: "2019-10-31", 2020: "2020-10-29",
        2021: "2021-10-28", 2022: "2022-10-27",
    },
    # ── Butterfield Bermuda Championship ──────────────────────────────────────
    "butterfield bermuda": {
        2021: "2021-10-28", 2022: "2022-10-27",
    },
    # ── Mayakoba Golf Classic (El Camaleon, Nov) ──────────────────────────────
    "mayakoba": {
        2015: "2015-10-29", 2016: "2016-11-10", 2017: "2017-11-09",
        2018: "2018-11-08",
        2019: "2019-11-07", 2020: "2020-11-05",
        2021: "2021-11-04", 2022: "2022-11-03", 2023: "2023-11-02",
        2024: "2024-10-31",
    },
    # ── Cadence Bank Houston Open ─────────────────────────────────────────────
    "cadence bank houston": {
        2022: "2022-11-10",
    },
    "hewlett packard enterprise houston": {
        2021: "2021-11-11",
    },
    # ── Mexico Open at Vidanta ────────────────────────────────────────────────
    "mexico open": {
        2022: "2022-04-28",
    },
    # ── Puerto Rico Open ──────────────────────────────────────────────────────
    "puerto rico open": {
        2015: "2015-03-05", 2016: "2016-03-03", 2017: "2017-03-02",
        2018: "2018-03-22",
        2019: "2019-03-21", 2020: "2020-02-27",
        2021: "2021-02-25", 2022: "2022-03-03",
    },
    # ── Corales Puntacana ─────────────────────────────────────────────────────
    "corales puntacana": {
        2019: "2019-03-28", 2020: "2020-03-26",
        2021: "2021-03-25", 2022: "2022-03-24",
    },
    # ── Desert Classic (now The American Express) ─────────────────────────────
    "desert classic": {
        2018: "2018-01-18", 2019: "2019-01-17",
    },
    # ── Charles Schwab Challenge (Colonial, renamed from Crowne Plaza) ────────
    "crowne plaza invitational": {
        2015: "2015-05-21", 2016: "2016-05-26", 2017: "2017-05-25",
    },
    # ── Alternate name for Masters ─────────────────────────────────────────────
    "the masters": {
        2015: "2015-04-09", 2016: "2016-04-07", 2017: "2017-04-06",
        2018: "2018-04-05", 2019: "2019-04-11", 2020: "2020-11-12",
        2021: "2021-04-08", 2022: "2022-04-07", 2023: "2023-04-06",
        2024: "2024-04-11", 2025: "2025-04-10",
    },
    # ── Wyndham Championship (Sedgefield, Aug) ────────────────────────────────
    "wyndham": {
        2015: "2015-08-13", 2016: "2016-08-18", 2017: "2017-08-17",
        2018: "2018-08-16", 2019: "2019-08-15",
        2020: "2020-08-13",
        2021: "2021-08-12", 2022: "2022-08-04", 2023: "2023-08-03",
        2024: "2024-08-08",
    },
    # ── Workday Charity Open ──────────────────────────────────────────────────
    "workday charity": {
        2020: "2020-07-09",
    },
    # ── Palmetto Championship / South Carolina events ─────────────────────────
    "palmetto championship": {
        2021: "2021-06-10",
    },
    # ── Wells Fargo Championship (Quail Hollow, May) ──────────────────────────
    "wells fargo": {
        2015: "2015-05-07", 2016: "2016-05-05", 2017: "2017-05-04",
        2018: "2018-05-03", 2019: "2019-05-02",
        2021: "2021-05-06", 2022: "2022-05-05", 2023: "2023-05-04",
        2024: "2024-05-09", 2025: "2025-05-08",
    },
    # ── WGC-Mexico Championship ───────────────────────────────────────────────
    "wgc-mexico championship": {
        2019: "2019-02-21",
    },
    "world golf championships-mexico": {
        2019: "2019-02-21", 2020: "2020-02-20",
    },
    # ── WGC-Workday Championship ──────────────────────────────────────────────
    "wgc-workday championship": {
        2021: "2021-03-04",
    },
    "world golf championships-workday": {
        2021: "2021-03-04",
    },
    # ── Honda Classic already defined above ───────────────────────────────────
    # ── 2020 COVID rescheduled events ─────────────────────────────────────────
    "charles schwab challenge": {
        2020: "2020-06-11",
    },
    "colonial country club": {  # alternate name fallback
        2020: "2020-06-11",
    },
    "sanderson farms championship": {
        2020: "2020-09-17",
    },
    "shriners hospitals for children open": {
        2020: "2020-10-08",
    },
    "shriners children": {
        2020: "2020-10-08",
    },
    # honda classic 2020 already merged above
    "vivint houston open": {
        2020: "2020-11-05",
    },
    "travelers championship": {
        2020: "2020-06-25",
    },
    # ── Rocket Mortgage Classic (Detroit, July) ───────────────────────────────
    "rocket mortgage classic": {
        2019: "2019-07-04", 2020: "2020-07-02",
        2021: "2021-07-01", 2022: "2022-06-30", 2023: "2023-06-29",
        2024: "2024-06-27", 2025: "2025-06-26",
    },
    # ── Hewlett Packard / Cadence Bank alternate names ────────────────────────
    "hp byron nelson": {
        2015: "2015-05-21", 2016: "2016-04-28",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# EXPLICIT OVERRIDES for tricky cases (takes priority over SCHEDULE lookup)
# ─────────────────────────────────────────────────────────────────────────────
EXPLICIT_OVERRIDES = {
    # (event_name_exact or substring, season): "YYYY-MM-DD"
    # 2020 Masters moved to November due to COVID
    ("masters tournament", 2020): "2020-11-12",
    # 2020 US Open moved to September
    ("u.s. open", 2020): "2020-09-17",
    # 2020 PGA Championship moved to August
    ("pga championship", 2020): "2020-08-06",
    # Players 2020 cancelled after round 1
    ("the players championship", 2020): "2020-03-12",
    ("players championship", 2020): "2020-03-12",
}


def get_real_date(event_name: str, season: int):
    """Return real date string for an event, or None if unknown."""
    en = event_name.lower().strip() if event_name else ""

    # 1. Check explicit overrides first
    for (key, s), date in EXPLICIT_OVERRIDES.items():
        if s == season and key in en:
            return date

    # 2. Check SCHEDULE via key fragment matching
    for fragment, season_map in SCHEDULE.items():
        if fragment in en and season in season_map:
            return season_map[season]

    return None


def fix_dates(dry_run: bool = False):
    print(f"Loading {DATA_PATH.name}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"  {len(df):,} rows, {df['date'].dtype} date column")

    # Parse existing dates
    df["_parsed"] = pd.to_datetime(df["date"], errors="coerce")

    # Identify placeholder rows: June 15 of any year
    is_placeholder = (df["_parsed"].dt.month == 6) & (df["_parsed"].dt.day == 15)
    n_placeholder = is_placeholder.sum()
    print(f"  Placeholder dates (June 15): {n_placeholder:,} rows ({n_placeholder/len(df)*100:.1f}%)")

    # Build fix map
    fixes = {}       # index → new_date_str
    no_fix = set()   # event_names we couldn't resolve (for reporting)

    placeholder_df = df[is_placeholder].copy()
    for idx, row in placeholder_df.iterrows():
        real = get_real_date(row["event_name"], row["season"])
        if real:
            fixes[idx] = real
        else:
            no_fix.add(f"{row['event_name']} (S{row['season']})")

    print(f"\n  Fixed: {len(fixes):,} rows")
    print(f"  No lookup found: {len(no_fix)} distinct event+season combos")
    if no_fix:
        print("  Could not fix:")
        for ev in sorted(no_fix)[:30]:
            print(f"    {ev}")
        if len(no_fix) > 30:
            print(f"    ... and {len(no_fix) - 30} more")

    if dry_run:
        print("\n  DRY RUN — no changes saved.")
        return

    # Apply fixes
    for idx, new_date in fixes.items():
        df.at[idx, "date"] = new_date

    # Drop temp column
    df = df.drop(columns=["_parsed"])

    # Verify no placeholder dates remain for recent years (2024-2026)
    df["_check"] = pd.to_datetime(df["date"], errors="coerce")
    remaining = df[(df["_check"].dt.month == 6) & (df["_check"].dt.day == 15) &
                   (df["season"] >= 2024)]
    if len(remaining) > 0:
        print(f"\n  WARNING: {len(remaining)} rows in 2024-2026 still have placeholder dates:")
        print(remaining.groupby(["season", "event_name"]).size())

    df = df.drop(columns=["_check"])

    # Save
    df.to_parquet(DATA_PATH, index=False)
    print(f"\n  Saved {len(df):,} rows to {DATA_PATH.name}")

    # Final stats
    df["_final"] = pd.to_datetime(df["date"], errors="coerce")
    remaining_all = ((df["_final"].dt.month == 6) & (df["_final"].dt.day == 15)).sum()
    print(f"  Remaining June-15 placeholders: {remaining_all:,} rows ({remaining_all/len(df)*100:.1f}%)")
    df = df.drop(columns=["_final"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print fixes without saving")
    args = parser.parse_args()
    fix_dates(dry_run=args.dry_run)
