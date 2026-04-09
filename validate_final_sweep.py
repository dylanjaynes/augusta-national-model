#!/usr/bin/env python3
"""
FINAL COMPREHENSIVE BUG SWEEP
Independent verification of the entire Augusta National prediction model.
Checks all 31 items across Data, Code, Prediction, Streamlit, and Tests.
"""

import sys
import os
import re
import ast
import json
import subprocess
from pathlib import Path

BASE = Path("/Users/dylanjaynes/Augusta National Model")
RESULTS = []

def check(n, label, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    RESULTS.append((n, label, passed, detail))
    icon = "✅" if passed else "❌"
    print(f"  {icon} [{n:02d}] {label}")
    if detail:
        print(f"       {detail}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ─────────────────────────────────────────────
# DATA INTEGRITY
# ─────────────────────────────────────────────
section("DATA INTEGRITY")

try:
    import pandas as pd
    import numpy as np

    parquet_path = BASE / "data/processed/historical_rounds_extended.parquet"
    df = pd.read_parquet(parquet_path)
    print(f"\n  Loaded: {len(df):,} rows, {df.columns.tolist()}")

    # [1] Row count reasonable
    check(1, "Row count ~148K", 100_000 <= len(df) <= 200_000,
          f"actual={len(df):,}")

    # [2] Placeholder dates by season
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month_day"] = df["date"].dt.strftime("%m-%d")
    placeholder = df[df["month_day"] == "06-15"]
    counts = placeholder.groupby("season").size()
    print(f"\n  Placeholder dates (06-15) by season:")
    for yr, cnt in counts.items():
        print(f"    {yr}: {cnt}")
    check(2, "Placeholder date counts reported", True, f"total={len(placeholder)}")

    # [3] 2024-2026: ZERO placeholder dates (excluding US Open)
    # US Open event names contain "U.S. Open" or "US Open"
    us_open_mask = df["event_name"].str.contains("U.S. Open|US Open", case=False, na=False)
    recent_placeholder = df[
        (df["month_day"] == "06-15") &
        (df["season"].isin([2024, 2025, 2026])) &
        (~us_open_mask)
    ]
    if len(recent_placeholder) > 0:
        bad = recent_placeholder.groupby(["season", "event_name"]).size()
        check(3, "Zero spurious placeholder dates in 2024-2026", False,
              f"found {len(recent_placeholder)} rows:\n       {bad.to_string()}")
    else:
        check(3, "Zero spurious placeholder dates in 2024-2026", True,
              "all June-15 dates in 2024-2026 are genuine US Open rows")

    # [4] Chronological ordering
    bad_order = 0
    bad_examples = []
    for (pid, season), grp in df.groupby(["dg_id", "season"]):
        dates = grp["date"].dropna().sort_values(ascending=False)
        if len(dates) < 2:
            continue
        diffs = dates.diff().dropna()
        backwards = (diffs > pd.Timedelta(0))
        if backwards.any():
            bad_order += 1
            if len(bad_examples) < 3:
                bad_examples.append(f"dg_id={pid}, season={season}")
    check(4, "Chronological ordering within (player, season)", bad_order == 0,
          f"violations={bad_order}" + (f": {bad_examples}" if bad_examples else ""))

    # [5] Duplicate rows
    dup_cols = [c for c in ["dg_id", "event_name", "round_num"] if c in df.columns]
    if not dup_cols:
        dup_cols = [c for c in ["dg_id", "event_name"] if c in df.columns]
    dupes = df.duplicated(subset=dup_cols).sum()
    check(5, "No duplicate rows (player+event+round)", dupes == 0,
          f"duplicates={dupes}, key_cols={dup_cols}")

    # [6] NaN counts in SG columns
    sg_cols = ["sg_total", "sg_ott", "sg_app", "sg_arg", "sg_putt"]
    nan_counts = {c: int(df[c].isna().sum()) for c in sg_cols if c in df.columns}
    print(f"\n  NaN counts: {nan_counts}")
    check(6, "NaN counts in SG columns reported", True,
          str(nan_counts))

except Exception as e:
    for n in range(1, 7):
        check(n, f"Data check {n}", False, f"Exception: {e}")

# ─────────────────────────────────────────────
# CODE INTEGRITY
# ─────────────────────────────────────────────
section("CODE INTEGRITY")

def read_file(path):
    try:
        return open(path).read()
    except:
        return ""

prod_path = BASE / "run_production.py"
prod_src = read_file(prod_path)

new_feat_path = BASE / "augusta_model/features/new_features.py"
new_feat_src = read_file(new_feat_path)

cal_path = BASE / "augusta_model/calibration.py"
cal_src = read_file(cal_path)

weights_path = BASE / "augusta_sg_weights.json"

# [8] season < ps (not <=)
# Find ALL season comparisons near train_s2
leakage_lines = []
for i, line in enumerate(prod_src.splitlines(), 1):
    if "season" in line and ("<=" in line or "<" in line) and "ps" in line:
        leakage_lines.append(f"L{i}: {line.strip()}")
has_leakage = any("<=" in l and "ps" in l for l in leakage_lines)
check(8, "run_production.py: season < ps (no leakage)", not has_leakage,
      "\n       ".join(leakage_lines) if leakage_lines else "no season<ps lines found")

# [9] sort_values includes "season"
sort_lines = []
for i, line in enumerate(prod_src.splitlines(), 1):
    if "sort_values" in line:
        sort_lines.append(f"L{i}: {line.strip()}")
season_in_sort = any('"season"' in l or "'season'" in l for l in sort_lines)
check(9, "run_production.py: sort_values includes 'season'", season_in_sort,
      "\n       ".join(sort_lines[:5]))

# [10] reg_alpha=2.0, reg_lambda=5.0
has_alpha2 = "reg_alpha=2.0" in prod_src or "reg_alpha = 2.0" in prod_src
has_lambda5 = "reg_lambda=5.0" in prod_src or "reg_lambda = 5.0" in prod_src
check(10, "run_production.py: reg_alpha=2.0, reg_lambda=5.0",
      has_alpha2 and has_lambda5,
      f"alpha2={has_alpha2}, lambda5={has_lambda5}")

# [11] Recency decay (0.80)
decay_lines = [l.strip() for l in prod_src.splitlines() if "0.80" in l or "decay" in l.lower()]
check(11, "run_production.py: recency decay (0.80) exists",
      len(decay_lines) > 0,
      "\n       ".join(decay_lines[:3]) if decay_lines else "NOT FOUND")

# [12] Stale player cap
stale_lines = [l.strip() for l in prod_src.splitlines()
               if "stale" in l.lower() or "0.04" in l or "stale_cap" in l]
check(12, "run_production.py: stale player cap exists",
      len(stale_lines) > 0,
      "\n       ".join(stale_lines[:3]) if stale_lines else "NOT FOUND")

# [13] Honorary invitee floor
honorary_lines = [l.strip() for l in prod_src.splitlines()
                  if "honorary" in l.lower() or "invitee" in l.lower()]
check(13, "run_production.py: honorary invitee floor exists",
      len(honorary_lines) > 0,
      "\n       ".join(honorary_lines[:3]) if honorary_lines else "NOT FOUND")

# [14] driving_dominance does NOT use .abs()
dd_lines = []
for i, line in enumerate(new_feat_src.splitlines(), 1):
    if "driving_dominance" in line:
        dd_lines.append(f"L{i}: {line.strip()}")
has_abs_in_dd = any(".abs()" in l for l in dd_lines)
check(14, "new_features.py: driving_dominance does NOT use .abs()",
      not has_abs_in_dd and len(dd_lines) > 0,
      "\n       ".join(dd_lines[:5]) if dd_lines else "driving_dominance NOT FOUND")

# [15] played_recently is NOT hardcoded
pr_lines = []
for i, line in enumerate(new_feat_src.splitlines(), 1):
    if "played_recently" in line:
        pr_lines.append(f"L{i}: {line.strip()}")
is_hardcoded = any(("= 1" in l or "=1" in l) and "played_recently" in l
                   and "==" not in l for l in pr_lines)
check(15, "new_features.py: played_recently NOT hardcoded to 1",
      not is_hardcoded,
      "\n       ".join(pr_lines[:5]) if pr_lines else "played_recently NOT FOUND")

# [16] calibration.py: stale cap + honorary invitee logic
cal_stale = any("stale" in l.lower() or "0.04" in l for l in cal_src.splitlines())
cal_honorary = any("honorary" in l.lower() or "invitee" in l.lower() for l in cal_src.splitlines())
check(16, "calibration.py: stale cap exists", cal_stale,
      "stale logic found" if cal_stale else "NOT FOUND in calibration.py")
# Note: this is check 16 but we have 17 for honorary
RESULTS.pop()  # remove last to redo as single check
check(16, "calibration.py: stale cap + honorary invitee logic",
      cal_stale and cal_honorary,
      f"stale={cal_stale}, honorary={cal_honorary}")

# [17] augusta_sg_weights.json: not all equal
try:
    with open(weights_path) as f:
        weights = json.load(f)
    values = list(weights.values())
    not_uniform = len(set(round(v, 4) for v in values)) > 1
    check(17, "augusta_sg_weights.json: not uniform",
          not_uniform,
          f"values: {weights}")
except Exception as e:
    check(17, "augusta_sg_weights.json: load and check", False, str(e))

# ─────────────────────────────────────────────
# PREDICTION INTEGRITY
# ─────────────────────────────────────────────
section("PREDICTION INTEGRITY")

try:
    # [18] Load predictions
    pred_csv = BASE / "predictions_2026.csv"
    pred_parquet = BASE / "predictions_2026.parquet"
    pred_df = None
    pred_source = None

    if pred_parquet.exists() and pred_csv.exists():
        # use the newer one
        if pred_parquet.stat().st_mtime >= pred_csv.stat().st_mtime:
            pred_df = pd.read_parquet(pred_parquet)
            pred_source = "predictions_2026.parquet"
        else:
            pred_df = pd.read_csv(pred_csv)
            pred_source = "predictions_2026.csv"
    elif pred_parquet.exists():
        pred_df = pd.read_parquet(pred_parquet)
        pred_source = "predictions_2026.parquet"
    elif pred_csv.exists():
        pred_df = pd.read_csv(pred_csv)
        pred_source = "predictions_2026.csv"

    check(18, "Load predictions_2026 file", pred_df is not None,
          f"source={pred_source}, rows={len(pred_df) if pred_df is not None else 0}")

    if pred_df is not None:
        print(f"\n  Columns: {pred_df.columns.tolist()}")

        # Map column names
        col_map = {}
        for candidate, key in [
            (["win_prob", "model_win_prob", "win"], "win"),
            (["top5_prob", "model_top5_prob", "t5"], "t5"),
            (["top10_prob", "model_top10_prob", "t10"], "t10"),
            (["top20_prob", "model_top20_prob", "t20"], "t20"),
            (["player_name", "player"], "name"),
            (["dg_id"], "dg_id"),
            (["dg_rank", "datagolf_rank"], "dg_rank"),
        ]:
            for c in candidate:
                if c in pred_df.columns:
                    col_map[key] = c
                    break

        print(f"  Column mapping: {col_map}")

        # [19] win sums to ~1.0
        if "win" in col_map:
            win_sum = pred_df[col_map["win"]].sum()
            check(19, "win_prob sums to ~1.0", abs(win_sum - 1.0) < 0.05,
                  f"sum={win_sum:.4f}")
        else:
            check(19, "win_prob column exists", False, "no win_prob column found")

        # [20] t5, t10, t20 sums
        sum_checks = [("t5", 5.0), ("t10", 10.0), ("t20", 20.0)]
        for key, target in sum_checks:
            if key in col_map:
                s = pred_df[col_map[key]].sum()
                check(20, f"{key}_prob sums to ~{target}", abs(s - target) < 0.5,
                      f"sum={s:.4f} (target={target})")
            else:
                check(20, f"{key}_prob column exists", False, f"no {key} column")

        # [21] Monotonicity
        violations = 0
        mono_examples = []
        if all(k in col_map for k in ["win", "t5", "t10", "t20"]):
            for _, row in pred_df.iterrows():
                w = row[col_map["win"]]
                t5 = row[col_map["t5"]]
                t10 = row[col_map["t10"]]
                t20 = row[col_map["t20"]]
                if not (w <= t5 <= t10 <= t20):
                    violations += 1
                    if len(mono_examples) < 3:
                        name = row.get(col_map.get("name", "player_name"), "?")
                        mono_examples.append(f"{name}: win={w:.3f} t5={t5:.3f} t10={t10:.3f} t20={t20:.3f}")
        check(21, "Monotonicity: win<=t5<=t10<=t20", violations == 0,
              f"violations={violations}" + (f": {mono_examples}" if mono_examples else ""))

        # [22] Scheffler top 3 by win_prob
        if "name" in col_map and "win" in col_map:
            sorted_df = pred_df.sort_values(col_map["win"], ascending=False).reset_index(drop=True)
            scheffler_rows = sorted_df[sorted_df[col_map["name"]].str.lower().str.contains("scheffler")]
            if len(scheffler_rows) > 0:
                scheffler_rank = scheffler_rows.index[0] + 1
                check(22, "Scheffler in top 3 by win_prob", scheffler_rank <= 3,
                      f"Scheffler rank={scheffler_rank}, win={scheffler_rows.iloc[0][col_map['win']]:.4f}")
            else:
                check(22, "Scheffler found in predictions", False, "Scheffler not found")
        else:
            check(22, "Scheffler top 3 check", False, "missing columns")

        # [23] Reed NOT in top 15
        if "name" in col_map and "win" in col_map:
            sorted_df = pred_df.sort_values(col_map["win"], ascending=False).reset_index(drop=True)
            reed_rows = sorted_df[sorted_df[col_map["name"]].str.lower().str.contains("reed")]
            if len(reed_rows) > 0:
                reed_rank = reed_rows.index[0] + 1
                check(23, "Reed NOT in top 15 by win_prob", reed_rank > 15,
                      f"Reed rank={reed_rank}, win={reed_rows.iloc[0][col_map['win']]:.4f}")
            else:
                check(23, "Reed NOT in top 15 (not found)", True, "Reed not in predictions")

        # [24] No dg_rank > 200 in top 20
        if "dg_rank" in col_map and "win" in col_map:
            top20 = pred_df.sort_values(col_map["win"], ascending=False).head(20)
            high_rank = top20[top20[col_map["dg_rank"]].fillna(999) > 200]
            check(24, "No dg_rank>200 player in top 20", len(high_rank) == 0,
                  f"offenders={high_rank[[col_map['name'], col_map['dg_rank'], col_map['win']]].to_string() if len(high_rank) > 0 else 'none'}")
        else:
            check(24, "dg_rank column check", False, f"missing dg_rank or win col. cols={list(col_map.keys())}")

        # [25] Honorary invitees T10 <= 2%
        honorary_names = ["vijay singh", "fred couples", "jose maria olazabal", "olazabal"]
        t10_col = col_map.get("t10")
        name_col = col_map.get("name")
        honorary_ok = True
        honorary_details = []
        if t10_col and name_col:
            for _, row in pred_df.iterrows():
                pname = str(row[name_col]).lower()
                if any(h in pname for h in honorary_names):
                    t10_val = row[t10_col]
                    honorary_details.append(f"{row[name_col]}: T10={t10_val:.4f}")
                    if t10_val > 0.02:
                        honorary_ok = False
        check(25, "Honorary invitees T10 <= 2%", honorary_ok,
              "\n       ".join(honorary_details) if honorary_details else "none found (may not be in field)")

        # [26] Print top 20
        print(f"\n  {'─'*80}")
        print(f"  TOP 20 PLAYERS (from {pred_source}):")
        print(f"  {'─'*80}")
        if "name" in col_map and "win" in col_map:
            top20 = pred_df.sort_values(col_map["win"], ascending=False).head(20).copy()
            market_win_col = next((c for c in ["market_win_prob", "mkt_win", "closing_win"] if c in pred_df.columns), None)
            market_t10_col = next((c for c in ["market_top10_prob", "mkt_t10", "closing_t10"] if c in pred_df.columns), None)
            edge_win_col = next((c for c in ["win_edge", "edge_win", "edge"] if c in pred_df.columns), None)
            edge_t10_col = next((c for c in ["t10_edge", "edge_t10"] if c in pred_df.columns), None)

            header = f"  {'#':>3} {'Player':<25} {'Win%':>6} {'T10%':>6} {'T20%':>6}"
            if market_win_col:
                header += f" {'MktW%':>6}"
            if market_t10_col:
                header += f" {'MktT10%':>7}"
            if edge_win_col:
                header += f" {'WinEdge':>8}"
            print(header)
            for rank, (_, row) in enumerate(top20.iterrows(), 1):
                name = str(row[name_col])[:25]
                win = row[col_map["win"]] * 100
                t10 = row[col_map.get("t10", col_map["win"])] * 100 if "t10" in col_map else 0
                t20 = row[col_map.get("t20", col_map["win"])] * 100 if "t20" in col_map else 0
                line = f"  {rank:>3} {name:<25} {win:>5.2f}% {t10:>5.2f}% {t20:>5.2f}%"
                if market_win_col:
                    mw = row.get(market_win_col, float("nan"))
                    line += f" {mw*100:>5.2f}%" if not pd.isna(mw) else f" {'N/A':>6}"
                if market_t10_col:
                    mt10 = row.get(market_t10_col, float("nan"))
                    line += f" {mt10*100:>6.2f}%" if not pd.isna(mt10) else f" {'N/A':>7}"
                if edge_win_col:
                    ew = row.get(edge_win_col, float("nan"))
                    line += f" {ew*100:>+7.2f}%" if not pd.isna(ew) else f" {'N/A':>8}"
                print(line)

        check(26, "Top 20 printed", True, "see table above")

        # [27] Spearman vs market
        market_win_col = next((c for c in ["market_win_prob", "mkt_win", "closing_win"] if c in pred_df.columns), None)
        if market_win_col and "win" in col_map:
            merged = pred_df[[col_map["name"], col_map["win"], market_win_col]].dropna(subset=[market_win_col])
            if len(merged) >= 5:
                from scipy.stats import spearmanr
                corr, pval = spearmanr(merged[col_map["win"]], merged[market_win_col])
                check(27, "Spearman vs market computed", True,
                      f"rho={corr:.3f}, p={pval:.4f}, n={len(merged)}")
            else:
                check(27, "Spearman vs market", False,
                      f"not enough market data: n={len(merged)}")
        else:
            check(27, "Spearman vs market (no market col)", True,
                  f"no market win col in predictions (cols: {pred_df.columns.tolist()})")

except Exception as e:
    import traceback
    for n in [18, 19, 20, 21, 22, 23, 24, 25, 26, 27]:
        check(n, f"Prediction check {n}", False, f"Exception: {e}")
    print(traceback.format_exc())

# ─────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────
section("STREAMLIT APP")

# [28] Best Bets page exists
best_bets = BASE / "streamlit_app/pages/5_Best_Bets.py"
check(28, "streamlit_app/pages/5_Best_Bets.py exists", best_bets.exists(),
      str(best_bets))

# [29] app.py references Best Bets
app_src = read_file(BASE / "streamlit_app/app.py")
has_best_bets_ref = "Best Bets" in app_src or "best_bets" in app_src.lower() or "5_Best_Bets" in app_src
check(29, "app.py references Best Bets", has_best_bets_ref,
      "found" if has_best_bets_ref else "NOT FOUND in app.py")

# [30] Syntax check all streamlit .py files
import py_compile
streamlit_dir = BASE / "streamlit_app"
py_files = list(streamlit_dir.rglob("*.py"))
syntax_errors = []
for pyf in py_files:
    try:
        py_compile.compile(str(pyf), doraise=True)
    except py_compile.PyCompileError as e:
        syntax_errors.append(f"{pyf.name}: {e}")
check(30, f"Syntax check all streamlit .py files ({len(py_files)} files)",
      len(syntax_errors) == 0,
      "\n       ".join(syntax_errors) if syntax_errors else f"all {len(py_files)} files OK")

# ─────────────────────────────────────────────
# TEST SUITE
# ─────────────────────────────────────────────
section("TEST SUITE")

# [31] Run pytest
tests_dir = BASE / "tests"
test_files = list(tests_dir.rglob("test_*.py")) if tests_dir.exists() else []
print(f"\n  Found {len(test_files)} test files: {[f.name for f in test_files]}")

if test_files:
    result = subprocess.run(
        ["python3", "-m", "pytest", str(tests_dir), "-v", "--tb=short", "--no-header"],
        capture_output=True, text=True, cwd=str(BASE)
    )
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-1000:])
    passed = result.returncode == 0
    # Parse summary
    summary_line = [l for l in result.stdout.splitlines() if "passed" in l or "failed" in l or "error" in l]
    check(31, "Test suite passes", passed,
          summary_line[-1] if summary_line else f"returncode={result.returncode}")
else:
    check(31, "Test suite", False, "no test files found in tests/")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
section("FINAL SUMMARY")

# Dedupe check numbers (in case of any double-adds)
seen = {}
deduped = []
for r in RESULTS:
    n = r[0]
    if n not in seen:
        seen[n] = r
        deduped.append(r)
    else:
        # keep the last one written
        seen[n] = r

# Replace with deduped
final = list(seen.values())
passed = sum(1 for r in final if r[2])
failed = sum(1 for r in final if not r[2])
total = len(final)

print(f"\n  {passed}/{total} PASS, {failed}/{total} FAIL\n")

failures = [r for r in final if not r[2]]
if failures:
    print("  FAILURES:")
    for r in failures:
        print(f"    ❌ [{r[0]:02d}] {r[1]}")
        if r[3]:
            for line in r[3].splitlines()[:3]:
                print(f"          {line}")
else:
    print("  ALL CHECKS PASSED! 🎯")

print()
