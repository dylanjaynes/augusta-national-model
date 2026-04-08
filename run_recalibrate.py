#!/usr/bin/env python3
"""
Recalibrate existing model outputs using the new calibration pipeline.

Fixes:
1. S1/S2 ranking disconnect (Scheffler 0.002% win + 73.8% top-10)
2. Temperature scaling compression (top-10 floor at 17%)
3. Probabilities not summing to physical constraints (top-10 sum=34, should be 10)
4. Debutant inflation (Marco Penge 57.6% top-10 as a debutant)

Reads: backtest_results_v7.parquet, predictions_2026.parquet
Writes: predictions_2026_calibrated.parquet/csv
"""
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import spearmanr

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from augusta_model.calibration import (
    fit_platt_calibrator,
    calibrate_s2_platt,
    apply_debutant_adjustment,
    normalize_to_sum,
    run_unified_mc,
    enforce_monotonic,
    calibrate_full_pipeline,
    fit_and_calibrate_backtest,
)

PROCESSED = Path("data/processed")
N_SIMS = 50000
NOISE = 0.16   # higher noise = more upsets (realistic for golf)
TPRED = 0.06   # lower signal = flatter distribution (~12% max win for 82 players)
SEED = 42


def precision_at_k(probs, actuals, k):
    """Top-k precision: of the k highest-predicted, how many were actual top-10?"""
    top_k_idx = np.argsort(-probs)[:k]
    return actuals[top_k_idx].mean()


def evaluate_calibration(probs, actuals, label=""):
    """Print calibration metrics."""
    valid = ~np.isnan(probs) & ~np.isnan(actuals)
    p, a = probs[valid], actuals[valid]
    if len(p) == 0:
        print(f"  {label}: no valid data")
        return {}

    auc = roc_auc_score(a, p) if a.sum() > 0 and a.sum() < len(a) else 0.5
    brier = brier_score_loss(a, np.clip(p, 0, 1))
    sp, _ = spearmanr(p, a)
    prec10 = precision_at_k(p, a, 10)
    prec13 = precision_at_k(p, a, 13)

    print(f"  {label}: AUC={auc:.3f} Brier={brier:.4f} Spearman={sp:.3f} "
          f"Prec@10={prec10:.0%} Prec@13={prec13:.0%} "
          f"range=[{p.min():.3f}, {p.max():.3f}] sum={p.sum():.1f}")
    return {"auc": auc, "brier": brier, "spearman": sp, "prec10": prec10, "prec13": prec13}


# ═══════════════════════════════════════════════════════════
# PART 1: BACKTEST VALIDATION
# ═══════════════════════════════════════════════════════════

def run_backtest_validation():
    print("=" * 60)
    print("PART 1: BACKTEST CALIBRATION VALIDATION")
    print("=" * 60)

    bt = pd.read_parquet(PROCESSED / "backtest_results_v7.parquet")
    bt["actual_top10"] = (bt["actual_finish_num"] <= 10).astype(int)

    years = sorted(bt["season"].unique())
    print(f"Backtest years: {years}, total rows: {len(bt)}")
    print(f"Overall top-10 rate: {bt['actual_top10'].mean():.3f}")

    # ── Evaluate BEFORE calibration (old temperature scaling) ──
    print("\n--- BEFORE (temperature scaling T=2.5) ---")
    for y in years:
        ydf = bt[bt["season"] == y]
        evaluate_calibration(ydf["blend_top10"].values, ydf["actual_top10"].values, f"{y}")
    evaluate_calibration(bt["blend_top10"].values, bt["actual_top10"].values, "ALL ")

    # ── Walk-forward Platt calibration ──
    print("\n--- AFTER (Platt scaling + debutant adj + sum normalization) ---")
    bt_cal, platt_params = fit_and_calibrate_backtest(bt)

    print(f"\nPlatt parameters by year: {platt_params}")

    all_metrics = []
    for y in years:
        ydf = bt_cal[bt_cal["season"] == y]

        if ydf["s2_adjusted"].isna().all():
            print(f"  {y}: skipped (cold start, no prior training data)")
            continue

        # Normalize to sum constraint
        raw_adj = ydf["s2_adjusted"].values
        normed = normalize_to_sum(raw_adj, target_sum=10.0)

        m = evaluate_calibration(normed, ydf["actual_top10"].values, f"{y}")
        m["year"] = y
        all_metrics.append(m)

        # Run unified MC for this year
        mc = run_unified_mc(raw_adj, n_sims=N_SIMS, noise_std=NOISE,
                            target_pred_std=TPRED, seed=SEED)
        win_norm = normalize_to_sum(mc["win"], 1.0)

        # Check the winner
        actual_winner_mask = ydf["actual_finish_num"] == 1
        if actual_winner_mask.any():
            winner_name = ydf.loc[actual_winner_mask, "player_name"].iloc[0]
            winner_idx = actual_winner_mask.values.nonzero()[0][0]
            winner_win_prob = win_norm[winner_idx]
            winner_rank = (-win_norm).argsort().argsort()[winner_idx] + 1
            print(f"    Winner: {winner_name}, model win%={winner_win_prob:.2%}, rank={winner_rank}")

    if all_metrics:
        avg = {k: np.mean([m[k] for m in all_metrics if k in m and not np.isnan(m[k])])
               for k in ["auc", "brier", "spearman", "prec10", "prec13"]}
        print(f"\n  AVG : AUC={avg['auc']:.3f} Brier={avg['brier']:.4f} "
              f"Spearman={avg['spearman']:.3f} Prec@10={avg['prec10']:.0%} "
              f"Prec@13={avg['prec13']:.0%}")

    return bt_cal, platt_params


# ═══════════════════════════════════════════════════════════
# PART 2: RECALIBRATE 2026 PREDICTIONS
# ═══════════════════════════════════════════════════════════

def recalibrate_2026(platt_params_by_year):
    print(f"\n{'=' * 60}")
    print("PART 2: RECALIBRATE 2026 PREDICTIONS")
    print("=" * 60)

    preds = pd.read_parquet(PROCESSED / "predictions_2026.parquet")
    print(f"Loaded {len(preds)} players from predictions_2026.parquet")

    # Fit final Platt calibrator on ALL backtest data (excluding cold start)
    bt = pd.read_parquet(PROCESSED / "backtest_results_v7.parquet")
    bt["actual_top10"] = (bt["actual_finish_num"] <= 10).astype(int)
    valid = bt.dropna(subset=["s2_top10", "actual_top10"])
    valid = valid[valid["s2_top10"] > 0.001]  # exclude 2021 cold start

    final_params = fit_platt_calibrator(
        valid["s2_top10"].values,
        valid["actual_top10"].values
    )
    print(f"Final Platt params: a={final_params[0]:.4f}, b={final_params[1]:.4f}")
    print(f"Trained on {len(valid)} rows ({valid['actual_top10'].sum()} positives)")

    # Get S2 raw and experience tiers
    s2_raw = preds["stage2_prob_raw"].values
    tiers = preds.get("augusta_experience_tier",
                      pd.Series(np.zeros(len(preds)))).fillna(0).values

    print(f"\nBEFORE calibration:")
    print(f"  Win% range: [{preds['win_prob'].min():.4f}, {preds['win_prob'].max():.4f}], "
          f"sum={preds['win_prob'].sum():.3f}")
    print(f"  Top10% range: [{preds['top10_prob_calibrated'].min():.4f}, "
          f"{preds['top10_prob_calibrated'].max():.4f}], "
          f"sum={preds['top10_prob_calibrated'].sum():.1f}")

    # ── Full calibration pipeline ──
    cal_result = calibrate_full_pipeline(
        s2_raw, tiers,
        platt_params=final_params,
        n_sims=N_SIMS, noise_std=NOISE,
        target_pred_std=TPRED, seed=SEED,
        field_size=len(preds)
    )

    # Merge calibrated columns
    preds_new = preds.copy()
    preds_new["win_prob"] = cal_result["win_prob"].values
    preds_new["top5_prob"] = cal_result["top5_prob"].values
    preds_new["top10_prob"] = cal_result["top10_prob"].values
    preds_new["top20_prob"] = cal_result["top20_prob"].values
    preds_new["make_cut_prob"] = cal_result["make_cut_prob"].values
    preds_new["s2_platt"] = cal_result["s2_calibrated"].values
    preds_new["s2_debutant_adj"] = cal_result["s2_adjusted"].values

    # Keep old columns for comparison
    preds_new["top10_prob_old"] = preds["top10_prob_calibrated"]
    preds_new["win_prob_old"] = preds["win_prob"]

    # Recompute Kelly edges if DK odds exist
    if "dk_fair_prob_win" in preds_new.columns:
        dk_win = preds_new["dk_fair_prob_win"].fillna(0)
        dk_t10 = preds_new["dk_fair_prob_top10"].fillna(0)
        preds_new["kelly_edge_win"] = np.where(
            dk_win > 0, (preds_new["win_prob"] - dk_win) / dk_win, 0
        )
        preds_new["kelly_edge_top10"] = np.where(
            dk_t10 > 0, (preds_new["top10_prob"] - dk_t10) / dk_t10, 0
        )

    # Sort by top-10 probability
    preds_new = preds_new.sort_values("top10_prob", ascending=False).reset_index(drop=True)

    print(f"\nAFTER calibration:")
    print(f"  Win% range: [{preds_new['win_prob'].min():.4f}, {preds_new['win_prob'].max():.4f}], "
          f"sum={preds_new['win_prob'].sum():.3f}")
    print(f"  Top5% range: [{preds_new['top5_prob'].min():.4f}, {preds_new['top5_prob'].max():.4f}], "
          f"sum={preds_new['top5_prob'].sum():.1f}")
    print(f"  Top10% range: [{preds_new['top10_prob'].min():.4f}, {preds_new['top10_prob'].max():.4f}], "
          f"sum={preds_new['top10_prob'].sum():.1f}")
    print(f"  Top20% range: [{preds_new['top20_prob'].min():.4f}, {preds_new['top20_prob'].max():.4f}], "
          f"sum={preds_new['top20_prob'].sum():.1f}")

    # ── Print results ──
    print(f"\n{'=' * 60}")
    print("2026 CALIBRATED PREDICTIONS")
    print(f"{'=' * 60}")
    print(f"{'#':<4} {'Player':<25} {'Win%':>6} {'T5%':>6} {'T10%':>6} {'T20%':>6} "
          f"{'Tier':>4} {'S2raw':>6} {'S2plt':>6} {'old_t10':>8}")
    print("-" * 95)
    for i, r in preds_new.head(30).iterrows():
        tier = r.get("augusta_experience_tier", 0)
        print(f"{i+1:<4} {r['player_name']:<25} {r['win_prob']:>5.1%} {r['top5_prob']:>5.1%} "
              f"{r['top10_prob']:>5.1%} {r['top20_prob']:>5.1%} "
              f"{tier:>4.0f} {r['stage2_prob_raw']:>5.3f} {r['s2_platt']:>5.3f} "
              f"{r['top10_prob_old']:>7.1%}")

    # ── Debutant check ──
    debs = preds_new[preds_new.get("augusta_experience_tier", 0) == 0]
    if len(debs) > 0:
        print(f"\n--- DEBUTANTS ({len(debs)} players) ---")
        for _, r in debs.iterrows():
            rank = preds_new.index.get_loc(r.name) + 1
            print(f"  #{rank:<3} {r['player_name']:<25} t10={r['top10_prob']:>5.1%} "
                  f"(was {r['top10_prob_old']:.1%}) s2_raw={r['stage2_prob_raw']:.3f}")

    # ── Key player check ──
    print(f"\n--- KEY PLAYER CONSISTENCY CHECK ---")
    for name in ["Scottie Scheffler", "Rory McIlroy", "Jon Rahm", "Bryson DeChambeau",
                  "Ludvig Aberg", "Xander Schauffele", "Marco Penge", "Fred Couples"]:
        row = preds_new[preds_new["player_name"] == name]
        if len(row) > 0:
            r = row.iloc[0]
            rank = preds_new.index.get_loc(row.index[0]) + 1
            print(f"  #{rank:<3} {name:<25} win={r['win_prob']:.2%} t5={r['top5_prob']:.1%} "
                  f"t10={r['top10_prob']:.1%} t20={r['top20_prob']:.1%} "
                  f"(was win={r['win_prob_old']:.2%} t10={r['top10_prob_old']:.1%})")

    # ── Sanity checks ──
    print(f"\n--- SANITY CHECKS ---")
    violations = 0
    for _, r in preds_new.iterrows():
        if r["win_prob"] > r["top5_prob"] + 0.001:
            violations += 1
        if r["top5_prob"] > r["top10_prob"] + 0.001:
            violations += 1
        if r["top10_prob"] > r["top20_prob"] + 0.001:
            violations += 1
    print(f"  Monotonic violations: {violations}")
    print(f"  Win sum: {preds_new['win_prob'].sum():.3f} (target: 1.0)")
    print(f"  Top5 sum: {preds_new['top5_prob'].sum():.1f} (target: 5.0)")
    print(f"  Top10 sum: {preds_new['top10_prob'].sum():.1f} (target: 10.0)")
    print(f"  Top20 sum: {preds_new['top20_prob'].sum():.1f} (target: 20.0)")

    # ── Save ──
    out_path = PROCESSED / "predictions_2026_calibrated.parquet"
    preds_new.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")

    csv_path = PROCESSED / "predictions_2026_calibrated.csv"
    preds_new.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")

    return preds_new


if __name__ == "__main__":
    bt_cal, platt_params = run_backtest_validation()
    preds = recalibrate_2026(platt_params)
