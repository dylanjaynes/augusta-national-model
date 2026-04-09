"""Tests for augusta_model/calibration.py"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from augusta_model.calibration import (
    _to_logits,
    _sigmoid,
    fit_platt_calibrator,
    calibrate_s2_platt,
    apply_debutant_adjustment,
    normalize_to_sum,
    run_unified_mc,
    enforce_monotonic,
    calibrate_full_pipeline,
    TIER_BASE_RATES,
    OVERALL_BASE_RATE,
)


# ── _to_logits ──

class TestToLogits:
    def test_midpoint(self):
        assert _to_logits(np.array([0.5]))[0] == pytest.approx(0.0, abs=1e-6)

    def test_symmetric(self):
        lo = _to_logits(np.array([0.2]))[0]
        hi = _to_logits(np.array([0.8]))[0]
        assert lo == pytest.approx(-hi, abs=1e-6)

    def test_clips_extremes(self):
        # Should not produce inf for 0.0 or 1.0
        result = _to_logits(np.array([0.0, 1.0]))
        assert np.all(np.isfinite(result))

    def test_monotonic(self):
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        logits = _to_logits(probs)
        assert all(logits[i] < logits[i + 1] for i in range(len(logits) - 1))


# ── _sigmoid ──

class TestSigmoid:
    def test_midpoint(self):
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive(self):
        assert _sigmoid(100.0) == pytest.approx(1.0, abs=1e-6)

    def test_large_negative(self):
        assert _sigmoid(-100.0) == pytest.approx(0.0, abs=1e-6)

    def test_inverse_of_logit(self):
        probs = np.array([0.1, 0.5, 0.9])
        recovered = _sigmoid(_to_logits(probs))
        np.testing.assert_allclose(recovered, probs, atol=1e-5)

    def test_array_input(self):
        result = _sigmoid(np.array([-10, 0, 10]))
        assert result.shape == (3,)
        assert result[0] < 0.5 < result[2]


# ── fit_platt_calibrator ──

class TestFitPlatt:
    def test_returns_two_params(self):
        s2 = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        actual = np.array([0, 0, 0, 1, 1])
        a, b = fit_platt_calibrator(s2, actual)
        assert isinstance(a, float)
        assert isinstance(b, float)

    def test_separable_data_preserves_order(self):
        rng = np.random.RandomState(42)
        n = 100
        s2 = np.concatenate([rng.uniform(0.05, 0.3, n // 2),
                             rng.uniform(0.6, 0.95, n // 2)])
        actual = np.array([0] * (n // 2) + [1] * (n // 2))
        a, b = fit_platt_calibrator(s2, actual)
        cal = calibrate_s2_platt(s2, (a, b))
        # Calibrated high-input players should still be higher than low-input
        assert cal[n // 2:].mean() > cal[:n // 2].mean()


# ── calibrate_s2_platt ──

class TestCalibrateS2Platt:
    def test_output_in_0_1(self):
        s2 = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
        cal = calibrate_s2_platt(s2)
        assert np.all(cal >= 0) and np.all(cal <= 1)

    def test_preserves_ordering(self):
        s2 = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        cal = calibrate_s2_platt(s2)
        assert all(cal[i] <= cal[i + 1] for i in range(len(cal) - 1))

    def test_custom_params(self):
        s2 = np.array([0.2, 0.8])
        cal_default = calibrate_s2_platt(s2)
        cal_custom = calibrate_s2_platt(s2, platt_params=(1.0, 0.0))
        # Different params should give different results
        assert not np.allclose(cal_default, cal_custom)


# ── apply_debutant_adjustment ──

class TestDebutantAdjustment:
    def test_tier0_shrinks_toward_base_rate(self):
        probs = np.array([0.5, 0.5, 0.5])
        tiers = np.array([0, 0, 0])
        adj = apply_debutant_adjustment(probs, tiers)
        # Should shrink toward 0.097 (debutant base rate)
        assert np.all(adj < 0.5)
        expected = 0.55 * 0.5 + 0.45 * TIER_BASE_RATES[0]
        np.testing.assert_allclose(adj, expected, atol=1e-6)

    def test_tier2_no_change(self):
        probs = np.array([0.3, 0.5, 0.7])
        tiers = np.array([2, 3, 4])
        adj = apply_debutant_adjustment(probs, tiers)
        np.testing.assert_allclose(adj, probs, atol=1e-6)

    def test_tier1_partial_shrinkage(self):
        probs = np.array([0.5])
        tiers = np.array([1])
        adj = apply_debutant_adjustment(probs, tiers)
        expected = 0.85 * 0.5 + 0.15 * TIER_BASE_RATES[1]
        np.testing.assert_allclose(adj, expected, atol=1e-6)

    def test_preserves_ordering_within_tier(self):
        probs = np.array([0.2, 0.4, 0.6])
        tiers = np.array([0, 0, 0])
        adj = apply_debutant_adjustment(probs, tiers)
        assert adj[0] < adj[1] < adj[2]


# ── normalize_to_sum ──

class TestNormalizeToSum:
    def test_sums_to_target(self):
        probs = np.array([0.3, 0.2, 0.1, 0.05, 0.05])
        normed = normalize_to_sum(probs, 1.0)
        assert normed.sum() == pytest.approx(1.0, abs=0.01)

    def test_preserves_ordering(self):
        probs = np.array([0.5, 0.3, 0.1, 0.05])
        normed = normalize_to_sum(probs, 1.0)
        assert all(normed[i] >= normed[i + 1] for i in range(len(normed) - 1))

    def test_zero_sum_input(self):
        probs = np.array([0.0, 0.0, 0.0])
        normed = normalize_to_sum(probs, 1.0)
        assert normed.sum() == pytest.approx(1.0, abs=0.01)


# ── run_unified_mc ──

class TestUnifiedMC:
    @pytest.fixture
    def mc_result(self):
        # 20-player field, player 0 is best, player 19 is worst
        skills = np.linspace(0.8, 0.2, 20)
        return run_unified_mc(skills, n_sims=10000, seed=42)

    def test_win_sums_to_1(self, mc_result):
        assert mc_result["win"].sum() == pytest.approx(1.0, abs=0.01)

    def test_top5_sums_to_5(self, mc_result):
        assert mc_result["top5"].sum() == pytest.approx(5.0, abs=0.05)

    def test_top10_sums_to_10(self, mc_result):
        assert mc_result["top10"].sum() == pytest.approx(10.0, abs=0.1)

    def test_top20_sums_to_20(self, mc_result):
        assert mc_result["top20"].sum() == pytest.approx(20.0, abs=0.1)

    def test_best_player_wins_most(self, mc_result):
        assert mc_result["win"].argmax() == 0

    def test_monotonic_across_markets(self):
        # Use 80-player field so cut_line (55% = 44) > 20, ensuring top20 <= make_cut
        skills = np.linspace(0.8, 0.2, 80)
        mc = run_unified_mc(skills, n_sims=10000, seed=42)
        for i in range(len(mc["win"])):
            assert mc["win"][i] <= mc["top5"][i] + 1e-6
            assert mc["top5"][i] <= mc["top10"][i] + 1e-6
            assert mc["top10"][i] <= mc["top20"][i] + 1e-6
            assert mc["top20"][i] <= mc["make_cut"][i] + 1e-6

    def test_deterministic_with_seed(self):
        skills = np.array([0.8, 0.5, 0.2])
        r1 = run_unified_mc(skills, n_sims=1000, seed=42)
        r2 = run_unified_mc(skills, n_sims=1000, seed=42)
        np.testing.assert_array_equal(r1["win"], r2["win"])

    def test_equal_skills_uniform(self):
        skills = np.full(10, 0.5)
        result = run_unified_mc(skills, n_sims=20000, seed=42)
        # Each player should win ~10% of the time
        np.testing.assert_allclose(result["win"], 0.1, atol=0.03)


# ── enforce_monotonic ──

class TestEnforceMonotonic:
    def test_fixes_violations(self):
        df = pd.DataFrame({
            "win_prob": [0.10, 0.05],
            "top5_prob": [0.08, 0.04],   # violation: top5 < win
            "top10_prob": [0.20, 0.10],
            "top20_prob": [0.15, 0.08],   # violation: top20 < top10
            "make_cut_prob": [0.50, 0.40],
        })
        fixed = enforce_monotonic(df)
        for _, row in fixed.iterrows():
            assert row["win_prob"] <= row["top5_prob"]
            assert row["top5_prob"] <= row["top10_prob"]
            assert row["top10_prob"] <= row["top20_prob"]
            assert row["top20_prob"] <= row["make_cut_prob"]

    def test_no_change_when_valid(self):
        df = pd.DataFrame({
            "win_prob": [0.05],
            "top5_prob": [0.15],
            "top10_prob": [0.30],
            "top20_prob": [0.50],
            "make_cut_prob": [0.70],
        })
        fixed = enforce_monotonic(df)
        pd.testing.assert_frame_equal(fixed, df)


# ── calibrate_full_pipeline ──

class TestFullPipeline:
    def test_output_shape(self):
        s2 = np.random.uniform(0.1, 0.9, 40)
        tiers = np.random.choice([0, 1, 2, 3], 40)
        result = calibrate_full_pipeline(s2, tiers, n_sims=1000, seed=42)
        assert len(result) == 40
        assert "win_prob" in result.columns
        assert "top10_prob" in result.columns
        assert "make_cut_prob" in result.columns

    def test_sums_correct(self):
        s2 = np.random.uniform(0.1, 0.9, 80)
        tiers = np.random.choice([0, 1, 2, 3, 4], 80)
        result = calibrate_full_pipeline(s2, tiers, n_sims=10000, seed=42)
        assert result["win_prob"].sum() == pytest.approx(1.0, abs=0.01)
        assert result["top5_prob"].sum() == pytest.approx(5.0, abs=0.1)
        assert result["top10_prob"].sum() == pytest.approx(10.0, abs=0.2)
        assert result["top20_prob"].sum() == pytest.approx(20.0, abs=0.3)

    def test_no_monotonic_violations(self):
        s2 = np.random.uniform(0.1, 0.9, 30)
        tiers = np.array([0] * 10 + [1] * 10 + [2] * 10)
        result = calibrate_full_pipeline(s2, tiers, n_sims=5000, seed=42)
        for _, row in result.iterrows():
            assert row["win_prob"] <= row["top5_prob"] + 1e-6
            assert row["top5_prob"] <= row["top10_prob"] + 1e-6
            assert row["top10_prob"] <= row["top20_prob"] + 1e-6
            assert row["top20_prob"] <= row["make_cut_prob"] + 1e-6
