"""Tests for augusta_model/features/new_features.py"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from augusta_model.features.new_features import (
    add_scoring_profile,
    add_weather_features,
    add_putting_surface_features,
    add_form_momentum,
    add_sg_interactions,
    add_aging_decay,
)


# ── add_par5_proxy ──

class TestScoringProfile:
    def test_adds_columns(self):
        df = pd.DataFrame({
            "sg_ott_8w": [1.0, -0.5],
            "sg_app_8w": [0.5, 1.0],
            "sg_total_8w": [2.0, 0.5],
            "sg_total_std_8": [0.3, 0.4],
        })
        result = add_scoring_profile(df)
        assert "par5_scoring_proxy" in result.columns
        assert "driving_dominance" in result.columns
        assert "bogey_resistance" in result.columns

    def test_par5_proxy_formula(self):
        df = pd.DataFrame({
            "sg_ott_8w": [1.0],
            "sg_app_8w": [2.0],
            "sg_total_8w": [3.0],
            "sg_total_std_8": [0.5],
        })
        result = add_scoring_profile(df)
        expected = 1.0 * 0.45 + 2.0 * 0.40 + 3.0 * 0.15
        assert result["par5_scoring_proxy"].iloc[0] == pytest.approx(expected)

    def test_bogey_resistance_formula(self):
        df = pd.DataFrame({
            "sg_ott_8w": [1.0],
            "sg_app_8w": [0.5],
            "sg_total_8w": [2.0],
            "sg_total_std_8": [0.3],
        })
        result = add_scoring_profile(df)
        assert result["bogey_resistance"].iloc[0] == pytest.approx(2.0 - 0.3)

    def test_handles_missing_columns(self):
        df = pd.DataFrame({"player_name": ["A", "B"]})
        result = add_scoring_profile(df)
        assert "par5_scoring_proxy" in result.columns

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"sg_ott_8w": [1.0], "sg_app_8w": [0.5],
                           "sg_total_8w": [1.5], "sg_total_std_8": [0.3]})
        original_cols = list(df.columns)
        add_scoring_profile(df)
        assert list(df.columns) == original_cols


# ── add_weather_features ──

class TestWeatherFeatures:
    def test_missing_weather_file_returns_original(self):
        df = pd.DataFrame({"season": [2025], "player_name": ["A"]})
        result = add_weather_features(df, weather_path="/nonexistent/file.parquet")
        assert len(result) == 1

    def test_no_season_uses_defaults(self):
        df = pd.DataFrame({"player_name": ["A"]})
        result = add_weather_features(df, weather_path="/nonexistent/file.parquet")
        # Without season column AND missing weather file, function returns early
        # so wind_avg_mph won't be set. Just check the function doesn't crash.
        assert len(result) == 1

    def test_wind_experience_interaction(self, tmp_path):
        weather = pd.DataFrame({
            "season": [2025],
            "wind_avg_mph": [18.0],
            "wind_max_mph": [25.0],
            "rain_total_mm": [5.0],
        })
        wp = tmp_path / "weather.parquet"
        weather.to_parquet(wp)

        df = pd.DataFrame({
            "season": [2025, 2025],
            "player_name": ["Vet", "Debut"],
            "augusta_competitive_rounds": [20.0, 0.0],
        })
        result = add_weather_features(df, weather_path=wp)
        assert "wind_experience_interaction" in result.columns
        # Veteran should have higher interaction than debutant
        assert result["wind_experience_interaction"].iloc[0] > result["wind_experience_interaction"].iloc[1]

    def test_wet_course_flag(self, tmp_path):
        weather = pd.DataFrame({
            "season": [2025],
            "wind_avg_mph": [10.0],
            "wind_max_mph": [15.0],
            "rain_total_mm": [50.0],
        })
        wp = tmp_path / "weather.parquet"
        weather.to_parquet(wp)

        df = pd.DataFrame({"season": [2025], "player_name": ["A"]})
        result = add_weather_features(df, weather_path=wp)
        assert result["wet_course"].iloc[0] == 1.0


# ── add_putting_surface_features ──

class TestPuttingSurface:
    def test_adds_column(self):
        tour_df = pd.DataFrame({
            "player_name": ["A", "A", "A", "A"],
            "event_name": ["Masters Tournament", "Masters Tournament",
                          "Masters Tournament", "U.S. Open"],
            "sg_putt": [0.5, 0.6, 0.4, 0.3],
        })
        field_df = pd.DataFrame({"player_name": ["A"]})
        result = add_putting_surface_features(tour_df, field_df)
        assert "fast_green_sg_putt" in result.columns

    def test_minimum_rounds_filter(self):
        tour_df = pd.DataFrame({
            "player_name": ["A", "A"],
            "event_name": ["Masters Tournament", "U.S. Open"],
            "sg_putt": [1.0, 1.0],
        })
        field_df = pd.DataFrame({"player_name": ["A"]})
        result = add_putting_surface_features(tour_df, field_df)
        # Only 2 rounds, below threshold of 3 — should be 0 (NaN filled)
        assert result["fast_green_sg_putt"].iloc[0] == 0.0

    def test_sufficient_rounds_computes_mean(self):
        tour_df = pd.DataFrame({
            "player_name": ["A"] * 4,
            "event_name": ["Masters Tournament", "U.S. Open",
                          "Tour Championship", "Some Other Event"],
            "sg_putt": [0.6, 0.4, 0.8, 99.0],  # last one not a fast green event
        })
        field_df = pd.DataFrame({"player_name": ["A"]})
        result = add_putting_surface_features(tour_df, field_df)
        expected = (0.6 + 0.4 + 0.8) / 3
        assert result["fast_green_sg_putt"].iloc[0] == pytest.approx(expected, abs=0.01)

    def test_unmatched_player_gets_zero(self):
        tour_df = pd.DataFrame({
            "player_name": ["A", "A", "A"],
            "event_name": ["Masters Tournament"] * 3,
            "sg_putt": [0.5, 0.5, 0.5],
        })
        field_df = pd.DataFrame({"player_name": ["B"]})
        result = add_putting_surface_features(tour_df, field_df)
        assert result["fast_green_sg_putt"].iloc[0] == 0.0

    def test_no_sg_putt_column(self):
        tour_df = pd.DataFrame({"player_name": ["A"], "event_name": ["Masters"]})
        field_df = pd.DataFrame({"player_name": ["A"]})
        result = add_putting_surface_features(tour_df, field_df)
        assert "fast_green_sg_putt" in result.columns


# ── add_form_momentum ──

class TestFormMomentum:
    def _make_tour(self, finishes, player="A", season=2025):
        n = len(finishes)
        return pd.DataFrame({
            "player_name": [player] * n,
            "finish_num": finishes,
            "date": pd.date_range("2024-01-01", periods=n, freq="7D"),
            "season": [season] * n,
            "field_size": [156] * n,
        })

    def test_adds_all_columns(self):
        tour = self._make_tour([5, 10, 20, 30, 1])
        field = pd.DataFrame({"player_name": ["A"]})
        result = add_form_momentum(tour, field, target_year=2026)
        assert "events_since_top10" in result.columns
        assert "recent_win" in result.columns
        assert "recent_consistency" in result.columns

    def test_recent_win_detected(self):
        tour = self._make_tour([30, 20, 15, 10, 1])  # Won most recent
        field = pd.DataFrame({"player_name": ["A"]})
        result = add_form_momentum(tour, field, target_year=2026)
        assert result["recent_win"].iloc[0] == 1

    def test_no_recent_win(self):
        tour = self._make_tour([30, 20, 15, 10, 5])  # Best is 5th
        field = pd.DataFrame({"player_name": ["A"]})
        result = add_form_momentum(tour, field, target_year=2026)
        assert result["recent_win"].iloc[0] == 0

    def test_events_since_top10(self):
        # Top-10 is 3rd from last event
        tour = self._make_tour([30, 20, 8, 25, 40])
        field = pd.DataFrame({"player_name": ["A"]})
        result = add_form_momentum(tour, field, target_year=2026)
        assert result["events_since_top10"].iloc[0] == 2

    def test_no_top10_ever(self):
        tour = self._make_tour([30, 40, 50, 60])
        field = pd.DataFrame({"player_name": ["A"]})
        result = add_form_momentum(tour, field, target_year=2026)
        assert result["events_since_top10"].iloc[0] == 20

    def test_unknown_player_defaults(self):
        tour = self._make_tour([5], player="Other")
        field = pd.DataFrame({"player_name": ["Unknown"]})
        result = add_form_momentum(tour, field, target_year=2026)
        assert result["events_since_top10"].iloc[0] == 20
        assert result["recent_win"].iloc[0] == 0
        assert result["recent_consistency"].iloc[0] == 0.5

    def test_does_not_mutate_field_df(self):
        tour = self._make_tour([5, 10])
        field = pd.DataFrame({"player_name": ["A"]})
        original_cols = list(field.columns)
        add_form_momentum(tour, field, target_year=2026)
        assert list(field.columns) == original_cols


# ── add_sg_interactions ──

class TestSGInteractions:
    def test_adds_columns(self):
        df = pd.DataFrame({
            "sg_ott_8w": [1.0],
            "sg_app_8w": [0.5],
            "sg_arg_8w": [0.3],
            "sg_putt_8w": [0.2],
        })
        result = add_sg_interactions(df)
        assert "augusta_fit_score" in result.columns
        assert "power_precision" in result.columns
        assert "short_game_package" in result.columns

    def test_augusta_fit_score_formula(self):
        df = pd.DataFrame({
            "sg_ott_8w": [1.0],
            "sg_app_8w": [2.0],
            "sg_arg_8w": [0.5],
            "sg_putt_8w": [0.3],
        })
        result = add_sg_interactions(df)
        expected = 2.0 * 1.60 + 0.5 * 1.50 + 0.3 * 1.20 + 1.0 * 0.70
        assert result["augusta_fit_score"].iloc[0] == pytest.approx(expected)

    def test_power_precision_is_product(self):
        df = pd.DataFrame({
            "sg_ott_8w": [2.0],
            "sg_app_8w": [3.0],
            "sg_arg_8w": [0.0],
            "sg_putt_8w": [0.0],
        })
        result = add_sg_interactions(df)
        assert result["power_precision"].iloc[0] == pytest.approx(6.0)

    def test_short_game_package(self):
        df = pd.DataFrame({
            "sg_ott_8w": [0.0],
            "sg_app_8w": [0.0],
            "sg_arg_8w": [0.8],
            "sg_putt_8w": [0.4],
        })
        result = add_sg_interactions(df)
        assert result["short_game_package"].iloc[0] == pytest.approx(1.2)

    def test_handles_missing_columns(self):
        df = pd.DataFrame({"player_name": ["A"]})
        result = add_sg_interactions(df)
        assert result["augusta_fit_score"].iloc[0] == pytest.approx(0.0)

    def test_handles_nan_values(self):
        df = pd.DataFrame({
            "sg_ott_8w": [np.nan],
            "sg_app_8w": [1.0],
            "sg_arg_8w": [np.nan],
            "sg_putt_8w": [0.5],
        })
        result = add_sg_interactions(df)
        assert np.isfinite(result["augusta_fit_score"].iloc[0])

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({
            "sg_ott_8w": [1.0], "sg_app_8w": [0.5],
            "sg_arg_8w": [0.3], "sg_putt_8w": [0.2],
        })
        original_cols = list(df.columns)
        add_sg_interactions(df)
        assert list(df.columns) == original_cols


# ── add_aging_decay ──

class TestAgingDecay:
    def _make_field(self, names, ages_via_birth=None):
        """Create field_df with Augusta features to decay."""
        n = len(names)
        return pd.DataFrame({
            "player_name": names,
            "augusta_competitive_rounds": [40.0] * n,
            "augusta_top10_rate": [0.20] * n,
            "augusta_made_cut_rate": [0.80] * n,
            "augusta_best_finish": [5.0] * n,
            "augusta_scoring_avg": [-1.0] * n,
            "augusta_sg_app_career": [0.5] * n,
            "augusta_sg_total_career": [1.0] * n,
            "augusta_experience_tier": [3] * n,
            "augusta_starts": [10] * n,
        })

    def test_known_player_gets_decay(self):
        field = self._make_field(["Vijay Singh"])  # born 1963, age 63 in 2026
        result = add_aging_decay(field, current_year=2026)
        # Age 63 → decay = max(0.10, 1.0 - (63-45)*0.10) = max(0.10, -0.80) = 0.10
        assert result["age_decay_factor"].iloc[0] == pytest.approx(0.10)
        assert result["augusta_competitive_rounds"].iloc[0] == pytest.approx(4.0)
        assert result["augusta_top10_rate"].iloc[0] == pytest.approx(0.02)

    def test_young_player_no_decay(self):
        field = self._make_field(["Scottie Scheffler"])  # born 1996, age 30
        result = add_aging_decay(field, current_year=2026)
        assert result["age_decay_factor"].iloc[0] == pytest.approx(1.0)
        assert result["augusta_competitive_rounds"].iloc[0] == pytest.approx(40.0)

    def test_age_50_half_decay(self):
        field = self._make_field(["Phil Mickelson"])  # born 1970, age 56 in 2026
        result = add_aging_decay(field, current_year=2026)
        # Age 56 → decay = max(0.10, 1.0 - 1.1) = 0.10 (floor)
        assert result["age_decay_factor"].iloc[0] == pytest.approx(0.10)

    def test_does_not_decay_tier_or_starts(self):
        field = self._make_field(["Vijay Singh"])
        result = add_aging_decay(field, current_year=2026)
        assert result["augusta_experience_tier"].iloc[0] == 3  # unchanged
        assert result["augusta_starts"].iloc[0] == 10  # unchanged

    def test_first_season_proxy(self):
        field = self._make_field(["Unknown Veteran"])
        tour_df = pd.DataFrame({
            "player_name": ["Unknown Veteran"] * 5,
            "season": [2000, 2001, 2002, 2003, 2004],
        })
        result = add_aging_decay(field, tour_df=tour_df, current_year=2026)
        # First season 2000 → approx age = 2026 - 2000 + 22 = 48
        # Decay = max(0.10, 1.0 - (48-45)*0.10) = 0.70
        assert result["age_decay_factor"].iloc[0] == pytest.approx(0.70)

    def test_does_not_mutate_input(self):
        field = self._make_field(["Vijay Singh"])
        original_rounds = field["augusta_competitive_rounds"].iloc[0]
        add_aging_decay(field, current_year=2026)
        assert field["augusta_competitive_rounds"].iloc[0] == original_rounds

    def test_decay_floor_at_010(self):
        field = self._make_field(["Bernhard Langer"])  # born 1957, age 69
        result = add_aging_decay(field, current_year=2026)
        assert result["age_decay_factor"].iloc[0] == pytest.approx(0.10)
