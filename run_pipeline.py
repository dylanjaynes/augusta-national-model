#!/usr/bin/env python3
"""
Augusta National Model — Full Pipeline
Tasks 1-6: Data ingestion, feature engineering, backtesting
"""
import sys
import os

# Ensure we're in the project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from augusta_model.data.ingest import task1_pull_masters_sg, task2_scrape_historical_scores, task3_build_unified
from augusta_model.features.augusta_features import task4_augusta_history_features, task5_course_weights
from augusta_model.model.backtest import task6_backtest


def main():
    print("=" * 60)
    print("  AUGUSTA NATIONAL MODEL — FULL PIPELINE")
    print("=" * 60)

    # Task 1
    sg_df, rounds_df = task1_pull_masters_sg()

    # Task 2
    scores_df = task2_scrape_historical_scores()

    # Task 3
    unified_df = task3_build_unified(sg_df, scores_df)

    # Task 4
    features_df = task4_augusta_history_features(unified_df)

    # Task 5
    weights = task5_course_weights(unified_df)

    # Task 6
    metrics, bets = task6_backtest()

    print("\n" + "=" * 60)
    print("  ALL TASKS COMPLETE")
    print("=" * 60)

    return metrics, bets


if __name__ == "__main__":
    main()
