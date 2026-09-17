from pathlib import Path

import numpy as np
import pandas as pd

# set pathing
DATA_PATH = "data/modeling_data.csv"

FORECAST_ARCHIVE_DIR = (
    "data/forecasts/archive"
)

EVALUATION_DIR = (
    "data/forecast_evaluations"
)

COMPARISON_PATH = (
    "data/forecast_evaluations/"
    "forecast_actual_comparison.csv"
)

SUMMARY_PATH = (
    "data/forecast_evaluations/"
    "evaluation_summary.csv"
)

GROUP_PATH = (
    "data/forecast_evaluations/"
    "evaluation_by_group.csv"
)

def run_forecast_evaluation(
        data_path=DATA_PATH,
        forecast_archive_dir=FORECAST_ARCHIVE_DIR
):
    print(
        "\n================================"
    )
    print(
        "Evaluating previous forecasts..."
    )
    print(
        "================================"
    )

    Path(
        EVALUATION_DIR
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    # Load in real data
    actual = pd.read_csv(
        data_path
    )

    actual['toll_10_minute_block'] = (
        pd.to_datetime(
            actual['toll_10_minute_block']
        )
    )

    actual = actual[
        [
            'toll_10_minute_block',
            'region_id',
            'group_id',
            'traffic_volume'
        ]
    ].rename(
        columns={
            'traffic_volume':'actual_volume'
        }
    )

    # load existing comparisons
    if Path(COMPARISON_PATH).exists():

        existing = pd.read_csv(
            COMPARISON_PATH
        )

        existing[
            'toll_10_minute_block'
        ] = pd.to_datetime(
            existing['toll_10_minute_block']
        )

    else:

        existing = pd.DataFrame()

    # archived forecasts
    forecast_files = sorted(
        Path(
            forecast_archive_dir
        ).glob(
            'traffic_forecast_*.csv'
        )
    )

    print(
        f"Found {len(forecast_files)} archived forecast files."
    )

    new_comparisons = []

    # compare forecasts
    for forecast_file in forecast_files:

        print(
            "\nChecking:",
            forecast_file.name
        )

        forecast = pd.read_csv(
            forecast_file
        )

        forecast[
            'toll_10_minute_block'
        ] = pd.to_datetime(
            forecast['toll_10_minute_block']
        )

        forecast_origin = (
            forecast['toll_10_minute_block'].min() - pd.Timedelta(minutes=10)
        )

        comparison = forecast[
            [
                'toll_10_minute_block',
                'region_id',
                'group_id',
                'predicted_volume'
            ]
        ].merge(
            actual,
            on=[
                'toll_10_minute_block',
                'region_id',
                'group_id'
            ],
            how='inner'
        )

        if comparison.empty:

            print(
                "No data available yet."
            )

            continue

        # identify forecast
        forecast_date = (
            forecast_file
            .stem
            .replace(
                'traffic_forecast_',
                ''
            )
        )

        comparison[
            'forecast_generated_date'
        ] = forecast_date

        # calculate forecast horizon
        comparison['forecast_horizon_hours'] = (
            (
                comparison['toll_10_minute_block'] - forecast_origin
            )
            .dt.total_seconds() / 3600
        )

        comparison['forecast_horizon_days'] = (
            comparison['forecast_horizon_hours'] / 24
        )

        # calculate error rate
        comparison['error'] = (
            comparison['predicted_volume']
            - comparison['actual_volume']
        )

        comparison['absolute_error'] = (
            comparison['error'].abs()
        )

        new_comparisons.append(
            comparison
        )

    # combine comparisons
    if not new_comparisons:

        print(
            '\nNo forecast/actual overlap found.'
        )

        return {
            'new_rows':0
        }

    new_comparisons = pd.concat(
        new_comparisons,
        ignore_index=True
    )

    if not existing.empty:

        comparison_history = pd.concat(
            [
                existing,
                new_comparisons
            ],
            ignore_index=True
        )

    else:

        comparison_history = (
            new_comparisons.copy()
        )

    # prevent duplicate evaluations

    comparison_history = (
        comparison_history
        .drop_duplicates(
            subset=[
                'forecast_generated_date',
                'toll_10_minute_block',
                'group_id'
            ],
            keep='last'
        )
        .sort_values(
            [
                'forecast_generated_date',
                'toll_10_minute_block',
                'group_id'
            ]
        )
        .reset_index(drop=True)
    )

    comparison_history.to_csv(
        COMPARISON_PATH,
        index=False
    )

    overall_summary = (
        comparison_history
        .groupby(
            'forecast_generated_date',
            as_index=False
        )
        .agg(
            overall_mae=(
                'absolute_error',
                'mean'
            ),

            average_error=(
                'error',
                'mean'
            ),

            mean_actual_volume=(
                'actual_volume',
                'mean'
            ),

            observations=(
                'actual_volume',
                'size'
            ),

            actual_through=(
                'toll_10_minute_block',
                'max'
            )
        )
    )

    overall_summary['mae_pct_of_mean'] = (
        overall_summary['overall_mae'] / overall_summary['mean_actual_volume'] * 100
    )

    group_summary = (
        comparison_history
        .groupby(
            [
                'forecast_generated_date',
                'region_id',
                'group_id'
            ],
            as_index=False
        )
        .agg(
            mae=(
                'absolute_error',
                'mean'
            ),

            average_error=(
                'error',
                'mean'
            ),

            mean_actual_volume=(
                'actual_volume',
                'mean'
            ),

            observations=(
                'actual_volume',
                'size'
            )
        )
    )

    group_summary[
        'mae_pct_of_mean'
    ] = (
        (group_summary['mae'] / group_summary['mean_actual_volume']) * 100
    )

    # best predicted group
    best_groups = (
        group_summary.loc[
            group_summary
            .groupby('forecast_generated_date')['mae']
            .idxmin(),
            [
                'forecast_generated_date',
                'group_id',
                'mae'
            ]
        ]
        .rename(
            columns={
                'group_id': 'best_group_id',
                'mae': 'best_group_mae'
            }
        )
    )

    # worst predicted group
    worst_groups = (
        group_summary.loc[
            group_summary
            .groupby('forecast_generated_date')['mae']
            .idxmax(),
            [
                'forecast_generated_date',
                'group_id',
                'mae'
            ]
        ]
        .rename(
            columns={
                'group_id': 'worst_group_id',
                'mae': 'worst_group_mae'
            }
        )
    )

    overall_summary = (
        overall_summary
        .merge(
            best_groups,
            on='forecast_generated_date',
            how='left'
        )
        .merge(
            worst_groups,
            on='forecast_generated_date',
            how='left'
        )
    )

    overall_summary.to_csv(
        SUMMARY_PATH,
        index=False
    )

    group_summary.to_csv(
        GROUP_PATH,
        index=False
    )

    print(
        f"\nSaved comparison history to:"
        f"\n{COMPARISON_PATH}"
    )

    print(
        f"\nOverall evaluations:"
        f"\n{SUMMARY_PATH}"
    )

    print(
        f"\nGroup evaluations:"
        f"\n{GROUP_PATH}"
    )

    return {
        "new_rows":
            len(new_comparisons),

        "comparison":
            comparison_history,

        "summary":
            overall_summary,

        "by_group":
            group_summary
    }

if __name__ == "__main__":
    run_forecast_evaluation()