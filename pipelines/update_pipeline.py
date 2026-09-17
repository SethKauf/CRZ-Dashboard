import pandas as pd
from model_utils.prepare_data import run_prepare_data
from model_utils.lstm_model import run_model_update
from model_utils.generate_forecast import run_forecast_generation
from model_utils.evaluate_forecasts import run_forecast_evaluation


def main():

    print("\n========================================")
    print("STEP 1: PREPARE DATA")
    print("========================================")

    data = run_prepare_data()

    print("\n========================================")
    print("STEP 2: UPDATE LSTM MODEL")
    print("========================================")

    model_result = run_model_update()

    print("\n========================================")
    print("STEP 3: GENERATE FORECAST")
    print("========================================")

    forecast_result = run_forecast_generation()

    print(
        f"\nModel updated: "
        f"{model_result['updated']}"
    )

    print(
        f"Forecast rows: "
        f"{forecast_result['rows']:,}"
    )

    print(
        f"Forecast saved to: "
        f"{forecast_result['forecast_path']}"
    )

    print(
        f"Archive saved to: "
        f"{forecast_result['archive_path']}"
    )

    print("\n========================================")
    print("STEP 4: EVALUATE PREVIOUS FORECAST")
    print("========================================")

    evaluation_result = run_forecast_evaluation()

    evaluation_summary = evaluation_result["summary"].copy()

    evaluation_summary["forecast_generated_date_dt"] = pd.to_datetime(
        evaluation_summary["forecast_generated_date"].astype(str),
        format="%Y%m%d"
    )

    most_recent_evaluation = (
        evaluation_summary
        .sort_values("forecast_generated_date_dt")
        .iloc[-1]
    )

    forecast_date = (
        most_recent_evaluation["forecast_generated_date_dt"]
        .strftime("%Y-%b-%d")
    )

    overall_mae = most_recent_evaluation["overall_mae"]
    mae_pct = most_recent_evaluation["mae_pct_of_mean"]

    print(
        "\nEvaluation completed. "
        f"Average overall MAE of previous forecast "
        f"({forecast_date}) was {overall_mae:.1f} vehicles "
        f"({mae_pct:.1f}% of mean traffic volume)."
    )

    print("\n========================================")
    print("PIPELINE COMPLETE")
    print("========================================")



if __name__ == "__main__":
    main()