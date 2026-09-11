from model_utils.prepare_data import run_prepare_data
from model_utils.lstm_model import run_model_update
from model_utils.generate_forecast import run_forecast_generation


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


    print("\n========================================")
    print("PIPELINE COMPLETE")
    print("========================================")

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


if __name__ == "__main__":
    main()