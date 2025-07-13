import pandas as pd
from core.models import train_and_evaluate_model


def test_train_and_evaluate_model_linear():
    df = pd.DataFrame({
        "Close": [100, 102, 104, 106, 108, 110, 112],
        "Volume": [1000, 1050, 1100, 1150, 1200, 1250, 1300],
        "tradecount": [10, 11, 12, 13, 14, 15, 16],
    })
    df["target"] = df["Close"].shift(-1).fillna(method="ffill")
    results_df = train_and_evaluate_model(df, model_type="linear", kfolds=2)
    assert not results_df.empty
    assert all(col in results_df.columns for col in ["MSE", "MAE", "R2", "fold"])
