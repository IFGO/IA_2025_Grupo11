import pandas as pd
from core.features import generate_features


def test_generate_features_basic():
    df = pd.DataFrame({
        "Close": [100, 102, 101, 105, 107],
        "Volume": [1000, 1100, 1050, 1150, 1200],
        "tradecount": [10, 12, 11, 14, 13]
    })
    features_df = generate_features(df, set_type="basic")
    assert not features_df.empty
    assert "Close" in features_df.columns
