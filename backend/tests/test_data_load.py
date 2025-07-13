import os
from core.data_load import download_crypto_data


def test_download_crypto_data(tmp_path):
    df = download_crypto_data("BTCUSDT", save_dir=tmp_path)
    assert df is not None
    assert not df.empty
    assert "Close" in df.columns
    assert "Date" in df.columns

    file_path = tmp_path / "BTCUSDT.csv"
    assert file_path.exists()


