import pandas as pd
from typing import Optional


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calcula o RSI (Relative Strength Index) de uma série."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / (avg_loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def generate_features(df: pd.DataFrame, set_type: str = "basic", include_original: bool = True) -> pd.DataFrame:
    """
    Gera features para o DataFrame de preços.

    Args:
        df: DataFrame com colunas ['Close', 'Volume', 'Volume USDT', 'Tradecount'].
        set_type: Tipo de features ('basic', 'rolling', 'technical', 'all').
        include_original: Se True, inclui também colunas originais úteis.

    Returns:
        DataFrame com as features e, opcionalmente, colunas originais.
    """
    df = df.copy()

    features = pd.DataFrame(index=df.index)  # cria DataFrame vazio com mesmo índice

    # Inclui colunas originais úteis
    if include_original:
        features["Close"] = df["Close"]
        features["Volume"] = df["Volume"]  # Volume principal unificado
        features["Tradecount"] = df["tradecount"]
        if "Volume USDT" in df.columns:
            features["Volume USDT"] = df["Volume USDT"]  # Inclui só se existir

    # Features geradas
    if set_type in ["basic", "rolling", "technical", "all"]:
        if set_type in ["rolling", "all"]:
            features["sma_7"] = df["Close"].rolling(window=7).mean()
            features["volatility_7"] = df["Close"].rolling(window=7).std()
        if set_type in ["technical", "all"]:
            features["ema_14"] = df["Close"].ewm(span=14).mean()
            features["rsi_14"] = compute_rsi(df["Close"])

    # Remove linhas com NaNs gerados por janelas móveis
    features.dropna(inplace=True)

    return features
