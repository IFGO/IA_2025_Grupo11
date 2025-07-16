import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calcula o RSI (Relative Strength Index) de uma série."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    # Evita divisão por zero adicionando um pequeno valor ao denominador
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calcula o MACD (Moving Average Convergence Divergence)."""
    exp1 = series.ewm(span=fast_period, adjust=False).mean()
    exp2 = series.ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def compute_bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calcula as Bandas de Bollinger."""
    mavg = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = mavg + (std * num_std)
    lower = mavg - (std * num_std)
    return mavg, upper, lower, std

def compute_stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> tuple[pd.Series, pd.Series]:
    """Calcula o Oscilador Estocástico (%K e %D)."""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    
    # Evita divisão por zero
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent, d_percent

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calcula o ADX (Average Directional Index)."""
    # True Range (TR)
    high_low = high - low
    high_prev_close = abs(high - close.shift(1))
    low_prev_close = abs(low - close.shift(1))
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    atr = tr.ewm(span=window, adjust=False).mean()

    # Directional Movement (DM)
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    plus_dm = plus_dm.ewm(span=window, adjust=False).mean()
    minus_dm = abs(minus_dm).ewm(span=window, adjust=False).mean()

    # Directional Index (DI)
    plus_di = 100 * (plus_dm / (atr + 1e-10))
    minus_di = 100 * (minus_dm / (atr + 1e-10))

    # Directional Movement Index (DX)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))

    # Average Directional Index (ADX)
    adx = dx.ewm(span=window, adjust=False).mean()
    return adx

def generate_features(df: pd.DataFrame, set_type: str = "basic", include_original: bool = True) -> pd.DataFrame:
    """
    Gera features para o DataFrame de preços.

    Args:
        df: DataFrame com colunas ['Close', 'High', 'Low', 'Open', 'Volume', 'Volume USDT', 'Tradecount'].
            É importante que 'High', 'Low', 'Open' existam para algumas features.
        set_type: Tipo de features ('basic', 'rolling', 'technical', 'all').
        include_original: Se True, inclui também colunas originais úteis.

    Returns:
        DataFrame com as features e, opcionalmente, colunas originais.
    """
    df = df.copy()

    # Garante que 'Date' é datetime e define como índice para operações de séries temporais
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("Índice do DataFrame não é DatetimeIndex. Algumas features de tempo podem falhar.")


    features = pd.DataFrame(index=df.index)

    # Inclui colunas originais úteis
    if include_original:
        features["Close"] = df["Close"]
        features["Volume"] = df["Volume"]
        if "Tradecount" in df.columns:
            features["Tradecount"] = df["Tradecount"]
        if "Volume USDT" in df.columns:
            features["Volume USDT"] = df["Volume USDT"]

    # Features geradas
    if set_type in ["basic", "rolling", "technical", "all"]:

        # Transformação Logarítmica do Preço (para features)
        features["log_close"] = np.log(df["Close"])
        if 'High' in df.columns and 'Low' in df.columns and 'Open' in df.columns:
            features["log_high"] = np.log(df["High"])
            features["log_low"] = np.log(df["Low"])
            features["log_open"] = np.log(df["Open"])

        # Features de Retorno e Log Retorno
        features["daily_return"] = df["Close"].pct_change()
        features["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        if all(col in df.columns for col in ['High', 'Low', 'Open', 'Close']):
            features["log_ret_hl"] = np.log(df["High"] / df["Low"])
            features["log_ret_oc"] = np.log(df["Open"] / df["Close"])
            features["log_ret_ho"] = np.log(df["High"] / df["Open"])
            features["log_ret_lc"] = np.log(df["Low"] / df["Close"])

        # Features de Variação de Preço
        if all(col in df.columns for col in ['High', 'Low', 'Open', 'Close']):
            features["high_low_diff"] = df["High"] - df["Low"]
            features["close_open_diff"] = df["Close"] - df["Open"]
            features["high_close_diff"] = df["High"] - df["Close"]
            features["low_close_diff"] = df["Low"] - df["Close"]
            features["open_close_ratio"] = df["Open"] / df["Close"]
            features["high_low_ratio"] = df["High"] / df["Low"]
            features["range_ratio"] = (df["High"] - df["Low"]) / df["Close"]

        # Features de Volume
        features["volume_change"] = df["Volume"].pct_change()
        features["log_volume"] = np.log(df["Volume"] + 1e-10)
        features["volume_sma_7"] = df["Volume"].rolling(window=7).mean()
        features["volume_ema_14"] = df["Volume"].ewm(span=14, adjust=False).mean()
        features["positive_volume"] = df["Volume"].where(df["Close"].diff() > 0, 0)
        features["negative_volume"] = df["Volume"].where(df["Close"].diff() < 0, 0)
        features["positive_negative_volume_ratio"] = features["positive_volume"] / (features["negative_volume"].abs() + 1e-10)


        # Features de Médias Móveis e Volatilidade (Rolling)
        if set_type in ["rolling", "all"]:
            windows_rolling = [7, 14, 20, 50, 100, 200]
            for window in windows_rolling:
                features[f"sma_{window}"] = df["Close"].rolling(window=window).mean()
                features[f"ema_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()
                features[f"volatility_{window}"] = df["Close"].rolling(window=window).std()
                features[f"rolling_max_{window}"] = df["High"].rolling(window=window).max()
                features[f"rolling_min_{window}"] = df["Low"].rolling(window=window).min()
                features[f"price_vs_sma_{window}"] = (df["Close"] / features[f"sma_{window}"]) - 1
                
                # Certifica-se que volume_sma_7 existe antes de usar
                if f"volume_sma_7" in features.columns:
                     features[f"volume_vs_sma_{window}"] = (df["Volume"] / features[f"volume_sma_7"]) - 1
                else:
                    logger.warning(f"volume_sma_7 não encontrado para 'volume_vs_sma_{window}'.")


        # Features Técnicas (Indicadores de Trading Comuns)
        if set_type in ["technical", "all"]:
            # RSI
            features["rsi_14"] = compute_rsi(df["Close"])
            features["rsi_7"] = compute_rsi(df["Close"], window=7)
            features["rsi_21"] = compute_rsi(df["Close"], window=21)

            # MACD
            macd, macd_signal, macd_hist = compute_macd(df["Close"])
            features["macd"] = macd
            features["macd_signal"] = macd_signal
            features["macd_hist"] = macd_hist
            features["macd_cross_signal"] = ((macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1))).astype(int)

            # Bollinger Bands
            mavg_bb, upper_bb, lower_bb, std_bb = compute_bollinger_bands(df["Close"])
            features["bollinger_mavg"] = mavg_bb
            features["bollinger_upper"] = upper_bb
            features["bollinger_lower"] = lower_bb
            features["bollinger_std"] = std_bb
            features["bollinger_band_width"] = (upper_bb - lower_bb) / mavg_bb
            features["price_position_bb"] = (df["Close"] - lower_bb) / (upper_bb - lower_bb + 1e-10)

            # ATR
            if all(col in df.columns for col in ['High', 'Low', 'Close']):
                high_low = df['High'] - df['Low']
                high_prev_close = np.abs(df['High'] - df['Close'].shift(1))
                low_prev_close = np.abs(df['Low'] - df['Close'].shift(1))
                tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
                features['atr_14'] = tr.ewm(span=14, adjust=False).mean()
                features['atr_ratio_close'] = features['atr_14'] / df['Close']
            else:
                logger.warning("Aviso: Colunas 'High', 'Low' ou 'Close' não encontradas para calcular ATR.")

            # OBV
            features["obv"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
            features["obv_ema_20"] = features["obv"].ewm(span=20, adjust=False).mean()
            features["obv_slope"] = features["obv"].diff(periods=5)

            # Stochastic Oscillator
            if all(col in df.columns for col in ['High', 'Low', 'Close']):
                k_stoch, d_stoch = compute_stochastic_oscillator(df['High'], df['Low'], df['Close'])
                features["stoch_k"] = k_stoch
                features["stoch_d"] = d_stoch
                features["stoch_cross"] = ((k_stoch > d_stoch) & (k_stoch.shift(1) <= d_stoch.shift(1))).astype(int)
            else:
                logger.warning("Aviso: Colunas 'High', 'Low' ou 'Close' não encontradas para calcular Stochastic Oscillator.")

            # ADX
            if all(col in df.columns for col in ['High', 'Low', 'Close']):
                features["adx_14"] = compute_adx(df['High'], df['Low'], df['Close'])
            else:
                logger.warning("Aviso: Colunas 'High', 'Low' ou 'Close' não encontradas para calcular ADX.")
            
            # Aroon Indicator (CORRIGIDO)
            if 'High' in df.columns and 'Low' in df.columns:
                aroon_window = 25
                # Certifica-se que o índice é DatetimeIndex para .dt.days
                if not isinstance(df.index, pd.DatetimeIndex):
                    logger.warning("Índice não é DatetimeIndex, Aroon pode não ser calculado corretamente.")
                
                # Calcula dias desde o máximo/mínimo dentro da janela
                days_since_high = df['High'].rolling(window=aroon_window).apply(lambda x: (aroon_window - 1 - x.argmax()), raw=False)
                days_since_low = df['Low'].rolling(window=aroon_window).apply(lambda x: (aroon_window - 1 - x.argmin()), raw=False)

                features[f"aroon_up_{aroon_window}"] = 100 * (aroon_window - days_since_high) / aroon_window
                features[f"aroon_down_{aroon_window}"] = 100 * (aroon_window - days_since_low) / aroon_window
                features[f"aroon_oscillator_{aroon_window}"] = features[f"aroon_up_{aroon_window}"] - features[f"aroon_down_{aroon_window}"]
            else:
                logger.warning("Aviso: Colunas 'High' ou 'Low' não encontradas para calcular Aroon Indicator.")

            # Commodity Channel Index (CCI)
            if all(col in df.columns for col in ['High', 'Low', 'Close']):
                cci_window = 20
                typical_price = (df['High'] + df['Low'] + df['Close']) / 3
                cci_sma = typical_price.rolling(window=cci_window).mean()
                cci_md = typical_price.rolling(window=cci_window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
                features[f"cci_{cci_window}"] = (typical_price - cci_sma) / (0.015 * cci_md + 1e-10)
            else:
                logger.warning("Aviso: Colunas 'High', 'Low' ou 'Close' não encontradas para calcular CCI.")


        # Lagged Features
        lag_windows = [1, 2, 3, 5, 7, 14, 21, 30, 60]
        for i in lag_windows:
            features[f"close_lag_{i}"] = df["Close"].shift(i)
            features[f"volume_lag_{i}"] = df["Volume"].shift(i)
            features[f"daily_return_lag_{i}"] = features["daily_return"].shift(i)
            features[f"log_return_lag_{i}"] = features["log_return"].shift(i)
            
            if set_type in ["technical", "all"]:
                if "rsi_14" in features.columns: features[f"rsi_14_lag_{i}"] = features["rsi_14"].shift(i)
                if "macd" in features.columns: features[f"macd_lag_{i}"] = features["macd"].shift(i)
                if "stoch_k" in features.columns: features[f"stoch_k_lag_{i}"] = features["stoch_k"].shift(i)
                if "bollinger_band_width" in features.columns: features[f"bollinger_band_width_lag_{i}"] = features["bollinger_band_width"].shift(i)
                if "adx_14" in features.columns: features[f"adx_14_lag_{i}"] = features["adx_14"].shift(i) # Novo lag
                if "cci_20" in features.columns: features[f"cci_20_lag_{i}"] = features["cci_20"].shift(i) # Novo lag

        # Features de Tempo
        if isinstance(df.index, pd.DatetimeIndex):
            features["day_of_week"] = df.index.dayofweek
            features["month"] = df.index.month
            features["day_of_year"] = df.index.dayofyear
            features["week_of_year"] = df.index.isocalendar().week.astype(int)
            features["is_month_start"] = df.index.is_month_start.astype(int)
            features["is_month_end"] = df.index.is_month_end.astype(int)
            features["is_quarter_start"] = df.index.is_quarter_start.astype(int)
            features["is_quarter_end"] = df.index.is_quarter_end.astype(int)
            features["day_of_month"] = df.index.day
            features["year"] = df.index.year
            features["quarter"] = df.index.quarter
            features["day_of_year_sin"] = np.sin(2 * np.pi * features["day_of_year"] / 365) # Sazonalidade (Senoidal)
            features["day_of_year_cos"] = np.cos(2 * np.pi * features["day_of_year"] / 365)
            features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
            features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)
        else:
            logger.warning("Índice do DataFrame não é DatetimeIndex. Features de tempo não geradas.")

    # Remove linhas com NaNs gerados por janelas móveis ou shifts.
    features.dropna(inplace=True)

    return features