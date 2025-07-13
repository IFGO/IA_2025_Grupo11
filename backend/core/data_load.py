import pandas as pd
import logging
import os
import requests
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_crypto_data(
        symbol: str,
        base_url: str = "https://www.cryptodatadownload.com/cdd/",
        save_dir: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Baixa e carrega o CSV da criptomoeda especificada.

    Args:
        symbol: Símbolo da criptomoeda (ex: 'BTCUSDT')
        base_url: URL base do dataset da CryptoDataDownload.
        save_dir: Diretório onde o arquivo CSV será salvo (útil para testes).

    Returns:
        DataFrame com os dados de preço ou None se falhar.
    """
    url = f"{base_url}Binance_{symbol}_d.csv"

    if save_dir:
        local_file = os.path.join(save_dir, f"{symbol}.csv")
    else:
        local_file = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "data", f"{symbol}.csv"
        )
        local_file = os.path.abspath(local_file)

    try:
        logger.info(f"Baixando dados de {url}")
        response = requests.get(url)
        response.raise_for_status()

        os.makedirs(os.path.dirname(local_file), exist_ok=True)

        with open(local_file, "wb") as f:
            f.write(response.content)

        # Leitura do CSV
        df = pd.read_csv(local_file, skiprows=1, encoding="utf-8-sig")
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values("Date", inplace=True)

        # Renomeia automaticamente a coluna de volume principal (exceto Volume USDT)
        volume_cols = [
            col for col in df.columns
            if col.startswith("Volume ") and col != "Volume USDT"
        ]
        if volume_cols:
            df = df.rename(columns={volume_cols[0]: "Volume"})
            logger.info(f"Coluna de volume '{volume_cols[0]}' renomeada para 'Volume'")

        logger.info(f"{len(df)} registros carregados para {symbol}")
        return df

    except Exception as e:
        logger.error(f"Erro ao baixar ou processar dados de {symbol}: {e}")
        return None
