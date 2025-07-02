import pandas as pd
import logging
import os
import requests
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_crypto_data(
        symbol: str,
        base_url: str = "https://www.cryptodatadownload.com/cdd/"
) -> Optional[pd.DataFrame]:
    """
    Baixa e carrega o CSV da criptomoeda especificada.
    
    Args:
        symbol: Símbolo da criptomoeda (ex: 'BTC_USDT')
        base_url: URL base do dataset da CryptoDataDownload.

    Returns:
        DataFrame com os dados de preço ou None se falhar.
    """
    url = f"{base_url}Binance_{symbol}_d.csv"
    local_file = os.path.join(
        os.path.dirname(__file__), 
        "..", "..", 
        "data", f"{symbol}.csv"
    )
    local_file = os.path.abspath(local_file)

    try:
        logger.info(f"Baixando dados de {url}")
        response = requests.get(url)
        response.raise_for_status()

        os.makedirs(os.path.dirname(local_file), exist_ok=True)

        with open(local_file, "wb") as f:
            f.write(response.content)

        df = pd.read_csv(local_file, skiprows=1, encoding="utf-8-sig")
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values("Date", inplace=True)
        logger.info(f"{len(df)} registros carregados para {symbol}")
        return df

    except Exception as e:
        logger.error(f"Erro ao baixar ou processar dados de {symbol}: {e}")
        return None