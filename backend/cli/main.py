import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging

from core.data_load import download_crypto_data



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Executa o pipeline de previsão de criptomoedas."
    )

    parser.add_argument(
        "--crypto",
        type=str,
        required=True,
        help="Símbolo da criptomoeda (ex: BTC, ETH)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="mlp", 
        help="Modelo preditivo: mlp, linear, poly"
    )
    parser.add_argument(
        "--kfolds", 
        type=int, 
        default=5, 
        help="Número de folds para validação cruzada"
    )
    parser.add_argument(
        "--feature-set", 
        type=str, 
        default="basic", 
        help="Conjunto de features"
    )

    args = parser.parse_args()

    logger.info(
        f"Rodando pipeline para {args.crypto}usando modelo {args.model} " 
        f"com {args.kfolds} folds..."
    )

    df = download_crypto_data(args.crypto)

    if df is not None:
        logger.info(f"\n{df.head()}")
    else:
        logger.error("Falha ao carregar os dados.")

if __name__ == "__main__":
    main()