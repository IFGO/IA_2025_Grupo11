import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging

from core.data_load import download_crypto_data
from core.features import generate_features
from core.analysis import (
    basic_statistics, correlation_analysis,
    plot_boxplot, plot_histogram, plot_price_over_time
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Executa o pipeline de previsão de criptomoedas."
    )

    parser.add_argument(
        "--crypto",
        type=str,
        nargs="+",
        required=True,
        help="Lista de criptomoedas (ex: BTCUSDT ETHUSDT DOGEUSDT...)"
    )

    parser.add_argument(
        "--feature-set", 
        type=str, 
        default="basic", 
        help="Conjunto de features: basic, rolling, technical ou all"
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

    args = parser.parse_args()

    logger.info(
        f"Rodando pipeline para {args.crypto} usando modelo {args.model} "
        f"com {args.kfolds} folds..."
    )

    for symbol in args.crypto:
        logger.info(f"\nProcessando criptomoeda: {symbol}")

        df = download_crypto_data(symbol)

        if df is None:
            logger.error(f"Falha ao carregar os dados de {symbol}.")
            continue

        features_df = generate_features(df, set_type=args.feature_set, include_original=True)
        features_df["target"] = features_df["Close"].shift(-1)  # Target: preço de amanhã
        features_df.dropna(inplace=True)

        logger.info(f"Features geradas para {symbol}:\n{features_df.head()}")

        # Salva as features para cada cripto em CSV (opcional)
        features_df.to_csv(f"backend/data/features_{symbol}.csv", index=False)
        logger.info(f"Features salvas em backend/data/features_{symbol}.csv")

        # Análises estatísticas e gráficos.
        basic_statistics(df, symbol)
        plot_boxplot(df, symbol)
        plot_histogram(df, symbol)
        plot_price_over_time(df, symbol)

        # (Futuro) Treinamento de modelos também será incluído aqui.

    dfs_dict = {symbol: download_crypto_data(symbol) for symbol in args.crypto}
    correlation_analysis(dfs_dict)

if __name__ == "__main__":
    main()
