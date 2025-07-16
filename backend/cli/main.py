import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np

# Importações para otimização de hiperparâmetros e pipelines
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV # Adicionado RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso # Adicionado Ridge e Lasso para regularização
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # Adicionado GradientBoostingRegressor

# Ajusta o caminho do sistema para importar módulos do diretório 'core'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importações dos seus módulos existentes
from core.analysis import (
    basic_statistics, correlation_analysis,
    plot_boxplot, plot_histogram, plot_price_over_time
)
from core.anova import run_anova
from core.data_load import download_crypto_data
from core.features import generate_features
from core.hypothesis import test_return_hypothesis
from core.models import evaluate_model # Mantemos evaluate_model
from core.plots import plot_profit
from core.profit import simulate_profit
from core.report import generate_html_report


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Função Auxiliar para Otimização de Hiperparâmetros ---
def perform_hyperparameter_optimization(
    X: np.ndarray, y: np.ndarray, model_type: str, kfolds: int, use_random_search: bool = False
) -> Pipeline:
    """
    Realiza a otimização de hiperparâmetros usando GridSearchCV ou RandomizedSearchCV e TimeSeriesSplit.
    Retorna o melhor modelo treinado.

    Args:
        X: Features de entrada.
        y: Target (preço de amanhã).
        model_type: Tipo do modelo a ser otimizado ('mlp', 'linear', 'poly', 'random_forest', 'gradient_boosting').
        kfolds: Número de folds para validação cruzada.
        use_random_search: Se True, usa RandomizedSearchCV em vez de GridSearchCV.

    Returns:
        Um objeto Pipeline contendo o scaler e o melhor modelo treinado.
    """
    # Configura o TimeSeriesSplit para validação cruzada adequada para séries temporais
    tscv = TimeSeriesSplit(n_splits=kfolds)
    pipeline = None
    param_grid = {}

    if model_type == "mlp":
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(random_state=42, early_stopping=True, validation_fraction=0.1))
        ])
        param_grid = {
            'mlp__hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50), (200, 100)], # Mais opções
            'mlp__activation': ['relu', 'tanh'],
            'mlp__solver': ['adam', 'lbfgs'], # Incluído 'lbfgs'
            'mlp__alpha': [0.00001, 0.0001, 0.001, 0.01], # Mais opções de regularização
            'mlp__learning_rate_init': [0.001, 0.0005, 0.00001], # Para solver='adam'
            'mlp__max_iter': [1000, 2000, 3000] # Aumentado
        }
    elif model_type == "random_forest":
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(random_state=42, n_jobs=-1))
        ])
        param_grid = {
            'rf__n_estimators': [100, 200, 300, 500], # Mais árvores
            'rf__max_depth': [None, 10, 20, 30], # Mais profundidade
            'rf__min_samples_leaf': [1, 2, 4],
            'rf__min_samples_split': [2, 5, 10],
            'rf__max_features': ['sqrt', 'log2', 0.5, 0.7] # Mais opções para max_features
        }
    elif model_type == "gradient_boosting": # Novo modelo
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('gb', GradientBoostingRegressor(random_state=42))
        ])
        param_grid = {
            'gb__n_estimators': [100, 200, 300],
            'gb__learning_rate': [0.01, 0.1, 0.2],
            'gb__max_depth': [3, 5, 7],
            'gb__subsample': [0.7, 1.0]
        }
    elif model_type == "linear":
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('linear_reg', LinearRegression())
        ])
        param_grid = {} # Para LinearRegression simples
    elif model_type == "poly":
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('poly_features', PolynomialFeatures(include_bias=False)),
            ('linear_reg', LinearRegression())
        ])
        param_grid = {
            'poly_features__degree': [1, 2, 3] # Adicionado grau 3
        }
    else:
        raise ValueError(f"Modelo desconhecido para otimização: {model_type}. Escolha entre 'mlp', 'linear', 'poly', 'random_forest', 'gradient_boosting'.")

    if not pipeline:
        raise ValueError(f"Pipeline não definida para o modelo: {model_type}")

    logger.info(f"Iniciando {'RandomizedSearchCV' if use_random_search else 'GridSearchCV'} para o modelo {model_type}...")

    if param_grid:
        if use_random_search:
            search_cv = RandomizedSearchCV(
                pipeline,
                param_grid,
                n_iter=50, # Número de combinações para testar (ajustável)
                cv=tscv,
                scoring='r2',
                n_jobs=-1, # Usa todos os núcleos da CPU disponíveis
                verbose=1, # Maior verbosidade
                random_state=42
            )
        else:
            search_cv = GridSearchCV(
                pipeline,
                param_grid,
                cv=tscv,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
        search_cv.fit(X, y)
        logger.info(f"Melhores parâmetros para {model_type}: {search_cv.best_params_}")
        logger.info(f"Melhor score (R2) para {model_type}: {search_cv.best_score_:.4f}")
        return search_cv.best_estimator_
    else:
        logger.warning(f"Nenhum hiperparâmetro para otimizar para o modelo {model_type}. Treinando com configurações padrão.")
        pipeline.fit(X, y)
        return pipeline

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
        default="all", 
        choices=["basic", "rolling", "technical", "all"],
        help="Conjunto de features: basic, rolling, technical ou all"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest", 
        choices=["mlp", "linear", "poly", "random_forest", "gradient_boosting"], 
        help="Modelo preditivo: mlp, linear, poly, random_forest, gradient_boosting"
    )
    parser.add_argument(
        "--kfolds",
        type=int,
        default=5,
        help="Número de folds para validação cruzada para otimização e avaliação"
    )

    parser.add_argument(
        "--expected-return",
        type=float,
        default=1.0,
        help="Retorno esperado (%%) para o teste de hipótese (ex: 1.0 para 1%%)"
    )
    
    parser.add_argument(
        "--use-random-search",
        action="store_true",
        help="Use RandomizedSearchCV em vez de GridSearchCV para otimização de hiperparâmetros."
    )

    args = parser.parse_args()

    logger.info(
        f"Rodando pipeline para {args.crypto} usando modelo {args.model} "
        f"com {args.kfolds} folds..."
    )

    hypothesis_results = []
    all_metrics_results = []

    for symbol in args.crypto:
        logger.info(f"\nProcessando criptomoeda: {symbol}")

        df = download_crypto_data(symbol)

        if df is None:
            logger.error(f"Falha ao carregar os dados de {symbol}.")
            continue

        features_df = generate_features(df, set_type=args.feature_set, include_original=True)
        
        # Garante que as colunas essenciais estejam presentes
        if "Close" not in features_df.columns or "log_close" not in features_df.columns:
            logger.error(f"Colunas 'Close' ou 'log_close' não encontradas em features_df para {symbol}. Verifique generate_features.")
            continue
            
        # O target agora será o log do preço de fechamento do próximo dia
        features_df["target"] = features_df["log_close"].shift(-1)
        features_df.dropna(inplace=True) # Remove NaNs gerados pelo shift

        logger.info(f"Features geradas para {symbol}:\n{features_df.head()}")

        # Salva as features para cada cripto em CSV (opcional)
        features_df.to_csv(f"backend/data/features_{symbol}.csv", index=False)
        logger.info(f"Features salvas em backend/data/features_{symbol}.csv")

        # Análises estatísticas e gráficos.
        basic_statistics(df, symbol)
        plot_boxplot(df, symbol)
        plot_histogram(df, symbol)
        plot_price_over_time(df, symbol)

        # --- Otimização de Hiperparâmetros e Treinamento do Modelo ---
        # Excluir 'target' do X_data. 'log_close' já é uma feature e será usada em X_data.
        feature_columns_for_X = [col for col in features_df.columns if col != "target"]
        X_data = features_df[feature_columns_for_X].values
        y_data = features_df["target"].values

        # Realiza a otimização e obtém o melhor modelo
        best_model = perform_hyperparameter_optimization(X_data, y_data, args.model, args.kfolds, args.use_random_search)

        # Agora, avalia o melhor modelo usando TimeSeriesSplit para obter métricas por fold
        tscv_eval = TimeSeriesSplit(n_splits=args.kfolds)
        fold_results = []
        all_predictions_for_profit_sim_log_scale = np.zeros_like(y_data, dtype=float)

        for fold, (train_index, test_index) in enumerate(tscv_eval.split(X_data)):
            X_train, X_test = X_data[train_index], X_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index] # Corrigido y_test[test_index] para y_data[test_index]

            # Treina o melhor modelo (pipeline) nos dados de treinamento de cada fold
            best_model.fit(X_train, y_train)
            y_pred_fold_log_scale = best_model.predict(X_test)

            metrics = evaluate_model(y_test, y_pred_fold_log_scale) # Avalia em escala logarítmica
            metrics["fold"] = fold + 1
            metrics["crypto"] = symbol
            fold_results.append(metrics)
            logger.info(f"Avaliação do melhor modelo - Fold {fold + 1} para {symbol}: {metrics}")

            all_predictions_for_profit_sim_log_scale[test_index] = y_pred_fold_log_scale

        results_df = pd.DataFrame(fold_results)
        all_metrics_results.append(results_df)
        logger.info(f"Métricas médias do melhor modelo para {symbol}:\n{results_df.mean(numeric_only=True)}")

        results_path = f"backend/data/metrics_{symbol}.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Métricas de avaliação salvas em: {results_path}")

        # Reverte as previsões de log-escala para escala de preço real antes da simulação de lucro
        features_df["predicted"] = np.exp(all_predictions_for_profit_sim_log_scale)
        
        # Simula lucro e salva gráfico
        features_df["profit"] = simulate_profit(features_df)

        profit_dir = os.path.join("backend", "data")
        os.makedirs(profit_dir, exist_ok=True)
        profit_fig = os.path.join(profit_dir, f"lucro_{symbol}.png")

        plot_profit(features_df["profit"], symbol, profit_fig)
        logger.info(f"Gráfico de lucro salvo em {profit_fig}")

        # Teste de hipótese
        hypothesis_result = test_return_hypothesis(
            features_df, expected_return=args.expected_return, symbol=symbol
        )
        hypothesis_results.append(hypothesis_result)
        logger.info(f"Teste de hipótese realizado para {symbol}: {hypothesis_result}")

    hypothesis_df = pd.DataFrame(hypothesis_results)
    hypothesis_df.to_csv("backend/data/hypothesis_results.csv", index=False)
    logger.info("Resultados do teste de hipótese salvos em backend/data/hypothesis_results.csv")

    if all_metrics_results:
        combined_metrics_df = pd.concat(all_metrics_results, ignore_index=True)
        combined_metrics_df.to_csv("backend/data/combined_metrics_results.csv", index=False)
        logger.info("Métricas combinadas de todos os modelos salvas em backend/data/combined_metrics_results.csv")

    logger.info("Executando análise de correlação geral...")
    dfs_dict_corr = {}
    for symbol in args.crypto:
        temp_df = download_crypto_data(symbol)
        if temp_df is not None:
            dfs_dict_corr[symbol] = temp_df
        else:
            logger.warning(f"Não foi possível baixar dados para {symbol} para análise de correlação.")

    if dfs_dict_corr:
        correlation_analysis(dfs_dict_corr)
        run_anova(dfs_dict_corr)
    else:
        logger.error("Nenhum dado disponível para análise de correlação e ANOVA.")

    generate_html_report(args.crypto)


if __name__ == "__main__":
    main()