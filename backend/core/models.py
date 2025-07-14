import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

logger = logging.getLogger(__name__)


def evaluate_model(y_true, y_pred) -> dict:
    """Calcula as métricas de avaliação."""
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def train_and_evaluate_model(
    features_df: pd.DataFrame, model_type: str = "mlp", kfolds: int = 5, return_predictions: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Treina e avalia o modelo de previsão.

    Args:
        features_df: DataFrame com features + coluna 'target'.
        model_type: Tipo do modelo ('mlp', 'linear', 'poly').
        kfolds: Número de folds para validação cruzada.
        return_predictions: Se True, retorna o features_df com coluna 'predicted'.

    Returns:
        DataFrame com resultados por fold ou (results_df, features_df) com previsões.
    """
    logger.info(f"Treinando o modelo {model_type}")

    X = features_df.drop(columns=["target"]).values
    y = features_df["target"].values

    kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)
    results = []

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if model_type == "mlp":
            model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        elif model_type == "linear":
            model = LinearRegression()
        elif model_type == "poly":
            model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        else:
            raise ValueError(f"Modelo desconhecido: {model_type}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = evaluate_model(y_test, y_pred)
        metrics["fold"] = fold + 1
        results.append(metrics)

        logger.info(f"Fold {fold + 1}: {metrics}")

    results_df = pd.DataFrame(results)
    logger.info(f"Resultados médios:\n{results_df.mean()}")

    if return_predictions:
        # Treina no dataset todo para gerar previsões para simular o lucro
        if model_type == "mlp":
            model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        elif model_type == "linear":
            model = LinearRegression()
        elif model_type == "poly":
            model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        model.fit(X, y)
        features_df["predicted"] = model.predict(X)

        return results_df, features_df

    return results_df
