import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import TimeSeriesSplit # Importante para séries temporais
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler # Adicionado StandardScaler
from sklearn.pipeline import Pipeline # Adicionado Pipeline
from sklearn.ensemble import RandomForestRegressor # Novo modelo para experimentar

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
                     As features devem ser numéricas e sem NaNs.
        model_type: Tipo do modelo ('mlp', 'linear', 'poly', 'random_forest').
        kfolds: Número de folds para validação cruzada.
        return_predictions: Se True, retorna o features_df com coluna 'predicted'.

    Returns:
        DataFrame com resultados por fold ou (results_df, features_df) com previsões.
    """
    logger.info(f"Treinando o modelo {model_type}")

    # Certifica-se de que a coluna 'Date' não é uma feature, se ela estiver presente
    # e ainda não tiver sido removida pelo generate_features.
    # Assumimos que generate_features já lida com colunas não numéricas se necessário.
    feature_columns = [col for col in features_df.columns if col != "target"]
    X = features_df[feature_columns].values
    y = features_df["target"].values

    # Validação cruzada para séries temporais
    kf = TimeSeriesSplit(n_splits=kfolds)
    results = []
    
    # Lista para armazenar as previsões de teste em cada fold
    # Isso será usado para construir 'features_df["predicted"]' de forma mais robusta
    # sem re-treinar o modelo no dataset completo no final (que pode levar a overfitting
    # na métrica reportada para o *pipeline* de lucro).
    # Uma abordagem melhor é acumular as previsões out-of-sample da validação cruzada.
    all_predictions = np.zeros_like(y, dtype=float)


    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Definição dos modelos dentro de Pipelines para incluir o StandardScaler
        if model_type == "mlp":
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                                     random_state=42, early_stopping=True, validation_fraction=0.1))
            ])
        elif model_type == "linear":
            # LinearRegression é menos sensível ao scaling, mas ainda pode se beneficiar
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('linear_reg', LinearRegression())
            ])
        elif model_type == "poly":
            # PolynomialFeatures e LinearRegression são sensíveis ao scaling
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('poly_features', PolynomialFeatures(degree=2, include_bias=False)), # include_bias=False para evitar colunas redundantes
                ('linear_reg', LinearRegression())
            ])
        elif model_type == "random_forest":
            # Random Forest é menos sensível ao scaling, mas mantemos para consistência e futuras transformações
            model = Pipeline([
                ('scaler', StandardScaler()), # StandardScaler ainda é útil se houver outras transformações na pipeline
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)) # n_jobs=-1 usa todos os cores
            ])
        else:
            raise ValueError(f"Modelo desconhecido: {model_type}. Escolha entre 'mlp', 'linear', 'poly', 'random_forest'.")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = evaluate_model(y_test, y_pred)
        metrics["fold"] = fold + 1
        results.append(metrics)

        logger.info(f"Fold {fold + 1}: {metrics}")

        # Armazena as previsões do fold de teste
        all_predictions[test_index] = y_pred

    results_df = pd.DataFrame(results)
    logger.info(f"Resultados médios da validação cruzada:\n{results_df.mean(numeric_only=True)}")


    if return_predictions:
        # Usa as previsões out-of-sample da validação cruzada para o 'predicted'
        # Isso reflete um desempenho mais realista do modelo em dados "não vistos" durante o treinamento.
        # Note que as primeiras linhas (que não são usadas como teste em TimeSeriesSplit) terão previsão zero.
        # Se precisar de previsão para todas as linhas, o modelo precisaria ser treinado no dataset completo *após* a validação.
        # Para simulação de lucro, as previsões out-of-sample são mais relevantes.
        features_df["predicted"] = all_predictions
        
        # Opcional: Treinar o modelo no conjunto completo para ter previsões em todo o dataset
        # Isso pode ser útil para visualização, mas a simulação de lucro deve focar em previsões realistas.
        # Se for para o lucro, o "all_predictions" da validação cruzada é o mais correto.
        # Se você ainda quer o modelo treinado em tudo:
        # final_model_for_full_data = model # Re-instancie o modelo ou use o último treinado
        # final_model_for_full_data.fit(X, y)
        # features_df["predicted_full_data"] = final_model_for_full_data.predict(X)

        return results_df, features_df

    return results_df