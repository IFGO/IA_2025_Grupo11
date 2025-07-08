import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scipy import stats
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_SAVE_DIR = os.path.join(BASE_DIR, "analysis_results")

logger = logging.getLogger(__name__)


def basic_statistics(df: pd.DataFrame, symbol: str, save_dir: str = DEFAULT_SAVE_DIR) -> pd.DataFrame:
    """Gera e salva estatísticas descritivas básicas para uma criptomoeda."""
    stats_df = df.describe()
    os.makedirs(save_dir, exist_ok=True)
    stats_path = os.path.join(save_dir, f"stats_{symbol}.csv")
    stats_df.to_csv(stats_path)
    logger.info(f"Estatísticas salvas em: {stats_path}")
    return stats_df


def correlation_analysis(dfs: dict, save_dir: str = DEFAULT_SAVE_DIR):
    """
    Analisa a variabilidade entre criptomoedas com base nas medidas de dispersão (desvio padrão).

    Args:
        dfs: Dicionário {symbol: dataframe} com os dados de cada criptomoeda.
    """
    dispersions = {}
    for symbol, df in dfs.items():
        dispersion = df["Close"].std()
        dispersions[symbol] = dispersion
        logger.info(f"Desvio padrão (variabilidade) de {symbol}: {dispersion}")

    # Salva dispersões em CSV
    dispersions_df = pd.DataFrame.from_dict(dispersions, orient='index', columns=["Desvio Padrão"])
    os.makedirs(save_dir, exist_ok=True)
    dispersions_path = os.path.join(save_dir, "variabilidade_criptos.csv")
    dispersions_df.to_csv(dispersions_path)
    logger.info(f"Variabilidades salvas em: {dispersions_path}")
    return dispersions_df


def plot_boxplot(df: pd.DataFrame, symbol: str, save_dir: str = DEFAULT_SAVE_DIR):
    """Gera e salva um boxplot do preço de fechamento."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df["Close"])
    plt.title(f"Boxplot do preço de fechamento: {symbol}")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"boxplot_{symbol}.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Boxplot salvo em: {path}")


def plot_histogram(df: pd.DataFrame, symbol: str, save_dir: str = DEFAULT_SAVE_DIR):
    """Gera e salva um histograma do preço de fechamento."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df["Close"], bins=50, kde=True)
    plt.title(f"Histograma do preço de fechamento: {symbol}")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"histograma_{symbol}.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Histograma salvo em: {path}")


def plot_price_over_time(df: pd.DataFrame, symbol: str, save_dir: str = DEFAULT_SAVE_DIR):
    """Gera e salva gráfico de linha com preço, média, mediana e moda ao longo do tempo."""
    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["Close"], label="Preço de Fechamento", color="blue")
    plt.plot(df["Date"], df["Close"].rolling(window=7).mean(), label="Média Móvel (7 dias)", color="green")
    plt.plot(df["Date"], df["Close"].rolling(window=7).median(), label="Mediana Móvel (7 dias)", color="orange")

    # Moda: para dados contínuos pode não ser muito informativa, mas incluída para cumprir requisito
    try:
        moda = stats.mode(df["Close"], keepdims=True)[0][0]
        plt.axhline(y=moda, color="red", linestyle="--", label=f"Moda: {moda:.2f}")
    except Exception as e:
        logger.warning(f"Não foi possível calcular a moda para {symbol}: {e}")

    plt.title(f"Preço ao longo do tempo: {symbol}")
    plt.xlabel("Data")
    plt.ylabel("Preço de Fechamento")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"grafico_linha_{symbol}.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Gráfico de linha salvo em: {path}")