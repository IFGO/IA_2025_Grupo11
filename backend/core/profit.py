import pandas as pd
import matplotlib.pyplot as plt


def simulate_profit(df: pd.DataFrame, initial_investment: float = 1000.0) -> pd.Series:
    """
    Simula o lucro acumulado ao longo do tempo com base nas previsões.
    Compra/reinveste apenas se a previsão indica alta no fechamento.

    Args:
        df: DataFrame com colunas ["Close", "predicted"].
        initial_investment: Capital inicial.

    Returns:
        Série com saldo acumulado ao longo do tempo.
    """
    saldo = initial_investment
    saldo_history = []

    for i in range(len(df) - 1):
        preco_hoje = df.iloc[i]["Close"]
        preco_amanha = df.iloc[i + 1]["Close"]
        previsao_amanha = df.iloc[i]["predicted"]

        if previsao_amanha > preco_hoje:
            retorno = preco_amanha / preco_hoje
            saldo *= retorno
        saldo_history.append(saldo)

    saldo_history.append(saldo)
    return pd.Series(saldo_history, index=df.index)

def plot_profit(profit_series: pd.Series, symbol: str, save_path: str):
    plt.figure(figsize=(10, 5))
    plt.plot(profit_series.index, profit_series.values, label="Lucro Acumulado")
    plt.title(f"Evolução do Lucro - {symbol}")
    plt.xlabel("Data")
    plt.ylabel("Saldo (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
