# backend/core/plots.py

import matplotlib.pyplot as plt
import pandas as pd


def plot_profit(profit_series: pd.Series, symbol: str, save_path: str) -> None:
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


def plot_predictions_vs_actual(df: pd.DataFrame, symbol: str, save_path: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(df["target"], df["predicted"], alpha=0.5)
    plt.plot(
        [df["target"].min(), df["target"].max()],
        [df["target"].min(), df["target"].max()],
        'r--',
    )
    plt.title(f"Predições vs Reais - {symbol}")
    plt.xlabel("Preço Real")
    plt.ylabel("Preço Previsto")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
