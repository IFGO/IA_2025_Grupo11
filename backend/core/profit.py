import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List

def simulate_profit(
    df: pd.DataFrame,
    initial_investment: float = 1000.0,
    trading_fee_rate: float = 0.00075,  # Ex: 0.1% de taxa por operação (compra + venda)
    min_profit_threshold: float = 0.002 # Ex: 0.5% de alta esperada mínima para operar
) -> pd.Series:
    """
    Simula o lucro acumulado ao longo do tempo com base nas previsões,
    considerando custos de transação e um limiar mínimo de lucro esperado.

    Args:
        df: DataFrame com colunas "Close" (preço real de fechamento) e "predicted"
            (previsão de preço de fechamento para o dia seguinte).
        initial_investment: Capital inicial para a simulação.
        trading_fee_rate: Taxa de transação aplicada em cada compra e venda (porcentagem, e.g., 0.001 para 0.1%).
        min_profit_threshold: Limiar de lucro esperado para acionar uma operação (porcentagem, e.g., 0.005 para 0.5%).

    Returns:
        pd.Series: Uma série temporal com o saldo acumulado ao longo do tempo.
    """
    saldo = initial_investment
    saldo_history: List[float] = [initial_investment]
    
    # Criar uma coluna com o preço de fechamento do dia seguinte para facilitar a lógica
    df['Close_next_day'] = df['Close'].shift(-1)

    # Itera sobre o DataFrame para simular as operações
    # A iteração vai até o penúltimo dia, pois a previsão do último dia não tem um 'Close' no dia seguinte
    for i in range(len(df) - 1):
        preco_hoje = df.iloc[i]["Close"]
        preco_amanha_real = df.iloc[i]['Close_next_day']
        previsao_amanha = df.iloc[i]["predicted"]

        # O modelo opera apenas se houver uma previsão. NaNs são ignorados.
        if pd.isna(previsao_amanha) or pd.isna(preco_amanha_real):
            saldo_history.append(saldo)
            continue

        # Lógica de operação: Apenas opera se a previsão de alta for maior que um limiar
        # O limiar de lucro mínimo é crucial para cobrir a taxa de transação e o "ruído" do modelo
        required_predicted_price = preco_hoje * (1 + min_profit_threshold + trading_fee_rate)

        if previsao_amanha > required_predicted_price:
            # Simulamos a compra no final do dia 'i'
            # Aplicamos a taxa de transação (compra)
            # custo_compra = saldo * trading_fee_rate

            # Calculamos o retorno real do preço
            retorno_real = preco_amanha_real / preco_hoje
            
            # Atualiza o saldo após a operação
            saldo = saldo * retorno_real * (1 - trading_fee_rate) * (1 - trading_fee_rate)
            # O cálculo acima aplica a taxa de compra e de venda em uma única operação.
            # Alternativa:
            # saldo = (saldo - custo_compra) * retorno_real
            # saldo = saldo - (saldo * trading_fee_rate) # custo de venda

        # Se a condição não for atendida, o saldo permanece o mesmo (não há operação)
        saldo_history.append(saldo)

    # Garante que a série de histórico tenha o mesmo tamanho do DataFrame original
    return pd.Series(saldo_history, index=df.index[:len(saldo_history)])


def plot_profit(profit_series: pd.Series, symbol: str, save_path: str) -> None:
    """
    Plota o gráfico de evolução do lucro.

    Args:
        profit_series: Série com o saldo acumulado ao longo do tempo.
        symbol: Símbolo da criptomoeda.
        save_path: Caminho para salvar o gráfico.
    """
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