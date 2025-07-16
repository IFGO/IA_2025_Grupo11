import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def simulate_profit(
    df: pd.DataFrame,
    initial_investment: float = 1000.0,
    trading_fee_rate: float = 0.001,  # Ex: 0.1% de taxa por operação (compra + venda)
    min_profit_threshold: float = 0.005 # Ex: 0.5% de alta esperada mínima para operar
    # stop_loss_percent: float = None,  # Ex: 0.05 para 5% de stop loss (opcional)
    # take_profit_percent: float = None # Ex: 0.10 para 10% de take profit (opcional)
) -> pd.Series:
    """
    Simula o lucro acumulado ao longo do tempo com base nas previsões.
    Compra/reinveste apenas se a previsão indica alta no fechamento,
    considerando taxas de transação e um limiar mínimo de lucro esperado.

    Args:
        df: DataFrame com colunas ["Close", "predicted"].
            'Close' deve ser o preço de fechamento real.
            'predicted' deve ser o preço de fechamento previsto para o dia seguinte (na escala real).
        initial_investment: Capital inicial.
        trading_fee_rate: Taxa de transação (ex: 0.001 para 0.1%). Aplicada na compra e na venda.
        min_profit_threshold: Percentual de lucro esperado mínimo para abrir uma posição.
                              Ex: 0.005 significa que a previsão_amanha deve ser 0.5% maior que preco_hoje
                              para considerar uma compra.
        # stop_loss_percent: Percentual de queda que aciona uma venda para limitar perdas.
        # take_profit_percent: Percentual de ganho que aciona uma venda para realizar lucros.

    Returns:
        Série com saldo acumulado ao longo do tempo.
    """
    saldo = initial_investment
    saldo_history = []
    
    # O Close do dia `i+1` é o Close real do dia que a previsão `i` se refere.
    df['Close_next_day'] = df['Close'].shift(-1)


    # Itera até o penúltimo dia, pois a previsão do último dia não tem um 'próximo dia' real
    for i in range(len(df) - 1):
        preco_hoje = df.iloc[i]["Close"] # Preço de fechamento no dia atual (para comparação)
        preco_amanha_real = df.iloc[i]["Close_next_day"] # Preço de fechamento real do dia que a previsão se refere
        previsao_amanha = df.iloc[i]["predicted"] # Previsão do modelo para o preco_amanha_real

        # Verifica se há NaNs nas colunas críticas antes de operar
        if pd.isna(preco_hoje) or pd.isna(preco_amanha_real) or pd.isna(previsao_amanha):
            saldo_history.append(saldo) # Mantém o saldo, não opera
            continue

        # Regra de Compra: Previsão indica alta E a alta esperada supera o limiar
        # O limiar é aplicado sobre o preço de hoje para ter uma "margem de segurança"
        required_predicted_price = preco_hoje * (1 + min_profit_threshold)

        if previsao_amanha > required_predicted_price:
            # Compra no final do dia 'i' e a venda no final do dia 'i+1'
            # Custo da COMPRA (reduz o saldo)
            saldo *= (1 - trading_fee_rate)

            # Cálculo do retorno baseado no preço real de amanhã
            # Retorno é calculado sobre o capital após a taxa de compra
            retorno_real = preco_amanha_real / preco_hoje
            saldo *= retorno_real

            # Custo da VENDA (reduz o saldo novamente)
            saldo *= (1 - trading_fee_rate)
        
        # Se não houver operação, o saldo permanece o mesmo.
        saldo_history.append(saldo)

    # Adiciona o saldo final para o último dia do DataFrame para ter o mesmo tamanho do índice
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