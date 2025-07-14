from scipy.stats import ttest_1samp
import pandas as pd

def test_return_hypothesis(df: pd.DataFrame, expected_return: float, symbol: str) -> dict:
    """
    Realiza teste de hipótese para verificar se o retorno médio é maior que o valor esperado.

    Args:
        df: DataFrame com a coluna 'profit' ou 'return' (variação percentual diária/acumulada).
        expected_return: Valor de retorno esperado em percentual (ex: 1.0 para 1%).
        symbol: Nome da criptomoeda.

    Returns:
        Dicionário com estatísticas do teste.
    """
    # Supondo que 'profit' já é percentual (ex: 0.01 = 1%)
    returns = df["profit"].pct_change().dropna() * 100  # Retorno diário em %
    t_stat, p_value = ttest_1samp(returns, expected_return)

    # Como é teste unilateral (queremos 'maior que'):
    p_value_one_sided = p_value / 2
    reject_h0 = (t_stat > 0) and (p_value_one_sided < 0.05)

    return {
        "Crypto": symbol,
        "t-Statistic": t_stat,
        "p-Value (one-sided)": p_value_one_sided,
        "Mean Return %": returns.mean(),
        "Expected Return %": expected_return,
        "Reject H0?": reject_h0
    }
