import os
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd

def run_anova(dfs_dict: dict, save_dir="backend/data") -> None:
    os.makedirs(save_dir, exist_ok=True)

    data = []
    for symbol, df in dfs_dict.items():
        if df is None:
            continue
        returns = df["Close"].pct_change().dropna() * 100  # Retorno diário em %
        for ret in returns:
            data.append({"Crypto": symbol, "Return %": ret})

    data_df = pd.DataFrame(data)

    # ANOVA
    grouped = [group["Return %"].values for _, group in data_df.groupby("Crypto")]
    f_stat, p_value = f_oneway(*grouped)

    print(f"ANOVA F-statistic: {f_stat:.4f} | p-value: {p_value:.4f}")

    anova_result = pd.DataFrame([{"F-statistic": f_stat, "p-value": p_value}])

    anova_path = os.path.join(save_dir, "anova_result.csv")
    try:
        anova_result.to_csv(anova_path, index=False)
        print(f"Arquivo anova_result.csv salvo em: {anova_path}")
    except Exception as e:
        print(f"Erro ao salvar anova_result.csv: {e}")

    # Post hoc (Tukey HSD)
    tukey = pairwise_tukeyhsd(data_df["Return %"], data_df["Crypto"])
    tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])

    tukey_path = os.path.join(save_dir, "tukey_results.csv")
    try:
        tukey_df.to_csv(tukey_path, index=False)
        print(f"Arquivo tukey_results.csv salvo em: {tukey_path}")
    except Exception as e:
        print(f"Erro ao salvar tukey_results.csv: {e}")

    # Também salva os dados usados na ANOVA
    anova_data_path = os.path.join(save_dir, "anova_data.csv")
    try:
        data_df.to_csv(anova_data_path, index=False)
        print(f"Arquivo anova_data.csv salvo em: {anova_data_path}")
    except Exception as e:
        print(f"Erro ao salvar anova_data.csv: {e}")

    print("ANOVA e Tukey HSD concluídos e salvos.")
