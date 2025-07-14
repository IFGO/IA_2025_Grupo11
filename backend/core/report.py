import os
import webbrowser
import pandas as pd


def generate_html_report(crypto_list, save_dir="backend/reports"):
    """Gera um relatório HTML com os gráficos, métricas, teste de hipótese e ANOVA."""
    os.makedirs(save_dir, exist_ok=True)

    # Lê os resultados do teste de hipótese, ANOVA e Tukey
    hypothesis_path = "backend/data/hypothesis_results.csv"
    hypothesis_df = pd.read_csv(hypothesis_path) if os.path.exists(hypothesis_path) else None

    anova_path = "backend/data/anova_result.csv"
    tukey_path = "backend/data/tukey_results.csv"
    anova_df = pd.read_csv(anova_path) if os.path.exists(anova_path) else None
    tukey_df = pd.read_csv(tukey_path) if os.path.exists(tukey_path) else None

    html = """
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <title>Relatório de Análise de Criptomoedas</title>
        <style>
            body {
                font-family: sans-serif;
                background-color: #f7fafc;
                color: #1a202c;
                margin: 0;
                padding: 2rem;
                text-align: center;
            }
            h1 {
                font-size: 2rem;
                font-weight: bold;
                margin-bottom: 2rem;
            }
            h2 {
                font-size: 1.5rem;
                font-weight: bold;
                margin: 2rem 0 1rem;
            }
            h3 {
                font-weight: bold;
                margin-top: 1rem;
            }
            table {
                margin: 1rem auto;
                width: 80%;
                border-collapse: collapse;
            }
            th, td {
                border: 1px solid #cbd5e0;
                padding: 0.5rem;
                text-align: center;
            }
            th {
                background-color: #edf2f7;
            }
            img {
                max-width: 90%;
                height: auto;
                border: 1px solid #e2e8f0;
                margin: 0.5rem 0;
            }
            section {
                margin-bottom: 4rem;
                padding: 1rem;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                background-color: #fff;
                width: 90%;
                margin-left: auto;
                margin-right: auto;
            }
            .grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 2rem;
                justify-items: center;
            }
        </style>
    </head>
    <body>
        <h1>📊 Relatório de Análise de Criptomoedas</h1>
    """

    for symbol in crypto_list:
        html += f"""
        <section>
            <h2>{symbol}</h2>

            <div class="grid">
                <div>
                    <h3>Boxplot</h3>
                    <img src="../analysis_results/boxplot_{symbol}.png" alt="Boxplot {symbol}">
                </div>
                <div>
                    <h3>Histograma</h3>
                    <img src="../analysis_results/histograma_{symbol}.png" alt="Histograma {symbol}">
                </div>
            </div>

            <div>
                <h3>Gráfico de Linha (Preço + Média + Mediana + Moda)</h3>
                <img src="../analysis_results/grafico_linha_{symbol}.png" alt="Gráfico de Linha {symbol}">
            </div>

            <div>
                <h3>Lucro Simulado</h3>
                <img src="../data/lucro_{symbol}.png" alt="Lucro {symbol}">
            </div>

            <h3>Métricas de Modelagem (por Fold)</h3>
        """

        metrics_path = f"backend/data/metrics_{symbol}.csv"
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            html += df.to_html(index=False)
        else:
            html += "<p style='color: red;'>Métricas não encontradas para esta moeda.</p>"

        # Inclui resultado do teste de hipótese (se existir)
        if hypothesis_df is not None:
            hypo_row = hypothesis_df[hypothesis_df["Crypto"] == symbol]
            if not hypo_row.empty:
                html += "<h3>Resultado do Teste de Hipótese (Retorno Esperado)</h3>"
                html += hypo_row.to_html(index=False)

        html += "</section>"

    # Bloco geral ANOVA e Tukey
    html += "<section><h2>ANOVA - Comparação Entre Criptomoedas</h2>"
    if anova_df is not None:
        html += "<h3>Resultado da ANOVA (F-test)</h3>"
        html += anova_df.to_html(index=False)
    else:
        html += "<p>ANOVA não realizada ou não encontrada.</p>"

    if tukey_df is not None:
        html += "<h3>Post Hoc (Tukey HSD)</h3>"
        html += tukey_df.to_html(index=False)
    else:
        html += "<p>Tukey HSD não realizado ou não encontrado.</p>"
    html += "</section>"

    html += """
    </body>
    </html>
    """

    report_path = os.path.join(save_dir, "relatorio_cripto.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Relatório HTML gerado em: {report_path}")
    webbrowser.open(f"file://{os.path.abspath(report_path)}")
