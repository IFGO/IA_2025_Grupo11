import os
import webbrowser
import pandas as pd


def generate_html_report(crypto_list, save_dir="backend/reports"):
    """Gera um relatório HTML com os gráficos e métricas."""
    os.makedirs(save_dir, exist_ok=True)

    html = """
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <title>Relatório de Análise de Criptomoedas</title>
        <style>
            body { font-family: sans-serif; background-color: #f7fafc; color: #1a202c; margin: 0; padding: 2rem; }
            h1 { font-size: 2rem; font-weight: bold; margin-bottom: 2rem; }
            h2 { font-size: 1.5rem; font-weight: bold; margin-bottom: 1rem; }
            h3 { font-weight: bold; margin-top: 1rem; }
            table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
            th, td { border: 1px solid #cbd5e0; padding: 0.5rem; text-align: center; }
            th { background-color: #edf2f7; }
            img { max-width: 100%; height: auto; border: 1px solid #e2e8f0; margin-top: 0.5rem; }
            section { margin-bottom: 4rem; }
        </style>
    </head>
    <body>
        <h1>📊 Relatório de Análise de Criptomoedas</h1>
    """

    for symbol in crypto_list:
        html += f"""
        <section>
            <h2>{symbol}</h2>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
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