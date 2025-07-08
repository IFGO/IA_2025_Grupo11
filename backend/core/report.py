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
        <link rel="stylesheet" href="tailwind.min.css">
    </head>
    <body class="bg-gray-100 text-gray-900">
        <div class="max-w-6xl mx-auto p-8">
            <h1 class="text-4xl font-bold mb-8">📊 Relatório de Análise de Criptomoedas</h1>
    """

    for symbol in crypto_list:
        html += f"""
        <section class="mb-16">
            <h2 class="text-2xl font-semibold mb-4">{symbol}</h2>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div>
                    <h3 class="font-semibold mb-2">Boxplot</h3>
                    <img src="../analysis_results/boxplot_{symbol}.png" alt="Boxplot {symbol}">
                </div>
                <div>
                    <h3 class="font-semibold mb-2">Histograma</h3>
                    <img src="../analysis_results/histograma_{symbol}.png" alt="Histograma {symbol}">
                </div>
                <div class="col-span-2">
                    <h3 class="font-semibold mb-2">Gráfico de Linha (Preço + Média + Mediana + Moda)</h3>
                    <img src="../analysis_results/grafico_linha_{symbol}.png" alt="Gráfico de Linha {symbol}">
                </div>
            </div>

            <h3 class="font-semibold mt-8 mb-2">Métricas de Modelagem (por Fold)</h3>
        """

        metrics_path = f"backend/data/metrics_{symbol}.csv"
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            html += df.to_html(classes="table-auto border border-gray-300", index=False)
        else:
            html += "<p class='text-red-500'>Métricas não encontradas para esta moeda.</p>"

        html += "</section>"

    html += """
        </div>
    </body>
    </html>
    """

    report_path = os.path.join(save_dir, "relatorio_cripto.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Relatório HTML gerado em: {report_path}")

    webbrowser.open(f"file://{os.path.abspath(report_path)}")