# 📊 Crypto Predictor — Pipeline de Previsão de Criptomoedas

Projeto de Machine Learning para análise e predição de preços de criptomoedas com:
- Coleta automática de dados
- Análises estatísticas completas
- Geração de gráficos
- Treinamento e avaliação de modelos de ML (MLP, Linear, Polinomial)
- Pipeline automatizado via CLI

---

## 🚀 Funcionalidades

- ✅ Download automático dos dados (site CryptoDataDownload)
- ✅ Limpeza e normalização dos dados
- ✅ Geração de features (médias móveis, volatilidade, RSI, etc.)
- ✅ Análises estatísticas e gráficos (salvos automaticamente)
- ✅ Treinamento e avaliação de modelos preditivos com validação k-fold
- ✅ Relatórios de métricas por criptomoeda
- ✅ Pipeline modular e configurável via CLI

---

## 📂 Estrutura do Projeto

## 📂 Estrutura do Projeto

```plaintext
crypto-predictor/
│
├── backend/
│   ├── cli/                     # CLI principal do pipeline
│   ├── core/                    # Módulos do pipeline (data_load, features, models, analysis)
│   └── data/                    # CSVs de features e métricas gerados automaticamente
│
├── analysis_results/            # Relatórios e gráficos gerados automaticamente
│
├── frontend/                    # (Opcional) Interface web em React (não obrigatório no trabalho)
│
├── docker-compose.yml           # (Opcional) Configuração Docker
│
└── readme.md                    # Este arquivo
```

---

## 🏗️ Como Executar o Pipeline

### Pré-requisitos:
- Python 3.10+ (recomendado ambiente virtual)
- Instalar dependências:
```bash
pip install -r requirements.txt
```
Execução:
```bash
python backend/cli/main.py --crypto BTCUSDT ETHUSDT DOGEUSDT ADAUSDT \
--feature-set all --model mlp --kfolds 5
```
### 📄 Explicação dos parâmetros:

| Parâmetro     | Descrição                                            |
|---------------|------------------------------------------------------|
| `--crypto`    | Lista de criptomoedas a processar                    |
| `--feature-set` | Conjunto de features: `basic`, `rolling`, `technical`, `all` |
| `--model`     | Modelo preditivo: `mlp`, `linear`, `poly`            |
| `--kfolds`    | Número de folds para validação cruzada               |

---

### ✅ Saídas Geradas:

- 📈 Gráficos e estatísticas: `analysis_results/`
- 📊 Features e métricas: `backend/data/`

---

### 📝 Observação:
Este projeto foi desenvolvido para fins acadêmicos, com foco em pipelines automatizados de previsão de preços de criptomoedas.
