import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Executa o pipeline de previsão de criptomoedas."
    )

    parser.add_argument(
        "--crypto",
        type=str,
        required=True,
        help="Símbolo da criptomoeda (ex: BTC, ETH)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="mlp", 
        help="Modelo preditivo: mlp, linear, poly"
    )
    parser.add_argument(
        "--kfolds", 
        type=int, 
        default=5, 
        help="Número de folds para validação cruzada"
    )
    parser.add_argument(
        "--feature-set", 
        type=str, 
        default="basic", 
        help="Conjunto de features"
    )

    args = parser.parse_args()

    print(f"Rodando pipeline para {args.crypto} usando modelo {args.model} com {args.kfolds} folds...")

if __name__ == "__main__":
    main()