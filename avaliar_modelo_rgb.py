"""Avaliação exploratória do modelo RGB no teste já conhecido."""

from pathlib import Path

from src.avaliacao import avaliar_modelo
from src.preprocessamento_rgb import carregar_divisao_rgb


if __name__ == "__main__":
    raiz = Path(__file__).resolve().parent
    dados = carregar_divisao_rgb(raiz, "teste")
    avaliar_modelo(
        raiz, modelo_relativo="modelos/mlp_rgb_final.npz",
        sufixo="teste_rgb_exploratorio", dados_teste=dados,
        observacao=("Resultado exploratório: o conjunto de teste já foi acessado "
                    "na avaliação de modelos anteriores; não é um novo teste independente."))
