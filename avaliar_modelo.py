"""Avalia o modelo principal de textura no teste já conhecido."""

from pathlib import Path

from src.avaliacao import avaliar_modelo
from src.recursos_textura import carregar_divisao_textura


if __name__ == "__main__":
    raiz = Path(__file__).resolve().parent
    avaliar_modelo(
        raiz, dados_teste=carregar_divisao_textura(raiz, "teste"),
        sufixo="teste_textura_exploratorio",
        observacao=("Resultado exploratório: o conjunto de teste já foi acessado "
                    "na avaliação do modelo anterior; não é um novo teste independente."))
