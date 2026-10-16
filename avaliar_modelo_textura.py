"""Avalia exploratoriamente a MLP de textura no teste já conhecido."""

from pathlib import Path

from src.avaliacao import avaliar_modelo
from src.recursos_textura import carregar_divisao_textura


if __name__ == "__main__":
    raiz = Path(__file__).resolve().parent
    dados = carregar_divisao_textura(raiz, "teste")
    avaliar_modelo(
        raiz, modelo_relativo="modelos/mlp_textura_final.npz",
        sufixo="teste_textura_exploratorio", dados_teste=dados,
        observacao=("Resultado exploratório: o conjunto de teste já foi acessado "
                    "em avaliações anteriores; não é um novo teste independente."))
