"""Classifica uma foto nova com a MLP principal do projeto.

Exemplo: python classificar_imagem.py imagens_teste/minha_nuvem.jpg
"""

import argparse
from pathlib import Path

from src.config import CLASSES
from src.mlp import MLP
from src.preprocessamento import processar_imagem
from src.preprocessamento_rgb import processar_imagem_rgb
from src.recursos_textura import processar_imagem_textura


def preparar_entrada(caminho, n_entradas):
    """Aplica o pré-processamento que corresponde à entrada do modelo."""
    if n_entradas == 936:
        return processar_imagem_textura(caminho)
    if n_entradas == 3072:
        return processar_imagem_rgb(caminho)
    if n_entradas == 1024:
        return processar_imagem(caminho)
    raise ValueError(f"Arquitetura de entrada não reconhecida: {n_entradas}")


def main():
    raiz = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Classifica uma imagem de nuvem")
    parser.add_argument("imagem", type=Path, help="Caminho de uma imagem JPG ou PNG")
    parser.add_argument("--modelo", type=Path,
                        default=raiz / "modelos" / "mlp_etapa4.npz")
    args = parser.parse_args()
    if not args.imagem.is_file():
        parser.error(f"Imagem não encontrada: {args.imagem}")
    rede = MLP.carregar(args.modelo)
    vetor = preparar_entrada(args.imagem, rede.tamanhos[0])
    probabilidades = rede.forward(vetor[None, :])[0]
    indice = int(probabilidades.argmax())
    print(f"Classe prevista: {CLASSES[indice]} ({probabilidades[indice]:.1%})")
    for classe, probabilidade in zip(CLASSES, probabilidades):
        print(f"  {classe:15s}: {probabilidade:.1%}")


if __name__ == "__main__":
    main()
