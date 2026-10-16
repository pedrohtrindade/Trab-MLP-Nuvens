"""Etapa 4 — execute: python treinar_modelo.py"""

import argparse
from pathlib import Path

from src.treinamento import executar_treinamento


def main():
    parser = argparse.ArgumentParser(description="Treina a MLP de nuvens com NumPy")
    parser.add_argument("--epocas", type=int, default=180)
    parser.add_argument("--lote", type=int, default=32)
    parser.add_argument("--taxa", type=float, default=0.003)
    parser.add_argument("--paciencia", type=int, default=25)
    parser.add_argument("--ocultas", type=int, nargs="+", default=[256, 64])
    parser.add_argument("--semente", type=int, default=42)
    args = parser.parse_args()
    executar_treinamento(
        Path(__file__).resolve().parent, epocas=args.epocas,
        tamanho_lote=args.lote, taxa_aprendizado=args.taxa,
        paciencia=args.paciencia, camadas_ocultas=tuple(args.ocultas),
        semente=args.semente)


if __name__ == "__main__":
    main()
