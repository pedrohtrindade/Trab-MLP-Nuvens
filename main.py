"""Ponto de entrada inicial: verifica as imagens do dataset."""

from pathlib import Path
from src.config import CLASSES


def main():
    raiz = Path(__file__).resolve().parent
    dataset = raiz / "dataset" / "original"
    print("CLASSIFICADOR DE NUVENS — VERIFICAÇÃO INICIAL")
    print("-" * 48)
    total = 0
    for classe in CLASSES:
        pasta = dataset / classe
        quantidade = sum(1 for arquivo in pasta.iterdir() if arquivo.suffix.lower() in {".jpg", ".jpeg", ".png"}) if pasta.is_dir() else 0
        print(f"{classe.capitalize():15s}: {quantidade:4d} imagens")
        total += quantidade
    print("-" * 48)
    print(f"TOTAL: {total} imagens")
    print("\nDataset organizado. Forward propagation da MLP disponível em src/mlp.py.")


if __name__ == "__main__":
    main()
