"""Etapa 2 — execute: python preparar_dados.py"""
from pathlib import Path
from src.preprocessamento import preparar_dataset

if __name__ == '__main__':
    preparar_dataset(Path(__file__).resolve().parent)
