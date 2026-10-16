"""Entrada RGB alternativa para a MLP, sem alterar o pré-processamento cinza."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def processar_imagem_rgb(caminho, tamanho=32):
    """Converte uma foto em vetor RGB normalizado de tamanho 3×tamanho²."""
    with Image.open(caminho) as imagem:
        imagem = ImageOps.exif_transpose(imagem).convert("RGB")
        imagem = imagem.resize((tamanho, tamanho), Image.Resampling.LANCZOS)
        valores = np.asarray(imagem, dtype=np.float32) / 255.0
    return valores.reshape(-1)


def carregar_divisao_rgb(raiz, conjunto, tamanho=32):
    """Carrega as imagens da divisão original, preservando a ordem do CSV."""
    if conjunto not in {"treino", "validacao", "teste"}:
        raise ValueError("Conjunto deve ser treino, validacao ou teste")
    raiz = Path(raiz)
    caminho_csv = raiz / "dataset" / "processado" / "divisao.csv"
    with caminho_csv.open(newline="", encoding="utf-8") as arquivo:
        registros = [r for r in csv.DictReader(arquivo) if r["conjunto"] == conjunto]
    if not registros:
        raise ValueError(f"Nenhuma imagem encontrada para {conjunto}")
    X = np.stack([processar_imagem_rgb(raiz / r["arquivo"], tamanho)
                  for r in registros]).astype(np.float32)
    y = np.array([int(r["rotulo"]) for r in registros], dtype=np.int64)
    return X, y


def aumentar_espelho_rgb(X, y, tamanho=32):
    """Acrescenta uma cópia horizontalmente espelhada de cada imagem."""
    n_entradas = tamanho * tamanho * 3
    if X.ndim != 2 or X.shape[1] != n_entradas or y.shape != (len(X),):
        raise ValueError("Formato de imagens RGB ou rótulos inválido")
    espelho = X.reshape(-1, tamanho, tamanho, 3)[:, :, ::-1, :].reshape(-1, n_entradas)
    return np.concatenate([X, espelho]).astype(np.float32), np.concatenate([y, y])
