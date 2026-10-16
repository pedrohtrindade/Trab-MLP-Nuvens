"""Descritores de cor e textura em NumPy para entrada da MLP."""

import numpy as np

from .preprocessamento_rgb import carregar_divisao_rgb, processar_imagem_rgb


def extrair_recursos_rgb(vetores):
    """Retorna 936 recursos: HOG 144, histograma RGB 24 e RGB 16×16 768.

    A entrada contém imagens RGB 32×32 achatadas, normalizadas em [0, 1].
    """
    vetores = np.asarray(vetores, dtype=np.float32)
    if vetores.ndim != 2 or vetores.shape[1] != 3072:
        raise ValueError("Esperada matriz de imagens RGB 32×32 com 3072 colunas")
    rgb = vetores.reshape(-1, 32, 32, 3)
    cinza = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    gx = np.zeros_like(cinza)
    gy = np.zeros_like(cinza)
    gx[:, :, 1:-1] = cinza[:, :, 2:] - cinza[:, :, :-2]
    gy[:, 1:-1, :] = cinza[:, 2:, :] - cinza[:, :-2, :]
    magnitude = np.sqrt(gx * gx + gy * gy)
    angulo = np.mod(np.arctan2(gy, gx), np.pi)
    indices = np.minimum((angulo * (9 / np.pi)).astype(np.int32), 8)

    celulas = []
    for linha in range(4):
        for coluna in range(4):
            m = magnitude[:, linha*8:(linha+1)*8, coluna*8:(coluna+1)*8]
            b = indices[:, linha*8:(linha+1)*8, coluna*8:(coluna+1)*8]
            hist = np.stack([(m * (b == i)).sum(axis=(1, 2)) for i in range(9)], axis=1)
            hist /= np.sqrt((hist * hist).sum(axis=1, keepdims=True) + 1e-6)
            celulas.append(hist)
    hog = np.concatenate(celulas, axis=1)

    bins_rgb = np.minimum((rgb * 8).astype(np.int32), 7)
    hist_cor = np.stack([np.mean(bins_rgb[..., canal] == i, axis=(1, 2))
                         for canal in range(3) for i in range(8)], axis=1)
    pequeno = rgb.reshape(-1, 16, 2, 16, 2, 3).mean(axis=(2, 4)).reshape(-1, 768)
    return np.concatenate([hog, hist_cor, pequeno], axis=1).astype(np.float32)


def processar_imagem_textura(caminho):
    """Prepara uma imagem externa exatamente como no treinamento."""
    return extrair_recursos_rgb(processar_imagem_rgb(caminho)[None, :])[0]


def carregar_divisao_textura(raiz, conjunto):
    """Lê uma divisão e extrai recursos sem alterar suas imagens."""
    X, y = carregar_divisao_rgb(raiz, conjunto)
    return extrair_recursos_rgb(X), y
