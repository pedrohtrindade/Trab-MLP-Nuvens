"""Prepara o CCSN para uma MLP clássica (sem treinar a rede)."""
from pathlib import Path
import csv
import hashlib

import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

from .config import CLASSES, TAMANHO_IMAGEM, SEMENTE

EXTENSOES = {'.jpg', '.jpeg', '.png'}


def processar_imagem(caminho):
    """Converte imagem em vetor float32 normalizado de tamanho 32*32."""
    with Image.open(caminho) as imagem:
        imagem = ImageOps.exif_transpose(imagem).convert('L')
        imagem = imagem.resize(TAMANHO_IMAGEM, Image.Resampling.LANCZOS)
        valores = np.asarray(imagem, dtype=np.float32) / 255.0
    return valores.reshape(-1)


def preparar_dataset(raiz):
    raiz = Path(raiz)
    origem = raiz / 'dataset' / 'original'
    destino = raiz / 'dataset' / 'processado'
    destino.mkdir(parents=True, exist_ok=True)

    vetores, rotulos, caminhos, hashes = [], [], [], set()
    for indice, classe in enumerate(CLASSES):
        pasta = origem / classe
        if not pasta.is_dir():
            raise FileNotFoundError(f'Pasta da classe não encontrada: {pasta}')
        arquivos = sorted(p for p in pasta.iterdir() if p.suffix.lower() in EXTENSOES)
        if not arquivos:
            raise ValueError(f'Nenhuma imagem encontrada em {pasta}')
        for arquivo in arquivos:
            try:
                conteudo = arquivo.read_bytes()
                assinatura = hashlib.sha256(conteudo).hexdigest()
                if assinatura in hashes:
                    print(f'Aviso: imagem idêntica ignorada: {arquivo.name}')
                    continue
                vetor = processar_imagem(arquivo)
            except (OSError, ValueError) as erro:
                print(f'Aviso: erro ao abrir {arquivo.name}: {erro}')
                continue
            hashes.add(assinatura)
            vetores.append(vetor)
            rotulos.append(indice)
            caminhos.append(arquivo.relative_to(raiz).as_posix())

    X = np.stack(vetores).astype(np.float32)
    y = np.asarray(rotulos, dtype=np.int64)
    indices = np.arange(len(y))

    # 70% treino, 15% validação, 15% teste, aproximadamente, com estratificação.
    treino, restante = train_test_split(indices, test_size=0.30, random_state=SEMENTE, stratify=y)
    validacao, teste = train_test_split(restante, test_size=0.50, random_state=SEMENTE, stratify=y[restante])
    partes = {'treino': treino, 'validacao': validacao, 'teste': teste}

    for nome, idx in partes.items():
        np.savez_compressed(destino / f'{nome}.npz', X=X[idx], y=y[idx])

    with (destino / 'divisao.csv').open('w', newline='', encoding='utf-8') as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(['arquivo', 'classe', 'rotulo', 'conjunto'])
        for nome, idx in partes.items():
            for i in idx:
                escritor.writerow([caminhos[i], CLASSES[y[i]], int(y[i]), nome])

    print(f'Imagens processadas: {len(y)} | Entradas por imagem: {X.shape[1]}')
    print('Classes (rótulos numéricos):', ', '.join(f'{i}={c}' for i, c in enumerate(CLASSES)))
    for nome, idx in partes.items():
        distribuicao = ', '.join(f'{classe}={int(np.sum(y[idx] == i))}' for i, classe in enumerate(CLASSES))
        print(f'{nome.capitalize():10s}: {len(idx):3d} imagens | {distribuicao}')
    print(f'Arquivos salvos em: {destino}')
    return partes
