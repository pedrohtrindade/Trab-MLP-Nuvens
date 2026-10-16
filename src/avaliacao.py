"""Avaliação final da MLP: usa o teste somente após a escolha do modelo."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import CLASSES
from .mlp import MLP
from .treinamento import carregar_conjunto


def matriz_confusao(y_verdadeiro, y_previsto, n_classes):
    """Linhas = classes reais; colunas = classes previstas."""
    y_verdadeiro = np.asarray(y_verdadeiro)
    y_previsto = np.asarray(y_previsto)
    if (y_verdadeiro.ndim != 1 or y_previsto.shape != y_verdadeiro.shape
            or not np.issubdtype(y_verdadeiro.dtype, np.integer)
            or not np.issubdtype(y_previsto.dtype, np.integer)
            or y_verdadeiro.size == 0):
        raise ValueError("Os rótulos reais e previstos devem ser vetores inteiros não vazios")
    if (np.any(y_verdadeiro < 0) or np.any(y_verdadeiro >= n_classes)
            or np.any(y_previsto < 0) or np.any(y_previsto >= n_classes)):
        raise ValueError("Rótulo fora do intervalo das classes")
    matriz = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matriz, (y_verdadeiro, y_previsto), 1)
    return matriz


def metricas_por_classe(matriz, classes):
    """Calcula acertos, precisão, recall e F1 para cada classe."""
    if matriz.shape != (len(classes), len(classes)):
        raise ValueError("Matriz e lista de classes incompatíveis")
    linhas = []
    for indice, classe in enumerate(classes):
        acertos = int(matriz[indice, indice])
        suporte = int(matriz[indice, :].sum())
        previstos = int(matriz[:, indice].sum())
        precisao = acertos / previstos if previstos else 0.0
        recall = acertos / suporte if suporte else 0.0
        f1 = 2 * precisao * recall / (precisao + recall) if precisao + recall else 0.0
        linhas.append({
            "classe": classe, "suporte": suporte, "acertos": acertos,
            "precisao": precisao, "recall": recall, "f1": f1,
        })
    return linhas


def salvar_csv(caminho, campos, linhas):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(linhas)


def salvar_grafico(matriz, classes, caminho):
    """Cria figura da matriz com números legíveis para o artigo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figura, eixo = plt.subplots(figsize=(7, 6))
    imagem = eixo.imshow(matriz, cmap="Blues", vmin=0)
    figura.colorbar(imagem, ax=eixo, label="Número de imagens")
    eixo.set_xticks(range(len(classes)), labels=[c.capitalize() for c in classes], rotation=30)
    eixo.set_yticks(range(len(classes)), labels=[c.capitalize() for c in classes])
    eixo.set_xlabel("Classe prevista")
    eixo.set_ylabel("Classe real")
    eixo.set_title("Matriz de confusão — conjunto de teste")
    limite = matriz.max() / 2
    for linha in range(len(classes)):
        for coluna in range(len(classes)):
            eixo.text(coluna, linha, str(matriz[linha, coluna]),
                      ha="center", va="center",
                      color="white" if matriz[linha, coluna] > limite else "black")
    figura.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(figura)


def avaliar_modelo(raiz, modelo_relativo="modelos/mlp_etapa4.npz",
                  sufixo="teste", observacao=None, dados_teste=None):
    """Avalia o modelo indicado e grava resultados com nomes próprios."""
    raiz = Path(raiz)
    caminho_modelo = raiz / modelo_relativo
    rede = MLP.carregar(caminho_modelo)
    if rede.tamanhos[-1] != len(CLASSES):
        raise ValueError("A saída do modelo não corresponde às classes configuradas")
    if dados_teste is None:
        X, y = carregar_conjunto(
            raiz / "dataset" / "processado" / "teste.npz",
            n_entradas=rede.tamanhos[0], n_classes=len(CLASSES))
    else:
        X, y = dados_teste

    probabilidades, cache = rede.forward(X, retornar_cache=True)
    previstos = np.argmax(probabilidades, axis=1)
    matriz = matriz_confusao(y, previstos, len(CLASSES))
    por_classe = metricas_por_classe(matriz, CLASSES)
    acertos = int(np.trace(matriz))
    resumo = {
        "conjunto": "teste",
        "modelo": caminho_modelo.relative_to(raiz).as_posix(),
        "sha256_modelo": hashlib.sha256(caminho_modelo.read_bytes()).hexdigest(),
        "classes": list(CLASSES),
        "total": len(y),
        "acertos": acertos,
        "acuracia": acertos / len(y),
        "perda_entropia_cruzada": rede.entropia_cruzada(cache["pre_ativacoes"][-1], y),
        "precisao_macro": float(np.mean([linha["precisao"] for linha in por_classe])),
        "recall_macro": float(np.mean([linha["recall"] for linha in por_classe])),
        "f1_macro": float(np.mean([linha["f1"] for linha in por_classe])),
    }
    if observacao:
        resumo["observacao"] = observacao

    tabelas = raiz / "resultados" / "tabelas"
    tabelas.mkdir(parents=True, exist_ok=True)
    (tabelas / f"avaliacao_{sufixo}.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    salvar_csv(tabelas / f"metricas_por_classe_{sufixo}.csv",
               ("classe", "suporte", "acertos", "precisao", "recall", "f1"), por_classe)
    with (tabelas / f"matriz_confusao_{sufixo}.csv").open(
            "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["classe_real\\prevista", *CLASSES])
        for classe, valores in zip(CLASSES, matriz):
            escritor.writerow([classe, *valores.tolist()])
    salvar_grafico(matriz, CLASSES,
                   raiz / "resultados" / "graficos" / f"matriz_confusao_{sufixo}.png")

    print(f"Teste: {acertos}/{len(y)} acertos; acurácia {resumo['acuracia']:.1%}; "
          f"perda {resumo['perda_entropia_cruzada']:.4f}; F1 macro {resumo['f1_macro']:.3f}")
    for linha in por_classe:
        print(f"{linha['classe']:15s}: {linha['acertos']:2d}/{linha['suporte']:2d} "
              f"acertos; precisão {linha['precisao']:.3f}; "
              f"recall {linha['recall']:.3f}; F1 {linha['f1']:.3f}")
    print(f"Resultados salvos em: {tabelas} e {raiz / 'resultados' / 'graficos'}")
    return resumo, matriz, por_classe
