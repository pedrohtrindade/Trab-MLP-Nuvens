"""Treinamento em mini-lotes de uma MLP implementada apenas com NumPy."""

import csv
from pathlib import Path

import numpy as np

from .mlp import MLP


def carregar_conjunto(caminho, n_entradas=1024, n_classes=4):
    """Lê um conjunto preparado na etapa 2 e confere formato e rótulos."""
    with np.load(caminho, allow_pickle=False) as dados:
        X = dados["X"].astype(np.float32)
        y = dados["y"]
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != n_entradas:
        raise ValueError(f"X em {caminho} deve ter formato (amostras, {n_entradas})")
    if y.shape != (len(X),) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError(f"y em {caminho} deve conter rótulos inteiros")
    if not np.all(np.isfinite(X)) or np.any(y < 0) or np.any(y >= n_classes):
        raise ValueError(f"Valores inválidos em {caminho}")
    return X, y


def metricas(rede, X, y):
    """Retorna perda e acurácia para um conjunto completo."""
    probabilidades, cache = rede.forward(X, retornar_cache=True)
    perda = rede.entropia_cruzada(cache["pre_ativacoes"][-1], y)
    acuracia = float(np.mean(np.argmax(probabilidades, axis=1) == y))
    return perda, acuracia


def treinar(rede, X_treino, y_treino, X_validacao, y_validacao, *,
            epocas=180, tamanho_lote=32, taxa_aprendizado=0.003,
            paciencia=25, semente=42, criterio="perda"):
    """Treina por SGD e restaura os pesos da melhor época na validação.

    O critério pode ser menor perda (padrão) ou maior acurácia; empates em
    acurácia são resolvidos pela menor perda.
    """
    if epocas <= 0 or tamanho_lote <= 0 or paciencia <= 0:
        raise ValueError("Épocas, tamanho do lote e paciência devem ser positivos")
    if not np.isfinite(taxa_aprendizado) or taxa_aprendizado <= 0:
        raise ValueError("A taxa de aprendizado deve ser positiva e finita")
    if criterio not in {"perda", "acuracia"}:
        raise ValueError("O critério deve ser 'perda' ou 'acuracia'")
    if len(X_treino) != len(y_treino) or len(X_validacao) != len(y_validacao):
        raise ValueError("Cada imagem deve ter um rótulo correspondente")

    gerador = np.random.default_rng(semente)
    historico = []
    melhor_perda = float("inf")
    melhor_acuracia = -1.0
    melhor_epoca = 0
    melhores_pesos = None
    melhores_bias = None

    for epoca in range(1, epocas + 1):
        ordem = gerador.permutation(len(X_treino))
        for inicio in range(0, len(ordem), tamanho_lote):
            indices = ordem[inicio:inicio + tamanho_lote]
            _, cache = rede.forward(X_treino[indices], retornar_cache=True)
            grad_pesos, grad_bias = rede.backward(cache, y_treino[indices])
            rede.atualizar(grad_pesos, grad_bias, taxa_aprendizado)

        perda_treino, acuracia_treino = metricas(rede, X_treino, y_treino)
        perda_validacao, acuracia_validacao = metricas(rede, X_validacao, y_validacao)
        if not np.isfinite(perda_treino) or not np.isfinite(perda_validacao):
            raise FloatingPointError("A perda ficou não finita; reduza a taxa de aprendizado")
        historico.append({
            "epoca": epoca,
            "perda_treino": perda_treino,
            "acuracia_treino": acuracia_treino,
            "perda_validacao": perda_validacao,
            "acuracia_validacao": acuracia_validacao,
        })
        print(f"Época {epoca:3d} | treino: perda {perda_treino:.4f}, "
              f"acurácia {acuracia_treino:.1%} | validação: "
              f"perda {perda_validacao:.4f}, acurácia {acuracia_validacao:.1%}")

        melhorou = (perda_validacao < melhor_perda - 1e-4 if criterio == "perda"
                   else (acuracia_validacao > melhor_acuracia or
                         (acuracia_validacao == melhor_acuracia and
                          perda_validacao < melhor_perda)))
        if melhorou:
            melhor_perda = perda_validacao
            melhor_acuracia = acuracia_validacao
            melhor_epoca = epoca
            melhores_pesos = [w.copy() for w in rede.pesos]
            melhores_bias = [b.copy() for b in rede.bias]
        if epoca - melhor_epoca >= paciencia:
            print(f"Parada antecipada: sem melhora na validação por {paciencia} épocas.")
            break

    rede.pesos = melhores_pesos
    rede.bias = melhores_bias
    return historico, melhor_epoca


def salvar_historico(historico, caminho):
    """Grava as métricas de cada época em CSV para inspeção posterior."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=(
            "epoca", "perda_treino", "acuracia_treino",
            "perda_validacao", "acuracia_validacao"))
        escritor.writeheader()
        escritor.writerows(historico)


def executar_treinamento(raiz, *, epocas=180, tamanho_lote=32,
                         taxa_aprendizado=0.003, paciencia=25,
                         camadas_ocultas=(256, 64), semente=42):
    """Carrega treino e validação, treina e salva o melhor modelo."""
    raiz = Path(raiz)
    dados = raiz / "dataset" / "processado"
    X_treino, y_treino = carregar_conjunto(dados / "treino.npz")
    X_validacao, y_validacao = carregar_conjunto(dados / "validacao.npz")
    rede = MLP(X_treino.shape[1], camadas_ocultas, 4, semente)
    historico, melhor_epoca = treinar(
        rede, X_treino, y_treino, X_validacao, y_validacao,
        epocas=epocas, tamanho_lote=tamanho_lote,
        taxa_aprendizado=taxa_aprendizado, paciencia=paciencia, semente=semente)

    caminho_modelo = raiz / "modelos" / "mlp_etapa4.npz"
    caminho_modelo.parent.mkdir(parents=True, exist_ok=True)
    rede.salvar(caminho_modelo)
    caminho_historico = raiz / "resultados" / "tabelas" / "historico_etapa4.csv"
    salvar_historico(historico, caminho_historico)
    perda, acuracia = metricas(rede, X_validacao, y_validacao)
    print(f"Melhor época: {melhor_epoca}; validação: perda {perda:.4f}, "
          f"acurácia {acuracia:.1%}")
    print(f"Modelo salvo em: {caminho_modelo}")
    print(f"Histórico salvo em: {caminho_historico}")
    return rede, historico, melhor_epoca
