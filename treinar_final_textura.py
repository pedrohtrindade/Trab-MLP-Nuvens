"""Treina a MLP com recursos de cor/textura usando treino + validação."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.mlp import MLP
from src.preprocessamento_rgb import carregar_divisao_rgb, aumentar_espelho_rgb
from src.recursos_textura import extrair_recursos_rgb


def main():
    raiz = Path(__file__).resolve().parent
    X_treino, y_treino = carregar_divisao_rgb(raiz, "treino")
    X_validacao, y_validacao = carregar_divisao_rgb(raiz, "validacao")
    X_base = np.concatenate([X_treino, X_validacao])
    y_base = np.concatenate([y_treino, y_validacao])
    rgb_aumentado, y = aumentar_espelho_rgb(X_base, y_base)
    X = extrair_recursos_rgb(rgb_aumentado)
    rede = MLP(936, (256, 64), 4, semente=42)
    gerador = np.random.default_rng(42)
    historico = []

    for epoca in range(1, 75):
        ordem = gerador.permutation(len(y))
        for inicio in range(0, len(y), 32):
            indices = ordem[inicio:inicio + 32]
            _, cache = rede.forward(X[indices], retornar_cache=True)
            grad_pesos, grad_bias = rede.backward(cache, y[indices])
            rede.atualizar(grad_pesos, grad_bias, 0.003)
        probabilidades, cache = rede.forward(X, retornar_cache=True)
        perda = rede.entropia_cruzada(cache["pre_ativacoes"][-1], y)
        acuracia = float(np.mean(np.argmax(probabilidades, axis=1) == y))
        historico.append({"epoca": epoca, "perda_treino": perda,
                          "acuracia_treino": acuracia})
        if epoca == 1 or epoca % 10 == 0 or epoca == 74:
            print(f"Época {epoca:3d} | perda treino {perda:.4f} | "
                  f"acurácia treino {acuracia:.1%}")

    modelo = raiz / "modelos" / "mlp_textura_final.npz"
    rede.salvar(modelo)
    historico_csv = raiz / "resultados" / "tabelas" / "historico_textura_final.csv"
    with historico_csv.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=(
            "epoca", "perda_treino", "acuracia_treino"))
        escritor.writeheader()
        escritor.writerows(historico)
    resumo = {
        "modelo": modelo.relative_to(raiz).as_posix(),
        "sha256_modelo": hashlib.sha256(modelo.read_bytes()).hexdigest(),
        "entrada": "RGB 32x32 + HOG 144 + histograma RGB 24 + RGB reduzido 16x16 768",
        "arquitetura": list(rede.tamanhos),
        "amostras_treino": len(y_treino),
        "amostras_validacao": len(y_validacao),
        "amostras_com_espelhamento": len(y),
        "epocas": 74, "lote": 32, "taxa_aprendizado": 0.003, "semente": 42,
        "criterio_previo": "74 epocas escolhidas pela acuracia da validacao",
        "acertos_validacao_modelo_previo": 61,
        "total_validacao_modelo_previo": 113,
    }
    (raiz / "resultados" / "tabelas" / "resumo_textura_final.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Modelo de textura salvo em: {modelo}")


if __name__ == "__main__":
    main()
