"""Modelo RGB final, com treino + validação e parâmetros fixados antes do teste."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.mlp import MLP
from src.preprocessamento_rgb import carregar_divisao_rgb, aumentar_espelho_rgb


def main():
    raiz = Path(__file__).resolve().parent
    X_treino, y_treino = carregar_divisao_rgb(raiz, "treino")
    X_validacao, y_validacao = carregar_divisao_rgb(raiz, "validacao")
    X_base = np.concatenate([X_treino, X_validacao])
    y_base = np.concatenate([y_treino, y_validacao])
    X, y = aumentar_espelho_rgb(X_base, y_base)
    rede = MLP(3072, (256, 64), 4, semente=42)
    gerador = np.random.default_rng(42)
    historico = []

    for epoca in range(1, 95):
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
        if epoca == 1 or epoca % 10 == 0 or epoca == 94:
            print(f"Época {epoca:3d} | perda treino {perda:.4f} | "
                  f"acurácia treino {acuracia:.1%}")

    caminho_modelo = raiz / "modelos" / "mlp_rgb_final.npz"
    caminho_modelo.parent.mkdir(parents=True, exist_ok=True)
    rede.salvar(caminho_modelo)
    caminho_historico = raiz / "resultados" / "tabelas" / "historico_rgb_final.csv"
    with caminho_historico.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=(
            "epoca", "perda_treino", "acuracia_treino"))
        escritor.writeheader()
        escritor.writerows(historico)
    resumo = {
        "modelo": caminho_modelo.relative_to(raiz).as_posix(),
        "sha256_modelo": hashlib.sha256(caminho_modelo.read_bytes()).hexdigest(),
        "preprocessamento": "RGB 32x32 normalizado em [0,1]",
        "aumento_treino": "espelhamento horizontal",
        "arquitetura": list(rede.tamanhos),
        "amostras_treino": len(y_treino),
        "amostras_validacao": len(y_validacao),
        "amostras_com_aumento": len(y),
        "epocas": 94, "lote": 32, "taxa_aprendizado": 0.003,
        "semente": 42,
        "criterio_previo": "94 epocas escolhidas pela acuracia da validacao RGB",
        "acertos_validacao_modelo_previo": 59,
        "total_validacao_modelo_previo": 113,
    }
    caminho_resumo = raiz / "resultados" / "tabelas" / "resumo_rgb_final.json"
    caminho_resumo.write_text(json.dumps(resumo, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    print(f"Modelo RGB salvo em: {caminho_modelo}")


if __name__ == "__main__":
    main()
