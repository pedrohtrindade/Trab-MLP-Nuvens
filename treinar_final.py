"""Treina o modelo final com treino + validação, sem acessar o teste.

Os hiperparâmetros e as 102 épocas foram definidos antes, com a validação.
Execute: python treinar_final.py
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.mlp import MLP
from src.treinamento import carregar_conjunto
from treinar_melhorado import aumentar_espelho


def main():
    raiz = Path(__file__).resolve().parent
    dados = raiz / "dataset" / "processado"
    X_treino, y_treino = carregar_conjunto(dados / "treino.npz")
    X_validacao, y_validacao = carregar_conjunto(dados / "validacao.npz")
    X_base = np.concatenate([X_treino, X_validacao])
    y_base = np.concatenate([y_treino, y_validacao])
    X, y = aumentar_espelho(X_base, y_base)

    rede = MLP(1024, (256, 64), 4, semente=42)
    gerador = np.random.default_rng(42)
    historico = []
    for epoca in range(1, 103):
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
        if epoca == 1 or epoca % 10 == 0 or epoca == 102:
            print(f"Época {epoca:3d} | perda treino {perda:.4f} | "
                  f"acurácia treino {acuracia:.1%}")

    caminho_modelo = raiz / "modelos" / "mlp_final_treino_validacao.npz"
    caminho_modelo.parent.mkdir(parents=True, exist_ok=True)
    rede.salvar(caminho_modelo)
    caminho_historico = raiz / "resultados" / "tabelas" / "historico_final.csv"
    caminho_historico.parent.mkdir(parents=True, exist_ok=True)
    with caminho_historico.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=(
            "epoca", "perda_treino", "acuracia_treino"))
        escritor.writeheader()
        escritor.writerows(historico)
    resumo = {
        "modelo_principal": caminho_modelo.relative_to(raiz).as_posix(),
        "sha256_modelo": hashlib.sha256(caminho_modelo.read_bytes()).hexdigest(),
        "arquitetura": list(rede.tamanhos),
        "amostras_originais_treino": len(y_treino),
        "amostras_originais_validacao": len(y_validacao),
        "amostras_com_espelhamento": len(y),
        "epocas": 102, "lote": 32, "taxa_aprendizado": 0.003,
        "semente": 42,
        "criterio_previo": "102 epocas selecionadas pela acuracia da validacao sem treino conjunto",
        "acertos_validacao_modelo_previo": 58,
        "total_validacao_modelo_previo": 113,
        "historico": caminho_historico.relative_to(raiz).as_posix(),
    }
    caminho_resumo = raiz / "resultados" / "tabelas" / "resumo_final.json"
    caminho_resumo.write_text(json.dumps(resumo, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    print(f"Modelo final salvo em: {caminho_modelo}")
    print(f"Histórico salvo em: {caminho_historico}")


if __name__ == "__main__":
    main()
