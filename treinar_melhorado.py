"""Treina uma MLP com espelhamento horizontal, sem acessar o teste.

Execute: python treinar_melhorado.py
O modelo original da avaliação final permanece intacto.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from src.mlp import MLP
from src.treinamento import carregar_conjunto, salvar_historico, treinar


def aumentar_espelho(X, y):
    """Acrescenta uma versão espelhada de cada imagem 32×32 do treino."""
    if X.ndim != 2 or X.shape[1] != 1024 or y.shape != (len(X),):
        raise ValueError("Esperadas imagens 32×32 achatadas e um rótulo por imagem")
    espelhadas = X.reshape(-1, 32, 32)[:, :, ::-1].reshape(-1, 1024)
    return np.concatenate([X, espelhadas]).astype(np.float32), np.concatenate([y, y])


def main():
    raiz = Path(__file__).resolve().parent
    dados = raiz / "dataset" / "processado"
    X_treino, y_treino = carregar_conjunto(dados / "treino.npz")
    X_validacao, y_validacao = carregar_conjunto(dados / "validacao.npz")
    X_aumentado, y_aumentado = aumentar_espelho(X_treino, y_treino)

    rede = MLP(1024, (256, 64), 4, semente=42)
    historico, melhor_epoca = treinar(
        rede, X_aumentado, y_aumentado, X_validacao, y_validacao,
        epocas=300, tamanho_lote=32, taxa_aprendizado=0.003,
        paciencia=40, semente=42, criterio="acuracia")
    probabilidades, cache = rede.forward(X_validacao, retornar_cache=True)
    acertos = int(np.sum(np.argmax(probabilidades, axis=1) == y_validacao))
    perda = rede.entropia_cruzada(cache["pre_ativacoes"][-1], y_validacao)

    caminho_modelo = raiz / "modelos" / "mlp_melhorado_validacao.npz"
    rede.salvar(caminho_modelo)
    caminho_historico = raiz / "resultados" / "tabelas" / "historico_melhorado_validacao.csv"
    salvar_historico(historico, caminho_historico)
    resumo = {
        "conjunto": "validacao", "arquitetura": list(rede.tamanhos),
        "aumento_treino": "espelhamento horizontal",
        "criterio_selecao": "maior acuracia; empate pela menor perda",
        "taxa_aprendizado": 0.003, "lote": 32, "semente": 42,
        "epocas_executadas": len(historico), "melhor_epoca": melhor_epoca,
        "total_validacao": len(y_validacao), "acertos_validacao": acertos,
        "acuracia_validacao": acertos / len(y_validacao), "perda_validacao": perda,
        "modelo": caminho_modelo.relative_to(raiz).as_posix(),
        "sha256_modelo": hashlib.sha256(caminho_modelo.read_bytes()).hexdigest(),
    }
    caminho_resumo = raiz / "resultados" / "tabelas" / "resumo_melhorado_validacao.json"
    caminho_resumo.write_text(json.dumps(resumo, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    print(f"Melhor época: {melhor_epoca}; validação: {acertos}/{len(y_validacao)} "
          f"acertos ({acertos / len(y_validacao):.1%}), perda {perda:.4f}")
    print(f"Modelo salvo em: {caminho_modelo}")
    print(f"Histórico salvo em: {caminho_historico}")
    print("O modelo da avaliação final anterior não foi substituído.")


if __name__ == "__main__":
    main()
