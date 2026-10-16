"""Compara treinos de MLP usando apenas treino e validação.

Execute no projeto com: python comparar_experimentos.py
Cada execução conserva seu modelo, histórico e log. O resumo pode ser usado
para montar a tabela do artigo. O conjunto de teste não é carregado aqui.
"""

import contextlib
import csv
import shutil
from pathlib import Path

import numpy as np

from src.mlp import MLP
from src.treinamento import carregar_conjunto, treinar, salvar_historico


RAIZ = Path(__file__).resolve().parent
SEMENTE = 42
PACIENCIA = 25
EXPERIMENTOS = (
    # A execução 02 foi preservada do treino feito pelo usuário.
    ("01", (256, 64), 0.003, 32, 180),
    ("02", (256, 64), 0.002, 32, 250),
    ("03", (256, 64), 0.002, 32, 350),
    ("04", (256, 64), 0.003, 32, 350),
    ("05", (256, 64), 0.003, 16, 250),
    ("06", (256, 64), 0.003, 64, 250),
)
CAMPOS = (
    "execucao", "camadas_intermediarias", "neuronios", "taxa_aprendizado",
    "lote", "epocas_max", "epocas_executadas", "melhor_epoca",
    "perda_validacao", "acuracia_validacao", "acertos_validacao",
    "total_validacao", "semente", "modelo", "historico", "log",
)


def caminhos(numero):
    return (
        RAIZ / "modelos" / f"mlp_execucao_{numero}.npz",
        RAIZ / "resultados" / "tabelas" / f"historico_execucao_{numero}.csv",
        RAIZ / "resultados" / "tabelas" / f"log_execucao_{numero}.txt",
    )


def salvar_resumo(linhas):
    destino = RAIZ / "resultados" / "tabelas" / "resumo_experimentos.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=CAMPOS)
        escritor.writeheader()
        escritor.writerows(linhas)
    return destino


def ler_historico(caminho):
    with caminho.open(newline="", encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    if not linhas:
        raise ValueError(f"Histórico vazio: {caminho}")
    return linhas


def main():
    dados = RAIZ / "dataset" / "processado"
    X_treino, y_treino = carregar_conjunto(dados / "treino.npz")
    X_validacao, y_validacao = carregar_conjunto(dados / "validacao.npz")
    linhas = []

    for numero, ocultas, taxa, lote, epocas in EXPERIMENTOS:
        modelo, historico, log = caminhos(numero)
        if modelo.exists() and historico.exists():
            print(f"Execução {numero}: arquivos existentes preservados.", flush=True)
        elif modelo.exists() or historico.exists():
            raise RuntimeError(f"Execução {numero}: só um dos arquivos existe; confira antes de continuar")
        else:
            print(f"Execução {numero}: iniciando {ocultas}, taxa={taxa}, lote={lote}, "
                  f"até {epocas} épocas.", flush=True)
            rede = MLP(X_treino.shape[1], ocultas, 4, SEMENTE)
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w", encoding="utf-8") as arquivo, contextlib.redirect_stdout(arquivo):
                registros, melhor_epoca = treinar(
                    rede, X_treino, y_treino, X_validacao, y_validacao,
                    epocas=epocas, tamanho_lote=lote,
                    taxa_aprendizado=taxa, paciencia=PACIENCIA, semente=SEMENTE)
                print(f"Melhor época: {melhor_epoca}")
            rede.salvar(modelo)
            salvar_historico(registros, historico)

        rede = MLP.carregar(modelo)
        probabilidades, cache = rede.forward(X_validacao, retornar_cache=True)
        perda = rede.entropia_cruzada(cache["pre_ativacoes"][-1], y_validacao)
        acertos = int(np.sum(np.argmax(probabilidades, axis=1) == y_validacao))
        registros = ler_historico(historico)
        melhor = min(registros, key=lambda linha: float(linha["perda_validacao"]))
        linha = {
            "execucao": numero,
            "camadas_intermediarias": len(ocultas),
            "neuronios": "-".join(map(str, ocultas)),
            "taxa_aprendizado": taxa,
            "lote": lote,
            "epocas_max": epocas,
            "epocas_executadas": len(registros),
            "melhor_epoca": int(melhor["epoca"]),
            "perda_validacao": perda,
            "acuracia_validacao": acertos / len(y_validacao),
            "acertos_validacao": acertos,
            "total_validacao": len(y_validacao),
            "semente": SEMENTE,
            "modelo": modelo.relative_to(RAIZ).as_posix(),
            "historico": historico.relative_to(RAIZ).as_posix(),
            "log": log.relative_to(RAIZ).as_posix() if log.exists() else "",
        }
        linhas.append(linha)
        destino = salvar_resumo(linhas)
        print(f"Execução {numero}: melhor época {linha['melhor_epoca']}, "
              f"perda {perda:.4f}, {acertos}/{len(y_validacao)} acertos "
              f"({linha['acuracia_validacao']:.1%}).", flush=True)

    melhor = min(linhas, key=lambda linha: float(linha["perda_validacao"]))
    shutil.copy2(RAIZ / melhor["modelo"], RAIZ / "modelos" / "mlp_etapa4.npz")
    shutil.copy2(RAIZ / melhor["historico"],
                 RAIZ / "resultados" / "tabelas" / "historico_etapa4.csv")
    print(f"Selecionada execução {melhor['execucao']} pela menor perda de validação.")
    print(f"Resumo salvo em: {destino}")


if __name__ == "__main__":
    main()
