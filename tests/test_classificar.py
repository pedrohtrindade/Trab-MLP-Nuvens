"""Verifica o preparo das entradas para modelos já suportados."""

import csv
import unittest
from pathlib import Path

from classificar_imagem import preparar_entrada


class TesteClassificar(unittest.TestCase):
    def test_formatos_de_entrada(self):
        raiz = Path(__file__).resolve().parents[1]
        with (raiz / "dataset/processado/divisao.csv").open(encoding="utf-8", newline="") as arquivo:
            primeira = next(r for r in csv.DictReader(arquivo) if r["conjunto"] == "treino")
        caminho = raiz / primeira["arquivo"]
        self.assertEqual(preparar_entrada(caminho, 1024).shape, (1024,))
        self.assertEqual(preparar_entrada(caminho, 3072).shape, (3072,))
        self.assertEqual(preparar_entrada(caminho, 936).shape, (936,))


if __name__ == "__main__":
    unittest.main()
