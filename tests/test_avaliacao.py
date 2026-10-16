"""Testes das métricas sem acessar o conjunto de teste reservado."""

import unittest

import numpy as np

from src.avaliacao import matriz_confusao, metricas_por_classe


class TesteAvaliacao(unittest.TestCase):
    def test_matriz_e_metricas_por_classe(self):
        reais = np.array([0, 0, 1, 1, 2])
        previstos = np.array([0, 1, 1, 1, 0])
        matriz = matriz_confusao(reais, previstos, 3)
        np.testing.assert_array_equal(matriz, [[1, 1, 0], [0, 2, 0], [1, 0, 0]])
        linhas = metricas_por_classe(matriz, ("a", "b", "c"))
        self.assertEqual([linha["suporte"] for linha in linhas], [2, 2, 1])
        self.assertEqual([linha["acertos"] for linha in linhas], [1, 2, 0])
        self.assertAlmostEqual(linhas[0]["precisao"], 0.5)
        self.assertAlmostEqual(linhas[0]["recall"], 0.5)
        self.assertAlmostEqual(linhas[1]["precisao"], 2 / 3)
        self.assertAlmostEqual(linhas[1]["f1"], 0.8)
        self.assertEqual(linhas[2]["f1"], 0)

    def test_rotulos_invalidos(self):
        with self.assertRaises(ValueError):
            matriz_confusao(np.array([0, 1]), np.array([0, 3]), 3)


if __name__ == "__main__":
    unittest.main()
