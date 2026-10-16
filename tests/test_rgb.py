"""Testa a entrada RGB sem acessar o conjunto de teste."""

import unittest
from pathlib import Path

import numpy as np

from src.preprocessamento_rgb import (aumentar_espelho_rgb,
                                     carregar_divisao_rgb, processar_imagem_rgb)


class TesteRGB(unittest.TestCase):
    def test_espelho_rgb(self):
        X = np.arange(32 * 32 * 3, dtype=np.float32).reshape(1, -1)
        y = np.array([1])
        Xa, ya = aumentar_espelho_rgb(X, y)
        np.testing.assert_array_equal(ya, [1, 1])
        np.testing.assert_array_equal(Xa[1].reshape(32, 32, 3),
                                      X[0].reshape(32, 32, 3)[:, ::-1, :])

    def test_primeira_imagem_de_treino(self):
        raiz = Path(__file__).resolve().parents[1]
        X, y = carregar_divisao_rgb(raiz, "treino")
        self.assertEqual(X.shape, (527, 3072))
        self.assertEqual(y.shape, (527,))
        self.assertTrue(np.isfinite(X).all())
        self.assertGreaterEqual(float(X.min()), 0)
        self.assertLessEqual(float(X.max()), 1)


if __name__ == "__main__":
    unittest.main()
