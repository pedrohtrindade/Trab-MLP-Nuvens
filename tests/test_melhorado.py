"""Testes do aumento de dados e da seleção por acurácia."""

import unittest

import numpy as np

from src.mlp import MLP
from src.treinamento import treinar
from treinar_melhorado import aumentar_espelho


class TesteMelhorado(unittest.TestCase):
    def test_espelho_preserva_rotulos_e_inverte_colunas(self):
        X = np.arange(1024, dtype=np.float32).reshape(1, 1024)
        y = np.array([2])
        aumentado, rotulos = aumentar_espelho(X, y)
        self.assertEqual(aumentado.shape, (2, 1024))
        np.testing.assert_array_equal(rotulos, [2, 2])
        np.testing.assert_array_equal(aumentado[1].reshape(32, 32),
                                      X[0].reshape(32, 32)[:, ::-1])

    def test_selecao_por_acuracia(self):
        X = np.array([[1, 0], [0, 1], [0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
        y = np.array([0, 1, 0, 1])
        rede = MLP(2, (), 2, semente=5)
        historico, melhor_epoca = treinar(
            rede, X, y, X, y, epocas=12, tamanho_lote=2,
            taxa_aprendizado=0.2, paciencia=12, criterio="acuracia")
        esperado = min(historico, key=lambda linha: (
            -linha["acuracia_validacao"], linha["perda_validacao"]))
        self.assertEqual(melhor_epoca, esperado["epoca"])
        self.assertAlmostEqual(
            np.mean(rede.forward(X).argmax(axis=1) == y),
            esperado["acuracia_validacao"])


if __name__ == "__main__":
    unittest.main()
