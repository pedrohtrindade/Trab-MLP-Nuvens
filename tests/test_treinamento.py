"""Testes da etapa 4, inclusive comparação numérica dos gradientes."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.mlp import MLP
from src.treinamento import treinar


class TesteTreinamento(unittest.TestCase):
    def test_gradientes_por_diferencas_finitas(self):
        rede = MLP(2, (2,), 2, semente=3)
        # Parâmetros em float64 e ReLU longe de zero tornam a checagem precisa.
        rede.pesos = [np.array([[0.2, 0.4], [0.3, 0.5]], dtype=np.float64),
                      np.array([[0.3, -0.2], [-0.1, 0.4]], dtype=np.float64)]
        rede.bias = [np.array([[0.5, 0.6]], dtype=np.float64),
                     np.array([[0.1, -0.1]], dtype=np.float64)]
        X = np.array([[0.2, 0.7], [0.8, 0.3]], dtype=np.float32)
        y = np.array([0, 1])
        _, cache = rede.forward(X, retornar_cache=True)
        grad_pesos, grad_bias = rede.backward(cache, y)

        epsilon = 1e-5
        for parametros, gradientes in ((rede.pesos, grad_pesos), (rede.bias, grad_bias)):
            for matriz, gradiente in zip(parametros, gradientes):
                for indice in np.ndindex(matriz.shape):
                    valor = matriz[indice]
                    matriz[indice] = valor + epsilon
                    perda_mais = rede.perda(X, y)
                    matriz[indice] = valor - epsilon
                    perda_menos = rede.perda(X, y)
                    matriz[indice] = valor
                    derivada_numerica = (perda_mais - perda_menos) / (2 * epsilon)
                    self.assertAlmostEqual(gradiente[indice], derivada_numerica, places=6)

    def test_passo_reduz_perda(self):
        rede = MLP(2, (), 2, semente=5)
        X = np.array([[1, 0], [0, 1]], dtype=np.float32)
        y = np.array([0, 1])
        antes = rede.perda(X, y)
        _, cache = rede.forward(X, retornar_cache=True)
        rede.atualizar(*rede.backward(cache, y), taxa_aprendizado=0.1)
        self.assertLess(rede.perda(X, y), antes)

    def test_treinamento_e_restauracao_do_melhor_modelo(self):
        rede = MLP(2, (), 2, semente=5)
        X = np.array([[1, 0], [0, 1], [0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
        y = np.array([0, 1, 0, 1])
        historico, melhor_epoca = treinar(
            rede, X, y, X, y, epocas=30, tamanho_lote=2,
            taxa_aprendizado=0.2, paciencia=5)
        self.assertGreaterEqual(melhor_epoca, 1)
        self.assertLess(rede.perda(X, y), historico[0]["perda_treino"])

    def test_salvar_e_carregar(self):
        rede = MLP(3, (4,), 2, semente=8)
        X = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "modelo.npz"
            rede.salvar(caminho)
            restaurada = MLP.carregar(caminho)
        np.testing.assert_array_equal(rede.forward(X), restaurada.forward(X))
        self.assertEqual(rede.tamanhos, restaurada.tamanhos)

    def test_rotulos_invalidos(self):
        rede = MLP(2, (), 2)
        with self.assertRaises(ValueError):
            rede.perda(np.ones((1, 2)), np.array([2]))


if __name__ == "__main__":
    unittest.main()
