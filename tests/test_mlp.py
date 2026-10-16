"""Execute com: python -m unittest discover -s tests -v"""

import unittest
from pathlib import Path

import numpy as np

from src.mlp import MLP, relu, softmax


class TesteMLP(unittest.TestCase):
    def test_relu(self):
        np.testing.assert_array_equal(relu(np.array([[-2, 0, 3]])), [[0, 0, 3]])

    def test_softmax_estavel(self):
        saida = softmax(np.array([[1000., 1001., 1002.], [-1000., -1001., -1002.]]))
        self.assertTrue(np.isfinite(saida).all())
        np.testing.assert_allclose(saida.sum(axis=1), 1, atol=1e-7)
        self.assertEqual(np.argmax(saida, axis=1).tolist(), [2, 0])

    def test_inicializacao_reproduzivel_e_formatos(self):
        a = MLP(4, (3, 2), 2, semente=7)
        b = MLP(4, (3, 2), 2, semente=7)
        self.assertEqual([w.shape for w in a.pesos], [(4, 3), (3, 2), (2, 2)])
        self.assertEqual([v.shape for v in a.bias], [(1, 3), (1, 2), (1, 2)])
        for peso_a, peso_b, bias in zip(a.pesos, b.pesos, a.bias):
            np.testing.assert_array_equal(peso_a, peso_b)
            np.testing.assert_array_equal(bias, 0)

    def test_forward_com_pesos_conhecidos(self):
        rede = MLP(2, (2,), 2)
        rede.pesos[0][:] = [[1, -1], [1, 1]]
        rede.bias[0][:] = [[0, 0]]
        rede.pesos[1][:] = [[1, 0], [0, 1]]
        rede.bias[1][:] = [[0, 0]]
        resultado, cache = rede.forward([[2, -1]], retornar_cache=True)
        # z_oculta = [1, -3], ReLU = [1, 0], logits = [1, 0].
        np.testing.assert_allclose(cache["pre_ativacoes"][0], [[1, -3]])
        np.testing.assert_allclose(cache["ativacoes"][1], [[1, 0]])
        np.testing.assert_allclose(resultado, [[0.7310586, 0.2689414]], rtol=1e-6)

    def test_entrada_invalida(self):
        rede = MLP(4, (3,), 2)
        with self.assertRaises(ValueError):
            rede.forward(np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            rede.forward(np.array([[0, 0, np.nan, 0]]))

    def test_dataset_processado(self):
        caminho = Path(__file__).resolve().parents[1] / "dataset" / "processado" / "treino.npz"
        if not caminho.exists():
            self.skipTest("dataset processado indisponível")
        with np.load(caminho) as dados:
            X = dados["X"][:8]
        self.assertEqual(X.shape[1], 1024)
        probabilidades = MLP().forward(X)
        self.assertEqual(probabilidades.shape, (len(X), 4))
        self.assertTrue(np.isfinite(probabilidades).all())
        np.testing.assert_allclose(probabilidades.sum(axis=1), 1, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
