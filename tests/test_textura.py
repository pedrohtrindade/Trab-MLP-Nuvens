"""Testes do vetor de recursos de cor e textura."""

import unittest

import numpy as np

from src.recursos_textura import extrair_recursos_rgb


class TesteTextura(unittest.TestCase):
    def test_formato_e_valores(self):
        rng = np.random.default_rng(2)
        X = rng.random((3, 3072), dtype=np.float32)
        recursos = extrair_recursos_rgb(X)
        self.assertEqual(recursos.shape, (3, 936))
        self.assertEqual(recursos.dtype, np.float32)
        self.assertTrue(np.isfinite(recursos).all())
        np.testing.assert_allclose(recursos[:, 144:168].sum(axis=1), 3, atol=1e-6)

    def test_formato_invalido(self):
        with self.assertRaises(ValueError):
            extrair_recursos_rgb(np.zeros((1, 1024), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
