"""Pruebas de corrección, forma de la salida y robustez numérica de las 9 variantes de QDA.

Se ejecutan con la biblioteca estándar (sin dependencias extra), desde la raíz del repositorio:

    python -m unittest discover -s tests -t . -v

Con uv:  uv run python -m unittest discover -s tests -t . -v
"""
import tracemalloc
import unittest

import numpy as np

from base.cholesky import QDA_Chol1, QDA_Chol2, QDA_Chol3, TensorizedChol, EfficientChol
from base.qda import QDA, TensorizedQDA, FasterQDA, EfficientQDA
from utils.datasets import (get_iris_dataset, get_penguins_dataset, get_wine_dataset,
                            label_encode, split_transpose)
from utils.verification import (dataset_alta_dimension, prediccion_oraculo,
                                verificar_identidades_cholesky, verificar_truco_diagonal)

MODELOS = {
    'QDA': QDA, 'TensorizedQDA': TensorizedQDA, 'FasterQDA': FasterQDA, 'EfficientQDA': EfficientQDA,
    'QDA_Chol1': QDA_Chol1, 'QDA_Chol2': QDA_Chol2, 'QDA_Chol3': QDA_Chol3,
    'TensorizedChol': TensorizedChol, 'EfficientChol': EfficientChol,
}
SEMILLAS = (0, 1, 2)


def _split(cargador, semilla, test_size=0.3):
    X, y = cargador()
    return split_transpose(X, label_encode(y), test_size=test_size, random_state=semilla)


def _ajustar(clase, X_train, y_train):
    modelo = clase()
    modelo.fit(X_train, y_train)
    return modelo


class EquivalenciaConOraculo(unittest.TestCase):
    """Las 9 variantes deben predecir exactamente lo mismo que QDA calculado con scipy.stats."""

    def _verificar(self, cargador):
        for semilla in SEMILLAS:
            X_train, X_test, y_train, _ = _split(cargador, semilla)
            esperado = prediccion_oraculo(X_train, y_train, X_test)
            for nombre, clase in MODELOS.items():
                with self.subTest(modelo=nombre, semilla=semilla):
                    obtenido = _ajustar(clase, X_train, y_train).predict(X_test)
                    np.testing.assert_array_equal(obtenido, esperado)

    def test_iris(self):
        self._verificar(get_iris_dataset)

    def test_wine(self):
        self._verificar(get_wine_dataset)

    def test_penguins(self):
        try:
            get_penguins_dataset()                   # se descarga de OpenML
        except Exception as error:                   # sin red: no es una falla del código
            self.skipTest(f'no se pudo descargar penguins: {error}')
        self._verificar(get_penguins_dataset)


class FormaDeLaSalida(unittest.TestCase):
    """`predict` devuelve un vector fila (1, n) de enteros en [0, k-1], como `y`."""

    def test_shape_y_tipo(self):
        X_train, X_test, y_train, y_test = _split(get_wine_dataset, semilla=0)
        k = len(np.unique(y_train))
        for nombre, clase in MODELOS.items():
            with self.subTest(modelo=nombre):
                prediccion = _ajustar(clase, X_train, y_train).predict(X_test)
                self.assertEqual(prediccion.shape, y_test.shape)
                self.assertTrue(np.issubdtype(prediccion.dtype, np.integer))
                self.assertGreaterEqual(prediccion.min(), 0)
                self.assertLess(prediccion.max(), k)


class Tensorizacion(unittest.TestCase):
    """Ejercicio 1: TensorizedQDA elimina el factor k de las llamadas a la función de puntaje."""

    def test_cantidad_de_llamadas(self):
        X_train, X_test, y_train, _ = _split(get_wine_dataset, semilla=0)
        n, k = X_test.shape[1], len(np.unique(y_train))
        qda, tqda = _ajustar(QDA, X_train, y_train), _ajustar(TensorizedQDA, X_train, y_train)
        qda.predict(X_test)
        tqda.predict(X_test)
        self.assertEqual(qda.cc, n * k)
        self.assertEqual(tqda.cc, n)

    def test_shapes_de_los_tensores(self):
        X_train, _, y_train, _ = _split(get_wine_dataset, semilla=0)
        p, k = X_train.shape[0], len(np.unique(y_train))
        modelo = _ajustar(TensorizedQDA, X_train, y_train)
        self.assertEqual(modelo.tensor_inv_cov.shape, (k, p, p))
        self.assertEqual(modelo.tensor_means.shape, (k, p, 1))
        cholesky = _ajustar(TensorizedChol, X_train, y_train)
        self.assertEqual(cholesky.tensor_L_invs.shape, (k, p, p))
        self.assertEqual(cholesky.log_det_terms.shape, (k,))


class PropiedadesMatematicas(unittest.TestCase):
    """Resultados teóricos que justifican las optimizaciones, sobre entradas aleatorias."""

    def test_truco_de_la_diagonal(self):
        for semilla in range(50):
            with self.subTest(semilla=semilla):
                verificar_truco_diagonal(semilla)

    def test_identidades_de_cholesky(self):
        for semilla in range(50):
            with self.subTest(semilla=semilla):
                verificar_identidades_cholesky(semilla)


class EficienciaDeMemoria(unittest.TestCase):
    """Ejercicios 4 y 6: EfficientChol/EfficientQDA no materializan la matriz de k·n·n."""

    @staticmethod
    def _pico_predict_mb(clase, X_train, y_train, X_test):
        modelo = _ajustar(clase, X_train, y_train)
        tracemalloc.start()
        modelo.predict(X_test)
        _, pico = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return pico / 2 ** 20

    def test_memoria_lineal_en_n(self):
        rng = np.random.default_rng(0)
        p, k, n_train, n_test = 8, 3, 600, 1500
        X_train = rng.normal(size=(p, n_train)) + np.repeat(np.arange(k), n_train // k)
        y_train = np.repeat(np.arange(k), n_train // k).reshape(1, -1)
        X_test = rng.normal(size=(p, n_test))

        matriz_n_por_n_mb = k * n_test ** 2 * 8 / 2 ** 20          # lo que alocaría FasterQDA
        pico_faster = self._pico_predict_mb(FasterQDA, X_train, y_train, X_test)
        self.assertGreater(pico_faster, 0.9 * matriz_n_por_n_mb)

        for clase in (EfficientQDA, EfficientChol):
            with self.subTest(modelo=clase.__name__):
                pico = self._pico_predict_mb(clase, X_train, y_train, X_test)
                self.assertLess(pico, 0.1 * matriz_n_por_n_mb)


class EstabilidadNumerica(unittest.TestCase):
    """Con p grande y varianzas chicas, det(Σ) sale del rango de float64.

    El determinante como suma de logaritmos (variantes nuevas) es estable; las implementaciones
    originales (`log(det)` / `log(prod(diag))`) no lo son. Esas se declaran como falla conocida:
    si alguna se corrige, el test pasa a "unexpected success" y avisa que hay que actualizarlo.
    """

    @classmethod
    def setUpClass(cls):
        X, y = dataset_alta_dimension()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = split_transpose(X, y, test_size=0.3, random_state=0)

    def _accuracy(self, clase):
        with np.errstate(all='ignore'):
            prediccion = _ajustar(clase, self.X_train, self.y_train).predict(self.X_test)
        return float((prediccion.flatten() == self.y_test.flatten()).mean())

    def test_variantes_con_suma_de_logaritmos(self):
        for clase in (TensorizedChol, EfficientChol):
            with self.subTest(modelo=clase.__name__):
                self.assertGreaterEqual(self._accuracy(clase), 0.95)

    @unittest.expectedFailure
    def test_falla_conocida_qda(self):
        self.assertGreaterEqual(self._accuracy(QDA), 0.95)

    @unittest.expectedFailure
    def test_falla_conocida_chol1(self):
        self.assertGreaterEqual(self._accuracy(QDA_Chol1), 0.95)

    @unittest.expectedFailure
    def test_falla_conocida_chol2(self):
        self.assertGreaterEqual(self._accuracy(QDA_Chol2), 0.95)

    @unittest.expectedFailure
    def test_falla_conocida_chol3(self):
        self.assertGreaterEqual(self._accuracy(QDA_Chol3), 0.95)


if __name__ == '__main__':
    unittest.main(verbosity=2)
