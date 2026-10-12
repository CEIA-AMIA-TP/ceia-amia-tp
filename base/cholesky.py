import numpy as np
import numpy.linalg as LA
from scipy.linalg import cholesky, solve_triangular
from scipy.linalg.lapack import dtrtri

from base.bayesian import BaseBayesianClassifier


class QDA_Chol1(BaseBayesianClassifier):
  def _fit_params(self, X, y):
    self.L_invs = [
        LA.inv(cholesky(np.cov(X[:,y.flatten()==idx], bias=True), lower=True))
        for idx in range(len(self.log_a_priori))
    ]

    self.means = [X[:,y.flatten()==idx].mean(axis=1, keepdims=True)
                  for idx in range(len(self.log_a_priori))]

  def _predict_log_conditional(self, x, class_idx):
    L_inv = self.L_invs[class_idx]
    unbiased_x =  x - self.means[class_idx]

    y = L_inv @ unbiased_x

    return np.log(L_inv.diagonal().prod()) -0.5 * (y**2).sum()


class QDA_Chol2(BaseBayesianClassifier):
  def _fit_params(self, X, y):
    self.Ls = [
        cholesky(np.cov(X[:,y.flatten()==idx], bias=True), lower=True)
        for idx in range(len(self.log_a_priori))
    ]

    self.means = [X[:,y.flatten()==idx].mean(axis=1, keepdims=True)
                  for idx in range(len(self.log_a_priori))]

  def _predict_log_conditional(self, x, class_idx):
    L = self.Ls[class_idx]
    unbiased_x =  x - self.means[class_idx]

    y = solve_triangular(L, unbiased_x, lower=True)

    return -np.log(L.diagonal().prod()) -0.5 * (y**2).sum()


class QDA_Chol3(BaseBayesianClassifier):
  def _fit_params(self, X, y):
    self.L_invs = [
        dtrtri(cholesky(np.cov(X[:,y.flatten()==idx], bias=True), lower=True), lower=1)[0]
        for idx in range(len(self.log_a_priori))
    ]

    self.means = [X[:,y.flatten()==idx].mean(axis=1, keepdims=True)
                  for idx in range(len(self.log_a_priori))]

  def _predict_log_conditional(self, x, class_idx):
    L_inv = self.L_invs[class_idx]
    unbiased_x =  x - self.means[class_idx]

    y = L_inv @ unbiased_x

    return np.log(L_inv.diagonal().prod()) -0.5 * (y**2).sum()


class TensorizedChol(QDA_Chol3):
  """Cholesky + paralelización sobre las k clases.

  Hereda de QDA_Chol3 porque invierte L con `dtrtri` (rutina LAPACK para matrices
  triangulares), la forma más barata de obtener L^{-1}. Apila las k matrices en un
  tensor para resolver todas las clases con un único matmul batcheado por observación.
  Sigue recorriendo las n observaciones (ese for lo hereda `predict` de la base).
  """

  def _fit_params(self, X, y):
    super()._fit_params(X, y)

    # una "rebanada" por clase: (k, p, p) y (k, p, 1)
    self.tensor_L_invs = np.stack(self.L_invs)
    self.tensor_means = np.stack(self.means)

    # log|Σ_k|^(-1/2) = Σ_i log (L_k^{-1})_ii. Se suman logaritmos en lugar de
    # tomar el log del producto: con p grande el producto desborda (o underflowea) y da inf/0.
    self.log_det_terms = np.log(np.diagonal(self.tensor_L_invs, axis1=1, axis2=2)).sum(axis=1)

  def _predict_one(self, x):
    z = self.tensor_L_invs @ (x - self.tensor_means)                  # (k, p, 1)
    log_conditionals = self.log_det_terms - 0.5 * (z ** 2).sum(axis=(1, 2))  # (k,)
    return np.argmax(self.log_a_priori + log_conditionals)


class EfficientChol(TensorizedChol):
  """Cholesky + paralelización sobre clases y observaciones, sin matriz de n x n.

  QDA necesitaba el truco de la diagonal porque la forma cuadrática es x^T A x.
  Con Cholesky la forma cuadrática es ||L^{-1}(x - mu)||^2, una norma al cuadrado
  por columna: alcanza con elevar al cuadrado y sumar sobre las p features.
  Memoria O(k·p·n) en lugar de O(k·n^2).
  """

  def predict(self, X):
    z = self.tensor_L_invs @ (X - self.tensor_means)                  # (k, p, n)
    quadratic_terms = (z ** 2).sum(axis=1)                            # (k, n)

    class_scores = (self.log_a_priori + self.log_det_terms).reshape(-1, 1) - 0.5 * quadratic_terms

    return np.argmax(class_scores, axis=0, keepdims=True)             # (1, n), igual que y
