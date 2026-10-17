"""Verificaciones compartidas por el notebook TP_QDA_completo.ipynb y por tests/.

Se centralizan acá para que haya una única definición de cada prueba.
"""
import numpy as np
import numpy.linalg as LA
from scipy import stats


def prediccion_oraculo(X_train, y_train, X_test):
    """QDA calculado con scipy.stats: no comparte código con ninguna de las implementaciones del TP.

    Usa la misma convención que el código base (covarianza con bias=True, a priori por frecuencia).
    Devuelve las predicciones con shape (1, n), igual que `predict`.
    """
    y_flat = y_train.flatten().astype(int)
    log_scores = []
    for clase in np.unique(y_flat):
        Xc = X_train[:, y_flat == clase]
        distribucion = stats.multivariate_normal(mean=Xc.mean(axis=1), cov=np.cov(Xc, bias=True),
                                                 allow_singular=False)
        log_scores.append(np.log(Xc.shape[1] / y_flat.size) + distribucion.logpdf(X_test.T))
    return np.argmax(np.stack(log_scores), axis=0).reshape(1, -1)


def dataset_alta_dimension(p=400, k=3, n_por_clase=900, sigma2=0.01, separacion=0.35, seed=0):
    """k gaussianas con covarianza sigma2·I en dimensión p y medias separadas.

    Con p grande y sigma2 chico, det(Σ) ≈ sigma2^p queda fuera del rango de float64
    (p=400, sigma2=0.01 → ~1e-800): sirve para probar la estabilidad del cálculo del determinante.
    Devuelve X con shape (n, p) e y con shape (n, 1).
    """
    rng = np.random.default_rng(seed)
    medias = rng.normal(0, separacion, size=(k, p))
    X = np.vstack([medias[c] + rng.normal(0, np.sqrt(sigma2), size=(n_por_clase, p)) for c in range(k)])
    y = np.repeat(np.arange(k), n_por_clase).reshape(-1, 1)
    orden = rng.permutation(len(y))
    return X[orden], y[orden]


def verificar_truco_diagonal(seed):
    """diag(A·B) == sum(A ⊙ Bᵀ, axis=1) == sum(Aᵀ ⊙ B, axis=0) para matrices aleatorias."""
    rng = np.random.default_rng(seed)
    n, p = rng.integers(5, 60), rng.integers(1, 20)
    A, B = rng.normal(size=(n, p)), rng.normal(size=(p, n))
    assert np.allclose(np.diagonal(A @ B), np.sum(A * B.T, axis=1))
    assert np.allclose(np.diagonal(A @ B), np.sum(A.T * B, axis=0))


def verificar_identidades_cholesky(seed):
    """Con Σ = LLᵀ: L Lᵀ = Σ, xᵀΣ⁻¹x = ‖L⁻¹x‖² y ln|Σ| = 2·Σ ln L_ii, para Σ aleatoria definida positiva."""
    rng = np.random.default_rng(seed)
    p = int(rng.integers(2, 25))
    M = rng.normal(size=(p, p + 5))
    Sigma = M @ M.T / (p + 5) + 0.1 * np.eye(p)
    L = LA.cholesky(Sigma)
    x = rng.normal(size=(p, 1))
    assert np.allclose(L @ L.T, Sigma)
    assert np.allclose(x.T @ LA.inv(Sigma) @ x, np.sum(LA.solve(L, x) ** 2))
    assert np.isclose(LA.slogdet(Sigma)[1], 2 * np.log(np.diag(L)).sum())
