"""KNN y escalado estándar implementados desde cero.

Reglas:
- Los cálculos se hacen solo con NumPy. No se puede usar sklearn.neighbors, sklearn.preprocessing
  ni scipy.spatial.
- KNNClasificador hereda de ClassifierMixin y BaseEstimator solo para poder usarse dentro de un
  Pipeline y de GridSearchCV de scikit-learn.
- Nada de bucles sobre filas para calcular distancias: se calculan con operaciones vectorizadas,
  por lotes de `tamano_lote` filas para no llenar la memoria.

Verificá tu trabajo con:  python -m pytest -q tests/test_knn.py
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class EscaladorEstandar:
    """Estandariza cada columna: (x - media) / desvío.

    Atributos que crea fit: media_ y desvio_ (arrays de forma (n_columnas,)).
    - El desvío es el poblacional (np.std con ddof=0), igual que StandardScaler.
    - Si una columna es constante (desvío 0), se usa desvío 1 para no dividir por cero.
    """

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.media_ = X.mean(axis=0)
        desvio = X.std(axis=0)
        self.desvio_ = np.where(desvio == 0, 1.0, desvio)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.media_) / self.desvio_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class KNNClasificador(ClassifierMixin, BaseEstimator):
    """Clasificador de k vecinos más cercanos (búsqueda por fuerza bruta).

    Parámetros
    ----------
    k : int, por defecto 5
        Cantidad de vecinos.
    pesos : "uniforme" o "distancia"
        "uniforme": cada vecino vota con peso 1.
        "distancia": cada vecino vota con peso 1 / distancia. Si la consulta coincide exactamente con
        uno o más ejemplos (distancia 0), votan solo esos ejemplos, con peso 1 cada uno.
    metrica : "euclidea" o "manhattan"
    tamano_lote : int, por defecto 1000
        Cantidad de filas de consulta que se procesan juntas al calcular distancias.

    Atributos que crea fit
    ----------------------
    X_ : array (n_entrenamiento, n_variables) con los datos de entrenamiento, en float.
    classes_ : array (n_clases,) con las clases ordenadas (np.unique).
    y_codificada_ : array (n_entrenamiento,) con el índice de la clase de cada ejemplo en classes_.
    """

    def __init__(self, k=5, pesos="uniforme", metrica="euclidea", tamano_lote=1000):
        self.k = k
        self.pesos = pesos
        self.metrica = metrica
        self.tamano_lote = tamano_lote

    def fit(self, X, y):
        """KNN no estima parámetros: guarda los datos. Devuelve self."""
        self.X_ = np.asarray(X, dtype=float)
        self.classes_, self.y_codificada_ = np.unique(np.asarray(y), return_inverse=True)
        return self

    def _distancias(self, X):
        """Matriz (n_consultas, n_entrenamiento) con la distancia entre cada consulta y cada ejemplo.

        Pista para la euclídea: ||a - b||² = ||a||² + ||b||² - 2 a·b. Por redondeo puede dar valores
        levemente negativos: recortalos a 0 antes de la raíz cuadrada.
        """
        X = np.asarray(X, dtype=float)
        if self.metrica == "euclidea":
            cuad = (X ** 2).sum(axis=1)[:, None] + (self.X_ ** 2).sum(axis=1)[None, :] - 2.0 * X @ self.X_.T
            return np.sqrt(np.maximum(cuad, 0.0))
        if self.metrica == "manhattan":
            D = np.zeros((X.shape[0], self.X_.shape[0]))
            for j in range(X.shape[1]):  # por columna: evita un array 3D enorme
                D += np.abs(X[:, j][:, None] - self.X_[:, j][None, :])
            return D
        raise ValueError(f"Métrica desconocida: {self.metrica!r}")

    def _distancia_exacta(self, X, indices):
        """Distancia directa entre cada consulta y sus vecinos elegidos (n_consultas, k)."""
        dif = X[:, None, :] - self.X_[indices]
        if self.metrica == "euclidea":
            return np.sqrt((dif ** 2).sum(axis=2))
        return np.abs(dif).sum(axis=2)

    def kneighbors(self, X):
        """Devuelve (distancias, indices), ambos de forma (n_consultas, k), ordenados de menor a mayor distancia.

        Pista: np.argpartition encuentra los k menores sin ordenar toda la fila; después ordená solo esos k.
        """
        X = np.asarray(X, dtype=float)
        k = self.k
        lote = max(1, int(self.tamano_lote))
        dist_out = np.empty((len(X), k))
        idx_out = np.empty((len(X), k), dtype=int)
        for ini in range(0, len(X), lote):
            Xb = X[ini:ini + lote]
            D = self._distancias(Xb)
            if k < D.shape[1]:
                cand = np.argpartition(D, k - 1, axis=1)[:, :k]
            else:
                cand = np.tile(np.arange(D.shape[1]), (len(Xb), 1))
            # Se recalculan las distancias de los k elegidos de forma directa (sin el error de redondeo
            # de la expansión) y recién ahí se ordenan; desempates por índice.
            cand = np.sort(cand, axis=1)
            exacta = self._distancia_exacta(Xb, cand)
            orden = np.argsort(exacta, axis=1, kind="stable")
            idx_out[ini:ini + lote] = np.take_along_axis(cand, orden, axis=1)
            dist_out[ini:ini + lote] = np.take_along_axis(exacta, orden, axis=1)
        return dist_out, idx_out

    def predict_proba(self, X):
        """Proporción (ponderada según `pesos`) de vecinos de cada clase. Columnas en el orden de classes_."""
        if self.pesos not in ("uniforme", "distancia"):
            raise ValueError(f"Pesos desconocidos: {self.pesos!r}")
        dist, idx = self.kneighbors(X)
        if self.pesos == "uniforme":
            w = np.ones_like(dist)
        else:
            cero = dist == 0
            hay_cero = cero.any(axis=1, keepdims=True)
            with np.errstate(divide="ignore"):
                w = np.where(hay_cero, cero.astype(float), 1.0 / dist)
        clases = self.y_codificada_[idx]
        proba = np.zeros((len(dist), len(self.classes_)))
        for c in range(len(self.classes_)):
            proba[:, c] = (w * (clases == c)).sum(axis=1)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X):
        """Clase con mayor probabilidad. En caso de empate, la primera en classes_ (como np.argmax)."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]