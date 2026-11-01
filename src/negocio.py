"""Evaluación económica de la campaña de retención de Nodo Internet.

Completá las funciones. Solo NumPy. Verificá tu trabajo con:  python -m pytest -q tests/test_negocio.py
"""
import numpy as np

# Supuestos acordados por Finanzas y Fidelización (ver el enunciado del caso)
SUPUESTOS = {
    "costo_contacto": 10_000,            # Gs. por cliente contactado
    "costo_descuento": 200_000,          # Gs. por cliente que acepta el descuento
    "valor_cliente_retenido": 1_000_000,  # Gs. de margen de un cliente que se iba a ir y se queda
    "acepta_si_se_iba": 0.40,            # proporción que acepta la oferta entre los que se iban a ir
    "acepta_si_se_quedaba": 0.60,        # proporción que acepta la oferta entre los que se iban a quedar
}


def resultado_por_contacto(s=SUPUESTOS):
    """Devuelve una tupla (si_se_iba, si_se_quedaba) con el resultado esperado, en guaraníes, de contactar a:
    - un cliente que se iba a ir (si acepta, se queda y deja su margen, menos el descuento), y
    - un cliente que se iba a quedar (si acepta, la empresa resigna el descuento).
    En los dos casos se paga el costo del contacto. No contactar tiene resultado 0.
    """
    si_se_iba = s["acepta_si_se_iba"] * (s["valor_cliente_retenido"] - s["costo_descuento"]) - s["costo_contacto"]
    si_se_quedaba = -s["acepta_si_se_quedaba"] * s["costo_descuento"] - s["costo_contacto"]
    return si_se_iba, si_se_quedaba


def probabilidad_de_equilibrio(s=SUPUESTOS):
    """Probabilidad de baja p a partir de la cual contactar tiene resultado esperado positivo:
    el valor de p que hace p · si_se_iba + (1 - p) · si_se_quedaba = 0.
    """
    g_pos, g_neg = resultado_por_contacto(s)
    return float(-g_neg / (g_pos - g_neg))


def ganancia_campania(y_real, contactar, s=SUPUESTOS):
    """Ganancia total, en guaraníes, de contactar a los clientes marcados con True en `contactar`,
    sabiendo quiénes se fueron realmente (`y_real`, 1 = se fue). Devuelve un float.
    """
    y = np.asarray(y_real).astype(int)
    c = np.asarray(contactar).astype(bool)
    g_pos, g_neg = resultado_por_contacto(s)
    return float(g_pos * np.sum(c & (y == 1)) + g_neg * np.sum(c & (y == 0)))


def mejor_umbral(y_real, prob, s=SUPUESTOS, max_contactos=None):
    """Busca el umbral u que maximiza la ganancia al contactar a todos los clientes con prob >= u.

    - Los umbrales candidatos son los valores distintos de `prob`.
    - Si max_contactos no es None, se descartan los umbrales que contactan a más clientes.
    - Devuelve un dict {"umbral": u, "ganancia": g, "contactados": n}.
    - Si ningún umbral da ganancia positiva, devuelve {"umbral": np.inf, "ganancia": 0.0, "contactados": 0}.
    """
    y = np.asarray(y_real).astype(int)
    prob = np.asarray(prob, dtype=float)
    g_pos, g_neg = resultado_por_contacto(s)
    orden = np.argsort(-prob, kind="stable")
    p_ord = prob[orden]
    pos_acum = np.cumsum(y[orden] == 1)
    n_acum = np.arange(1, len(prob) + 1)
    # Solo cortes al final de un grupo de probabilidades iguales (contactar prob >= u)
    fin_grupo = np.append(p_ord[1:] != p_ord[:-1], True)
    idx = np.flatnonzero(fin_grupo)
    n = n_acum[idx]
    gan = g_pos * pos_acum[idx] + g_neg * (n - pos_acum[idx])
    if max_contactos is not None:
        valido = n <= max_contactos
        idx, n, gan = idx[valido], n[valido], gan[valido]
    if len(gan) == 0 or gan.max() <= 0:
        return {"umbral": np.inf, "ganancia": 0.0, "contactados": 0}
    j = int(np.argmax(gan))  # primer máximo: menor cantidad de contactos
    return {"umbral": float(p_ord[idx[j]]), "ganancia": float(gan[j]), "contactados": int(n[j])}


def lift(y_real, prob, fraccion):
    """Tasa de positivos entre la `fraccion` de clientes con mayor prob, dividida por la tasa general.

    Ejemplo: lift(y, prob, 0.07) compara la tasa de baja del 7% de mayor riesgo con la tasa de toda la base.
    La cantidad de clientes del grupo es round(fraccion · n), con un mínimo de 1.
    """
    y = np.asarray(y_real).astype(int)
    prob = np.asarray(prob, dtype=float)
    n_grupo = max(1, int(round(fraccion * len(y))))
    top = np.argsort(-prob, kind="stable")[:n_grupo]
    return float(y[top].mean() / y.mean())