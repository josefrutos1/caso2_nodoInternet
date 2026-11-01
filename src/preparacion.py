"""Preparación de los datos de clientes de Nodo Internet.

Completá las dos funciones con pandas y la biblioteca estándar.
Verificá tu trabajo con:  python -m pytest -q tests/test_preparacion.py
"""
import unicodedata

import numpy as np
import pandas as pd

METODOS_VALIDOS = ["debito_automatico", "tarjeta_credito", "transferencia", "efectivo"]
DIAS_POR_MES = 30.44


def normalizar_metodo_pago(valor):
    """Lleva las variantes con que el CRM guarda el medio de pago a uno de METODOS_VALIDOS.

    Ejemplos: "Débito automático", "DEBITO AUTOMATICO" y "Debito_Automatico" -> "debito_automatico";
    "Tarjeta de crédito" y "tarjeta credito" -> "tarjeta_credito"; "transferencia bancaria" -> "transferencia";
    "efectivo " -> "efectivo".
    Si el valor no corresponde a ninguno de los cuatro, lanzar ValueError.
    """
    texto = unicodedata.normalize("NFKD", str(valor))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.lower().replace("_", " ").strip()
    if texto.startswith("debito"):
        return "debito_automatico"
    if texto.startswith("tarjeta"):
        return "tarjeta_credito"
    if texto.startswith("transferencia"):
        return "transferencia"
    if texto == "efectivo":
        return "efectivo"
    raise ValueError(f"Método de pago desconocido: {valor!r}")


def agregar_variables(df):
    """Devuelve una COPIA de df (sin modificar el original) con metodo_pago normalizado y tres columnas nuevas.

    Todas se calculan a la fecha_corte de cada fila, nunca a la fecha de hoy:
    - antiguedad_meses: días entre fecha_alta y fecha_corte, divididos por DIAS_POR_MES.
    - meses_hasta_fin_permanencia: días entre fecha_corte y fin_permanencia, divididos por DIAS_POR_MES.
      Vale 0 en los contratos mensuales (no tienen fecha de fin de permanencia).
    - sin_permanencia_3m: 1 si el cliente puede irse sin penalidad en los próximos 3 meses, es decir,
      si su contrato es mensual o si meses_hasta_fin_permanencia <= 3; 0 en otro caso (entero).
    """
    d = df.copy()
    d["metodo_pago"] = d["metodo_pago"].map(normalizar_metodo_pago)
    corte = pd.to_datetime(d["fecha_corte"])
    alta = pd.to_datetime(d["fecha_alta"])
    fin = pd.to_datetime(d["fin_permanencia"])
    d["antiguedad_meses"] = (corte - alta).dt.days / DIAS_POR_MES
    d["meses_hasta_fin_permanencia"] = ((fin - corte).dt.days / DIAS_POR_MES).fillna(0.0)
    sin_perm = (d["tipo_contrato"] == "mensual") | (d["meses_hasta_fin_permanencia"] <= 3)
    d["sin_permanencia_3m"] = sin_perm.astype(int)
    return d