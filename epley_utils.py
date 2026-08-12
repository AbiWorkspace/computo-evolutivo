# epley_utils.py
# -*- coding: utf-8 -*-
"""
Utilidades basadas en la fórmula de Epley para relacionar:
- %1RM
- repeticiones hasta el fallo
- RIR / RPE

Incluye:
- Conversión básica reps ↔ %1RM (con clamps razonables).
- Cálculo de RIR y RPE a partir de %1RM + reps.
- Helper para "cerrar el triángulo" %–reps–RPE ajustando ligeramente
  la intensidad, pensado para usar dentro de las reparaciones del GA.
"""

import math
from typing import Tuple

# Rangos fisiológicamente razonables (se pueden ajustar si hace falta)
MIN_PCT = 40.0   # no nos interesa modelar por debajo de ~40%1RM
MAX_PCT = 100.0

MIN_REPS = 0.0   # reps "al fallo estimadas" pueden ser 0 (single pesado)
MAX_REPS = 30.0  # más allá normalmente ya no se usa en fuerza/hipertrofia

MIN_RPE = 5.0    # por debajo de 5 es muy alejado del fallo para este contexto
MAX_RPE = 10.0   # RPE máximo


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


# -------------------------------------------------------------------------
# 1) Conversión base Epley: reps ↔ %1RM
# -------------------------------------------------------------------------

def epley_reps_a_pct(reps_a_fallo: float) -> float:
    """
    %1RM ≈ 100 / (1 + 0.033 * reps_a_fallo)

    - reps_a_fallo: repeticiones que podrías hacer a tope (RIR=0).
    - Clampeamos a [MIN_REPS, MAX_REPS] para evitar valores absurdos.
    - Devolvemos %1RM en [MIN_PCT, MAX_PCT].
    """
    reps = _clamp(reps_a_fallo, MIN_REPS, MAX_REPS)
    pct = 100.0 / (1.0 + 0.033 * reps)
    return _clamp(pct, MIN_PCT, MAX_PCT)


def epley_pct_a_reps(pct: float) -> float:
    """
    reps_a_fallo ≈ (100/pct - 1) / 0.033

    - pct: porcentaje de 1RM.
    - Clampeamos a [MIN_PCT, MAX_PCT] para que no explote el modelo.
    - Devolvemos reps_a_fallo limitadas a [MIN_REPS, MAX_REPS].
    """
    p = _clamp(pct, MIN_PCT, MAX_PCT)
    reps = (100.0 / p - 1.0) / 0.033
    return _clamp(reps, MIN_REPS, MAX_REPS)


# -------------------------------------------------------------------------
# 2) RIR y RPE (modelo simple de fuerza)
# -------------------------------------------------------------------------

def rir_objetivo(rpe: float) -> float:
    """
    RIR ≈ 10 - RPE (capado a [0,∞))

    Se usa como modelo simple:
    - RPE 10  → RIR 0   (al fallo)
    - RPE 9   → RIR 1
    - RPE 8   → RIR 2
    - etc.
    """
    rpe_c = _clamp(rpe, MIN_RPE, MAX_RPE)
    return max(0.0, 10.0 - rpe_c)


def epley_pct_reps_a_rir(pct: float, reps: float) -> float:
    """
    Estima el RIR REAL a partir de:
    - pct: %1RM usado en la serie
    - reps: repeticiones realizadas

    Fórmula:
        reps_fallo_est = epley_pct_a_reps(pct)
        RIR_est = reps_fallo_est - reps
    """
    reps_fallo_est = epley_pct_a_reps(pct)
    rir_est = reps_fallo_est - float(reps)
    return max(0.0, rir_est)


def epley_pct_reps_a_rpe(pct: float, reps: float) -> float:
    """
    Convierte una combinación (%1RM, reps) a un RPE estimado usando Epley:

        RIR_est = reps_fallo_est - reps
        RPE_est ≈ 10 - RIR_est

    - Clampeamos RPE al rango [MIN_RPE, MAX_RPE].
    - Cuantizamos a pasos de 0.5 para que encaje con escalas habituales:
      7.0, 7.5, 8.0, 8.5, ..., 10.0
    """
    rir_est = epley_pct_reps_a_rir(pct, reps)
    rpe_est = 10.0 - rir_est
    rpe_est = _clamp(rpe_est, MIN_RPE, MAX_RPE)
    # redondeo a 0.5
    return round(rpe_est * 2.0) / 2.0


# -------------------------------------------------------------------------
# 3) Helper para "cerrar el triángulo" %–reps–RPE
# -------------------------------------------------------------------------

def cerrar_triangulo_pct_reps_rpe(
    pct: float,
    reps: int,
    rpe_obj: float,
    max_delta_pct: float = 3.0,
    max_delta_reps: int = 1,
    tolerancia_rir: float = 0.75,
) -> Tuple[float, int, float]:
    """
    Ajusta ligeramente %1RM y reps para que sean coherentes con un RPE objetivo.

    Uso típico dentro del GA:
    - El genoma decide reps y RPE objetivo (rpe_obj).
    - Este helper ajusta el %1RM (y opcionalmente ±1 rep) para que el RIR
      estimado por Epley no sea absurdo.

    Parámetros:
    - pct: %1RM actual de la sesión.
    - reps: repeticiones planeadas.
    - rpe_obj: RPE objetivo que quieres mantener.
    - max_delta_pct: cuánto podemos mover la intensidad en puntos porcentuales.
    - max_delta_reps: cuántas reps podemos subir/bajar como máximo (normalmente 0 o 1).
    - tolerancia_rir: margen aceptable entre RIR_target y RIR_est (en reps).

    Devuelve:
    - pct_corr: %1RM corregido (clamp + ajuste).
    - reps_corr: reps corregidas.
    - rpe_corr: RPE coherente (redondeado a 0.5), cercano a rpe_obj.

    NOTA:
    - Este helper es “suave”: no hace cambios gigantes.
    - El control duro de casos extremos (por ejemplo, prohibir 1 rep a 60% con RPE 10)
      se puede hacer en las funciones de reparación/penalización de tu GA.
    """
    # Clamps básicos
    pct = _clamp(pct, MIN_PCT, MAX_PCT)
    reps = int(max(1, min(30, reps)))
    rpe_obj = _clamp(rpe_obj, MIN_RPE, MAX_RPE)

    # 1) RIR objetivo a partir de RPE objetivo
    rir_t = rir_objetivo(rpe_obj)  # 10 - RPE_obj

    # 2) Reps al fallo objetivo = reps + RIR_target
    reps_fallo_target = float(reps) + float(rir_t)
    reps_fallo_target = _clamp(reps_fallo_target, MIN_REPS, MAX_REPS)

    # 3) %1RM objetivo según Epley para esas reps al fallo
    pct_target = epley_reps_a_pct(reps_fallo_target)

    # 4) Movemos pct hacia pct_target pero limitado por max_delta_pct
    if pct_target > pct:
        pct_corr = min(pct + max_delta_pct, pct_target)
    else:
        pct_corr = max(pct - max_delta_pct, pct_target)

    pct_corr = _clamp(pct_corr, MIN_PCT, MAX_PCT)

    # 5) Comprobamos RIR real con ese pct_corr; si sigue muy lejos, podemos
    #    ajustar reps en ±1 dentro del máximo permitido.
    rir_est = epley_pct_reps_a_rir(pct_corr, reps)
    diff_rir = rir_est - rir_t

    if abs(diff_rir) > tolerancia_rir and max_delta_reps > 0:
        # Si tenemos demasiado RIR, podemos subir reps (más lejos del fallo).
        # Si tenemos poco RIR, podemos bajar reps (más cerca del fallo).
        if diff_rir > 0.0:
            # tenemos más RIR del deseado → subir reps
            reps_corr = reps + min(max_delta_reps, int(round(diff_rir)))
        else:
            # tenemos menos RIR del deseado → bajar reps
            reps_corr = reps - min(max_delta_reps, int(round(-diff_rir)))

        reps_corr = int(max(1, min(30, reps_corr)))
    else:
        reps_corr = reps

    # 6) RPE coherente final a partir de pct_corr y reps_corr
    rpe_corr = epley_pct_reps_a_rpe(pct_corr, reps_corr)

    return pct_corr, reps_corr, rpe_corr


# -------------------------------------------------------------------------
# 4) Modelo de ESTÍMULO (fuerza / hipertrofia) y rendimientos decrecientes
# -------------------------------------------------------------------------
#
# Estas funciones son el núcleo de "cuánto vale realmente una serie".
# Se separan del algoritmo genético a propósito: son fisiología pura y se
# pueden probar de forma aislada.
#
# VENTANA_EFECTIVA: sólo las repeticiones cercanas al fallo son las que
# generan tensión mecánica suficiente para estimular hipertrofia. El consenso
# habitual sitúa esa ventana en las ~5 últimas reps antes del fallo.
VENTANA_EFECTIVA = 5.0

# Centro y anchura de la campana de hipertrofia (en %1RM).
HIPERTROFIA_CENTRO = 75.0
HIPERTROFIA_ANCHO = 18.0

# Rampa de fuerza: por debajo de FUERZA_PISO no hay apenas transferencia a
# fuerza máxima; a partir de FUERZA_TECHO el factor satura en 1.0.
FUERZA_PISO = 55.0
FUERZA_TECHO = 92.0

# RIR en el que el trabajo de fuerza es más rentable: cerca del fallo pero
# sin llegar a él (el fallo dispara la fatiga del SNC sin añadir fuerza).
FUERZA_RIR_OPTIMO = 2.0
FUERZA_RIR_ANCHO = 2.5


def reps_efectivas(reps: float, rpe: float) -> float:
    """
    Repeticiones ESTIMULANTES de una serie: las que caen dentro de la ventana
    de ~5 reps previas al fallo.

        RIR = 10 - RPE
        efectivas = clamp(VENTANA_EFECTIVA - RIR, 0, reps)

    Ejemplos:
    - 10 reps @ RPE 8 (RIR 2) → min(10, 3) = 3 reps efectivas
    -  3 reps @ RPE 10 (RIR 0) → min(3, 5)  = 3 reps efectivas
    - 12 reps @ RPE 5 (RIR 5) → min(12, 0) = 0 (volumen basura)

    Nótese que una serie pesada de 3 y una moderada de 10 pueden valer lo
    mismo: es justo lo que hace comparables fuerza e hipertrofia.
    """
    rir = rir_objetivo(rpe)
    return max(0.0, min(float(max(0.0, reps)), VENTANA_EFECTIVA - rir))


def factor_hipertrofia(pct: float) -> float:
    """
    Eficiencia hipertrófica por %1RM (campana centrada en ~75 %).

    Se estimula masa muscular en un rango amplio, pero el coste por unidad de
    estímulo es mínimo en cargas moderadas: muy ligero exige demasiadas reps y
    muy pesado acumula fatiga (sobre todo del SNC) sin más hipertrofia.
    """
    z = (float(pct) - HIPERTROFIA_CENTRO) / HIPERTROFIA_ANCHO
    return math.exp(-(z * z))


def factor_fuerza(pct: float) -> float:
    """
    Transferencia a fuerza máxima por %1RM: rampa suave y convexa que premia
    la exposición a cargas altas y satura al llegar a FUERZA_TECHO.

    Es continua en todo el rango (sin escalones) para que el algoritmo
    genético tenga gradiente en cualquier punto del espacio de búsqueda.
    """
    t = (float(pct) - FUERZA_PISO) / (FUERZA_TECHO - FUERZA_PISO)
    t = _clamp(t, 0.0, 1.0)
    return t * t


def calidad_fuerza(rpe: float) -> float:
    """
    Cuánto "vale" una serie pesada según lo cerca del fallo que se ejecute.

    El pico está en RIR≈2: llegar al fallo con cargas altas multiplica la
    fatiga del SNC sin añadir fuerza, y quedarse a RIR 5+ no da estímulo.
    """
    rir = rir_objetivo(rpe)
    z = (rir - FUERZA_RIR_OPTIMO) / FUERZA_RIR_ANCHO
    return math.exp(-(z * z))


def estimulo_serie(sets: int, reps: float, rpe: float, pct: float) -> Tuple[float, float]:
    """
    Estímulo bruto de un grupo de series idénticas, devuelto como
    (hipertrofia, fuerza).

    - Hipertrofia escala con reps efectivas (tensión mecánica acumulada).
    - Fuerza escala con la exposición a carga alta, no con el número de reps:
      una serie de 2@90 % vale más que una de 10@70 % aunque tenga menos reps.
    """
    n = int(max(0, sets))
    if n == 0:
        return 0.0, 0.0
    ef = reps_efectivas(reps, rpe)
    hip = n * ef * factor_hipertrofia(pct)
    fue = n * factor_fuerza(pct) * calidad_fuerza(rpe)
    return hip, fue


def saturar(x: float, techo: float) -> float:
    """
    Rendimientos decrecientes: x → techo * (1 - e^(-x/techo)).

    Es la pieza que hace que exista un óptimo interior de volumen. Sin ella,
    estímulo y fatiga crecen ambos de forma casi lineal con las series y la
    razón estímulo/fatiga se vuelve constante: al algoritmo le daría igual
    hacer 10 series que 40. Con saturación, pasado cierto volumen semanal
    cada serie extra aporta cada vez menos estímulo pero sigue costando
    fatiga completa, que es exactamente el comportamiento real (MEV → MRV).
    """
    techo = max(1e-9, float(techo))
    return techo * (1.0 - math.exp(-max(0.0, float(x)) / techo))
