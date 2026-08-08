# evolucion_entrenamiento.py (fachada generada)
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, sys, io, random, math
from typing import List, Tuple, Dict
from modelos import SemanaPlan, Plan12Semanas

from epley_utils import epley_pct_a_reps, epley_reps_a_pct, rir_objetivo

# ⬇️ mixins vecinos (generados por este script)
from evo_generacion_y_reparaciones import GeneracionDistribMixin
from evo_fitness_y_ga import FitnessGAMixin

# ⬇️ ya existía en tu proyecto
from penalizaciones import PenalizacionesMixin

# =============================================================================
# EvolucionEntrenamiento - logging + filtro de prints (SILENT por defecto)
# =============================================================================
import os, logging, builtins

EE_DEBUG = os.environ.get("EE_DEBUG", "").lower() in ("1", "true", "yes")
_DEBUG = EE_DEBUG  # alias para el código que aún usa _DEBUG

logger = logging.getLogger("EvolucionEntrenamiento")
logger.propagate = False

# Limpia handlers duplicados si se reimporta
for h in list(logger.handlers):
    logger.removeHandler(h)

if EE_DEBUG:
    # Modo debug: muestra TODO
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.disabled = False
else:
    # Modo silencioso: NO muestra logs
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL + 1)  # más alto que CRITICAL
    logger.disabled = True

_orig_print = builtins.print

def _filtered_print(*args, **kwargs):
    if args and isinstance(args[0], str) and args[0].startswith("EvolucionEntrenamiento - "):
        if _DEBUG:  # o EE_DEBUG, son equivalentes
            logger.debug(" ".join(str(a) for a in args))
        return
    return _orig_print(*args, **kwargs)

builtins.print = _filtered_print


class EvolucionEntrenamiento(GeneracionDistribMixin, FitnessGAMixin, PenalizacionesMixin):
    """Algoritmo genético para optimizar una periodización de 12 semanas (un patrón de movimiento).
    Integra: periodización lineal por bloques, ondulación H/M/L, progresión día vs día, deload intermedio, RPE por rol,
    series efectivas y límites de salto según nivel.
    """

    # Gaps mínimos para H/M/L
    _GAP_PCT_MIN = 6.0   # separación mínimas H-M y M-L (p.p.)
    _GAP_RPE_MIN = 0.5   # H ≥ M +0.5 ≥ L +0.5

    # Rangos de REGLA DURA por fase/rol
    # Esta es la "única fuente de verdad" para la identidad de los bloques.
    _PHASE_BOUNDS = {
        # B1: Hipertrofia (Acumulación)
        "acc": {
            "H": {"pct": (70, 78), "reps": (6, 12), "series": (12, 22), "rpe": (6.5, 8.0)},
            "M": {"pct": (65, 72), "reps": (8, 15), "series": (12, 22), "rpe": (6.5, 7.5)},
            "L": {"pct": (60, 68), "reps": (10, 20), "series": (12, 22), "rpe": (6.0, 7.0)}
        },
        # B2: Fuerza (Intensificación)
        "int": {
            "H": {"pct": (82, 90), "reps": (3, 6), "series": (10, 18), "rpe": (7.5, 8.5)},  # reps estrictas
            "M": {"pct": (76, 84), "reps": (4, 6), "series": (10, 18), "rpe": (7.0, 8.0)},  # reps estrictas
            "L": {"pct": (68, 76), "reps": (5, 8), "series": (10, 18), "rpe": (6.5, 7.5)}
        },
        # B3: Peaking
        "peak": {
            "H": {"pct": (88, 95), "reps": (1, 4), "series": (8, 14), "rpe": (8.0, 9.5)},  # reps estrictas
            "M": {"pct": (80, 88), "reps": (2, 4), "series": (8, 14), "rpe": (7.0, 8.5)},  # reps estrictas
            "L": {"pct": (60, 72), "reps": (2, 5), "series": (8, 14), "rpe": (6.0, 7.0)}
        },
        # B4: Taper (solo lo usan reparaciones finales/taper específico)
        "taper": {
            "H": {"pct": (90, 96), "reps": (1, 2), "series": (4, 8), "rpe": (8.0, 9.0)},
            "M": {"pct": (80, 85), "reps": (2, 3), "series": (4, 8), "rpe": (7.0, 8.0)},
            "L": {"pct": (60, 70), "reps": (2, 3), "series": (4, 8), "rpe": (6.0, 7.0)}
        },
    }

    def _fase_de_semana(self, semana_idx: int) -> str:
        """
        Mapea el índice de semana (1–12) a la fase del macro.
        Debe ser consistente con las claves de las reglas duras (acc/int/peak).
        """
        if semana_idx <= 4:
            return "acc"   # Bloque 1 – Acumulación / Hipertrofia
        elif semana_idx <= 8:
            return "int"   # Bloque 2 – Intensificación / Fuerza
        else:
            return "peak"  # Bloque 3 – Peaking / Taper

    def _roles_hml_por_pct(self, pct_sesion: list[float]) -> list[str]:
        # Ordena por %1RM y asigna H/M/L
        n = len(pct_sesion)
        if n == 0:
            return []
        orden = sorted(range(n), key=lambda j: pct_sesion[j], reverse=True)
        roles = ["M"] * n
        if n == 1:
            roles[orden[0]] = "H"
        elif n == 2:
            roles[orden[0]], roles[orden[1]] = "H", "M"
        else:
            roles[orden[0]], roles[orden[1]], roles[orden[2]] = "H", "M", "L"
        return roles

    def _es_dia_velocidad(self, fase: str, rol: str, semana) -> bool:
        # Si el plan marca explícito, respétalo
        if hasattr(semana, "dia_velocidad"):
            try:
                # bool o índice/sesión; soporta True en L
                return bool(semana.dia_velocidad) and (rol == "L")
            except Exception:
                pass
        # Por diseño, en Peak el día L es velocidad
        return (fase == "peak" and rol == "L")

    def _clamp(self, v, vmin, vmax):
        return max(vmin, min(vmax, v))

    def _ajustar_rampa_interbloque(self, semanas: List[SemanaPlan]) -> None:
        """
        Evita saltos W4→W5 > ~9 p.p. en % semanal. Si W5 se dispara,
        baja el % semanal y sus % por sesión de forma coherente, y revalida.
        """
        if not semanas or len(semanas) < 5:
            return
        w4, w5 = semanas[3], semanas[4]
        limite = float(w4.pct_1rm) + 9.0  # corta saltos tipo +11–12 p.p.
        if float(w5.pct_1rm) > limite:
            delta = float(w5.pct_1rm) - limite
            w5.pct_1rm = limite
            # Baja también % por sesión para mantener coherencia relativa
            if getattr(w5, "pct_1rm_sesion", None):
                for i in range(len(w5.pct_1rm_sesion)):
                    w5.pct_1rm_sesion[i] = max(50.0, float(w5.pct_1rm_sesion[i]) - delta)
            # Revalida RIR y realinea
            try:
                self._validar_coherencia_rir_rpe(w5)
            except Exception:
                pass
            try:
                self._realinear_sets_con_pct(w5)
            except Exception:
                pass

    # -------- Rangos objetivo por bloque (idénticos) --------
    BLOQUES = {
        1: {"idx": range(0, 4),  "series": (12, 18), "pct": (60, 75), "rpe": (6.5, 7.5), "reps": (8, 15)},
        2: {"idx": range(4, 8),  "series": (10, 14), "pct": (75, 85), "rpe": (7.0, 8.5), "reps": (4, 10)},
        3: {"idx": range(8, 12), "series": (6, 10),  "pct": (85, 95), "rpe": (7.5, 9.0), "reps": (1, 6)},
    }

    # -------- Rangos IDEALES (hard-guard global) --------
    IDEAL = {
        "reps": (1, 30),
        "series_semana": (6, 20),
        "series_sesion": (2, 10),
        "pct_sesion": (60.0, 100.0),
        "rpe_sesion": (6.0, 10.0),
    }

    # IDEALES por bloque
    IDEAL_BLOQUE = {
        1: {
            "series_semana": (14, 22),
            "reps_semana":   (6, 20),
            "pct_sesion":    (55.0, 90.0),
            "rpe_sesion":    (6.0, 9.0),
            "reps_sesion":   (5, 15),
        },
        2: {
            "series_semana": (12, 12),
            "reps_semana":   (3, 14),
            "pct_sesion":    (65.0, 95.0),
            "rpe_sesion":    (6.5, 9.0),
            "reps_sesion":   (3, 12),
        },
        3: {
            "series_semana": (12, 14),
            "reps_semana":   (1, 8),
            "pct_sesion":    (75.0, 100.0),
            "rpe_sesion":    (7.0, 9.5),
            "reps_sesion":   (1, 6),
        },
    }

    # Estímulo mínimo por bloque (semana)
    STIM_MIN = {1: 75.0, 2: 60.0, 3: 35.0}

    # --- RPE discretos 6.0..10.0 en pasos de 0.5
    RPE_SET_AMPLIADO = tuple(6.0 + 0.5 * k for k in range(9))

    # --- Objetivo de RPE medio por bloque (sube dentro del bloque)
    RPE_OBJ_BLOQUE = {
        1: (6.5, 7.5),
        2: (7.0, 8.5),
        3: (8.0, 9.0),
    }

    RPE_ROL_RANGOS = {
        1: {"H": (7.0, 8.0), "M": (6.5, 7.5), "L": (6.0, 7.0)},
        2: {"H": (7.5, 8.5), "M": (7.0, 8.0), "L": (6.5, 7.5)},
        3: {"H": (8.0, 9.0), "M": (7.5, 8.5), "L": (7.0, 8.0)},
    }

    SERIES_EFECTIVAS_BLOQUE = {1: (8, 16), 2: (8, 12), 3: (4, 8)}

    NIVELES = {
        "novato":       (0.025, 0.050),
        "principiante": (0.020, 0.040),
        "intermedio":   (0.010, 0.025),
        "avanzado":     (0.005, 0.010),
    }

    RPE_SET_MINIMO = (6.0, 6.5)

    def __init__(
        self,
        patron: str = "ampliado",
        poblacion: int = 180,
        generaciones: int = 260,
        prob_cruce: float = 0.85,
        prob_mutacion: float = 0.50,
        semilla: int = 21,
        nivel_usuario: str = "intermedio",
        modo_rpe: str = "ampliado",  # DEFAULT correcto
        usar_deload_intermedio: bool = True,
    ):
        self.patron = patron
        self.poblacion = poblacion
        self.generaciones = generaciones
        self.prob_cruce = prob_cruce
        self.prob_mutacion = prob_mutacion
        self.nivel_usuario = nivel_usuario.lower() if nivel_usuario else "intermedio"
        # Acepta sólo valores válidos; si no, cae a "ampliado"
        self.modo_rpe = modo_rpe if modo_rpe in ("minimo", "ampliado") else "ampliado"
        self.usar_deload_intermedio = usar_deload_intermedio
        # ---- Parámetros para shock intersemanal (salto entre semanas) ----
        self._EPS_DIV = 1e-6              # Para evitar división por cero
        self._LAMBDA_SHOCK_VOL = 10.0     # Peso del shock de volumen (tonelaje)
        self._LAMBDA_SHOCK_STIM = 5.0     # Peso del shock de estímulo (S_sem)

        # Últimas métricas calculadas del plan (para penalizaciones)
        self._metricas_last = None

        random.seed(semilla)
        print(
            f"EvolucionEntrenamiento - __init__: nivel_usuario={self.nivel_usuario}, "
            f"modo_rpe='{self.modo_rpe}', semilla={semilla}."
        )

    def _epley_reps_a_pct(self, reps_a_fallo: float) -> float:
        return epley_reps_a_pct(reps_a_fallo)

    def _epley_pct_a_reps(self, pct: float) -> float:
        return epley_pct_a_reps(pct)

    def _rir_objetivo(self, fase: str, rol: str, es_velocidad: bool) -> tuple[float, float, float]:
        """
        Devuelve (rir_min, rir_max, peso_penalizacion) para una combinación
        de fase (acc/int/peak), rol (H/M/L) y si el día es de velocidad.

        - 'peso' modula cuánto pesa esa violación en la penalización de RIR.
        """

        # Días de velocidad: tolerancia distinta, peso más bajo
        if es_velocidad:
            # L-velocidad típica: lejos del fallo, y con poco peso en la penalización
            if rol == "L":
                return 3.0, 6.0, 0.2
            # H/M de velocidad (si existen en tu diseño)
            return 2.0, 5.0, 0.3

        # Tabla por fase / rol
        tabla = {
            "acc": {   # Acumulación (hipertrofia)
                "H": (1.0, 3.0, 1.0),
                "M": (2.0, 4.0, 0.8),
                "L": (3.0, 5.0, 0.6),
            },
            "int": {   # Intensificación (fuerza)
                "H": (0.0, 2.0, 1.1),
                "M": (1.0, 3.0, 0.9),
                "L": (2.0, 4.0, 0.7),
            },
            "peak": {  # Peaking / Taper
                "H": (0.0, 1.0, 1.2),
                "M": (0.5, 2.0, 1.0),
                "L": (1.5, 3.0, 0.8),
            },
        }

        fase_cfg = tabla.get(fase)
        if not fase_cfg:
            # Fallback si alguien llama aún con "hipertrofia"/"fuerza"/"pico"
            if fase in ("hipertrofia", "acc"):
                fase_cfg = tabla["acc"]
            elif fase in ("fuerza", "int"):
                fase_cfg = tabla["int"]
            else:
                fase_cfg = tabla["peak"]

        # Si el rol es raro, devolvemos algo neutro
        return fase_cfg.get(rol, (1.0, 3.0, 0.5))

    def _estimar_rir_simple(self, pct_1rm: float, reps: int, rpe: float) -> float:
        tabla = [(100,1),(95,2),(90,4),(85,6),(80,8),(75,10),(70,12),(65,15),(60,18)]
        r_max = None
        for p, r in tabla:
            if pct_1rm >= p:
                r_max = r
                break
        if r_max is None:
            r_max = 20
        rir_aprox = max(0.0, float(r_max - reps))
        rir_por_rpe = max(0.0, 10.0 - rpe)  # RPE 10→0; RPE 7→3
        return 0.6 * rir_aprox + 0.4 * rir_por_rpe

    def _lerp(self, a: float, b: float, t: float) -> float:
        return a + (b - a) * max(0.0, min(1.0, t))

    def _rpe_round(self, x: float) -> float:
        return min(self.RPE_SET_AMPLIADO, key=lambda v: abs(v - x))

    def _rpe_set(self):
        """Devuelve el set discreto de RPE según el modo."""
        minimo = getattr(self, "RPE_SET_MINIMO", (6.0, 6.5))
        ampliado = getattr(self, "RPE_SET_AMPLIADO", tuple(6.0 + 0.5 * k for k in range(9)))
        return ampliado if getattr(self, "modo_rpe", "ampliado") == "ampliado" else minimo

    def _rpe_discreto(self, x: float) -> float:
        """Cuantiza a los valores permitidos del set activo."""
        return min(self._rpe_set(), key=lambda v: abs(v - x))

    # fallback: delega al mixin si está disponible
    def _ajustar_deload_semana(self, semanas):
        try:
            # si el mixin lo tiene, úsalo
            return GeneracionDistribMixin._ajustar_deload_semana(self, semanas)
        except Exception:
            return None

    # wrappers a funciones de módulo si no están como métodos del mixin
    def _enforce_role_separation_strict(self, s):
        try:
            return _enforce_role_separation_strict(self, s)
        except NameError:
            return None

    def _enforce_phase_ranges(self, s, semana_idx_1b: int):
        """
        Aplica los rangos de _PHASE_BOUNDS ignorando la función global
        obsoleta _enforce_phase_ranges del módulo.
        """
        try:
            return self._enforce_phase_ranges_local(s, semana_idx_1b)
        except Exception:
            # Si algo falla, no rompemos la corrida del GA
            return None

    def _enforce_phase_ranges_local(self, semana: SemanaPlan, semana_idx_1b: int) -> None:
        """
        Repara una semana aplicando los rangos de _PHASE_BOUNDS por FASE (acc/int/peak/taper)
        y por ROL (H/M/L), forzando además DUP H≥M≥L con brecha mínima.
        También recalcula los agregados semanales coherentes.
        """
        fase = self._fase_de_semana(semana_idx_1b)
        bounds_fase = self._PHASE_BOUNDS.get(fase)
        if not bounds_fase:
            return

        n = int(getattr(semana, "sesiones", 0) or 0)
        if n <= 0:
            return

        pct  = list(getattr(semana, "pct_1rm_sesion", []) or [])
        rpe  = list(getattr(semana, "rpe_sesion", []) or [])
        reps = list(getattr(semana, "reps_sesion", []) or [])
        distrib = list(getattr(semana, "distrib_series", []) or [])

        pct_default  = float(getattr(semana, "pct_1rm", 70.0))
        rpe_default  = float(getattr(semana, "rpe", 7.0))
        reps_default = int(getattr(semana, "reps", 8))

        while len(pct)  < n: pct.append(pct_default)
        while len(rpe)  < n: rpe.append(rpe_default)
        while len(reps) < n: reps.append(reps_default)

        roles = self._roles_hml_por_pct(pct)

        # 1) Clamp por fase/rol (+ día de velocidad en peak-L)
        for j in range(n):
            rol = roles[j]
            rol_bounds = bounds_fase.get(rol)
            if not rol_bounds:
                continue

            es_vel = self._es_dia_velocidad(fase, rol, semana)

            pct_low, pct_high   = rol_bounds["pct"]
            reps_low, reps_high = rol_bounds["reps"]
            rpe_low, rpe_high   = rol_bounds["rpe"]

            # Día de velocidad (fase peak, rol L): forzamos menos reps
            if es_vel and fase == "peak" and rol == "L":
                reps_high = min(reps_high, 3)

            pct[j]  = self._clamp(float(pct[j]),  pct_low, pct_high)
            reps[j] = int(self._clamp(int(reps[j]), reps_low, reps_high))
            rpe[j]  = self._rpe_discreto(self._clamp(float(rpe[j]), rpe_low, rpe_high))

        # 2) DUP H≥M≥L con brecha mínima en %1RM
        idx_orden = sorted(range(n), key=lambda j: pct[j], reverse=True)
        gap = float(self._GAP_PCT_MIN)

        if n >= 3:
            jH, jM, jL = idx_orden[0], idx_orden[1], idx_orden[2]
            bH = bounds_fase["H"]["pct"]
            bM = bounds_fase["M"]["pct"]
            bL = bounds_fase["L"]["pct"]

            if (pct[jH] - pct[jM]) < gap:
                objetivo_M = pct[jH] - gap
                pct[jM] = self._clamp(objetivo_M, bM[0], bM[1])

            if (pct[jM] - pct[jL]) < gap:
                objetivo_L = pct[jM] - gap
                pct[jL] = self._clamp(objetivo_L, bL[0], bL[1])

        elif n == 2:
            jH, jM = idx_orden[0], idx_orden[1]
            bM = bounds_fase["M"]["pct"]
            if (pct[jH] - pct[jM]) < gap:
                objetivo_M = pct[jH] - gap
                pct[jM] = self._clamp(objetivo_M, bM[0], bM[1])

        # 3) Volcar cambios y discretizar RPE definitivamente
        semana.pct_1rm_sesion = [float(x) for x in pct]
        semana.rpe_sesion     = [self._rpe_discreto(float(x)) for x in rpe]
        semana.reps_sesion    = [int(x) for x in reps]

        # 4) Recalcular agregados semanales coherentes
        try:
            if distrib and len(distrib) == n and sum(distrib) > 0:
                pesos = [int(x) for x in distrib]
            else:
                # Igual peso por sesión si no hay distrib_series coherente
                pesos = [1] * n

            total_sets = int(sum(pesos)) or 1
            semana.series = total_sets

            semana.pct_1rm = float(
                sum(w * p for w, p in zip(pesos, semana.pct_1rm_sesion)) / total_sets
            )
            semana.rpe = self._rpe_discreto(
                sum(w * r for w, r in zip(pesos, semana.rpe_sesion)) / total_sets
            )
            semana.reps = int(
                round(sum(w * rr for w, rr in zip(pesos, semana.reps_sesion)) / total_sets)
            )
        except Exception:
            # Si algo falla, al menos mantiene los arrays ya reparados
            pass

        return None

    def _limit_weekly_deltas_por_rol(self, plan):
        # asegúrate de que esta función NO esté anidada por error; si lo está, este wrapper evita el crash
        try:
            return self.__class__.__dict__.get("_limit_weekly_deltas_por_rol", lambda *_: None)(self, plan)
        except Exception:
            return None

    @staticmethod
    def _rol_por_share(share: float) -> str:
        # H si share > 45%; 45% exacto pasa a M
        return "H" if share > 0.45 else ("M" if share >= 0.30 else "L")

    def _bloque_de_semana(self, semana_idx: int) -> int:
        print("EvolucionEntrenamiento - _bloque_de_semana: resolviendo bloque por índice de semana.")
        # idx 0..11 → W1–4=B1, W5–8=B2, W9–12=B3
        return 1 if semana_idx < 4 else (2 if semana_idx < 8 else 3)

# Fin de archivo generado.
