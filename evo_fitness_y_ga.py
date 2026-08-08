from __future__ import annotations

import os
import logging
import random
import math
import time
from typing import List, Tuple, Dict

from modelos import SemanaPlan, Plan12Semanas

# =========================
# Configuración de debug EE
# =========================

# Activa o desactiva debug global:  EE_DEBUG=1
EE_DEBUG = os.environ.get("EE_DEBUG", "").lower() in ("1", "true", "yes")

# Filtrar por secciones:
#   EE_DEBUG_SECTIONS=metricas,penal,roles,trend,resumen
EE_DEBUG_SECTIONS = os.environ.get("EE_DEBUG_SECTIONS", "").strip()
if EE_DEBUG_SECTIONS:
    EE_DEBUG_SECTIONS = {s.strip() for s in EE_DEBUG_SECTIONS.split(",")}
else:
    EE_DEBUG_SECTIONS = set()

# Límite de mensajes por etiqueta (para no inundar la consola)
try:
    EE_DEBUG_LIMIT = int(os.environ.get("EE_DEBUG_LIMIT", "5"))
except ValueError:
    EE_DEBUG_LIMIT = 5

logger = logging.getLogger("EvolucionEntrenamiento")
logger.setLevel(logging.DEBUG if EE_DEBUG else logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _fmt = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
    _h.setFormatter(_fmt)
    logger.addHandler(_h)




class FitnessGAMixin:
    """
    Mixin con lógica de fitness/GA. Ahora incluye _dbg(...) para depuración adaptable:
    - Formato: "EvolucionEntrenamiento - nombre_metodo: mensaje".
    - Usa logger "IronProgress" si existe (rotativo/rate-limit/anti-spam del main).
    - Respeta EE_LOGS_PERMITIR / EE_LOGS_BLOQUEAR por prefijo de método.
    - Degrada a print si no hay handlers configurados.
    """

    # ------------------------------
    # Infra de depuración adaptable
    # ------------------------------
    _dbg_cache_conf = None  # cachea prefijos permitir/bloquear para no re-leer env

    def _dbg__cargar_conf(self):
        """Carga/cacha listas de prefijos permitir/bloquear (env) y defaults sensatos."""
        if self._dbg_cache_conf is not None:
            return self._dbg_cache_conf

        bloquear_defecto = [
            "_reparar_semana", "_asegurar_roles_hml", "_normalizar_arrays_por_sesion",
            "_coherencia_pct_reps_rpe", "_limitar_series_por_rol", "_bloque_de_semana",
            "_ordenar_y_gap_intensidades", "_semana_aleatoria", "_individuo_aleatorio",
        ]
        permitir_defecto = [
            "evolucionar", "_fitness", "_fatiga_semana", "_progreso_fuerza_semana",
            "_distribuir_series", "_penalizar_", "imprimir_diagnostico",
            "_cruzar", "_mutar",
        ]
        permitir_env = [p.strip() for p in (os.environ.get("EE_LOGS_PERMITIR") or "").split(",") if p.strip()]
        bloquear_env = [p.strip() for p in (os.environ.get("EE_LOGS_BLOQUEAR") or "").split(",") if p.strip()]
        self._dbg_cache_conf = {
            "permitir": tuple(permitir_defecto + permitir_env),
            "bloquear": tuple(bloquear_defecto + bloquear_env),
        }
        return self._dbg_cache_conf

        # ============
    # COHERENCIA %1RM ↔ REPS ↔ RPE
    # ============

    def _pct_desde_reps_rpe(self, reps: int, rpe: float) -> float:
        """
        Estima %1RM coherente a partir de (reps, RPE) usando RIR y Epley inverso.
        - El AG elige reps y RPE.
        - Aquí derivamos el %1RM que “cuadra” con esa combinación.
        """
        reps = max(1, int(reps))
        rpe = float(max(5.0, min(10.0, rpe)))
        # RIR aproximado: 10 - RPE
        rir = max(0.0, 10.0 - rpe)
        reps_fallo = reps + rir
        # Epley inverso: %1RM ≈ 100 / (1 + 0.033 * reps_fallo)
        pct = 100.0 / (1.0 + 0.033 * reps_fallo)
        # Clamp fisiológico
        return max(50.0, min(98.0, pct))

    def _sincronizar_plan_pct_vs_reps_rpe(self, plan) -> None:
        """
        Fuerza coherencia fisiológica en TODO el plan antes de evaluar:
        - Mantiene reps y RPE tal como los propuso el GA.
        - Recalcula %1RM de cada sesión mediante _pct_desde_reps_rpe.
        - Actualiza pct_1rm_sesion y pct_1rm semanal (media ponderada por sets).
        Así evitamos “exploits” tipo 1 rep @ 92% @ RPE 8.0.
        """
        for s in plan.semanas:
            n = int(getattr(s, "sesiones", 0) or 0)
            if n <= 0:
                continue

            sets, pct, rpe_obj, rpe_real, reps = self._get_arrays_semana(s)

            total_sets = max(1, sum(int(x) for x in sets))
            if total_sets <= 0:
                continue

            pct_new = []
            acum_pct_sets = 0.0

            for st, rp, rpe_o in zip(sets, reps, rpe_obj):
                st = int(st)
                if st <= 0:
                    pct_j = float(s.pct_1rm)
                else:
                    pct_j = self._pct_desde_reps_rpe(rp, rpe_o)
                pct_new.append(pct_j)
                acum_pct_sets += float(st) * pct_j

            # Escribimos de vuelta en la semana
            s.pct_1rm_sesion = pct_new
            s.pct_1rm = acum_pct_sets / float(total_sets)


    def _dbg__permitido(self, metodo: str) -> bool:
        """Decide si se emite el mensaje según prefijos permitir/bloquear."""
        conf = self._dbg__cargar_conf()
        m = str(metodo or "")
        if any(m.startswith(p) for p in conf["bloquear"]):
            return False
        # Si está permitido explícitamente, sí; si no, también (permite por defecto)
        return True

    def _dbg(self, metodo: str, *args, **kwargs):
        """
        Wrapper tolerante para logs de depuración.

        Acepta TODOS estos formatos sin tronar:
          _dbg("met", "msg")
          _dbg("met", "msg", nivel=logging.INFO)
          _dbg("met", logging.INFO, "msg")
          _dbg("met", "msg", logging.INFO)
          _dbg("met", "msg", nivel=logging.INFO, forzar=True)

        Donde:
          - metodo: nombre corto del método que loguea.
          - msg:    texto a imprimir.
          - nivel:  nivel de logging (DEBUG, INFO, WARNING...).
        """
        import logging

        # --- Defaults ---
        nivel_default = logging.DEBUG
        nivel = kwargs.pop("nivel", None)
        forzar = kwargs.pop("forzar", False)  # por si en algún lado lo usabas
        msg = ""

        # --- Parseo flexible de *args ---
        if len(args) == 0:
            pass
        elif len(args) == 1:
            # Puede ser solo msg, o solo nivel
            if isinstance(args[0], int):
                nivel = args[0]
            else:
                msg = str(args[0])
        else:
            # Tenemos al menos 2: pueden venir como (msg, nivel) o (nivel, msg)
            a0, a1 = args[0], args[1]
            if isinstance(a0, int) and not isinstance(a1, int):
                nivel = a0
                msg = str(a1)
            elif not isinstance(a0, int) and isinstance(a1, int):
                msg = str(a0)
                nivel = a1
            else:
                # Ambiguo → tomamos el primero como msg
                msg = str(a0)

            # Si hay más args, los concatenamos al mensaje
            if len(args) > 2:
                extra = " | ".join(str(a) for a in args[2:])
                msg = f"{msg} | {extra}" if msg else extra

        if nivel is None:
            nivel = nivel_default

        # Seguridad: si msg quedó vacío y alguien lo pasó por kwargs
        if not msg and "msg" in kwargs:
            msg = str(kwargs["msg"])

        # --- Enrutado final del log ---
        linea = f"EvolucionEntrenamiento - {metodo}: {msg}"

        # Si la clase expone log_debug (y el main lo envuelve con filtros), úsalo
        ld = getattr(self, "log_debug", None)
        if callable(ld):
            try:
                ld(metodo, msg)
                return
            except Exception:
                pass

        # Fallback: logger estándar
        logger = logging.getLogger("EvolucionEntrenamiento")
        logger.log(nivel, linea)

    


    # ============================== LÓGICA ORIGINAL (no alterada) ==============================

    def _hard_guard_ideales(self, s: SemanaPlan, semana_idx: int) -> Tuple[bool, str]:
        """
        EvolucionEntrenamiento - _hard_guard_ideales: valida rangos 'ideales' por BLOQUE (suave).
        Si NO cumple, devolvemos (False, motivo) para abortar el individuo.
        """
        bloque = 1 if semana_idx < 4 else (2 if semana_idx < 8 else 3)
        cfg = self.IDEAL_BLOQUE[bloque]

        lo_ws, hi_ws = cfg["series_semana"]
        lo_rw, hi_rw = cfg["reps_semana"]

        # Reps semanales como media ponderada por sets si hay desglose
        reps_sem = int(s.reps)
        if s.distrib_series and s.reps_sesion:
            total_sets = max(1, int(s.series))
            reps_sem = int(round(
                sum(sets * reps for sets, reps in zip(s.distrib_series, s.reps_sesion)) / total_sets
            ))

        # Series y reps semanales
        if not (lo_ws <= int(s.series) <= hi_ws):
            return False, f"[B{bloque}] series semanales fuera de [{lo_ws},{hi_ws}] (tiene {s.series})"
        if not (lo_rw <= reps_sem <= hi_rw):
            return False, f"[B{bloque}] reps semanales fuera de [{lo_rw},{hi_rw}] (tiene {reps_sem})"

        # Chequeos por sesión si existen arrays
        if s.distrib_series and s.pct_1rm_sesion and s.rpe_sesion and s.reps_sesion:
            lo_pct, hi_pct = cfg["pct_sesion"]
            lo_rpe, hi_rpe = cfg["rpe_sesion"]
            lo_rs,  hi_rs  = cfg["reps_sesion"]

            n = min(len(s.distrib_series), len(s.pct_1rm_sesion), len(s.rpe_sesion), len(s.reps_sesion))
            for j in range(n):
                sets = int(s.distrib_series[j])
                pct  = float(s.pct_1rm_sesion[j])
                rpe  = float(s.rpe_sesion[j])
                reps = int(s.reps_sesion[j])

                if sets < 0:
                    return False, f"[B{bloque}] sets negativos en sesión {j+1}"
                if not (lo_pct <= pct <= hi_pct):
                    return False, f"[B{bloque}] %1RM sesión {j+1} fuera de [{lo_pct},{hi_pct}] (tiene {pct:.1f})"
                if not (lo_rpe <= rpe <= hi_rpe):
                    return False, f"[B{bloque}] RPE sesión {j+1} fuera de [{lo_rpe},{hi_rpe}] (tiene {rpe:.1f})"
                if not (lo_rs <= reps <= hi_rs):
                    return False, f"[B{bloque}] reps sesión {j+1} fuera de [{lo_rs},{hi_rs}] (tiene {reps})"
        else:
            # Fallback si no hay desglose por sesión
            lo_pct, hi_pct = cfg["pct_sesion"]
            lo_rpe, hi_rpe = cfg["rpe_sesion"]
            if not (lo_pct <= float(s.pct_1rm) <= hi_pct):
                return False, f"[B{bloque}] %1RM semanal fuera de [{lo_pct},{hi_pct}] (tiene {s.pct_1rm:.1f})"
            if not (lo_rpe <= float(s.rpe) <= hi_rpe):
                return False, f"[B{bloque}] RPE semanal fuera de [{lo_rpe},{hi_rpe}] (tiene {s.rpe:.1f})"

        return True, ""

    # --- Guardarraíl "duro" real (solo incoherencias físicas graves) ---
    def _violacion_dura(self, pct: float, rpe: float) -> bool:
        return (pct < 50.0 or pct > 98.0) or (rpe < 5.0 or rpe > 10.0)

    def _fatiga_semana(self, semana: SemanaPlan) -> Tuple[float, float]:
        """
        Calcula los dos componentes de carga semanal:
        1) Fatiga muscular (volumen efectivo)
        2) Carga del SNC
        """
        EXP_PESO_INTENSIDAD = 1.6
        RPE_UMBRAL_BASE = 6.0
        RPE_INCREMENTO = 0.25

        modo_musc = getattr(self, "modelo_fatiga_muscular", "original")  # "original" | "por_rir" | "mixto"
        modo_snc  = getattr(self, "modelo_fatiga_snc", "original")       # "original" | "suave"

        def epley_pct_a_reps_a_fallo(pct: float) -> float:
            pct = max(1e-6, min(100.0, float(pct)))
            return max(0.0, (100.0 / pct - 1.0) / 0.033)

        def rir_objetivo_desde_rpe(rpe: float) -> float:
            return max(0.0, 10.0 - float(rpe))

        def interp_lineal(x: float, xs: list, ys: list) -> float:
            if not xs or not ys or len(xs) != len(ys):
                return ys[-1] if ys else 0.0
            if x <= xs[0]:
                return ys[0]
            if x >= xs[-1]:
                return ys[-1]
            for i in range(1, len(xs)):
                if x <= xs[i]:
                    x0, y0 = xs[i-1], ys[i-1]
                    x1, y1 = xs[i], ys[i]
                    t = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
                    return y0 + t * (y1 - y0)
            return ys[-1]

        def peso_snc_por_porcentaje(pct_1rm: float) -> float:
            if modo_snc == "suave":
                cortes_pct = [0.0, 80.0, 85.0, 90.0, 95.0, 100.0]
                pesos_pct  = [0.0, 0.0, 0.8, 1.6, 3.0, 4.5]
                return interp_lineal(float(pct_1rm), cortes_pct, pesos_pct)
            if pct_1rm < 80.0: return 0.0
            if pct_1rm < 85.0: return 0.8
            if pct_1rm < 90.0: return 1.6
            if pct_1rm < 95.0: return 3.0
            return 4.5

        def peso_snc_por_rpe(rpe: float) -> float:
            if modo_snc == "suave":
                cortes_rpe = [0.0, 7.0, 8.5, 10.0]
                pesos_rpe  = [0.5, 0.5, 1.0, 1.5]
                return interp_lineal(float(rpe), cortes_rpe, pesos_rpe)
            if rpe <= 7.0: return 0.5
            if rpe <= 8.5: return 1.0
            return 1.5

        def factor_rpe_lineal(rpe: float) -> float:
            return 1.0 + max(0.0, (float(rpe) - RPE_UMBRAL_BASE)) * RPE_INCREMENTO

        def factor_por_rir_est(pct: float, reps: int) -> float:
            reps_fallo = epley_pct_a_reps_a_fallo(pct)
            rir_est = max(0.0, reps_fallo - float(reps))
            return 1.0 + max(0.0, 4.0 - rir_est) / 4.0

        def volumen_muscular_sesion(sets: int, pct: float, rpe: float, reps: int) -> float:
            intensidad_rel = max(0.0, float(pct)) / 100.0
            base = float(sets) * float(reps) * (intensidad_rel ** EXP_PESO_INTENSIDAD)

            if modo_musc == "original":
                return base * factor_rpe_lineal(rpe)
            elif modo_musc == "por_rir":
                return base * factor_por_rir_est(pct, reps)
            elif modo_musc == "mixto":
                f_rpe = factor_rpe_lineal(rpe)
                f_rir = factor_por_rir_est(pct, reps)
                return base * (0.5 * f_rpe + 0.5 * f_rir)
            else:
                return base * factor_rpe_lineal(rpe)

        tiene_arrays = (
            bool(semana.distrib_series) and semana.series > 0
            and bool(semana.pct_1rm_sesion) and bool(semana.rpe_sesion) and bool(semana.reps_sesion)
            and len(semana.distrib_series) == len(semana.pct_1rm_sesion) == len(semana.rpe_sesion) == len(semana.reps_sesion)
        )

        if tiene_arrays:
            # ✅ aquí corregimos la llamada a _dbg
            self._dbg("fatiga", "arrays", "usando arrays por sesión (H/M/L).")

            fatiga_muscular_total = 0.0
            carga_snc_total = 0.0
            sets_con_pct85_o_mas = 0

            for sets_ses, pct_ses, rpe_ses, reps_ses in zip(
                semana.distrib_series, semana.pct_1rm_sesion, semana.rpe_sesion, semana.reps_sesion
            ):
                fatiga_muscular_total += volumen_muscular_sesion(sets_ses, pct_ses, rpe_ses, reps_ses)
                carga_snc_total += float(sets_ses) * peso_snc_por_porcentaje(pct_ses) * peso_snc_por_rpe(rpe_ses)
                if pct_ses >= 85.0:
                    sets_con_pct85_o_mas += int(sets_ses)

            if sets_con_pct85_o_mas >= 3:
                carga_snc_total *= 1.1

            fatiga_muscular_total /= (semana.sesiones ** 0.15)

            return fatiga_muscular_total, carga_snc_total

        no_hay_arrays = (not semana.distrib_series) or (semana.series <= 0)
        if no_hay_arrays:
            # ✅ corregimos esta también
            self._dbg("fatiga", "simple", "estimador simple (sin arrays).")

            fatiga_muscular_simple = (
                volumen_muscular_sesion(semana.series, semana.pct_1rm, semana.rpe, semana.reps)
                / (semana.sesiones ** 0.2)
            )

            carga_snc_simple = (
                float(semana.series) * peso_snc_por_porcentaje(semana.pct_1rm) * peso_snc_por_rpe(semana.rpe)
            )

            return fatiga_muscular_simple, carga_snc_simple

        # ✅ y esta tercera
        self._dbg("fatiga", "aprox", "aproximación al vuelo (derivar H/M/L por cuota de sets).")

        volumen_muscular_total: float = 0.0
        carga_snc_total: float = 0.0
        sets_con_85_o_mas: int = 0

        for sets_en_sesion in semana.distrib_series:
            cuota_relativa = sets_en_sesion / max(1.0, semana.series)

            if cuota_relativa >= 0.45:
                pct_1rm_sesion = min(100.0, semana.pct_1rm + 2.5)
                rpe_sesion = min(9.0, semana.rpe + 0.3)
                reps_sesion = max(1, semana.reps - 2)
            elif cuota_relativa >= 0.30:
                pct_1rm_sesion = semana.pct_1rm
                rpe_sesion = semana.rpe
                reps_sesion = semana.reps
            else:
                pct_1rm_sesion = max(50.0, semana.pct_1rm - 2.5)
                rpe_sesion = max(6.0, semana.rpe - 0.3)
                reps_sesion = min(30, semana.reps + 2)

            intensidad_relativa = pct_1rm_sesion / 100.0
            factor_rpe = 1.0 + max(0.0, (rpe_sesion - 6.0)) / 4.0
            volumen_por_sesion = sets_en_sesion * reps_sesion * (intensidad_relativa ** 1.6) * factor_rpe
            volumen_muscular_total += volumen_por_sesion

            carga_snc_total += sets_en_sesion * peso_snc_por_porcentaje(pct_1rm_sesion) * peso_snc_por_rpe(rpe_sesion)

            if pct_1rm_sesion >= 85.0:
                sets_con_85_o_mas += sets_en_sesion

        if sets_con_85_o_mas >= 3:
            carga_snc_total *= 1.2

        volumen_muscular_total /= (semana.sesiones ** 0.15)

        return volumen_muscular_total, carga_snc_total


    def _progreso_fuerza_semana(self, idx: int, s: SemanaPlan) -> float:
        """
        Componente de progreso de fuerza por semana.
          - Peso por bloque: B1=0.6, B2=1.0, B3=1.5
          - i = max(%1RM_sesion)/100; expo_int = max(0, i - 0.75)
          - reps_factor: 1.0 en B3; en B1/B2 es 0.85 si 5<=reps_ref<=10, si no 0.65
          - rpe_factor = 0.9 + 0.1*(rpe_ref - 6.0)
          - sesiones_factor = 0.95 + 0.05*(sesiones - 2)
          - sets_factor = log(1 + series)
          - Bonus patrón H/M/L cercano a [0.50, 0.33, 0.17] (tol ≤3%)
        """
        block_weight = 0.6 if idx < 4 else (1.0 if idx < 8 else 1.5)
        max_pct = max(s.pct_1rm_sesion) if s.pct_1rm_sesion else s.pct_1rm
        i = max_pct / 100.0
        expo_int = max(0.0, i - 0.75)

        if idx >= 8 and s.reps_sesion:
            reps_ref = min(s.reps_sesion)
        else:
            reps_ref = sum(s.reps_sesion) / len(s.reps_sesion) if s.reps_sesion else s.reps

        if idx >= 8:
            reps_factor = 1.0
        else:
            reps_factor = 0.85 if 5 <= reps_ref <= 10 else 0.65

        rpe_ref = max(s.rpe_sesion) if s.rpe_sesion else s.rpe
        rpe_factor = 0.9 + 0.1 * (rpe_ref - 6.0)

        sesiones_factor = 0.95 + 0.05 * (s.sesiones - 2)
        sets_factor = math.log(1 + s.series)

        bonus_hml = 1.0
        if s.distrib_series and s.series > 0:
            shares = sorted([x / s.series for x in s.distrib_series], reverse=True)
            if s.sesiones == 3:
                objetivo = [0.50, 0.33, 0.17]
                dist = sum(abs(a - b) for a, b in zip(shares, objetivo))
                bonus_hml = max(0.5, 1.0 - max(0.0, dist - 0.03) * 10.0)
            elif s.sesiones == 2:
                objetivo = [0.60, 0.40]
                dist = sum(abs(a - b) for a, b in zip(shares, objetivo))
                bonus_hml = max(0.5, 1.0 - max(0.0, dist - 0.03) * 10.0)

        return block_weight * expo_int * reps_factor * rpe_factor * sesiones_factor * sets_factor * bonus_hml

    def _penalizar_orden_roles(self, semanas) -> float:
        """
        Exige no solo H ≥ M ≥ L, sino un GAP mínimo entre roles:
        - Diferencia mínima 5 pp en %1RM y 0.5 en RPE entre H–M y M–L.
        Corrige el caso H y M casi iguales (falsa ondulación).
        """
        pen = 0.0
        TOL_PCT = 2.0      # tolerancia para violar el orden
        TOL_RPE = 0.25
        GAP_MIN_PCT = 5.0  # gap mínimo deseado
        GAP_MIN_RPE = 0.5

        for s in semanas:
            if not (s.distrib_series and s.series and s.pct_1rm_sesion and s.rpe_sesion):
                continue

            n = min(len(s.distrib_series), len(s.pct_1rm_sesion), len(s.rpe_sesion))
            if n == 0 or s.series <= 0:
                continue

            agg = {"H": [0.0, 0.0, 0], "M": [0.0, 0.0, 0], "L": [0.0, 0.0, 0]}
            for j in range(n):
                sets_j = int(s.distrib_series[j])
                if sets_j <= 0:
                    continue
                share = sets_j / max(1, s.series)
                rol = self._rol_por_share(share)
                pct_j = float(s.pct_1rm_sesion[j])
                rpe_j = float(s.rpe_sesion[j])

                agg[rol][0] += pct_j * sets_j
                agg[rol][1] += rpe_j * sets_j
                agg[rol][2] += sets_j

            rol_stats = {}
            total_sets = max(1, s.series)
            for rol, (sum_pct, sum_rpe, sum_sets) in agg.items():
                if sum_sets > 0:
                    rol_stats[rol] = {
                        "pct": sum_pct / sum_sets,
                        "rpe": sum_rpe / sum_sets,
                        "share": sum_sets / total_sets,
                    }

            def cmp(a: str, b: str) -> None:
                nonlocal pen
                if a in rol_stats and b in rol_stats:
                    pa, pb = rol_stats[a]["pct"], rol_stats[b]["pct"]
                    ra, rb = rol_stats[a]["rpe"], rol_stats[b]["rpe"]

                    # 1) orden básico H ≥ M ≥ L
                    if pa + TOL_PCT < pb:
                        pen += (pb - pa - TOL_PCT) * 0.8
                    if ra + TOL_RPE < rb:
                        pen += (rb - ra - TOL_RPE) * 1.0

                    # 2) GAP mínimo (ondulación real)
                    gap_pct = pa - pb
                    gap_rpe = ra - rb
                    if gap_pct < GAP_MIN_PCT:
                        pen += (GAP_MIN_PCT - gap_pct) * 0.6
                    if gap_rpe < GAP_MIN_RPE:
                        pen += (GAP_MIN_RPE - gap_rpe) * 0.8

            cmp("H", "M")
            cmp("M", "L")

        return pen

    def _penalizar_tendencias_y_suavidad(self, semanas) -> float:
        """
        Penaliza saltos bruscos entre semanas (suavidad W→W) y “zig-zags” raros.
        Implementa la idea de:
        - Δ%1RM moderados (≈2.5–5 pp)
        - ΔRPE moderados (≈0.5–1.0)
        - Δreps moderados (≈1–2)
        - Δseries moderados (≈4–6)
        Cualquier cambio que se pase de esos rangos suma penalización.
        """
        if not semanas or len(semanas) < 2:
            return 0.0

        # Penalizaciones separadas para poder loguear bonito
        pen_smooth = 0.0   # cambios bruscos semana→semana
        pen_trend  = 0.0   # zig-zags raros de %1RM

        # Umbrales “suaves” y “duros”
        SOFT_PCT   = 2.5
        HARD_PCT   = 5.0
        SOFT_RPE   = 0.5
        HARD_RPE   = 1.0
        SOFT_REPS  = 1.0
        HARD_REPS  = 2.0
        SOFT_SETS  = 4.0
        HARD_SETS  = 6.0

        # Pesos de penalización
        W_PCT      = 4.0
        W_PCT_HD   = 8.0
        W_RPE      = 20.0
        W_RPE_HD   = 40.0
        W_REPS     = 6.0
        W_REPS_HD  = 10.0
        W_SETS     = 1.5
        W_SETS_HD  = 2.5
        W_ZIGZAG   = 2.0

        # Resumen semanal: volumen, %ponderado, RPE medio, reps medias
        resumen = []
        for idx, s in enumerate(semanas):
            sets, pct, rpe_obj, rpe_real, reps = self._get_arrays_semana(s)
            total_sets = max(1, sum(int(x) for x in sets)) if sets else 1

            # %1RM semanal ponderado por sets
            pct_sem = float(self._pct_semana_ponderado(s))

            # RPE y reps medios ponderados por sets
            if sets and total_sets > 0:
                rpe_med = sum(float(r) * int(st) for r, st in zip(rpe_obj, sets)) / float(total_sets)
                reps_med = sum(int(rep) * int(st) for rep, st in zip(reps, sets)) / float(total_sets)
            else:
                # fallback defensivo
                rpe_med = float(getattr(s, "rpe", 7.0))
                reps_med = float(getattr(s, "reps", 8))

            resumen.append({
                "idx":  idx,
                "sets": float(total_sets),
                "pct":  pct_sem,
                "rpe":  float(rpe_med),
                "reps": float(reps_med),
            })

        # Penalización por cambios semana a semana
        for i in range(1, len(resumen)):
            prev = resumen[i - 1]
            cur  = resumen[i]

            d_pct  = cur["pct"]  - prev["pct"]
            d_rpe  = cur["rpe"]  - prev["rpe"]
            d_reps = cur["reps"] - prev["reps"]
            d_sets = cur["sets"] - prev["sets"]

            ad_pct  = abs(d_pct)
            ad_rpe  = abs(d_rpe)
            ad_reps = abs(d_reps)
            ad_sets = abs(d_sets)

            # =========================
            # 1) Suavidad semana→semana
            # =========================

            # %1RM
            if ad_pct > SOFT_PCT:
                exceso = ad_pct - SOFT_PCT
                pen_smooth += exceso * W_PCT
            if ad_pct > HARD_PCT:
                exceso = ad_pct - HARD_PCT
                pen_smooth += exceso * W_PCT_HD

            # RPE
            if ad_rpe > SOFT_RPE:
                exceso = ad_rpe - SOFT_RPE
                pen_smooth += exceso * W_RPE
            if ad_rpe > HARD_RPE:
                exceso = ad_rpe - HARD_RPE
                pen_smooth += exceso * W_RPE_HD

            # Reps
            if ad_reps > SOFT_REPS:
                exceso = ad_reps - SOFT_REPS
                pen_smooth += exceso * W_REPS
            if ad_reps > HARD_REPS:
                exceso = ad_reps - HARD_REPS
                pen_smooth += exceso * W_REPS_HD

            # Series totales
            if ad_sets > SOFT_SETS:
                exceso = ad_sets - SOFT_SETS
                pen_smooth += exceso * W_SETS
            if ad_sets > HARD_SETS:
                exceso = ad_sets - HARD_SETS
                pen_smooth += exceso * W_SETS_HD

            # ==============================
            # 2) Zig-zag de intensidad (%1RM)
            # ==============================
            if i >= 2:
                prev2 = resumen[i - 2]
                d_pct_prev = prev["pct"] - prev2["pct"]
                # cambia el signo (sube y luego baja o viceversa) y ambos saltos son grandes
                if d_pct_prev * d_pct < 0 and abs(d_pct_prev) > SOFT_PCT and ad_pct > SOFT_PCT:
                    pen_trend += (abs(d_pct_prev) + ad_pct) * W_ZIGZAG

        self._dbg(
        "trend",
        "tendencias",
        "pen_trend=%.2f | pen_smooth=%.2f",
        pen_trend,
        pen_smooth,
        )

        pen_total = float(pen_smooth + pen_trend)

        # Log compacto y protegido por _dbg si existe
        if hasattr(self, "_dbg"):
            self._dbg(
                "trend",
                "tendencias",
                "pen_trend=%.2f | pen_smooth=%.2f | pen_total=%.2f",
                pen_trend,
                pen_smooth,
                pen_total,
            )

        return pen_total



    def _penalizar_singles_pct_ML(self, semanas) -> float:
        """
        Penaliza:
        - Singles (reps == 1) en sesiones M o L.
        - %1RM >= 88 en M o L.
        Además controla el nº de singles por semana según BLOQUE.
        """
        PEN_SINGLE_ML = float(getattr(self, "pen_single_ml", 400.0))
        PEN_PCT88_ML  = float(getattr(self, "pen_pct88_ml", 300.0))

        pen = 0.0

        for idx, s in enumerate(semanas):
            if not (s.distrib_series and s.pct_1rm_sesion and s.rpe_sesion and s.reps_sesion):
                continue
            n = min(len(s.distrib_series), len(s.pct_1rm_sesion), len(s.rpe_sesion), len(s.reps_sesion))
            if n == 0 or s.series <= 0:
                continue

            try:
                roles = self._roles_por_intensidad(s)
            except Exception:
                total = max(1, s.series)
                roles = [self._rol_por_share((s.distrib_series[j] if j < len(s.distrib_series) else 0)/total) for j in range(n)]

            singles_week = 0

            for j in range(n):
                rol = roles[j] if j < len(roles) else "M"
                reps_j = int(s.reps_sesion[j])
                pct_j  = float(s.pct_1rm_sesion[j])

                if rol in ("M", "L"):
                    if reps_j == 1:
                        pen += PEN_SINGLE_ML
                        singles_week += int(s.distrib_series[j])
                    if pct_j >= 88.0:
                        pen += PEN_PCT88_ML

            # límite por BLOQUE
            if idx < 4:
                max_singles = 0          # B1: idealmente sin singles
            elif idx < 8:
                max_singles = 6          # B2: algunos singles OK
            else:
                max_singles = 10         # B3: peaking, pero sin locuras

            exceso = max(0, singles_week - max_singles)
            if exceso > 0:
                pen += exceso * 50.0 * (1 + idx * 0.1)

        return pen

   

    def _bonus_taper(self, plan: Plan12Semanas) -> float:
        """
        Score 0–10 para el taper (W12), usando W11 como referencia.

        Queremos:
        - Muy pocas series en W12 (≤4–6).
        - Pocas reps medias en W12 (≤3 aprox.).
        - Intensidad de W12 alta (≈82–90% 1RM),
          pero ligeramente menor o similar a W11 (no un fake deload).
        """
        semanas = plan.semanas
        if len(semanas) != 12:
            return 0.0

        w11 = semanas[10]
        w12 = semanas[11]

        def _metrics_semana(sem: SemanaPlan) -> Tuple[float, float, float]:
            sets, pct, rpe_obj, rpe_real, reps = self._get_arrays_semana(sem)
            total_sets = sum(int(s) for s in sets)
            if total_sets <= 0:
                # fallback por si algo raro
                return 0.0, float(getattr(sem, "reps", 0.0)), float(getattr(sem, "pct1RM", 0.0))

            reps_acum = sum(int(s) * int(r) for s, r in zip(sets, reps))
            pct_acum  = sum(float(p) * int(s) for s, p in zip(sets, pct))
            reps_prom = reps_acum / float(total_sets)
            pct_prom  = pct_acum  / float(total_sets)
            return float(total_sets), reps_prom, pct_prom

        sets11, reps11, pct11 = _metrics_semana(w11)
        sets12, reps12, pct12 = _metrics_semana(w12)

        score = 0.0

        # 1) Volumen muy bajo en W12
        if sets12 <= 4:
            score += 4.0
        elif sets12 <= 6:
            score += 2.0

        # 2) Reps bajas en W12 (más cerca de singles/dobles/triples)
        if reps12 <= 2.5:
            score += 3.0
        elif reps12 <= 3.5:
            score += 1.5

        # 3) Intensidad: alta pero compatible con deload
        #    Queremos algo tipo 82–90% y que no sea mucho más baja que W11.
        if 82.0 <= pct12 <= 90.0 and pct12 <= pct11 + 1.0:
            score += 3.0

        # Clamp 0–10
        if score < 0.0:
            score = 0.0
        if score > 10.0:
            score = 10.0

        return score



    def _cruzar(self, a: Plan12Semanas, b: Plan12Semanas) -> Tuple[Plan12Semanas, Plan12Semanas]:
        """Cruce de un punto por semanas completas."""
        if random.random() > self.prob_cruce:
            return a, b
        punto = random.randint(1, 11)
        hijo1 = Plan12Semanas(self.patron, a.semanas[:punto] + b.semanas[punto:])
        hijo2 = Plan12Semanas(self.patron, b.semanas[:punto] + a.semanas[punto:])

        self._dbg(
            "cruce",  # seccion
            "reparacion_hijo1",  # clave
            "aplicando reparación post-cruce para conservar tendencias por bloque (hijo1)."
        )
        self._reparar_y_marcar_bloques(hijo1)
        self._dbg(
            "cruce",  # seccion
            "reparacion_hijo2",  # clave
            "aplicando reparación post-cruce para conservar tendencias por bloque (hijo2)."
        )
        self._reparar_y_marcar_bloques(hijo2)

        return hijo1, hijo2

    def _mutar_semana(self, s: SemanaPlan, idx: int) -> SemanaPlan:
        """Ruido pequeño + clamps por bloque + reajuste de distribución + regeneración por sesión."""
        sesiones = s.sesiones
        series = s.series
        pct = s.pct_1rm
        rpe = s.rpe
        reps = s.reps
        distrib = s.distrib_series[:]

        if random.random() < self.prob_mutacion:
            sesiones = random.choice([2, 3])

        if random.random() < self.prob_mutacion:
            series = int(round(series + random.choice([-3, -2, -1, 1, 2, 3])))
        if random.random() < self.prob_mutacion:
            pct += random.uniform(-4.0, 4.0)
        if random.random() < self.prob_mutacion:
            rpe += random.uniform(-0.5, 0.5)
        if random.random() < self.prob_mutacion:
            reps = int(round(reps + random.choice([-4, -2, -1, 1, 2, 4])))

        # Clamps por bloque
        if idx < 4:
            series = min(20, max(12, series))
            pct = min(75.0, max(60.0, pct))
            rpe = min(9.0, max(6.0, rpe))
            reps = min(15, max(8, reps))
        elif idx < 8:
            series = min(16, max(10, series))
            pct = min(88.0, max(75.0, pct))
            rpe = min(9.0, max(6.0, rpe))
            reps = min(10, max(4, reps))
        else:
            series = min(12, max(6, series))
            pct = min(97.0, max(85.0, pct))
            rpe = min(9.0, max(6.0, rpe))
            reps = min(6, max(1, reps))

        # Clamps globales
        series = max(2, min(20, series))
        pct = max(50.0, min(100.0, pct))
        rpe = self._rpe_discreto(max(6.0, min(10.0, rpe)))
        reps = max(1, min(30, reps))

        # Reajustar distribución
        distrib = self._reajustar_distrib(distrib, series, sesiones)

        # Micro-mutación de distribución (conserva suma)
        if random.random() < (self.prob_mutacion * 0.6) and series > 0 and len(distrib) > 1:
            a, b = random.sample(range(len(distrib)), 2)
            if distrib[a] > 1:
                distrib[a] -= 1
                distrib[b] += 1

        nuevo = SemanaPlan(sesiones, series, pct, rpe, reps, distrib)
        self._generar_por_sesion(nuevo, idx)
        self._canonizar_orden_hml(s)
        self._validar_coherencia_rir_rpe(s, idx)

        return nuevo

    def _mutar(self, plan: Plan12Semanas) -> Plan12Semanas:
        semanas = [self._mutar_semana(s, idx) for idx, s in enumerate(plan.semanas)]
        if hasattr(self, "_ajustar_rampa_interbloque"):
            self._ajustar_rampa_interbloque(semanas)

        nuevo_plan = Plan12Semanas(plan.patron, semanas)
        self._dbg(
            "mutacion",              # seccion
            "reparacion_plan",       # clave
            "aplicando reparación post-mutación a nivel plan (H≥M≥L, gaps, velocidad, taper)."
        )
        self._reparar_y_marcar_bloques(nuevo_plan)
        return nuevo_plan

    


    def imprimir_plan(self, plan: Plan12Semanas, ancho: int = 96, colores: bool = True, ascii_only: bool = False):
        """
        Imprime el plan de 12 semanas en formato tabular legible.
        - ancho: anchura de línea aproximada.
        - colores: usa colores ANSI en terminales que lo soporten.
        - ascii_only: separadores ASCII simples.
        """
        def c(txt, code):
            if not colores:
                return txt
            return f"\033[{code}m{txt}\033[0m"

        C_BLOQUE = "1;36"
        C_SEMANA = "1;37"
        C_H = "1;31"
        C_M = "1;33"
        C_L = "1;36"

        if ascii_only:
            HLINE = "-" * ancho
            HL_B = "=" * ancho
            PIPE = "|"
        else:
            HLINE = "─" * ancho
            HL_B = "═" * ancho
            PIPE = "│"

        def cabecera_bloque(titulo: str) -> None:
            t = f" {titulo} "
            if ascii_only:
                print(HL_B)
                print(t.center(ancho))
                print(HL_B)
            else:
                pad = max(0, ancho - len(t) - 2)
                print(c("┌" + "─" * (ancho - 2) + "┐", C_BLOQUE))
                print(c("│" + t + " " * pad + "│", C_BLOQUE))
                print(c("└" + "─" * (ancho - 2) + "┘", C_BLOQUE))

        def titulo_semana(i: int, s) -> None:
            pct_semana = self._pct_semana_ponderado(s) if (s.distrib_series and s.pct_1rm_sesion) else s.pct_1rm
            linea = (f"Semana {i + 1:02d}  |  Sesiones: {s.sesiones}  |  Series: {s.series:02d}  |  "
                    f"%1RM: {pct_semana:5.1f}  |  RPE: {s.rpe:3.1f}  |  Reps: {s.reps:02d}")
            print(c(linea, C_SEMANA))
            print(f"Distribución de series: {s.distrib_series}")
            print(HLINE)

        def tabla_sesiones(s) -> None:
            n = s.sesiones
            pct = (s.pct_1rm_sesion or [])[:]
            rpe = (s.rpe_sesion or [])[:]
            reps = (s.reps_sesion or [])[:]
            while len(pct)  < n: pct.append(s.pct_1rm)
            while len(rpe)  < n: rpe.append(s.rpe)
            while len(reps) < n: reps.append(s.reps)

            # ← NUEVO: obtenemos los roles por intensidad; fallback a cuota de sets
            try:
                roles_print = self._roles_por_intensidad(s)  # ['H','M','L'] según % por sesión
            except Exception:
                roles_print = None
            if not roles_print or len(roles_print) != n:
                total_sets = max(1, s.series)
                roles_print = [
                    self._rol_por_share((s.distrib_series[j] if j < len(s.distrib_series) else 0)/total_sets)
                    for j in range(n)
                ]

            hdr = f"{'Sesión':>6} {PIPE} {'Rol':^3} {PIPE} {'Sets':>4} {PIPE} {'%1RM':>6} {PIPE} {'RPE':>4} {PIPE} {'Reps':>4}"
            print(hdr)
            print(HLINE)

            total_sets = max(1, s.series)
            for j in range(n):
                sets_j = s.distrib_series[j] if j < len(s.distrib_series) else 0
                rol = roles_print[j] if j < len(roles_print) else self._rol_por_share(sets_j / total_sets)
                color = C_H if rol == "H" else (C_M if rol == "M" else C_L)
                fila = (f"{j + 1:>6} {PIPE} "
                        f"{c(rol, color):^3} {PIPE} "
                        f"{sets_j:>4} {PIPE} "
                        f"{float(pct[j]):>6.1f} {PIPE} "
                        f"{float(rpe[j]):>4.1f} {PIPE} "
                        f"{int(reps[j]):>4}")
                print(fila)

        bloques = {
            0: "Bloque 1 – Hipertrofia (Acumulación)  — Semanas 1–4",
            4: "Bloque 2 – Transición a fuerza (Intensificación)  — Semanas 5–8",
            8: "Bloque 3 – Intensificación / Peaking  — Semanas 9–12",
        }

        for i, s in enumerate(plan.semanas):
            if i in bloques:
                cabecera_bloque(bloques[i])
            necesita = (not s.pct_1rm_sesion or not s.rpe_sesion or not s.reps_sesion or
                        len(s.pct_1rm_sesion) != s.sesiones or
                        len(s.rpe_sesion) != s.sesiones or
                        len(s.reps_sesion) != s.sesiones)
            if necesita:
                s.pct_1rm_sesion = [float(s.pct_1rm)] * s.sesiones
                s.rpe_sesion = [float(s.rpe)] * s.sesiones
                s.reps_sesion = [int(s.reps)] * s.sesiones

            titulo_semana(i, s)
            tabla_sesiones(s)
            roles_print = self._roles_por_intensidad(s)

            print()

        # --- Roles H/M/L por intensidad y utilidades relacionadas ---

    def _rol_por_share(self, share: float) -> str:
        """
        Clasifica un día como H/M/L según la cuota de sets dentro de la semana.
        Umbrales coherentes con el resto del código.
        """
        if share >= 0.45:
            return "H"
        if share >= 0.30:
            return "M"
        return "L"

    def _roles_por_intensidad(self, s) -> List[str]:
        """
        Devuelve una lista de longitud s.sesiones con los roles ['H','M','L'] ordenados
        por sesión, calculados principalmente por intensidad (%1RM) y desempate
        por RPE (mayor es más “pesado”) y por reps (menos reps es más “pesado”).

        Fallbacks:
        - Si faltan arrays por sesión, usa la cuota de sets (distrib_series).
        - Para 2 sesiones, asigna ['H','M'].
        """
        n = int(getattr(s, "sesiones", 0) or 0)
        if n <= 0:
            return []

        # Si no hay arrays completos, caemos al reparto por sets
        arrays_ok = (
            bool(s.pct_1rm_sesion) and bool(s.rpe_sesion) and bool(s.reps_sesion) and
            len(s.pct_1rm_sesion) >= n and len(s.rpe_sesion) >= n and len(s.reps_sesion) >= n
        )
        if not arrays_ok:
            total = max(1, int(getattr(s, "series", 0) or 0))
            # Por defecto: según cuota de sets
            roles = []
            for j in range(n):
                sets_j = 0
                if s.distrib_series and j < len(s.distrib_series):
                    sets_j = int(s.distrib_series[j])
                share = float(sets_j) / float(total)
                roles.append(self._rol_por_share(share))

            # Para 2 sesiones forzamos ['H','M'] (la más grande es H)
            if n == 2:
                idx = sorted(range(n), key=lambda j: (s.distrib_series[j] if s.distrib_series and j < len(s.distrib_series) else 0), reverse=True)
                res = ["M", "M"]
                res[idx[0]] = "H"
                res[idx[1]] = "M"
                return res
            return roles

        # Arrays presentes: calculamos un score por sesión
        pct  = [float(s.pct_1rm_sesion[j] if j < len(s.pct_1rm_sesion) else s.pct_1rm) for j in range(n)]
        rpe  = [float(s.rpe_sesion[j]     if j < len(s.rpe_sesion)     else s.rpe)     for j in range(n)]
        reps = [int(s.reps_sesion[j]      if j < len(s.reps_sesion)    else s.reps)    for j in range(n)]

        # Score: prima %1RM, luego RPE y penaliza más reps
        scores = [pct[j] * 1000.0 + rpe[j] * 10.0 - reps[j] for j in range(n)]
        orden = sorted(range(n), key=lambda j: scores[j], reverse=True)

        # Asignación de roles según número de sesiones
        role_map = {}
        if n >= 3:
            role_map[orden[0]] = "H"
            role_map[orden[1]] = "M"
            # el resto (normalmente 1) como L
            for j in orden[2:]:
                role_map[j] = "L"
        elif n == 2:
            role_map[orden[0]] = "H"
            role_map[orden[1]] = "M"
        else:  # n == 1
            role_map[orden[0]] = "H"

        return [role_map.get(j, "M") for j in range(n)]

    def _pct_semana_ponderado(self, s) -> float:
        """
        %1RM semanal ponderado por sets de cada sesión.
        Si no hay datos por sesión, usa s.pct_1rm como respaldo.
        """
        total_sets = 0
        pct_semana: float

        distrib = getattr(s, "distrib_series", None)
        pct_ses = getattr(s, "pct_1rm_sesion", None)

        tiene_arrays = bool(distrib) and bool(pct_ses)

        if tiene_arrays:
            # Normalizamos distrib a enteros
            distrib_int = [int(x) for x in distrib]
            total_sets = sum(distrib_int)
            if total_sets <= 0:
                total_sets = 1  # evita división entre cero

            top = 0.0
            # Seguridad por si hay desajuste de longitudes
            n_sesiones = min(
                getattr(s, "sesiones", len(distrib_int)),
                len(distrib_int),
                len(pct_ses),
            )

            for j in range(n_sesiones):
                sets_j = distrib_int[j]
                pct_j = float(pct_ses[j])
                top += sets_j * pct_j

            pct_semana = top / float(total_sets)
        else:
            # Sin arrays por sesión → usamos el %1RM global de la semana
            pct_semana = float(getattr(s, "pct_1rm", 0.0))

        # Log de diagnóstico (limitado por _dbg)
        idx = getattr(s, "indice", -1)
        sesiones_sem = getattr(s, "sesiones", 0)

        self._dbg(
            "metricas",
            "pct_semana",
            "pct_semana_ponderado → semana=%02d, sesiones=%d, sets=%d, pct=%.2f",
            idx + 1 if idx >= 0 else -1,
            sesiones_sem,
            total_sets,
            pct_semana,
        )

        return pct_semana


    # ─────────────────────────────────────────────────────────────────────────────
# PARCHE: añadir a FitnessGAMixin (copia y pega dentro de la clase)
# ─────────────────────────────────────────────────────────────────────────────

    # ==========================
    # Config por defecto (NSGA)
    # ==========================
    alfa_fp: float = 1.0
    beta_fp: float = 0.15
    gamma_fp: float = 0.8
    delta_fp: float = 0.5
    epsilon_fp: float = 0.2
    b_sd: float = 0.8                  # bono por singles/dobles en f1
    elitismo_frente1: int = 4          # nº de élites del primer frente (reemplazo)
    eps_div: float = 1e-9              # epsilon numérico para divisiones

    # =========================================================================
    # Validación externa (constraint-domination): factible y magnitud de violación
    # =========================================================================
    def _validar_externa(self, plan) -> tuple[bool, float]:
        """
        EvolucionEntrenamiento - _validar_externa: obtiene (es_factible, violacion).
        - Si existe self.validador_externo(plan) → úsalo.
        - Si no, aproximamos: es_factible si pasan hard-guards “ideales” por semana.
          La “violación” suma fallos leves (0 si pasa todo).
        """
        # Preferencia: validador externo del usuario
        if hasattr(self, "validador_externo") and callable(getattr(self, "validador_externo")):
            try:
                ok, viol = self.validador_externo(plan)
                return bool(ok), float(max(0.0, viol))
            except Exception:
                pass

        # Fallback: usar tus chequeos ideales por semana como proxy
        viol = 0.0
        ok_global = True
        for idx, s in enumerate(plan.semanas):
            ok, motivo = self._hard_guard_ideales(s, idx)
            if not ok:
                ok_global = False
                viol += 1.0  # suma simple por cada semana fuera de ideales
        return ok_global, viol

    # ======================================================
    # Utilidades para derivar arrays por sesión (si faltan)
    # ======================================================
    def _get_arrays_semana(self, s):
        """
        Devuelve tuplas alineadas por sesión: (sets[], pct[], rpe_obj[], rpe_real[], reps[])
        - Si no hay arrays completos, genera fallbacks con valores semanales.
        - rpe_real usa s.rpe_real_sesion si existe; si no, rpe_obj.
        """
        n = int(getattr(s, "sesiones", 0) or 0)
        sets = list((s.distrib_series or [])[:n])
        while len(sets) < n:
            sets.append(0)

        pct = list((s.pct_1rm_sesion or [])[:n])
        rpe_obj = list((s.rpe_sesion or [])[:n])
        reps = list((s.reps_sesion or [])[:n])

        while len(pct) < n:     pct.append(float(s.pct_1rm))
        while len(rpe_obj) < n: rpe_obj.append(float(s.rpe))
        while len(reps) < n:    reps.append(int(s.reps))

        # RPE "real": si no existe atributo, asumimos igual al objetivo
        rpe_real = list(getattr(s, "rpe_real_sesion", [])[:n]) if hasattr(s, "rpe_real_sesion") else []
        while len(rpe_real) < n:
            rpe_real.append(float(rpe_obj[len(rpe_real)]))

        return sets, pct, rpe_obj, rpe_real, reps

   
    # ======================================================
    # Score auxiliar para "Factible-Primero" (padres)
    # ======================================================
    def _score_factible_primero(self, met: dict) -> float:
        """
        Escalar auxiliar para ordenar candidatos dentro del frente 0.
        NO vuelve a recombinar objetivos; solo usa met['score_fp'] +
        una penalización grande si no es factible.
        """
        if met is None:
            return -1e9

        # Si por lo que sea no viene score_fp, devolvemos muy malo.
        score = float(met.get("score_fp", -1e9))

        # “factible primero” si algún día añades met["factible"]
        factible = bool(met.get("factible", True))
        if not factible:
            score -= 1e6

        return score




    # ======================================================
    # No-dominación (NSGA-II) y crowding
    # ======================================================
    def _no_dominado(self, a, b) -> bool:
        """
        True si 'a' NO es dominado por 'b' para los objetivos ya alineados a maximizar.
        a y b son tuplas como (f1, f2, f3, f4).
        """
        A = a
        B = b
        eps = 1e-12
        # a no es peor en ningún objetivo...
        no_peor = all(A[i] >= B[i] - eps for i in range(len(A)))
        # ...y es mejor en al menos uno
        mejor_en_alguno = any(A[i] > B[i] + eps for i in range(len(A)))
        return no_peor and mejor_en_alguno


    def _ordenar_por_frentes(self, pobl_obj):
        """
        NSGA-II: devuelve lista de frentes, cada frente es lista de índices de 'pobl_obj'.
        pobl_obj: lista de dict con {"plan":…, "met":…, "f":(…)}.
        """
        n = len(pobl_obj)
        domina_a = [set() for _ in range(n)]
        dominado_por = [0] * n
        frentes = []

        f_vec = [x["met"]["f"] for x in pobl_obj]

        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if self._no_dominado(f_vec[p], f_vec[q]):      # p domina a q
                    domina_a[p].add(q)
                elif self._no_dominado(f_vec[q], f_vec[p]):    # q domina a p
                    dominado_por[p] += 1

        frente0 = [i for i in range(n) if dominado_por[i] == 0]
        frentes.append(frente0)

        i = 0
        while frentes[i]:
            sig = []
            for p in frentes[i]:
                for q in domina_a[p]:
                    dominado_por[q] -= 1
                    if dominado_por[q] == 0:
                        sig.append(q)
            i += 1
            frentes.append(sig)
        if not frentes[-1]:
            frentes.pop()
        return frentes

    def _crowding_por_frente(self, frente_idx: list[int], pobl_obj: list[dict]) -> dict[int, float]:
        """
        Distancia de crowding clásica de NSGA-II, usando el vector f de cada individuo,
        ya orientado a maximizar en todas las componentes.
        """
        if not frente_idx:
            return {}

        k = len(frente_idx)
        dist = {i: 0.0 for i in frente_idx}

        # objs = lista de (idx, (f1,f2,...,fm))
        objs = []
        for i in frente_idx:
            f_vec = tuple(pobl_obj[i]["met"]["f"])
            objs.append((i, f_vec))

        m = len(objs[0][1])  # nº de objetivos (4 en nuestro caso)

        for obj_idx in range(m):
            orden = sorted(objs, key=lambda t: t[1][obj_idx])
            dist[orden[0][0]] = float("inf")
            dist[orden[-1][0]] = float("inf")

            vmin = orden[0][1][obj_idx]
            vmax = orden[-1][1][obj_idx]
            rango = max(self.eps_div, (vmax - vmin))

            for j in range(1, k - 1):
                prev_v = orden[j - 1][1][obj_idx]
                next_v = orden[j + 1][1][obj_idx]
                dist[orden[j][0]] += (next_v - prev_v) / rango

        return dist


    # ======================================================
    # Selección de padres: Factible-Primero
    # ======================================================
    def _seleccionar_padres_factible_primero(self, poblacion: list, cuantos: int) -> list:
        """
        EvolucionEntrenamiento - _seleccionar_padres_factible_primero:
        prioriza factibles por score_FP; si faltan, completa con no-factibles por menor violación.
        Empates por mayor S_total.
        """
        self._dbg(
            "seleccion",                      # seccion
            "padres_factible_primero",        # clave
            "seleccionando %d padres (factible-primero).",  # msg
            cuantos
        )
        # Eval métricas y factibilidad
        info = []
        for p in poblacion:
            met = self._calcular_metricas_er5(p)
            ok, viol = self._validar_externa(p)
            score = self._score_factible_primero(met)
            info.append({
                "plan": p,
                "met": met,
                "ok": ok,
                "viol": viol,
                "score_fp": score
            })

        factibles = [x for x in info if x["ok"]]
        no_fact = [x for x in info if not x["ok"]]

        factibles.sort(
            key=lambda x: (
                -(x["score_fp"]),          # 1º: tu escalar de dominio
                -x["met"]["progreso"],     # 2º: más progreso de fuerza
                -x["met"]["S_total"],      # 3º: más estímulo total
            )
        )

        no_fact.sort(
            key=lambda x: (
                x["viol"],                 # menos violación primero
                -x["score_fp"],
                -x["met"]["progreso"],
                -x["met"]["S_total"],
            )
        )


        sel = factibles[:cuantos]
        if len(sel) < cuantos:
            sel.extend(no_fact[:max(0, cuantos - len(sel))])

        padres = [x["plan"] for x in sel]
        return padres

    # ======================================================
    # Reemplazo NSGA-II (elitismo + crowding + torneos binarios)
    # ======================================================
    def _reemplazo_nsga2(self, poblacion_actual: list, descendencia: list, tam: int) -> list:
        """
        EvolucionEntrenamiento - _reemplazo_nsga2: (µ+λ)
        1) Une población y descendencia.
        2) Ordena por frentes de no dominación.
        3) Inserta frentes completos mientras quepan.
        4) Último frente: ordenar por crowding y cortar.
        5) Si queda hueco (raro), completar por torneos binarios (frente > crowding > S_total).
        """
        self._dbg(
            "reemplazo",           # seccion
            "nsga2",               # clave
            "construyendo nueva población por Pareto + crowding."
        )
        combinada = []
        for p in (poblacion_actual + descendencia):
            met = self._calcular_metricas_er5(p)
            combinada.append({"plan": p, "met": met, "f": met["f"]})

        frentes = self._ordenar_por_frentes(combinada)
        nueva = []
        crowding_cache = {}

        for fr in frentes:
            if len(nueva) + len(fr) <= tam:
                nueva.extend([combinada[i]["plan"] for i in fr])
            else:
                # calcular crowding en este frente
                dist = self._crowding_por_frente(fr, combinada)
                crowding_cache.update(dist)
                # ordenar por crowding descendente; empate por mayor S_total
                fr_orden = sorted(fr, key=lambda i: (dist.get(i, 0.0), combinada[i]["met"]["S_total"]), reverse=True)
                espacio = tam - len(nueva)
                nueva.extend([combinada[i]["plan"] for i in fr_orden[:espacio]])
                break

        # Si por algún motivo (edge-case) quedan huecos, usar torneos binarios multi-objetivo
        if len(nueva) < tam:
            restantes = [i for fr in frentes for i in fr if combinada[i]["plan"] not in nueva]
            # preparar candidatos sobrantes
            cand = [combinada[i] for i in restantes]
            # ranking de frentes para torneos
            rank = {}
            for r, fr in enumerate(frentes):
                for i in fr:
                    rank[i] = r
            # prepara crowding si no existe
            for fr in frentes:
                d = self._crowding_por_frente(fr, combinada)
                crowding_cache.update(d)

            import random
            while len(nueva) < tam and cand:
                a, b = random.sample(cand, k=min(2, len(cand)))
                # indices originales
                ia = (poblacion_actual + descendencia).index(a["plan"])  # no perfecto pero suficiente aquí
                ib = (poblacion_actual + descendencia).index(b["plan"])
                ra = min(rank.get(ia, 1e9), 1e9)
                rb = min(rank.get(ib, 1e9), 1e9)
                if ra < rb:
                    ganador = a
                elif rb < ra:
                    ganador = b
                else:
                    ca = crowding_cache.get(ia, 0.0)
                    cb = crowding_cache.get(ib, 0.0)
                    if ca > cb:
                        ganador = a
                    elif cb > ca:
                        ganador = b
                    else:
                        # último desempate: mayor S_total
                        ganador = a if a["met"]["S_total"] >= b["met"]["S_total"] else b
                nueva.append(ganador["plan"])
                cand.remove(ganador)

        return nueva[:tam]

    # ======================================================
    # Mejor individuo "legible" de la generación (para logs)
    # ======================================================
    def _mejor_por_pareto_y_score(self, poblacion: list):
        """
        Elige un "mejor" representativo: top del primer frente con mayor score_FP.
        Útil para imprimir progreso en logs sin forzar un escalar único.
        """
        objs = [{"plan": p, "met": self._calcular_metricas_er5(p)} for p in poblacion]
        frentes = self._ordenar_por_frentes(objs)
        if not frentes:
            return poblacion[0], 0.0
        front0 = frentes[0]
        cand = sorted(front0, key=lambda i: self._score_factible_primero(objs[i]["met"]), reverse=True)
        best = objs[cand[0]]
        return best["plan"], self._score_factible_primero(best["met"])

    # ─────────────────────────────────────────────────────────────────────────
    # REEMPLAZO DEL MÉTODO evolucionar: usa padres Factible-Primero + NSGA-II
    # ─────────────────────────────────────────────────────────────────────────
    def evolucionar(self) -> "Plan12Semanas":
        # Log de inicio del proceso evolutivo
        self._dbg(
            "evolucionar",
            "inicio",
            "iniciando proceso evolutivo con Factible-Primero + NSGA-II.",
            nivel=logging.INFO,
        )

        pobl = self._inicializar_poblacion()

        # pool inicial “mejor” para log
        best, best_score = self._mejor_por_pareto_y_score(pobl)
        self._dbg(
            "evolucionar",
            "pobl_inicial",
            "población inicial evaluada; mejor score_FP=%.3f.",
            best_score,
            nivel=logging.INFO,
        )

        for gen in range(1, self.generaciones + 1):
            # 1) Selección de padres (Factible-Primero)
            padres = self._seleccionar_padres_factible_primero(pobl, self.poblacion)

            # 2) Reproducción (cruce + mutación) → descendencia
            descendencia = []
            import random
            random.shuffle(padres)

            # Emparejar consecutivos; si impar, duplicamos el último
            if len(padres) % 2 == 1:
                padres.append(padres[-1])

            for i in range(0, len(padres), 2):
                p1, p2 = padres[i], padres[i + 1]
                h1, h2 = self._cruzar(p1, p2)
                h1 = self._mutar(h1)
                h2 = self._mutar(h2)
                descendencia.extend([h1, h2])

            # 3) Reemplazo NSGA-II (elitismo + crowding)
            pobl = self._reemplazo_nsga2(pobl, descendencia, self.poblacion)

            # 4) Monitoreo de progreso (Pareto + score_FP sobre frente 0)
            cand, cand_score = self._mejor_por_pareto_y_score(pobl)
            if gen % 5 == 0 or gen == 1 or gen == self.generaciones:
                self._dbg(
                    "evolucionar",
                    "loop",
                    "generación %d/%d, mejor score_FP=%.3f.",
                    gen,
                    self.generaciones,
                    cand_score,
                    nivel=logging.INFO,
                )
            best, best_score = cand, cand_score

        self._dbg(
            "evolucionar",
            "fin",
            "finalizado, devolviendo mejor plan por Pareto/score_FP.",
            nivel=logging.INFO,
        )
        self._ultima_poblacion = list(pobl)

        try:
            objs = [
                {"plan": ind, "met": self._calcular_metricas_er5(ind)}
                for ind in pobl
            ]
            frentes = self._ordenar_por_frentes(objs)
            if frentes:
                # Sólo frente 0 (no dominados)
                self._ultima_pareto = [objs[i] for i in frentes[0]]
            else:
                self._ultima_pareto = objs
        except Exception:
            self._ultima_poblacion = list(pobl)
            self._ultima_pareto = None
        return best



    def imprimir_diagnostico(self,costos, plan: Plan12Semanas):
        self._dbg("imprimir_diagnostico", "desglosando costos del mejor plan.", nivel=logging.INFO)
        fat_musc, fat_snc, progreso = 0.0, 0.0, 0.0
        for idx, s in enumerate(plan.semanas):
            fm, snc = self._fatiga_semana(s)
            fat_musc += fm
            fat_snc += snc
            progreso += self._progreso_fuerza_semana(idx, s)

        pen_trend = self._penalizar_tendencias_y_suavidad(plan.semanas)
        pen_block = self._penalizar_objetivos_por_bloque(plan.semanas, reps_prom_semana)
        pen_intra = self._penalizar_distribucion_intra(plan.semanas)
        pen_micro = self._penalizar_microtendencias_intrablock(plan.semanas)
        pen_taper = self._penalizar_taper_final(plan.semanas, probar_1rm=False)
        pen_smooth = self._suavizado_L2(plan.semanas)
        pen_h_over = self._penalizar_exceso_dias_H(plan.semanas)
        pen_roles = self._penalizar_orden_roles(plan.semanas)
        pen_effsets = self._penalizar_series_efectivas(plan.semanas)
        pen_deload = self._penalizar_deload_intermedio(plan.semanas)
        pen_por_rol = self._penalizar_progresion_por_rol(plan.semanas)

        w_fm, w_snc, w_prog = 1.0, 35.0, 26.0
        costo = (w_fm * fat_musc + w_snc * fat_snc - w_prog * progreso
                 + pen_trend + pen_block + pen_intra + pen_micro + pen_taper + pen_smooth
                 + pen_h_over + pen_roles + pen_effsets + pen_deload + pen_por_rol)

        print(f"EvolucionEntrenamiento - imprimir_diagnostico: fat_musc={fat_musc:.2f} | fat_snc={fat_snc:.2f} | progreso={progreso:.2f}")
        print(f"EvolucionEntrenamiento - imprimir_diagnostico: pen_trend={pen_trend:.2f} | pen_block={pen_block:.2f} | pen_intra={pen_intra:.2f}")
        print(f"EvolucionEntrenamiento - imprimir_diagnostico: pen_micro={pen_micro:.2f} | pen_taper={pen_taper:.2f} | pen_smooth={pen_smooth:.2f}")
        print(f"EvolucionEntrenamiento - imprimir_diagnostico: pen_roles={pen_roles:.2f} | pen_effsets={pen_effsets:.2f} | pen_deload={pen_deload:.2f} | pen_por_rol={pen_por_rol:.2f}")
        print(f"EvolucionEntrenamiento - imprimir_diagnostico: costo_total={costo:.2f}")
        self._dbg(
            "resumen",
            "diagnostico",
            "fat_musc=%.2f | fat_snc=%.2f | progreso=%.2f",
            costos["fat_musc"], costos["fat_snc"], costos["progreso"],
        )
        self._dbg(
            "resumen",
            "diagnostico",
            "pen_trend=%.2f | pen_block=%.2f | pen_intra=%.2f",
            costos["pen_trend"], costos["pen_block"], costos["pen_intra"],
        )
        def mejores_planes(self, k: int = 3, usar_pareto: bool = True):
            """
            Devuelve hasta k planes buenos para compararlos.
            Por defecto usa el frente de Pareto final (si existe),
            ordenados por el score 'factible primero'.
            """
            # 1) Fuente de datos
            src = getattr(self, "_ultima_pareto", None) if usar_pareto else None
            if src is None:
                pobl = getattr(self, "_ultima_poblacion", None)
                if not pobl:
                    return []
                src = [
                    {"plan": ind, "met": self._calcular_metricas_er5(ind)}
                    for ind in pobl
                ]

            # 2) Ordenar por tu scalarizado 'factible-primero'
            def score(x):
                return self._score_factible_primero(x["met"])

            ordenados = sorted(src, key=score, reverse=True)

            # 3) Devolver sólo los planes (sin las métricas, si quieres)
            planes = [x["plan"] for x in ordenados[:k]]
            return planes

        def _penalizar_objetivos_por_bloque(
            self,
            semanas: List[SemanaPlan],
            reps_prom_semana: Optional[List[float]] = None,
        ) -> float:
            """
            Penaliza que cada bloque no respete su objetivo de rango de repeticiones.

            Bloque 1 (W1-4): Hipertrofia → 6–12 reps de media.
            Bloque 2 (W5-8): Fuerza      → 3–6 reps de media (penal fuerte si >8).
            Bloque 3 (W9-12): Peaking    → 1–4 reps de media.

            Si reps_prom_semana es None, se calcula aquí con _get_arrays_semana.
            """
            if len(semanas) != 12:
                return 0.0

            # ------------------------------------------------------------------
            # Si no nos pasaron reps_prom_semana, la calculamos aquí
            # ------------------------------------------------------------------
            if reps_prom_semana is None or len(reps_prom_semana) != 12:
                reps_prom_semana = []
                for s in semanas:
                    sets, pct, rpe_obj, rpe_real, reps = self._get_arrays_semana(s)
                    total_sets = sum(int(x) for x in sets)
                    if total_sets > 0:
                        reps_acum = sum(int(st) * int(r) for st, r in zip(sets, reps))
                        reps_prom = reps_acum / float(total_sets) if reps_acum > 0 else float(getattr(s, "reps", 0.0))
                    else:
                        reps_prom = float(getattr(s, "reps", 0.0))
                    reps_prom_semana.append(reps_prom)

            def penal_rango(rep: float, low: float, high: float,
                            peso_suave: float = 1.0,
                            peso_fuerte: float = 3.0) -> float:
                """
                Penalización suave si se sale poco del rango,
                y fuerte si se sale mucho (>1 rep de margen).
                """
                if low <= rep <= high:
                    return 0.0

                if rep < low:
                    delta = low - rep
                else:
                    delta = rep - high

                if delta <= 1.0:
                    return peso_suave * delta
                # si se sale más de 1 rep, añadimos extra
                return peso_suave + peso_fuerte * (delta - 1.0)

            pen = 0.0

            # --------------------------------------------------
            # Bloque 1 – Hipertrofia (W1–4, índices 0–3)
            # 6–12 reps de media, penalización suave.
            # --------------------------------------------------
            for i in range(0, 4):
                rep = float(reps_prom_semana[i])
                pen += penal_rango(rep, 6.0, 12.0,
                                peso_suave=0.5,
                                peso_fuerte=1.5)

            # --------------------------------------------------
            # Bloque 2 – Fuerza (W5–8, índices 4–7)
            # 3–6 reps de media, penal fuerte fuera de rango.
            # Castigo extra si sigue en 8–12 (hipertrofia 2.0).
            # --------------------------------------------------
            for i in range(4, 8):
                rep = float(reps_prom_semana[i])
                p = penal_rango(rep, 3.0, 6.0,
                                peso_suave=2.0,
                                peso_fuerte=5.0)

                if rep > 8.0:
                    # castigo extra por seguir en hipertrofia
                    p += (rep - 8.0) * 4.0

                pen += p

            # --------------------------------------------------
            # Bloque 3 – Peaking (W9–12, índices 8–11)
            # 1–4 reps de media, rango estrecho.
            # --------------------------------------------------
            for i in range(8, 12):
                rep = float(reps_prom_semana[i])
                pen += penal_rango(rep, 1.0, 4.0,
                                peso_suave=1.5,
                                peso_fuerte=4.0)

            return pen

    def _penalizar_bloques(self, plan) -> float:
        """
        Penaliza ERRORES MACRO de planificación por bloques:

        - Shock de volumen entre bloques (S4→S5, S8→S9) ⇒ evitar colapsos tipo 21→7.
        - Identidad del Bloque 2 (S5–S8): reps medias 2–6, sin hipertrofia pura (10–12 reps).
        - Identidad del Bloque 3 (S9–S11): reps medias 2–5, tendencia NO creciente en reps
          y NO decreciente en intensidad (%1RM).
        - Semana 12 (taper real): volumen 40–70 % de S11, pocas reps (1–3) e intensidad alta.
        """
        semanas = getattr(plan, "semanas", None)
        if not semanas or len(semanas) != 12:
            return 0.0

        pen = 0.0

        # ───────────────── RESUMEN SEMANAL ─────────────────
        resumen = []
        for idx, s in enumerate(semanas):
            sets, pct, rpe_obj, rpe_real, reps = self._get_arrays_semana(s)
            total_sets = max(1, sum(int(x) for x in sets))

            # %1RM ponderado por series
            pct_sem = self._pct_semana_ponderado(s)

            # Reps medias ponderadas por series
            if total_sets > 0 and sets:
                reps_med = sum(int(rep) * int(st) for rep, st in zip(reps, sets)) / float(total_sets)
            else:
                reps_med = float(getattr(s, "reps", 8))

            resumen.append({
                "idx": idx,              # 0-based
                "sets": float(total_sets),
                "pct": float(pct_sem),
                "reps": float(reps_med),
            })

        # ───────────────── 1) SHOCK ENTRE BLOQUES ─────────────────
        # Aquí reforzamos especialmente S4→S5 y S8→S9.
        # No aplicamos esta lógica a S11→S12 (porque ahí S12 es el taper).
        W_SHOCK = 0.8        # peso base sobre (Δsets)^2
        limites = [4, 8]     # índices donde empieza bloque nuevo: 4→(S5), 8→(S9)

        for i in limites:
            prev = resumen[i - 1]   # semana anterior (3→S4, 7→S8)
            cur  = resumen[i]       # semana de inicio de bloque (4→S5, 8→S9)

            sets_prev = max(1.0, prev["sets"])
            sets_cur  = cur["sets"]
            ratio = sets_cur / sets_prev

            # Queremos algo razonable (ej. 70–120 % del volumen de la semana anterior).
            if ratio < 0.7 or ratio > 1.2:
                delta = sets_cur - sets_prev
                pen += (delta * delta) * W_SHOCK  # penalización cuadrática fuerte

        # ───────────────── 2) IDENTIDAD BLOQUE 2 (S5–S8) ─────────────────
        # Bloque 2 = semanas 5–8 ⇒ índices 4–7
        BLOQUE2 = range(4, 8)
        W_B2_OUT    = 5.0   # salir del rango 2–6 reps medias
        W_B2_HIPER  = 2.0   # sets de 10+ reps dentro de B2

        for i in BLOQUE2:
            info = resumen[i]
            reps_med = info["reps"]

            # Rango objetivo de reps medias: 2–6
            if reps_med < 2.0:
                pen += (2.0 - reps_med) * W_B2_OUT
            elif reps_med > 6.0:
                pen += (reps_med - 6.0) * W_B2_OUT

            # Penalizar hipertrofia pura (10+ reps) dentro del bloque de fuerza
            s = semanas[i]
            _, _, _, _, reps_arr = self._get_arrays_semana(s)
            for r in reps_arr:
                r = int(r)
                if r >= 10:
                    pen += (r - 9) * W_B2_HIPER

        # ───────────────── 3) IDENTIDAD BLOQUE 3 (S9–S11) ─────────────────
        # Bloque 3 "peaking" = semanas 9–11 ⇒ índices 8–10
        BLOQUE3 = range(8, 11)
        W_B3_OUT    = 6.0    # salir del rango 2–5 reps medias
        W_B3_NONDEC = 4.0    # reps que SUBEN dentro del bloque (no debería)
        W_B3_INT    = 2.0    # intensidad que BAJA (no debería)

        prev_reps = None
        prev_pct  = None
        for i in BLOQUE3:
            info = resumen[i]
            reps_med = info["reps"]
            pct_sem  = info["pct"]

            # Rango de reps objetivo: 2–5
            if reps_med < 2.0:
                pen += (2.0 - reps_med) * W_B3_OUT
            elif reps_med > 5.0:
                pen += (reps_med - 5.0) * W_B3_OUT

            # Tendencia NO creciente en reps (cada semana igual o MENOS repeticiones)
            if prev_reps is not None and reps_med > prev_reps + 0.25:
                pen += (reps_med - prev_reps) * W_B3_NONDEC

            # Intensidad %1RM NO debería caer dentro del bloque 3
            if prev_pct is not None and pct_sem < prev_pct - 1.0:
                pen += (prev_pct - pct_sem) * W_B3_INT

            prev_reps = reps_med
            prev_pct  = pct_sem

        # ───────────────── 4) SEMANA 12 – TAPER REAL ─────────────────
        info11 = resumen[10]  # semana 11
        info12 = resumen[11]  # semana 12

        sets11, sets12 = info11["sets"], info12["sets"]
        reps12, pct12  = info12["reps"], info12["pct"]

        # Volumen de W12 ≈ 40–70 % de W11 (taper, no colapso brutal ni casi igual)
        ratio_vol = sets12 / max(1.0, sets11)
        if ratio_vol < 0.35 or ratio_vol > 0.75:
            pen += abs(ratio_vol - 0.55) * 15.0

        # Pocas repeticiones en el taper (ideal 1–3, toleramos hasta ~3.5)
        if reps12 > 3.5:
            pen += (reps12 - 3.5) * 10.0

        # Intensidad alta en el taper (ideal ≥ 85 %, penalizamos fuerte por < 82.5)
        if pct12 < 82.5:
            pen += (82.5 - pct12) * 2.5

        return float(pen)

    def _penalizar_microtendencias_intrablock(self, semanas) -> float:
        """
        Penaliza que las reps promedio vayan en sentido contrario al bloque,
        en especial que en el Peaking (B3) vuelvan a subir.
        - B2 (S5–S8): no debería pegar latigazos al alza.
        - B3 (S9–S12): reps bajas y sin escalada dentro del bloque.
        """
        pen = 0.0
        reps_med = []

        # Media de reps por semana (ponderada por sets)
        for s in semanas:
            sets, pct, rpe_obj, rpe_real, reps = self._get_arrays_semana(s)
            total_sets = max(1, sum(int(x) for x in sets))
            if total_sets > 0:
                reps_media = sum(int(st) * int(rp) for st, rp in zip(sets, reps)) / float(total_sets)
            else:
                reps_media = float(getattr(s, "reps", 0) or 0)
            reps_med.append(reps_media)

        # --- Bloque 2 (semanas 5–8, idx 4–7): evitar saltos fuertes hacia arriba ---
        for idx in range(4, min(8, len(reps_med))):
            prev = reps_med[idx - 1]
            cur = reps_med[idx]
            # tolerancia de 0.5 reps
            if cur > prev + 0.5:
                pen += (cur - prev - 0.5) * 20.0

        # --- Bloque 3 (semanas 9–12, idx 8–11): las reps no deberían ir subiendo ---
        for idx in range(9, min(12, len(reps_med))):
            prev = reps_med[idx - 1]
            cur = reps_med[idx]
            # tolerancia pequeña: 0.25 reps
            if cur > prev + 0.25:
                pen += (cur - prev - 0.25) * 40.0

        return pen

    def _penalizar_taper_final(self, semanas, probar_1rm: bool = False) -> float:
        pen = 0.0
        s11, s12 = semanas[10], semanas[11]
        max_pct12 = max(s12.pct_1rm_sesion) if s12.pct_1rm_sesion else s12.pct_1rm

        if not probar_1rm:
            # Intensidad igual que antes
            pen += max(0.0, 90.0 - max_pct12) * 0.8
            pen += max(0.0, max_pct12 - 92.0) * 0.8

            # NUEVO: objetivo ~65 % de las series de W11 (más volumen)
            objetivo = max(2, int(round(s11.series * 0.65)))

            diff_vol = abs(s12.series - objetivo)
            # NUEVO: tolera ±2 series "gratis", y pesa menos
            if diff_vol > 2:
                pen += 0.6 * (diff_vol - 2)

            # NUEVO: permite hasta 4 series por sesión sin multa fuerte
            if s12.distrib_series and s12.sesiones in (2, 3):
                exceso = sum(max(0, x - 4) for x in s12.distrib_series)  # antes x - 3
                pen += 0.6 * exceso

        else:
            # rama probar_1rm la puedes dejar igual o suavizar también
            if s12.series > 8:
                pen += (s12.series - 8) * 0.8
            if s12.reps > 2:
                pen += (s12.reps - 2) * 0.5

        return pen


    def _calcular_metricas_er5(self, plan: Plan12Semanas) -> dict:
        """
        EvolucionEntrenamiento - _calcular_metricas_er5:

        - Sincroniza %1RM–reps–RPE en todo el plan.
        - Calcula:
            S_total (estímulo tipo ER5),
            fatiga muscular total,
            fatiga SNC total,
            progreso de fuerza,
            tonelaje total,
            series y reps promedio por semana.
        - Define vector multiobjetivo f orientado a MAXIMIZAR:
            f1 = S_total                          (↑ estímulo útil)
            f2 = -exceso_fatiga_muscular          (↑ acepta fatiga hasta un techo)
            f3 = -exceso_fatiga_snc               (↑ idem pero para SNC)
            f4 = -Shock_intersemanal              (↓ saltos semana a semana)
            f5 = score_taper (0–10)               (taper correcto en W12)
            f6 = -(pen_soft + 4·pen_hard)         (↓ violaciones de estructura)
            f7 = progreso_total                    (↑ progreso de fuerza estimado)

        - pen_soft incluye (sin shock):
            pen_trend   → tendencias/suavidad global
            pen_bloques → respeto estructura macro por bloques
            pen_block   → reps/objetivo por bloque
            pen_roles   → separación H/M/L
            pen_micro   → microtendencias intra-bloque
            pen_taper   → coherencia del taper

        - pen_hard incluye cosas “peligrosas” como singles en M/L, etc.
        """

        # --- Constantes de referencia / normalización para el score escalar ---
        ER5_NORM     = 300.0
        FAT_M_NORM   = 700.0   # rango “normal” de fatiga muscular
        FAT_SNC_NORM = 80.0    # rango “normal” de fatiga SNC
        SHOCK_NORM   = 10000.0
        TAPER_NORM   = 10.0
        PROG_NORM    = 50.0    # escala razonable para progreso_total (ajustable)

        # 0) Coherencia fisiológica en TODO el plan
        self._sincronizar_plan_pct_vs_reps_rpe(plan)

        self._dbg(
            "_calcular_metricas_er5",
            f"inicio: {len(plan.semanas)} semanas en el plan."
        )

        series_por_semana: list[float] = []
        reps_prom_semana: list[float]  = []

        S_total        = 0.0   # estímulo tipo ER5
        fat_musc_total = 0.0
        fat_snc_total  = 0.0
        progreso_total = 0.0
        ton_total      = 0.0

        # -------- 1) Recorrido semana a semana --------
        for idx, s in enumerate(plan.semanas):
            sets, pct, rpe_obj, rpe_real, reps = self._get_arrays_semana(s)
            total_sets = sum(int(x) for x in sets)  # sin max(1, ...)

            stim_sem   = 0.0
            ton_sem    = 0.0
            reps_acum  = 0.0

            for st, p, r, rep in zip(sets, pct, rpe_obj, reps):
                st_int  = int(st)
                rep_int = int(rep)
                if st_int <= 0:
                    continue

                # ER5: reps por encima de 5 (no negativas)
                er5 = max(0, rep_int - 5)
                intensidad_rel = float(p) / 100.0

                stim_sem  += st_int * er5 * (intensidad_rel ** 1.2)
                ton_sem   += st_int * rep_int * intensidad_rel
                reps_acum += st_int * rep_int

            if total_sets > 0:
                reps_prom = reps_acum / float(total_sets)
            else:
                reps_prom = float(s.reps)

            S_total   += stim_sem
            ton_total += ton_sem
            series_por_semana.append(float(total_sets))
            reps_prom_semana.append(reps_prom)

            fm, snc = self._fatiga_semana(s)
            fat_musc_total += fm
            fat_snc_total  += snc
            progreso_total += self._progreso_fuerza_semana(idx, s)

            # Log por semana (útil para depuración fina)
            self._dbg(
                "_calcular_metricas_er5",
                (
                    f"semana {idx+1:02d}: sets={total_sets}, stim_sem={stim_sem:.2f}, "
                    f"ton_sem={ton_sem:.2f}, reps_prom={reps_prom:.2f}, "
                    f"fm={fm:.2f}, snc={snc:.2f}, prog_sem={self._progreso_fuerza_semana(idx, s):.3f}"
                )
            )

        # -------- 2) Penalizaciones estructurales --------
        pen_trend   = self._penalizar_tendencias_y_suavidad(plan.semanas)
        pen_bloques = self._penalizar_bloques(plan)
        pen_block   = self._penalizar_objetivos_por_bloque(plan.semanas)
        pen_roles   = self._penalizar_orden_roles(plan.semanas)
        pen_singles = self._penalizar_singles_pct_ML(plan.semanas)
        pen_shock   = self._penalizar_shocks(series_por_semana, reps_prom_semana)
        pen_micro   = self._penalizar_microtendencias_intrablock(plan.semanas)
        pen_taper   = self._penalizar_taper_final(plan.semanas, probar_1rm=False)

        # Shock NO entra en pen_soft: es objetivo explícito (f4)
        pen_soft = (
            pen_trend +
            pen_bloques +
            pen_block +
            pen_roles +
            pen_micro +
            pen_taper
        )

        # Suavizamos la magnitud de las penalizaciones suaves
        pen_soft *= 0.2

        # Hard-guards: cosas que directamente no quieres (singles M/L, etc.)
        pen_hard = pen_singles

        # ---------- Shock como objetivo explícito ----------
        shock_cost  = pen_shock
        score_taper = self._bonus_taper(plan)  # ~0..10

        # ---------- 3) Exceso de fatiga (queremos “zona amarilla”, no cero) ----------
        exceso_fm  = max(0.0, fat_musc_total - FAT_M_NORM)
        exceso_snc = max(0.0, fat_snc_total  - FAT_SNC_NORM)

        # ---------- 4) Objetivos crudos (multiobjetivo) ----------
        f1 = S_total               # ↑ estímulo útil
        f2 = -exceso_fm            # 0 si fatiga ≤ FAT_M_NORM, negativo si te pasas
        f3 = -exceso_snc           # idem para SNC
        f4 = -shock_cost           # menos shocks → mejor
        f5 = score_taper           # 0–10, más es mejor
        f6 = -(pen_soft + 4.0 * pen_hard)  # menos violaciones → mejor
        f7 = progreso_total        # ↑ progreso fuerza

        f_vec = (f1, f2, f3, f4, f5, f6, f7)

        # ---------- 5) Score escalar AUXILIAR (para selección Factible-Primero) ----------
        f1n = f1 / ER5_NORM
        # f2 y f3 ya son negativos o 0, los normalizamos por la misma referencia:
        f2n = f2 / FAT_M_NORM if FAT_M_NORM > 0 else 0.0
        f3n = f3 / FAT_SNC_NORM if FAT_SNC_NORM > 0 else 0.0
        f4n = f4 / SHOCK_NORM
        f5n = f5 / TAPER_NORM
        f7n = f7 / PROG_NORM if PROG_NORM > 0 else 0.0

        # Aquí priorizamos:
        # - estímulo (f1n) y progreso (f7n),
        # - shock/taper con peso moderado,
        # - fatigas solo como freno cuando se exceden (f2n,f3n son <=0),
        # - penalizaciones suaves/duras como “freno de emergencia”.
        score_fp = (
            1.2 * f1n +        # estímulo
            1.8 * f7n +        # progreso de fuerza (muy importante)
            0.4 * f4n +        # suavidad / pocos shocks
            0.5 * f5n +        # buen taper
            0.3 * f2n +        # exceso de fatiga muscular (<=0, resta cuando hay mucho exceso)
            0.3 * f3n -        # exceso de fatiga SNC
            0.003 * pen_soft - # estructura rara → resta un poco
            0.03  * pen_hard   # violaciones duras → resta más
        )

        # ---------- Logs resumen ----------
        self._dbg(
            "_calcular_metricas_er5",
            (
                "penalizaciones: trend={:.2f}, bloques={:.2f}, block={:.2f}, "
                "roles={:.2f}, singles={:.2f}, shock={:.2f}, micro={:.2f}, "
                "taper={:.2f}, pen_soft={:.2f}, pen_hard={:.2f}"
            ).format(
                pen_trend, pen_bloques, pen_block, pen_roles,
                pen_singles, pen_shock, pen_micro, pen_taper,
                pen_soft, pen_hard
            )
        )

        self._dbg(
            "_calcular_metricas_er5",
            (
                "resumen métricas: S_total={:.2f}, ton_total={:.2f}, "
                "fat_musc={:.2f}, fat_snc={:.2f}, progreso={:.2f}, "
                "shock_cost={:.2f}, score_taper={:.2f}, "
                "f=({}, {}, {}, {}, {}, {}, {}), score_fp={:.3f}"
            ).format(
                S_total, ton_total, fat_musc_total, fat_snc_total,
                progreso_total, shock_cost, score_taper,
                f1, f2, f3, f4, f5, f6, f7, score_fp
            ),
            nivel=logging.INFO,
        )

        self._dbg(
            "_calcular_metricas_er5",
            "series_por_semana=" + str(series_por_semana) +
            ", reps_prom_semana=" + str(reps_prom_semana)
        )

        met = {
            "f":                 f_vec,
            "score_fp":          score_fp,
            "S_total":           S_total,
            "ton_total":         ton_total,
            "fat_musc":          fat_musc_total,
            "fat_snc":           fat_snc_total,
            "progreso":          progreso_total,
            "series_por_semana": series_por_semana,
            "reps_prom_semana":  reps_prom_semana,
            "shock_cost":        shock_cost,
            "pen_soft":          pen_soft,
            "pen_hard":          pen_hard,
        }

        self._metricas_last = met
        return met



    def _penalizar_shocks(self, series_por_semana: list[float],
                          reps_prom_semana: list[float]) -> float:
        """
        Penaliza cambios BRUSCOS intersemanales en:
        - Volumen (series totales/semana)
        - Reps promedio/semana

        Regla base (con tolerancias):
            ΔSeries_ef  = max(0, |ΔSeries| - TOL_SERIES)
            ΔReps_ef    = max(0, |ΔReps|   - TOL_REPS)

            penal += (ΔSeries_ef)^2 * PEN_SERIES
            penal += (ΔReps_ef)^2   * PEN_REPS

        Con esto:
        - Cambios pequeños semana a semana NO se castigan.
        - Solo los “latigazos” grandes suman mucha penalización.
        """
        if len(series_por_semana) <= 1:
            return 0.0

        # Pesos moderados: el shock pesa, pero no destruye el fitness.
        # (Si quieres más peso, puedes multiplicar ambos por 2.)
        PEN_SERIES = 1.0   # peso para shocks de volumen
        PEN_REPS   = 5.0   # peso para shocks de reps

        # Tolerancias: hasta aquí consideramos “cambio normal”.
        TOL_SERIES = 2.0   # hasta ±2 series de diferencia no penaliza
        TOL_REPS   = 0.75  # hasta ±0.75 reps de diferencia no penaliza

        pen = 0.0
        for i in range(1, len(series_por_semana)):
            prev_s = float(series_por_semana[i - 1])
            curr_s = float(series_por_semana[i])
            prev_r = float(reps_prom_semana[i - 1])
            curr_r = float(reps_prom_semana[i])

            diff_s = curr_s - prev_s
            diff_r = curr_r - prev_r

            # Aplicar tolerancias: solo penalizamos lo que excede el umbral.
            delta_s_eff = max(0.0, abs(diff_s) - TOL_SERIES)
            delta_r_eff = max(0.0, abs(diff_r) - TOL_REPS)

            if delta_s_eff > 0.0:
                pen += (delta_s_eff ** 2) * PEN_SERIES
            if delta_r_eff > 0.0:
                pen += (delta_r_eff ** 2) * PEN_REPS

            # Log solo cuando el cambio bruto es llamativo (para debug visual).
            if abs(diff_r) > 1.5 or abs(diff_s) > 4.0:
                self._dbg(
                    "penal",
                    "shocks",
                    (
                        "ALERTA SHOCK: Sem %02d→%02d | "
                        "ΔReps=%.1f (Δef=%.2f, Pen+≈%.1f) | "
                        "ΔSeries=%.1f (Δef=%.2f, Pen+≈%.1f)"
                    ),
                    i,
                    i + 1,
                    diff_r,
                    delta_r_eff,
                    (delta_r_eff ** 2) * PEN_REPS,
                    diff_s,
                    delta_s_eff,
                    (delta_s_eff ** 2) * PEN_SERIES,
                )

        return pen

    
    def exportar_dict(self, plan: Plan12Semanas) -> dict:
        self._dbg("exportar_dict", "convirtiendo plan a diccionario (para JSON/Firestore si se desea).")
        return {
            "pattern": plan.patron,
            "weeks": [
                {
                    "week": i + 1,
                    "sessions": s.sesiones,
                    "weeklySets": s.series,
                    "setsBySession": s.distrib_series,
                    "intensityPct1RM": round(s.pct_1rm, 1),
                    "rpe": round(s.rpe, 1),
                    "reps": s.reps,
                    "bySession": {
                        "pct1RM": [round(x, 1) for x in (s.pct_1rm_sesion or [])],
                        "rpe": [round(x, 1) for x in (s.rpe_sesion or [])],
                        "reps": list(s.reps_sesion or []),
                    },
                }
                for i, s in enumerate(plan.semanas)
            ],
            "meta": {
                "nivelUsuario": self.nivel_usuario,
                "modoRPE": self.modo_rpe,
                "deloadIntermedio": self.usar_deload_intermedio,
            },
        }


