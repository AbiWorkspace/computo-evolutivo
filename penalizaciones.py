# penalizaciones.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List
import math

# Mixin con TODAS las penalizaciones (y helpers que solo usan ellas).
# Se apoya en métodos/atributos que debe tener la clase principal:
# - self.NIVELES, self.STIM_MIN, self.usar_deload_intermedio
# - self._rir_objetivo, self._epley_pct_a_reps
# - self._roles_por_intensidad (para snapshots)
# - (opcional) self._pct_semana_ponderado si la usas en diagnóstico

class PenalizacionesMixin:
    # -----------------------
    # Helpers usados por penalizaciones
    # -----------------------
    def _avg(self, xs):
        return sum(xs) / len(xs) if xs else 0.0

    def _rep_semana_ponderada_mediana(self, s) -> float:
        """Mediana ponderada por sets; fallback a s.reps si no hay arrays por sesión."""
        if not (s.distrib_series and s.reps_sesion and len(s.distrib_series) == len(s.reps_sesion)):
            return float(s.reps)
        pares = []
        for sets, r in zip(s.distrib_series, s.reps_sesion):
            pares += [int(r)] * int(max(1, sets))
        pares.sort()
        m = len(pares) // 2
        return float(pares[m]) if len(pares) % 2 == 1 else 0.5 * (pares[m - 1] + pares[m])

    def _roles_snapshot(self, s) -> dict:
        """Devuelve dict con información de la sesión H/M/L (índice, %1RM, RPE, sets, reps)."""
        snap = {"H": None, "M": None, "L": None}
        if not (s.distrib_series and s.series > 0 and
                s.pct_1rm_sesion and s.rpe_sesion and s.reps_sesion and
                len(s.distrib_series) == s.sesiones == len(s.pct_1rm_sesion) == len(s.rpe_sesion) == len(s.reps_sesion)):
            return snap

        roles = self._roles_por_intensidad(s)

        def pack(j):
            return {
                "idx": j,
                "pct": float(s.pct_1rm_sesion[j]),
                "rpe": float(s.rpe_sesion[j]),
                "sets": int(s.distrib_series[j]),
                "reps": int(s.reps_sesion[j]),
            }

        if "H" in roles:
            j = roles.index("H")
            snap["H"] = pack(j)
        if "M" in roles:
            idxs_m = [i for i, r in enumerate(roles) if r == "M"]
            j = max(idxs_m, key=lambda t: s.pct_1rm_sesion[t])
            snap["M"] = pack(j)
        if "L" in roles:
            j = roles.index("L")
            snap["L"] = pack(j)

        return snap

    # -----------------------
    # Penalizaciones
    # -----------------------
    def _penalizar_tendencia_rpe(self, semanas) -> float:
        pen = 0.0
        for k in range(11):
            r_k = self._rpe_medio_semana(semanas[k])
            r_k1 = self._rpe_medio_semana(semanas[k + 1])
            es_deload = self.usar_deload_intermedio and (k in (2, 6))  # W3->W4 y W7->W8
            if es_deload:
                if r_k1 < r_k - 0.75:
                    pen += (r_k - 0.75 - r_k1) * 6.0
            else:
                if r_k1 + 1e-6 < r_k:
                    pen += (r_k - r_k1) * 10.0

        for s in semanas:
            if s.sesiones == 3 and s.rpe_sesion and len(s.rpe_sesion) == 3:
                order = sorted(range(3), key=lambda j: s.distrib_series[j], reverse=True)
                iH, iM, iL = order
                rH, rM, rL = s.rpe_sesion[iH], s.rpe_sesion[iM], s.rpe_sesion[iL]
                if rH + 1e-6 < rM:
                    pen += (rM - rH) * 8.0
                if rM + 1e-6 < rL:
                    pen += (rL - rM) * 8.0
        return pen

    def _penalizar_coherencia_intersemanal(self, semanas) -> float:
        if not semanas or len(semanas) < 2:
            return 0.0
        pen = 0.0
        low_H, high_H = self.NIVELES.get(self.nivel_usuario, self.NIVELES["intermedio"])
        low_M, high_M = low_H * 0.7, high_H * 0.7
        low_L, high_L = low_H * 0.5, high_H * 0.5
        MAX_UP_RPE, MAX_DOWN_RPE, MAX_DOWN_DELOAD = 1.0, 0.25, 0.75

        for k in range(len(semanas) - 1):
            a, b = semanas[k], semanas[k + 1]
            sa, sb = self._roles_snapshot(a), self._roles_snapshot(b)
            es_deload = self.usar_deload_intermedio and (k in (2, 6))

            def penalizar_rol(rol, lo_jump, hi_jump):
                nonlocal pen
                ra, rb = sa.get(rol), sb.get(rol)
                if ra is None or rb is None:
                    return
                # %1RM
                delta_pct = (rb["pct"] - ra["pct"]) / 100.0
                mismo_bloque = (k // 4) == ((k + 1) // 4)
                if mismo_bloque and not es_deload and delta_pct < -1e-4:
                    pen += 12.0 * abs(delta_pct)
                if delta_pct > 0.0:
                    if delta_pct < lo_jump:
                        pen += (lo_jump - delta_pct) * 18.0
                    if delta_pct > hi_jump:
                        pen += (delta_pct - hi_jump) * 26.0
                # RPE
                d_rpe = rb["rpe"] - ra["rpe"]
                if es_deload:
                    if d_rpe < -MAX_DOWN_DELOAD:
                        pen += (abs(d_rpe) - MAX_DOWN_DELOAD) * 8.0
                else:
                    if d_rpe < -MAX_DOWN_RPE:
                        pen += (abs(d_rpe) - MAX_DOWN_RPE) * 10.0
                    if d_rpe > MAX_UP_RPE:
                        pen += (d_rpe - MAX_UP_RPE) * 6.0
                # Sets
                if ra["sets"] > 0:
                    var_sets = abs(rb["sets"] - ra["sets"]) / max(1.0, ra["sets"])
                    if var_sets > 0.40:
                        pen += (var_sets - 0.40) * 10.0
                # Estabilidad de posición
                if a.sesiones == b.sesiones and ra["idx"] is not None and rb["idx"] is not None:
                    shift = abs(rb["idx"] - ra["idx"])
                    if shift > 1:
                        pen += (shift - 1) * 2.5

            penalizar_rol("H", low_H, high_H)
            penalizar_rol("M", low_M, high_M)
            penalizar_rol("L", low_L, high_L)
        return pen

    def _penalizar_exceso_dias_H(self, semanas) -> float:
        pen = 0.0
        for s in semanas:
            if not s.distrib_series or s.series <= 0:
                continue
            shares = [(sets / s.series) for sets in s.distrib_series]
            dias_H = sum(1 for sh in shares if sh >= 0.45)
            if s.sesiones == 2 and dias_H > 1:
                pen += 2.0 * (dias_H - 1)
            if s.sesiones == 3 and dias_H > 2:
                pen += 1.5 * (dias_H - 2)
        return pen

    def _penalizar_series_efectivas(self, semanas) -> float:
        pen = 0.0
        for idx, s in enumerate(semanas):
            block = 1 if idx < 4 else (2 if idx < 8 else 3)
            lo, hi = self.SERIES_EFECTIVAS_BLOQUE[block]
            if s.distrib_series and s.rpe_sesion:
                eff = 0.0
                for sets, rpe in zip(s.distrib_series, s.rpe_sesion):
                    if rpe >= 7.0:
                        eff += sets
                    elif 6.0 <= rpe < 7.0:
                        eff += 0.5 * sets
                if eff < lo:
                    pen += (lo - eff) * 0.7
                if eff > hi:
                    pen += (eff - hi) * 0.7
        return pen

    def _penalizar_deload_intermedio(self, semanas) -> float:
        if not self.usar_deload_intermedio:
            return 0.0
        pen = 0.0
        for w in (3, 7):  # semanas 4 y 8
            prev = semanas[w - 1]
            cur = semanas[w]
            target_min = max(1, int(round(prev.series * 0.65)))
            target_max = int(round(prev.series * 0.85))
            if not (target_min <= cur.series <= target_max):
                pen += 0.8 * abs(cur.series - min(max(cur.series, target_min), target_max))
            if not (prev.pct_1rm - 10.0 <= cur.pct_1rm <= prev.pct_1rm - 5.0):
                dif = 0.0
                if cur.pct_1rm > prev.pct_1rm - 5.0:
                    dif = cur.pct_1rm - (prev.pct_1rm - 5.0)
                elif cur.pct_1rm < prev.pct_1rm - 10.0:
                    dif = (prev.pct_1rm - 10.0) - cur.pct_1rm
                pen += 0.6 * dif
        return pen

    def _penalizar_progresion_por_rol(self, semanas) -> float:
        pen = 0.0
        low, high = self.NIVELES.get(self.nivel_usuario, self.NIVELES["intermedio"])
        for start in (0, 4, 8):
            end = start + 3
            for k in range(start, start + 3):
                a, b = semanas[k], semanas[k + 1]

                def heavy_pct(sem) -> float:
                    if not sem.distrib_series or not sem.pct_1rm_sesion:
                        return sem.pct_1rm
                    j = max(range(len(sem.distrib_series)), key=lambda t: sem.distrib_series[t])
                    return sem.pct_1rm_sesion[j] if j < len(sem.pct_1rm_sesion) else sem.pct_1rm

                pct_a = heavy_pct(a)
                pct_b = heavy_pct(b)
                delta = (pct_b - pct_a) / 100.0

                if (k + 1) == end and self.usar_deload_intermedio:
                    continue

                if delta < -0.001:
                    pen += 4.0 * abs(delta)

                lo, hi = (low, high)
                if delta > 0:
                    if delta < lo:
                        pen += (lo - delta) * 20.0
                    if delta > hi:
                        pen += (delta - hi) * 30.0
        return pen

    def _penalizar_microtendencias_intrablock(self, semanas) -> float:
        """Micro-rampa por bloque."""
        pen = 0.0
        b1 = semanas[0:4]
        b2 = semanas[4:8]
        b3 = semanas[8:12]

        def tendencia(vals, delta_min):
            d = vals[-1] - vals[0]
            return max(0.0, delta_min - d)

        pen += 1.0 * tendencia([s.pct_1rm for s in b1], 1.0)
        pen += 0.8 * tendencia([-s.reps for s in b1], 1.0)

        pen += 1.2 * tendencia([s.pct_1rm for s in b2], 3.0)
        target_b2 = [10, 8, 7, 6]
        for (s, t) in zip(b2, target_b2):
            pen += max(0.0, abs(s.reps - t) - 2) * 0.3

        target_b3 = [(3, 5), (2, 3), (1, 2), (1, 2)]
        for s, (lo, hi) in zip(b3, target_b3):
            if s.reps < lo:
                pen += (lo - s.reps) * 0.6
            if s.reps > hi:
                pen += (s.reps - hi) * 0.6

        return pen

    def _penalizar_objetivos_por_bloque(self, semanas) -> float:
        pen = 0.0

        # -------------------------------
        #  A) LÓGICA ORIGINAL POR BLOQUES
        # -------------------------------
        for cfg in self.BLOQUES.values():
            idxs = cfg["idx"]
            ss = [semanas[i] for i in idxs]

            avg_series = self._avg([x.series for x in ss])
            avg_pct    = self._avg([x.pct_1rm for x in ss])
            avg_rpe    = self._avg([x.rpe for x in ss])
            avg_reps   = self._avg([self._rep_semana_ponderada_mediana(x) for x in ss])

            def rango_pen(v, r):
                lo, hi = r
                if v < lo:
                    return (lo - v)
                if v > hi:
                    return (v - hi)
                return 0.0

            # Penalización por PROMEDIOS de bloque
            pen += 8.0 * rango_pen(avg_series, cfg["series"])
            pen += 6.0 * rango_pen(avg_pct,    cfg["pct"])
            pen += 4.0 * rango_pen(avg_rpe,    cfg["rpe"])
            pen += 4.0 * rango_pen(avg_reps,   cfg["reps"])

            # Penalización por SEMANA dentro del bloque
            for s in ss:
                reps_mediana = self._rep_semana_ponderada_mediana(s)
                pen += 0.8 * rango_pen(s.series,        cfg["series"])
                pen += 1.0 * rango_pen(s.pct_1rm,       cfg["pct"])
                pen += 0.8 * rango_pen(reps_mediana,    cfg["reps"])
                pen += 0.5 * rango_pen(reps_mediana,    cfg["reps"])

        # -------------------------------------------------------
        #  B) AJUSTE EXTRA SOLO PARA BLOQUE 2 (FUERZA 2–6 REPS)
        # -------------------------------------------------------
        # Queremos obligar un poco más a que el bloque 2 NO sea
        # hipertrofia 2.0: evitar medias de 7–8 repeticiones y
        # caída de intensidad al final del bloque (W7–8).
        # self.BLOQUES[2]["idx"] debe ser algo como [4,5,6,7]
        # (índices 0-based -> semanas 5,6,7,8).
        try:
            idxs_b2 = self.BLOQUES[2]["idx"]
        except Exception:
            # Si por lo que sea no existe el bloque 2, nos salimos.
            return pen

        for pos, semana_idx in enumerate(idxs_b2, start=1):
            if semana_idx < 0 or semana_idx >= len(semanas):
                continue
            s = semanas[semana_idx]

            # Mediana ponderada por sets (más realista que s.reps a secas)
            reps_med = self._rep_semana_ponderada_mediana(s)
            pct_med  = float(getattr(s, "pct_1rm", 0.0))

            # 1) Bloque 2 NO debe tener reps medias de hipertrofia.
            #    Objetivo ≈ 3–6 reps. Penalizamos si se pasa de 6.5.
            if reps_med > 6.5:
                # Peso suave (5.0); si quieres que sea más "letal", sube a 8–10.
                pen += (reps_med - 6.5) * 5.0

            # 2) En la parte final del bloque (posición 3 y 4 dentro del bloque,
            #    que corresponden a semanas 7 y 8 globales) NO queremos que
            #    se caiga la intensidad por debajo de ~78%.
            if pos in (3, 4) and pct_med < 78.0:
                pen += (78.0 - pct_med) * 3.0

        return pen


    def _penalizar_distribucion_intra(self, semanas) -> float:
        pen = 0.0
        for s in semanas:
            if not s.distrib_series or s.series <= 0:
                continue
            rel = [x / s.series for x in s.distrib_series]
            share_max = max(rel)
            if share_max > 0.70:
                pen += (share_max - 0.70) * 10.0
            if s.sesiones == 3:
                targets = [[1 / 3, 1 / 3, 1 / 3], [0.50, 0.33, 0.17]]
            else:
                targets = [[0.5, 0.5], [0.6, 0.4]]
            rel_sorted = sorted(rel, reverse=True)
            best_dist = min(sum(abs(a - b) for a, b in zip(rel_sorted, sorted(t, reverse=True))) for t in targets)
            pen += 1.5 * best_dist
        return pen

    def _penalizar_bloques(self, plan) -> float:
        """
        Penaliza que cada bloque no respete su identidad:

        - Bloque 1 (Hipertrofia): ya lo estás haciendo bastante bien; aquí casi no tocamos nada.
        - Bloque 2 (Fuerza): reps medias 2–6 por semana.
        - Bloque 3 (Peaking): reps medias 2–5 y que NO suban dentro del bloque.
        - Semana 12 (Taper real): pocas reps (≤3) y alta intensidad (>= 80% 1RM).

        Devuelve un escalar de penalización (>= 0). Cuanto mayor, peor.
        """
        semanas = plan.semanas
        if len(semanas) != 12:
            # Por si cambias la duración del macro en algún experimento
            return 0.0

        # ------------------------------------------------------------------
        # 1) Sacar reps medias y %1RM medios por semana (12 valores cada uno)
        # ------------------------------------------------------------------

        reps_prom_semana = []
        pct_prom_semana = []

        # --- INICIO DE CORRECCIÓN ---
        # El código original no extraía los datos reales (usaba placeholders).
        # Ahora calculamos las medias ponderadas por series para cada semana.
        
        # Asumimos que 'self' tiene acceso a '_get_arrays_semana' 
        # (definido en FitnessGAMixin)
        if not hasattr(self, "_get_arrays_semana"):
             # Fallback muy básico si _get_arrays_semana no está
             self._dbg("_penalizar_bloques", "ADVERTENCIA: _get_arrays_semana no encontrado. Usando fallback simple.", 20)
             for semana in semanas:
                 reps_prom_semana.append(float(getattr(semana, "reps", 0)))
                 pct_prom_semana.append(float(getattr(semana, "pct_1rm", 0)))
        else:
            # Lógica correcta:
            self._dbg("_penalizar_bloques", "Calculando métricas ponderadas reales (reps/pct).")
            for semana in semanas:
                try:
                    sets, pct, _, _, reps = self._get_arrays_semana(semana)
                except Exception as e:
                    self._dbg("_penalizar_bloques", f"Error en _get_arrays_semana: {e}. Usando fallback para esta semana.", 30)
                    sets = getattr(semana, "distrib_series", [])
                    pct = getattr(semana, "pct_1rm_sesion", [])
                    reps = getattr(semana, "reps_sesion", [])

                total_sets = max(1.0, float(sum(int(x) for x in sets)))

                if total_sets > 1.0 and sets and pct and reps:
                    # Media ponderada por series
                    reps_med = sum(int(st) * int(rp) for st, rp in zip(sets, reps)) / total_sets
                    pct_med = sum(int(st) * float(p) for st, p in zip(sets, pct)) / total_sets
                else:
                    # Fallback si no hay desglose por sesión
                    reps_med = float(getattr(semana, "reps", 0))
                    pct_med = float(getattr(semana, "pct_1rm", 0))

                reps_prom_semana.append(reps_med)
                pct_prom_semana.append(pct_med)
        
        # --- FIN DE CORRECCIÓN ---


        # ------------------------------------------------------------------
        # 2) Penalizaciones por bloque
        # ------------------------------------------------------------------

        pen = 0.0

        # --- INICIO DE CORRECCIÓN 2: AUMENTAR PESOS ---
        # Aumentamos los pesos 10x para que estas reglas sean "letales"
        # y fuercen al algoritmo a respetar los bloques.
        PESO_FUERZA_REPS = 2000.0  # (Original 200.0)
        PESO_PEAK_REPS   = 2000.0  # (Original 200.0)
        PESO_PEAK_TEND   = 1500.0  # (Original 150.0)
        PESO_TAPER_INT   = 4000.0  # (Original 400.0)
        PESO_TAPER_REPS  = 4000.0  # (Original 400.0)
        # --- FIN DE CORRECCIÓN 2 ---


        # Índices 0-based
        # B1: S1–S4 -> [0,1,2,3]  (no tocamos casi nada aquí)
        # B2: S5–S8 -> [4,5,6,7]
        # B3: S9–S11 -> [8,9,10]
        # Taper real: S12 -> índice 11

        # ----- Bloque 2: Fuerza (reps medias 2–6) -----
        for idx in range(4, 8):  # semanas 5 a 8
            r = reps_prom_semana[idx]
            if r == 0.0: continue # Ignorar si la semana no tiene datos (aunque no debería pasar)
            
            if r > 6.0:
                # Penalización masiva por pasarte de reps "de fuerza"
                self._dbg("_penalizar_bloques", f"Penalizando B2 (Sem {idx+1}): reps_med={r:.1f} > 6.0", 20)
                pen += (r - 6.0) * PESO_FUERZA_REPS
            if r < 2.0:
                # Penalización masiva por quedarte ridículamente bajo
                self._dbg("_penalizar_bloques", f"Penalizando B2 (Sem {idx+1}): reps_med={r:.1f} < 2.0", 20)
                pen += (2.0 - r) * PESO_FUERZA_REPS

        # ----- Bloque 3: Peaking (reps medias 2–5, y que NO suban) -----
        for idx in range(8, 11):  # semanas 9 a 11
            r = reps_prom_semana[idx]
            if r == 0.0: continue
            
            if r > 5.0:
                self._dbg("_penalizar_bloques", f"Penalizando B3 (Sem {idx+1}): reps_med={r:.1f} > 5.0", 20)
                pen += (r - 5.0) * PESO_PEAK_REPS

        # No permitir que las reps SUBAN dentro del bloque de peaking
        # S10 vs S9, S11 vs S10
        for idx in range(9, 11):
            r_actual = reps_prom_semana[idx]
            r_prev = reps_prom_semana[idx - 1]
            if r_actual > r_prev + 0.1:  # margen pequeño de tolerancia
                self._dbg("_penalizar_bloques", f"Penalizando B3 (Sem {idx+1}): reps suben {r_prev:.1f} -> {r_actual:.1f}", 20)
                pen += (r_actual - r_prev) * PESO_PEAK_TEND

        # ----- Semana 12: Taper real (poca fatiga, alta intensidad) -----
        r12 = reps_prom_semana[11]
        pct12 = pct_prom_semana[11]

        # %1RM promedio < 80% -> penalización masiva
        if pct12 < 80.0 and pct12 > 0:  # Ignorar si es 0.0 (sin datos)
            self._dbg("_penalizar_bloques", f"Penalizando Taper (Sem 12): pct_med={pct12:.1f} < 80.0", 20)
            pen += (80.0 - pct12) * PESO_TAPER_INT

        # Reps promedio > 3 -> penalización masiva
        if r12 > 3.0:
            self._dbg("_penalizar_bloques", f"Penalizando Taper (Sem 12): reps_med={r12:.1f} > 3.0", 20)
            pen += (r12 - 3.0) * PESO_TAPER_REPS

        return pen

        
    def _penalizar_tendencias_y_suavidad(self,
                                        semanas,
                                        metricas: dict | None = None) -> float:
        pen = 0.0
        b1 = [semanas[i] for i in self.BLOQUES[1]["idx"]]
        b2 = [semanas[i] for i in self.BLOQUES[2]["idx"]]
        b3 = [semanas[i] for i in self.BLOQUES[3]["idx"]]

        s1, s2, s3 = self._avg([x.series for x in b1]), self._avg([x.series for x in b2]), self._avg([x.series for x in b3])
        i1, i2, i3 = self._avg([x.pct_1rm for x in b1]), self._avg([x.pct_1rm for x in b2]), self._avg([x.pct_1rm for x in b3])
        r1, r2, r3 = self._avg([x.reps for x in b1]), self._avg([x.reps for x in b2]), self._avg([x.reps for x in b3])

        # Tendencia de series por bloque (hipertrofia -> fuerza -> pico)
        if not (s1 >= s2 >= s3):
            pen += 10.0 + 2.0 * (abs(s1 - s2) + abs(s2 - s3))
        if (s1 - s3) < 4.0:
            pen += 6.0 * (4.0 - (s1 - s3))

        # Tendencia de intensidad por bloque
        if not (i1 <= i2 <= i3):
            pen += 10.0 + 0.5 * (abs(i1 - i2) + abs(i2 - i3))
        if (i3 - i1) < 12.0:
            pen += 3.0 * (12.0 - (i3 - i1))

        # Tendencia de repeticiones por bloque
        if not (r1 >= r2 >= r3):
            pen += 6.0 + 0.5 * (abs(r1 - r2) + abs(r2 - r3))

        # Cambios bruscos semana a semana en series e intensidad
        for a, b in zip(semanas, semanas[1:]):
            if a.series > 0:
                delta_series = abs(b.series - a.series) / max(1.0, a.series)
                if delta_series > 0.25:
                    pen += 3.0 * (delta_series - 0.25) * 10
            if abs(b.pct_1rm - a.pct_1rm) > 7.5:
                pen += 2.5 * (abs(b.pct_1rm - a.pct_1rm) - 7.5) / 5.0

        # Estímulo mínimo por bloque (volumen * %1RM)
        for idx, s in enumerate(semanas):
            block = 1 if idx < 4 else (2 if idx < 8 else 3)
            stim = 0.0
            if s.distrib_series and s.pct_1rm_sesion and s.reps_sesion and \
                    len(s.distrib_series) == len(s.pct_1rm_sesion) == len(s.reps_sesion):
                for sets, pctj, repsj in zip(s.distrib_series, s.pct_1rm_sesion, s.reps_sesion):
                    stim += sets * repsj * (pctj / 100.0)
            else:
                stim = s.series * s.reps * (s.pct_1rm / 100.0)

            deficit = self.STIM_MIN[block] - stim
            if deficit > 0:
                pen += 0.05 * deficit

        # Ramp-up raro entre semana 4 y 5 (seguridad extra)
        w4, w5 = semanas[3], semanas[4]
        delta_45 = abs(w5.pct_1rm - w4.pct_1rm)
        if delta_45 > 10.0:
            pen += 4.0 * (delta_45 - 10.0)
            print("EvolucionEntrenamiento - _penalizar_tendencias_y_suavidad: salto W4->W5 excesivo, penalizando ramp-up.")

        # 🔴 NUEVO: penalización por shock intersemanal usando métricas ER5
        if metricas is not None:
            ton_sem = metricas.get("ton_sem", [])
            stim_sem = metricas.get("S_sem", [])

            if ton_sem and stim_sem:
                pen += self._shock_intersemanal(ton_sem, stim_sem)

        return float(pen)

    # -------------------------------------------------------------------------
    # 1) Penalizar exceso de H en acumulación + días L ligeros + día de velocidad
    # -------------------------------------------------------------------------
    def _penalizar_roles(self, semanas):
        pen = 0.0

        for i, sem in enumerate(semanas):
            sets_H = sum(s.sets for s in sem.sesiones if s.rol == "H")
            sets_M = sum(s.sets for s in sem.sesiones if s.rol == "M")
            sets_L = sum(s.sets for s in sem.sesiones if s.rol == "L")

            # 1) ACUMULACIÓN (W1-4): castiga exceso de H y premia M/L 65-72 %
            if i < 4:
                pen += max(0, (sets_H - (sets_M + sets_L))) * 1.2
                for ses in sem.sesiones:
                    if ses.rol in ("M", "L") and 65 <= ses.pct1RM <= 72:
                        pen -= ses.sets * 0.8

            # 2) FUERZA (W5-8): L debe ser realmente ligero
            if 4 <= i < 8:
                for ses in sem.sesiones:
                    if ses.rol == "L":
                        if ses.pct1RM >= 73:            # demasiado % para L
                            pen += (ses.pct1RM - 72) * 0.6
                        if ses.RPE >= 7:                # RPE alto para L
                            pen += (ses.RPE - 6.5) * 1.0

            # 3) PEAK (W9-11): día de velocidad obligatorio (≤65 %, ≤2 reps, RPE ≤7)
            if 8 <= i < 11:
                hay_vel = any(
                    (s.pct1RM <= 65 and s.reps <= 2 and s.RPE <= 7)
                    for s in sem.sesiones
                )
                if not hay_vel:
                    pen += 7.0

        return pen

   

    def _penalizar_tendencias_y_suavidad(self, semanas):
        pen = 0.0
        for i in range(1, len(semanas)):
            prev, sem = semanas[i - 1], semanas[i]

            Δsets = abs(sem.total_sets() - prev.total_sets())
            Δreps = abs(sem.reps_prom - prev.reps_prom)

            if not self._cambio_de_fase(i):            # dentro del mismo bloque
                pen += max(0, Δsets - 3) * 0.8         # castiga saltos >3 series
                pen += max(0, Δreps - 1.0) * 2.0       # castiga saltos >1 rep media
            else:                                      # entre bloques
                pen += max(0, Δsets - 6) * 0.2         # más permisivo

        return pen

    def _penalizar_bloques(self, plan):
        semanas = plan.semanas
        pen = 0.0

        # ---- Peaking & Taper ----
        w11, w12 = semanas[10], semanas[11]
        vol11, vol12 = w11.total_sets(), w12.total_sets()
        drop = 1 - (vol12 / max(vol11, 1))

        
        if 0.15 <= drop <= 0.35:
            pen -= 12.0
        else:
            pen += abs(drop - 0.50) * 12.0

        # Top single @85-90 %, RPE 7-8.5, 1-2 reps
        for s in w12.sesiones:
            if "top" in (s.notas or "").lower():
                if 85 <= s.pct1RM <= 90 and 7 <= s.RPE <= 8.5 and s.reps in (1, 2):
                    pen -= 5.0
                else:
                    pen += 5.0

        return pen


    def _penalizar_taper_final(self, semanas, probar_1rm: bool = False) -> float:
        pen = 0.0
        s11, s12 = semanas[10], semanas[11]
        max_pct12 = max(s12.pct_1rm_sesion) if s12.pct_1rm_sesion else s12.pct_1rm

        if not probar_1rm:
            pen += max(0.0, 90.0 - max_pct12) * 0.8
            pen += max(0.0, max_pct12 - 92.0) * 0.8

            # Mismo ratio que en _enforce_taper, para no pelearse
            ratio = 0.80
            objetivo = max(6, int(round(s11.series * ratio)))

            # Solo penalizamos si te quedas POR DEBAJO del objetivo,
            # no por encima (más sets en taper no es un problema).
            if s12.series < objetivo:
                diff = objetivo - s12.series
                pen += 1.0 * (diff ** 2)


            if s12.distrib_series and s12.sesiones in (2, 3):
                exceso = sum(max(0, x - 3) for x in s12.distrib_series)
                pen += 1.2 * exceso
        else:
            if s12.series > 8:
                pen += (s12.series - 8) * 0.8
            if s12.reps > 2:
                pen += (s12.reps - 2) * 0.5
        return pen

    def _penalizar_incoherencia_rir(self, semanas) -> float:
        """
        Penaliza desajustes entre el RIR estimado (de %1RM y reps) y el RIR objetivo
        definido por FASE (semana), ROL (H/M/L) y si el día es de VELOCIDAD.
        - No penaliza L-velocidad cuando pct<=65 y reps<=2.
        - Usa un umbral de tolerancia de 0.5 RIR antes de sumar coste.
        - Pesa la multa por número de sets y por 'peso' del modo (ej. velocidad pesa 0.2).
        """
        penal_total: float = 0.0

        for semana_idx, semana in enumerate(semanas, start=1):
            tiene_arrays = bool(
                semana.distrib_series and semana.pct_1rm_sesion and semana.rpe_sesion and semana.reps_sesion
            )
            if not tiene_arrays:
                continue

            fase = self._fase_de_semana(semana_idx)

            # Longitud segura
            n = min(
                len(semana.distrib_series),
                len(semana.pct_1rm_sesion),
                len(semana.rpe_sesion),
                len(semana.reps_sesion),
            )
            if n <= 0:
                continue

            # Roles por intensidad; si falla, derivar por cuota de sets
            try:
                roles = self._roles_hml_por_pct(list(semana.pct_1rm_sesion[:n]))
                if not roles or len(roles) != n:
                    raise ValueError
            except Exception:
                total_sets = max(1, int(semana.series))
                roles = [
                    self._rol_por_share((semana.distrib_series[j] if j < len(semana.distrib_series) else 0) / total_sets)
                    for j in range(n)
                ]

            for j in range(n):
                sets_j = int(semana.distrib_series[j])
                pct_j  = float(semana.pct_1rm_sesion[j])
                rpe_j  = float(semana.rpe_sesion[j])
                reps_j = int(semana.reps_sesion[j])
                rol_j  = roles[j]

                es_vel = self._es_dia_velocidad(fase, rol_j, semana)

                # Excepción explícita para L-velocidad "6×2 @60–65% RPE 6–7"
                if es_vel and pct_j <= 65.0 and reps_j <= 2:
                    continue

                rir_min, rir_max, peso = self._rir_objetivo(fase, rol_j, es_vel)

                # RIR estimado con Epley inverso
                reps_fallo = self._epley_pct_a_reps(pct_j)
                rir_est = max(0.0, reps_fallo - float(reps_j))

                # Desajuste fuera del intervalo objetivo
                if rir_est < rir_min:
                    desajuste = rir_min - rir_est
                elif rir_est > rir_max:
                    desajuste = rir_est - rir_max
                else:
                    desajuste = 0.0

                # Tolerancia 0.5 RIR; peso reducido en velocidad (p.ej. 0.2)
                if desajuste > 0.5 and sets_j > 0:
                    penal_total += 6.0 * float(sets_j) * float(peso) * (desajuste - 0.5)

        return float(penal_total)


    def _penalizar_orden_roles(self, semanas) -> float:
        """Exige H≥M≥L (por medias ponderadas), en acumulación evita que H acapare todo el volumen
        y en fuerza empuja a que L sea realmente ligero. Tolerancias: 2 pp en % y 0.25 en RPE.
        """
        pen = 0.0
        TOL_PCT = 2.0
        TOL_RPE = 0.25
        W_PCT = 0.8
        W_RPE = 1.0

        # Usamos índice de semana para distinguir bloques: 0–3 acc, 4–7 fuerza
        for idx, s in enumerate(semanas):
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
                rol = "H" if share > 0.45 else ("M" if share >= 0.30 else "L")
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
                    if pa + TOL_PCT < pb:
                        pen += (pb - pa - TOL_PCT) * W_PCT
                    if ra + TOL_RPE < rb:
                        pen += (rb - ra - TOL_RPE) * W_RPE

            # Mantener H≥M≥L en intensidad y RPE
            cmp("H", "M")
            cmp("M", "L")

            # ─────────────────────────────────────────────
            # BLOQUE DE ACUMULACIÓN (SEMANAS 1–4, idx 0–3)
            # Evitar que H acapare casi todo el volumen.
            # ─────────────────────────────────────────────
            if idx < 4 and rol_stats:
                share_H = rol_stats.get("H", {}).get("share", 0.0)
                share_M = rol_stats.get("M", {}).get("share", 0.0)
                share_L = rol_stats.get("L", {}).get("share", 0.0)
                share_ML = share_M + share_L

                # Penaliza cuando H se lleva claramente >60 % de las series
                exceso_H = max(0.0, share_H - 0.60)
                if exceso_H > 0.0:
                    pen += exceso_H * 10.0

                # Pequeña "recompensa" cuando H está entre 40–60 %
                # y al menos 40 % del volumen va a M+L
                if 0.40 <= share_H <= 0.60 and share_ML >= 0.40:
                    pen -= 2.0

            # ─────────────────────────────────────────────
            # BLOQUE DE FUERZA (SEMANAS 5–8, idx 4–7)
            # Empujar a que L sea realmente ligero.
            # ─────────────────────────────────────────────
            if 4 <= idx < 8 and rol_stats:
                stats_L = rol_stats.get("L")
                if stats_L:
                    pct_L = stats_L["pct"]
                    rpe_L = stats_L["rpe"]

                    # Penalizar L demasiado pesado: >72 % o RPE >7
                    if pct_L > 72.0:
                        pen += (pct_L - 72.0) * 0.8
                    if rpe_L > 7.0:
                        pen += (rpe_L - 7.0) * 1.2

                    # Pequeña recompensa cuando L está en 68–72 % y RPE 6–7
                    if 68.0 <= pct_L <= 72.0 and 6.0 <= rpe_L <= 7.0:
                        pen -= 2.5

        return pen



    def _suavizado_L2(self, semanas) -> float:
        pen = 0.0
        for a, b in zip(semanas, semanas[1:]):
            d_series = (b.series - a.series) / max(1.0, a.series)
            d_pct = (b.pct_1rm - a.pct_1rm) / 100.0
            d_rpe = (b.rpe - a.rpe) / 3.0
            d_reps = (b.reps - a.reps) / 10.0
            pen += 0.6 * (d_series ** 2) + 1.2 * (d_pct ** 2) + 0.6 * (d_rpe ** 2) + 0.4 * (d_reps ** 2)
        return pen
    def _shock_intersemanal(self,
                            ton_sem: list[float],
                            stim_sem: list[float]) -> float:
        """
        Penaliza saltos bruscos (shock) de una semana a otra en:
        - Volumen (ton_sem): series*reps*%1RM
        - Estímulo (stim_sem): S_sem (ER5 ponderado por intensidad)

        Si el cambio relativo es <= 10% → sin penalización.
        Si es > 10%, penaliza cuadráticamente.
        Usa:
        - self._EPS_DIV
        - self._LAMBDA_SHOCK_VOL
        - self._LAMBDA_SHOCK_STIM

        Ajuste añadido:
        - En cambios de bloque (W4→W5, W8→W9) la penalización se
          reduce a factor 0.5 para permitir un poco más de "ruptura".
        """
        pen = 0.0
        n = min(len(ton_sem), len(stim_sem))

        if n <= 1:
            return 0.0

        for i in range(1, n):
            # i recorre 1..n-1, y comparamos semana (i-1) -> semana i
            semana_prev = i - 1  # 0-based
            semana_act  = i      # 0-based

            # Factor de suavizado: mitad de penalización en cambios de bloque
            factor = 1.0
            if (semana_prev, semana_act) in ((3, 4), (7, 8)):  # 3->4 = W4→W5, 7->8 = W8→W9
                factor = 0.5

            # ---- Volumen (ton_sem) ----
            base_vol = max(ton_sem[semana_prev], self._EPS_DIV)
            delta_vol = abs(ton_sem[semana_act] - ton_sem[semana_prev]) / base_vol

            if delta_vol > 0.10:  # 10%
                exceso = delta_vol - 0.10
                pen += factor * self._LAMBDA_SHOCK_VOL * (exceso ** 2)

            # ---- Estímulo (stim_sem = S_sem) ----
            base_st = max(stim_sem[semana_prev], self._EPS_DIV)
            delta_st = abs(stim_sem[semana_act] - stim_sem[semana_prev]) / base_st

            if delta_st > 0.10:  # 10%
                exceso = delta_st - 0.10
                pen += factor * self._LAMBDA_SHOCK_STIM * (exceso ** 2)

        return float(pen)

    