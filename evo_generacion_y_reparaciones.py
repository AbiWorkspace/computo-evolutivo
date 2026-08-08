# evo_generacion_y_reparaciones.py (generado)
# -*- coding: utf-8 -*-
from __future__ import annotations
import random, math
from typing import List, Tuple, Dict
from modelos import SemanaPlan, Plan12Semanas

class GeneracionDistribMixin:

    def _generar_por_sesion(self, s: SemanaPlan, semana_idx: int) -> None:
        """
        Crea %1RM, RPE y reps por sesión (H/M/L) a partir de los valores semanales.
        Reglas:
        - share >= 0.45  → H: %1RM += 2.5, RPE += 0.3, reps -= 2
        - 0.30 <= share < 0.45 → M: sin cambios
        - share  < 0.30 → L: %1RM -= 2.5, RPE -= 0.3, reps += 2
        Clamps: %1RM∈[50,100], RPE cuantizado a pasos de 0.5, reps∈[1,30].
        """
        s.pct_1rm_sesion, s.rpe_sesion, s.reps_sesion = [], [], []

        if semana_idx is None:
            semana_idx = 0

        # Guardas de seguridad si no hay distribución o parámetros inválidos
        if not s.distrib_series or s.series <= 0 or s.sesiones <= 0:
            n = max(1, s.sesiones)
            s.pct_1rm_sesion = [float(min(100.0, max(50.0, s.pct_1rm)))] * n
            s.rpe_sesion = [self._rpe_discreto(s.rpe)] * n
            s.reps_sesion = [int(max(1, min(30, s.reps)))] * n
        else:
            total_series = max(1.0, float(s.series))
            for sets in s.distrib_series:
                share = sets / total_series
                rol = self._rol_por_share(share)

                pct = float(s.pct_1rm)
                rpe = float(s.rpe)
                reps = int(s.reps)

                # Ajustes por rol
                if rol == "H":
                    pct += 2.5
                    rpe += 0.3
                    reps -= 2
                elif rol == "L":
                    pct -= 2.5
                    rpe -= 0.3
                    reps += 2
                # M se queda igual

                # Clamps/cuantización
                pct = float(min(100.0, max(50.0, pct)))
                rpe = self._rpe_discreto(rpe)
                reps = int(max(1, min(30, reps)))

                s.pct_1rm_sesion.append(pct)
                s.rpe_sesion.append(rpe)
                s.reps_sesion.append(reps)

        # Reparaciones/normalizaciones globales (una sola vez)
        self._normalizar_arrays_por_sesion(s)

        self._forzar_minimos_y_gaps_por_rol(s)

        # ===== 1) RPE según modo (definitivo para esta semana) =====
        if getattr(self, "modo_rpe", None) == "ampliado":
            self._asignar_rpe_progresivo(s, semana_idx)
            if s.rpe_sesion:
                s.rpe = self._rpe_discreto(sum(s.rpe_sesion) / len(s.rpe_sesion))
        else:
            base = s.rpe
            rpe_ses = s.rpe_sesion if s.rpe_sesion else [base] * max(1, s.sesiones)
            s.rpe_sesion = [self._rpe_discreto(x if x is not None else base) for x in rpe_ses]
            s.rpe = self._rpe_discreto(sum(s.rpe_sesion) / len(s.rpe_sesion))

        self._validar_coherencia_rir_rpe(s, semana_idx)

        # ===== 3) Sincronizar reps semanales con media ponderada =====
        if s.distrib_series and sum(s.distrib_series) > 0:
            total_sets = sum(s.distrib_series)
            s.reps = int(round(sum(sets * reps for sets, reps in zip(s.distrib_series, s.reps_sesion)) / total_sets))

        # Canonizar orden H/M/L y RPE por rol
        self._canonizar_orden_hml(s)
        self._ordenar_rpe_por_rol(s)

    def _cerrar_triangulo_pct_rpe(self, pct_1rm: float, rpe: float) -> int:
        """
        Compatibilidad: dado un %1RM y un RPE, devuelve unas repeticiones
        coherentes usando Epley + RIR ≈ 10 − RPE.

        Se usa en la versión vieja de `_canonizar_orden_hml`, así que este
        wrapper evita el AttributeError sin romper la lógica nueva.
        """
        try:
            pct = float(pct_1rm)
            rpe = float(rpe)
        except Exception:
            pct = float(pct_1rm or 70.0)
            rpe = float(rpe or 7.5)

        # reps a fallo estimadas por Epley
        reps_fallo = self._epley_pct_a_reps(pct)

        # RIR objetivo aproximado según RPE
        rir_t = max(0.0, 10.0 - rpe)

        # reps “objetivo” = reps a fallo - RIR
        reps_obj = reps_fallo - rir_t

        # límites razonables
        reps_obj = max(1.0, min(30.0, reps_obj))

        return int(round(reps_obj))

    def _rpe_base_semana(self, semana_idx: int) -> float:
        bloque = 1 if semana_idx < 4 else (2 if semana_idx < 8 else 3)
        pos = semana_idx % 4  # 0..3
        lo, hi = self.RPE_OBJ_BLOQUE[bloque]
        tmap = {0: 0.00, 1: 0.33, 2: 0.66, 3: 1.00}
        base = self._lerp(lo, hi, tmap[pos])
        if self.usar_deload_intermedio and pos == 3 and bloque in (1, 2):
            base = max(6.0, base - 0.5)  # deload W4/W8
        if semana_idx == 11:
            base = min(base, 8.8)  # taper/test W12
        return self._rpe_round(base)

    def _offsets_rpe_por_rol(self, sesiones: int, semana_idx: int) -> list:
        bloque = 1 if semana_idx < 4 else (2 if semana_idx < 8 else 3)
        offs = [0.5, 0.0, -0.5] if sesiones == 3 else [0.5, 0.0]
        if bloque == 3:
            offs[0] += 0.25  # empuja un poco más H en peaking
        return offs

    def _canonizar_orden_hml(self, s: SemanaPlan) -> None:
        """
        Ordena las sesiones por intensidad y garantiza una ondulación H > M > L
        sin reescribir por completo M y L.

        Idea:
        - Ordenar por %1RM descendente → índice 0 = H, 1 = M, 2 = L (si existe).
        - Asegurar gaps mínimos H–M y M–L (p.ej. 3 p.p.) solo bajando % de M y L.
        - Ajustar RPE de forma suave para que H ≥ M ≥ L, tocando lo mínimo posible.
        - NO recalcula reps con Epley: respeta las reps ya generadas.
        """
        # Asegurar tamaños consistentes
        n = min(
            len(s.pct_1rm_sesion or []),
            len(s.distrib_series or []),
            len(s.rpe_sesion or []),
            len(s.reps_sesion or []),
        )
        if n < 2:
            return

        # 1) Ordenar por %1RM descendente: 0=H, 1=M, 2=L
        idx_orden = sorted(
            range(n),
            key=lambda i: float(s.pct_1rm_sesion[i]),
            reverse=True,
        )

        s.pct_1rm_sesion = [float(s.pct_1rm_sesion[i]) for i in idx_orden]
        s.distrib_series = [int(s.distrib_series[i]) for i in idx_orden]
        s.rpe_sesion     = [float(s.rpe_sesion[i]) for i in idx_orden]
        s.reps_sesion    = [int(s.reps_sesion[i]) for i in idx_orden]

        # 2) Si solo hay 2 sesiones, basta con H > L
        if n == 2:
            gap_min = 3.0  # pp mínimos entre H y L
            if s.pct_1rm_sesion[0] <= s.pct_1rm_sesion[1]:
                s.pct_1rm_sesion[1] = max(50.0, s.pct_1rm_sesion[0] - gap_min)

            # RPE: asegurar que H ≥ L
            if s.rpe_sesion[0] < s.rpe_sesion[1]:
                s.rpe_sesion[1] = self._rpe_discreto(
                    min(s.rpe_sesion[1], s.rpe_sesion[0])
                )

            # Clamps finales
            s.pct_1rm_sesion = [float(self._clamp(x, 50.0, 100.0)) for x in s.pct_1rm_sesion]
            s.rpe_sesion     = [self._rpe_discreto(float(x)) for x in s.rpe_sesion]
            s.reps_sesion    = [int(self._clamp(int(x), 1, 30)) for x in s.reps_sesion]
            return

        # 3) Para 3 sesiones: H, M, L con gaps suaves
        gap_min = 3.0   # gap mínimo H–M y M–L en puntos porcentuales
        gap_max = 10.0  # para no abrir huecos absurdos

        # H es referencia
        pct_H = s.pct_1rm_sesion[0]

        # --- Ajustar M ---
        pct_M = s.pct_1rm_sesion[1]
        if pct_M >= pct_H:  # si M quedó igual o por encima, bajarlo un poco
            pct_M = pct_H - gap_min
        else:
            # Si el gap es muy pequeño, abrirlo hasta gap_min
            if pct_H - pct_M < gap_min:
                pct_M = pct_H - gap_min
            # Si el gap es exagerado, recortarlo a gap_max
            if pct_H - pct_M > gap_max:
                pct_M = pct_H - gap_max
        s.pct_1rm_sesion[1] = pct_M

        # --- Ajustar L ---
        pct_L = s.pct_1rm_sesion[2]
        # M es ahora la referencia para L
        if pct_L >= pct_M:
            pct_L = pct_M - gap_min
        else:
            if pct_M - pct_L < gap_min:
                pct_L = pct_M - gap_min
            if pct_M - pct_L > gap_max:
                pct_L = pct_M - gap_max
        s.pct_1rm_sesion[2] = pct_L

        # 4) Ajuste suave de RPE: H ≥ M ≥ L
        # No subimos RPE, solo bajamos si rompe el orden
        # H
        s.rpe_sesion[0] = self._rpe_discreto(s.rpe_sesion[0])

        # M
        if s.rpe_sesion[1] > s.rpe_sesion[0]:
            s.rpe_sesion[1] = self._rpe_discreto(s.rpe_sesion[0])
        # L
        if s.rpe_sesion[2] > s.rpe_sesion[1]:
            s.rpe_sesion[2] = self._rpe_discreto(s.rpe_sesion[1])

        # 5) Clamps finales por seguridad
        s.pct_1rm_sesion = [float(self._clamp(x, 50.0, 100.0)) for x in s.pct_1rm_sesion]
        s.rpe_sesion     = [self._rpe_discreto(float(x)) for x in s.rpe_sesion]
        s.reps_sesion    = [int(self._clamp(int(x), 1, 30)) for x in s.reps_sesion]
        # Actualizar total de series por si algo estaba raro
        s.series = int(sum(s.distrib_series or []))



    # >>> NUEVO
    def _ordenar_rpe_por_rol(self, s: SemanaPlan, tol: float = 0.0) -> None:
        """
        Asegura RPE H ≥ M ≥ L (en el orden canónico por %) con una tolerancia 'tol'.
        """
        if not (s.rpe_sesion and s.pct_1rm_sesion):
            return
        self._canonizar_orden_hml(s)
        for i in range(1, len(s.rpe_sesion)):
            if s.rpe_sesion[i] > s.rpe_sesion[i-1] - tol:
                s.rpe_sesion[i] = self._rpe_discreto(min(s.rpe_sesion[i], s.rpe_sesion[i-1]))

        
    
    def _asignar_rpe_progresivo(self, s: SemanaPlan, semana_idx: int) -> None:
        if not (s.distrib_series and s.series > 0 and s.sesiones in (2, 3)):
            base = self._rpe_base_semana(semana_idx)
            s.rpe_sesion = [self._rpe_round(base)] * max(1, s.sesiones)
            s.rpe = self._rpe_round(sum(s.rpe_sesion) / len(s.rpe_sesion))
            return

        base = self._rpe_base_semana(semana_idx)
        order = sorted(range(s.sesiones), key=lambda j: s.distrib_series[j], reverse=True)  # H,M,(L)
        offs = self._offsets_rpe_por_rol(s.sesiones, semana_idx)

        if not s.rpe_sesion or len(s.rpe_sesion) != s.sesiones:
            s.rpe_sesion = [base] * s.sesiones

        rpe_tmp = [0.0] * s.sesiones
        for rank, j in enumerate(order):
            rpe_tmp[j] = self._rpe_round(base + offs[rank])

        if s.sesiones == 3:
            iH, iM, iL = order
            if rpe_tmp[iH] < rpe_tmp[iM]:
                rpe_tmp[iH] = self._rpe_round(rpe_tmp[iM] + 0.5)
            if rpe_tmp[iM] < rpe_tmp[iL]:
                rpe_tmp[iM] = self._rpe_round(rpe_tmp[iL] + 0.5)

        s.rpe_sesion = rpe_tmp[:]
        s.rpe = self._rpe_round(
            sum(r * sets for r, sets in zip(s.rpe_sesion, s.distrib_series)) / max(1, s.series)
        )

    def _rpe_medio_semana(self, s: SemanaPlan) -> float:
        if s.distrib_series and s.rpe_sesion and len(s.rpe_sesion) == s.sesiones:
            return sum(r * sets for r, sets in zip(s.rpe_sesion, s.distrib_series)) / max(1, s.series)
        return s.rpe

    def _realinear_sets_con_pct(self, s: SemanaPlan) -> None:
        """Reordena la distribución de sets para que la mayor cuota caiga en la sesión de mayor %."""
        if not (s.distrib_series and s.pct_1rm_sesion and len(s.distrib_series) == len(s.pct_1rm_sesion)):
            return
        n = len(s.distrib_series)
        # Índices ordenados por % final (desc)
        order_pct = sorted(range(n), key=lambda i: s.pct_1rm_sesion[i], reverse=True)
        # Sets ordenados (desc)
        sets_sorted = sorted(s.distrib_series, reverse=True)
        nueva = [0] * n
        for rank, i in enumerate(order_pct):
            nueva[i] = sets_sorted[rank]
        s.distrib_series = nueva
        s.series = int(sum(s.distrib_series))

    def _validar_coherencia_rir_rpe(self, s, semana_idx: int | None = None) -> None:
        """
        Hace coherentes % - reps - RPE por sesión usando el rango de RIR objetivo
        según FASE (semana), ROL (H/M/L) y si el día es de velocidad.
        Ajusta primero reps (±1) y luego % (≤3 pp) hacia el objetivo.
        
        >>> ESTA ES LA MODIFICACIÓN CLAVE:
        Esta función AHORA respeta los rangos de repeticiones de _PHASE_BOUNDS.
        No puede "corregir" las repeticiones fuera del rango permitido para el bloque.
        """
        # Nada que hacer si no hay arrays por sesión
        if not (s.pct_1rm_sesion and s.reps_sesion and s.rpe_sesion):
            return

        # --- Cargar rangos de fase (los necesitamos para CLAMPEAR) ---
        if not hasattr(self, "_PHASE_BOUNDS"):
             # Fallback (copiado de la función _enforce_phase_ranges)
             self._PHASE_BOUNDS = {
                "acc": {"H": {"reps": (6, 8)}, "M": {"reps": (8, 10)}, "L": {"reps": (10,12)}},
                "int": {"H": {"reps": (2, 4)}, "M": {"reps": (3, 5)}, "L": {"reps": (5, 8)}},
                "peak": {"H": {"reps": (1, 3)}, "M": {"reps": (2, 4)}, "L": {"reps": (2, 3)}},
                "taper": {"H": {"reps": (1, 2)}, "M": {"reps": (2, 3)}, "L": {"reps": (2, 3)}},
             }
             # Añadir RPEs/PCTs por defecto si faltan
             for k, v in self._PHASE_BOUNDS.items():
                 for k2, v2 in v.items():
                     if "rpe" not in v2: v2["rpe"] = (6.0, 9.0)
                     if "pct" not in v2: v2["pct"] = (60.0, 95.0)

        # Determinar fase (acepta semana_idx 0-based o 1-based; si viene None, infiere por %)
        semana_1based = 1
        if semana_idx is not None:
            si = int(semana_idx)
            if 1 <= si <= 12:
                semana_1based = si
            elif 0 <= si <= 11:
                semana_1based = si + 1
            else:
                semana_1based = max(1, min(12, si))
        
        fase = self._fase_simple(semana_1based) # acc, int, peak, taper

        # Roles por intensidad (si no hay, derivar por cuota de sets)
        try:
            roles = self._roles_hml_por_pct(list(s.pct_1rm_sesion or []))
            if not roles or len(roles) != len(s.pct_1rm_sesion):
                raise ValueError
        except Exception:
            total_sets = max(1, int(getattr(s, "series", 0) or sum(s.distrib_series or [0])))
            roles = [
                self._rol_por_share((s.distrib_series[j] if j < len(s.distrib_series or []) else 0) / total_sets)
                for j in range(len(s.pct_1rm_sesion))
            ]

        # Iterar y corregir sesión por sesión
        for i in range(len(s.pct_1rm_sesion)):
            pct  = float(s.pct_1rm_sesion[i])
            reps = int(s.reps_sesion[i])
            rpe  = float(s.rpe_sesion[i])
            rol  = roles[i] if i < len(roles) else "M"
            es_vel = self._es_dia_velocidad(fase, rol, s)

            # --- INICIO DE LA LÓGICA DE CLAMP ---
            # 1. Obtener los rangos de repeticiones permitidos para este bloque/rol
            try:
                bounds = self._PHASE_BOUNDS[fase][rol]
                reps_lo, reps_hi = bounds["reps"]
            except KeyError:
                reps_lo, reps_hi = 1, 30 # Fallback si algo falla
            # --- FIN DE LA LÓGICA DE CLAMP ---

            # RIR objetivo como intervalo; usamos el punto medio para corregir
            rir_min, rir_max, _peso = self._rir_objetivo(fase, rol, es_vel)
            rir_t = 0.5 * (float(rir_min) + float(rir_max))

            # Techos conservadores
            if rpe <= 6.5 and reps <= 2 and pct >= 99.0:
                pct = 95.0

            # Hasta 3 intentos de corrección
            for _ in range(3):
                reps_afallo_est = self._epley_pct_a_reps(pct)
                rir_est = max(0.0, reps_afallo_est - float(reps))
                if abs(rir_est - rir_t) >= 0.5:
                    delta_rir = rir_t - rir_est
                    if delta_rir > 0.5:
                        reps_new = reps - 1   # más difícil (menos RIR)
                    elif delta_rir < -0.5:
                        reps_new = reps + 1  # más RIR
                    else:
                        reps_new = reps
                    
                    # >>> APLICAR CLAMP DE BLOQUE
                    # Forzamos la nueva repetición a estar DENTRO del rango del bloque
                    reps = int(self._clamp(reps_new, reps_lo, reps_hi))

                    pct_target = self._epley_reps_a_pct(reps + rir_t)
                    step = 3.0
                    if pct < pct_target - 0.1:
                        pct = min(pct + step, pct_target)
                    elif pct > pct_target + 0.1:
                        pct = max(pct - step, pct_target)
                else:
                    break

            # Clamp por Epley (respetando el rango del bloque)
            reps_afallo_est = self._epley_pct_a_reps(pct)
            reps_max_epley = int(math.floor(reps_afallo_est - rir_t + 1e-6))
            # El máximo es el MÍNIMO entre el max de Epley y el max del bloque
            reps_max = max(reps_lo, min(reps_hi, reps_max_epley))
            
            if reps > max(1, reps_max):
                reps = max(1, reps_max)
            
            reps = int(self._clamp(reps, reps_lo, reps_hi)) # Doble chequeo

            pct_target = self._epley_reps_a_pct(reps + rir_t)
            if pct > pct_target + 0.1:
                pct = max(pct_target, pct - 3.0)
            elif pct < pct_target - 0.1:
                pct = min(pct_target, pct + 3.0)

            # Techos conservadores sin test
            if rpe <= 6.5 and reps <= 2:
                pct = min(pct, 95.0)

            # Evita combinaciones irreales: singles/dobles con RPE muy bajo
            if int(reps) <= 1:
                rpe = max(rpe, 7.5)
            elif int(reps) == 2 and pct >= 85.0:
                rpe = max(rpe, 7.0)
            
            # >>> REGLA DURA FINAL: No singles en M o L
            if rol in ("M", "L") and reps < 2:
                reps = 2
                # Y re-validamos el RPE/PCT para 2 reps
                pct = self._epley_reps_a_pct(2.0 + rir_t)
                rpe = self._rpe_discreto(10.0 - rir_t)

            
            # Guardar (con cuantización de RPE)
            s.pct_1rm_sesion[i] = float(min(100.0, max(50.0, pct)))
            s.reps_sesion[i]    = int(max(1, min(30, reps)))
            s.rpe_sesion[i]     = self._rpe_discreto(rpe)

        # Sincroniza reps semanales
        if s.distrib_series and sum(s.distrib_series) > 0:
            total_sets = sum(s.distrib_series)
            s.reps = int(round(sum(sets * reps for sets, reps in zip(s.distrib_series, s.reps_sesion)) / total_sets))

        # Alinear cuotas H/M/L con ranking real de %
        try:
            self._realinear_sets_con_pct(s)
        except Exception:
            pass
        
    def _caps_por_fase(self, semana_1based: int) -> dict:
        """Devuelve caps de deltas por rol según fase."""
        if   1 <= semana_1based <= 4:   # Acumulación
            return {
                "pct": {"H": 3.0, "M": 3.0, "L": 3.0},
                "rpe": 0.5,
                "sets": 2
            }
        elif 5 <= semana_1based <= 8:   # Intensificación
            return {
                "pct": {"H": 4.0, "M": 3.0, "L": 3.0},
                "rpe": 0.5,
                "sets": None  # no cap específico pedido; dejamos como None
            }
        else:                            # Peaking (9–12)
            return {
                "pct": {"H": 3.0, "M": 2.0, "L": 2.0},  # 2–3 p.p. por rol
                "rpe": 0.5,
                "sets": None
            }

    def _clamp(self, x, lo, hi):
        return max(lo, min(hi, x))

    def _enforce_separacion_roles(self, s: SemanaPlan, semana_1based: int) -> None:
        """H≥M≥L por %; gaps mínimos (intensidad/RPE) y orden de reps (L≥M+2, M≥H+1)."""
        if not s.distrib_series: 
            return
        self._normalizar_arrays_por_sesion(s)
        self._canonizar_orden_hml(s)  # índice 0=H,1=M,2=L

        n = len(s.distrib_series)
        gap_pp_min = 6.0   # 6–8 p.p. pedido → usamos 6 como mínimo duro
        gap_rpe    = 0.5

        # 1) Intensidad: abrir gaps bajando el inferior (no subimos % innecesariamente)
        if n >= 2:
            if s.pct_1rm_sesion[0] < s.pct_1rm_sesion[1] + gap_pp_min:
                s.pct_1rm_sesion[1] = self._clamp(s.pct_1rm_sesion[0] - gap_pp_min, 50.0, 100.0)
        if n == 3:
            if s.pct_1rm_sesion[1] < s.pct_1rm_sesion[2] + gap_pp_min:
                s.pct_1rm_sesion[2] = self._clamp(s.pct_1rm_sesion[1] - gap_pp_min, 50.0, 100.0)

        # 2) RPE: H ≥ M +0.5 ≥ L +1.0
        if n >= 2 and s.rpe_sesion[0] < s.rpe_sesion[1] + gap_rpe:
            s.rpe_sesion[1] = self._rpe_discreto(min(s.rpe_sesion[1], s.rpe_sesion[0] - gap_rpe))
        if n == 3 and s.rpe_sesion[1] < s.rpe_sesion[2] + gap_rpe:
            s.rpe_sesion[2] = self._rpe_discreto(min(s.rpe_sesion[2], s.rpe_sesion[1] - gap_rpe))

        # 3) Reps: L ≥ M +2 y M ≥ H +1 (excepto peaking para H)
        fase = "pico" if semana_1based >= 9 else ("fuerza" if semana_1based >= 5 else "hipertrofia")
        H, M = 0, 1
        if n >= 2:
            objetivo_M = max(s.reps_sesion[H] + 1, 2 if fase == "pico" else 1)
            if s.reps_sesion[M] < objetivo_M:
                s.reps_sesion[M] = objetivo_M
        if n == 3:
            L = 2
            objetivo_L = max(s.reps_sesion[M] + 2, 2)
            if s.reps_sesion[L] < objetivo_L:
                s.reps_sesion[L] = objetivo_L

        # Cuantización/clamps y coherencia posterior
        s.pct_1rm_sesion = [float(self._clamp(x, 50.0, 100.0)) for x in s.pct_1rm_sesion]
        s.rpe_sesion     = [self._rpe_discreto(float(x)) for x in s.rpe_sesion]
        s.reps_sesion    = [int(self._clamp(int(x), 1, 30)) for x in s.reps_sesion]
        self._validar_coherencia_rir_rpe(s, semana_1based - 1)

    def _limitar_salto_intersemanal_por_rol(self, plan: Plan12Semanas) -> None:
        """Suaviza Δ por rol semana a semana (H,M,L ya canonizados por %)."""
        for w in range(1, len(plan.semanas)):
            prev = plan.semanas[w - 1]
            cur  = plan.semanas[w]
            self._normalizar_arrays_por_sesion(prev); self._canonizar_orden_hml(prev)
            self._normalizar_arrays_por_sesion(cur);  self._canonizar_orden_hml(cur)

            caps = self._caps_por_fase(w + 1)  # semana actual 1-based
            # PCT y RPE
            roles = ["H", "M"] + (["L"] if len(cur.distrib_series) == 3 else [])
            for i, rol in enumerate(roles):
                # %1RM
                max_delta = caps["pct"][rol]
                d = cur.pct_1rm_sesion[i] - prev.pct_1rm_sesion[i]
                if d >  max_delta: cur.pct_1rm_sesion[i] = prev.pct_1rm_sesion[i] + max_delta
                if d < -max_delta: cur.pct_1rm_sesion[i] = prev.pct_1rm_sesion[i] - max_delta
                # RPE (solo acumulación tiene regla explícita; aplicamos general por seguridad)
                rpe_cap = caps["rpe"]
                cur.rpe_sesion[i] = self._rpe_discreto(
                    self._clamp(cur.rpe_sesion[i], prev.rpe_sesion[i] - rpe_cap, prev.rpe_sesion[i] + rpe_cap)
                )

            # Series por rol (sólo acumulación pide ±2)
            if caps["sets"]:
                for i in range(len(cur.distrib_series)):
                    tgt_min = max(0, int(prev.distrib_series[i] - caps["sets"]))
                    tgt_max = int(prev.distrib_series[i] + caps["sets"])
                    tgt     = int(self._clamp(cur.distrib_series[i], tgt_min, tgt_max))
                    delta   = tgt - cur.distrib_series[i]
                    if delta != 0:
                        # Mantén total moviendo sobrante/faltante al resto (del más cargado/ligero)
                        cur.distrib_series[i] = tgt
                        resto_idx = [j for j in range(len(cur.distrib_series)) if j != i]
                        if delta < 0:  # quitamos sets → devolverlos al más ligero (para no crear doble H)
                            recv = sorted(resto_idx, key=lambda j: cur.distrib_series[j])
                            for j in recv:
                                if delta == 0: break
                                cur.distrib_series[j] += 1
                                delta += 1
                        else:          # añadimos sets → quitar del más cargado
                            donors = sorted(resto_idx, key=lambda j: cur.distrib_series[j], reverse=True)
                            for j in donors:
                                if delta == 0 or cur.distrib_series[j] <= 0: break
                                cur.distrib_series[j] -= 1
                                delta -= 1

            # Caso crítico W4→W5: forzar H ≈81–83 si venía de ~77
            if w == 4 - 0:  # w index 0-based=4 es W5
                H_prev = prev.pct_1rm_sesion[0]
                H_cur  = cur.pct_1rm_sesion[0]
                if H_cur - H_prev > 8.0:  # subida desmesurada
                    cur.pct_1rm_sesion[0] = self._clamp(H_prev + 4.0, 81.0, 83.0)

            self._forzar_minimos_y_gaps_por_rol(cur)
            self._validar_coherencia_rir_rpe(cur, w)

    def _reglas_por_fase_reps_y_pct(self, s: SemanaPlan, semana_1based: int) -> None:
        """Clamps por fase/rol para evitar singles en M/L y fijar rangos de % y reps."""
        self._normalizar_arrays_por_sesion(s); self._canonizar_orden_hml(s)
        fase = "hipertrofia" if semana_1based <= 4 else ("fuerza" if semana_1based <= 8 else "pico")
        n = len(s.distrib_series)

        def clamp_arr(i, pct_lo, pct_hi, reps_lo, reps_hi, rpe_lo=None, rpe_hi=None):
            s.pct_1rm_sesion[i] = float(self._clamp(s.pct_1rm_sesion[i], pct_lo, pct_hi))
            s.reps_sesion[i]    = int(self._clamp(s.reps_sesion[i],    reps_lo, reps_hi))
            if rpe_lo is not None:
                s.rpe_sesion[i] = self._rpe_discreto(self._clamp(s.rpe_sesion[i], rpe_lo, (rpe_hi or 9.0)))

        # H=0, M=1, L=2 (si existe)
        if fase == "hipertrofia":
            if n >= 1: clamp_arr(0, 70.0, 76.0, 6, 8, 6.5, 7.5)  # H
            if n >= 2: clamp_arr(1, 68.0, 72.0, 8,10, 6.5, 7.5)  # M
            if n >= 3: clamp_arr(2, 60.0, 68.0,10,12, 6.0, 7.0)  # L
        elif fase == "fuerza":
            if n >= 1: clamp_arr(0, 82.0, 88.0, 2, 4)            # H
            if n >= 2: clamp_arr(1, 76.0, 84.0, 3, 5)            # M (sin singles)
            if n >= 3: clamp_arr(2, 68.0, 76.0, 5, 8)            # L (sin singles)
            if n >= 2 and s.reps_sesion[1] == 1: s.reps_sesion[1] = 2
            if n >= 3 and s.reps_sesion[2] == 1: s.reps_sesion[2] = 2
        else:  # pico
            if n >= 1:
                # H: se permite single pero también 2–3 back-offs (modelo mono-parámetro: mantenemos reps=1 y sets)
                s.pct_1rm_sesion[0] = float(self._clamp(s.pct_1rm_sesion[0], 88.0, 92.0))
                s.reps_sesion[0]    = int(self._clamp(s.reps_sesion[0], 1, 3))
            if n >= 2:
                clamp_arr(1, 80.0, 87.0, 2, 4)                   # M sin singles
                if s.reps_sesion[1] == 1: s.reps_sesion[1] = 2
            if n >= 3:
                clamp_arr(2, 60.0, 72.0, 2, 3)                   # L velocidad
                if s.reps_sesion[2] == 1: s.reps_sesion[2] = 2

        self._enforce_separacion_roles(s, semana_1based)
        self._validar_coherencia_rir_rpe(s, semana_1based - 1)

    def _tope_singles_por_semana(self, s: SemanaPlan, semana_1based: int) -> None:
        """Limita singles por semana y sólo en H según fase."""
        self._normalizar_arrays_por_sesion(s); self._canonizar_orden_hml(s)
        n = len(s.distrib_series)
        if n == 0: 
            return
        iH = 0
        singles_sets_H = s.distrib_series[iH] if s.reps_sesion[iH] == 1 else 0
        singles_sets_M = (s.distrib_series[1] if n >= 2 and s.reps_sesion[1] == 1 else 0)
        singles_sets_L = (s.distrib_series[2] if n >= 3 and s.reps_sesion[2] == 1 else 0)

        def eliminar_singles(i, pct_cap=85.0):
            if i is None: return
            if s.reps_sesion[i] == 1:
                s.reps_sesion[i] = 2
                s.pct_1rm_sesion[i] = min(s.pct_1rm_sesion[i], pct_cap)

        if 1 <= semana_1based <= 8:
            # Máx 1 single set total y sólo en H
            eliminar_singles(1, 84.0)
            eliminar_singles(2, 80.0 if n >= 3 else 84.0)
            if s.reps_sesion[iH] == 1 and s.distrib_series[iH] > 1:
                s.distrib_series[iH] = 1
        elif 9 <= semana_1based <= 10:
            # 1–3 singles y sólo en H
            eliminar_singles(1, 85.0); eliminar_singles(2, 82.0 if n >=3 else 85.0)
            if s.reps_sesion[iH] == 1:
                s.distrib_series[iH] = int(self._clamp(s.distrib_series[iH], 1, 3))
        elif semana_1based == 11:
            # 2–4 singles en H
            eliminar_singles(1, 85.0); eliminar_singles(2, 82.0 if n >=3 else 85.0)
            if s.reps_sesion[iH] == 1:
                s.distrib_series[iH] = int(self._clamp(s.distrib_series[iH], 2, 4))
        else:  # 12
            # 1–2 top singles H; M y L sin singles (ligeros/velocidad)
            eliminar_singles(1, 75.0); eliminar_singles(2, 70.0 if n >=3 else 75.0)
            if s.reps_sesion[iH] == 1:
                s.distrib_series[iH] = int(self._clamp(s.distrib_series[iH], 1, 2))

        s.series = int(sum(s.distrib_series))
        self._validar_coherencia_rir_rpe(s, semana_1based - 1)

    def _taper_real_w12(self, plan: Plan12Semanas) -> None:
        """S12: Volumen 40–60% de S10; H con 1–2 singles 90–92; M 70–75×2–3; L 60–70×2–3."""
        if len(plan.semanas) < 12:
            return
        w10 = plan.semanas[9]
        w12 = plan.semanas[11]
        self._normalizar_arrays_por_sesion(w12); self._canonizar_orden_hml(w12)

        # Volumen objetivo vs S10
        objetivo = int(round(w10.series * 0.65))
        objetivo = int(self._clamp(objetivo, int(round(w10.series * 0.40)), int(round(w10.series * 0.60))))
        self._redistribuir_a_total(w12, objetivo)

        n = len(w12.distrib_series)
        order = list(range(n))  # ya canónico: 0=H
        iH = 0
        iM = 1 if n >= 2 else None
        iL = 2 if n >= 3 else (1 if n == 2 else None)

        # H: top singles (1–2) @90–92, RPE 8.0–8.5
        w12.pct_1rm_sesion[iH] = float(self._clamp(w12.pct_1rm_sesion[iH], 90.0, 92.0))
        w12.reps_sesion[iH]    = 1
        w12.rpe_sesion[iH]     = self._rpe_discreto(self._clamp(w12.rpe_sesion[iH], 8.0, 8.5))
        w12.distrib_series[iH] = int(self._clamp(w12.distrib_series[iH], 1, 2))

        # M: 70–75 ×2–3, RPE 6.5–7.5
        if iM is not None:
            w12.pct_1rm_sesion[iM] = float(self._clamp(w12.pct_1rm_sesion[iM], 70.0, 75.0))
            w12.reps_sesion[iM]    = int(self._clamp(w12.reps_sesion[iM], 2, 3))
            w12.rpe_sesion[iM]     = self._rpe_discreto(self._clamp(w12.rpe_sesion[iM], 6.5, 7.5))
            # sin singles
            if w12.reps_sesion[iM] == 1: w12.reps_sesion[iM] = 2

        # L: 60–70 ×2–3, RPE 6–7 (velocidad)
        if iL is not None:
            w12.pct_1rm_sesion[iL] = float(self._clamp(w12.pct_1rm_sesion[iL], 60.0, 70.0))
            w12.reps_sesion[iL]    = int(self._clamp(w12.reps_sesion[iL], 2, 3))
            w12.rpe_sesion[iL]     = self._rpe_discreto(self._clamp(w12.rpe_sesion[iL], 6.0, 7.0))
            if w12.reps_sesion[iL] == 1: w12.reps_sesion[iL] = 2

        w12.series = int(sum(w12.distrib_series))
        # Agregados
        if w12.distrib_series and w12.pct_1rm_sesion:
            w12.pct_1rm = float(self._pct_semana_ponderado(w12))
        if w12.distrib_series and w12.rpe_sesion:
            w12.rpe = self._rpe_discreto(
                sum(r * sets for r, sets in zip(w12.rpe_sesion, w12.distrib_series)) / max(1, w12.series)
            )
        if w12.distrib_series and w12.reps_sesion:
            w12.reps = int(round(
                sum(sets * reps for sets, reps in zip(w12.distrib_series, w12.reps_sesion)) / max(1, w12.series)
            ))
        setattr(w12, "top_single_objetivo", True)

    

    # ─────────────────────────────────────────────────────────────────────────────
    # Penalizaciones (para tu fitness) y operadores GA sugeridos
    # ─────────────────────────────────────────────────────────────────────────────
    def _penalizaciones_hard_rules(self, plan: Plan12Semanas) -> dict:
        """
        Calcula violaciones para que las ponderes en tu fitness.
        dev: {'trend': x, 'micro': y, 'roles': z, 'singles': k}
        """
        pen = {"trend": 0.0, "micro": 0.0, "roles": 0.0, "singles": 0.0}
        # Trend: deltas por rol
        for w in range(1, len(plan.semanas)):
            prev = plan.semanas[w - 1]; cur = plan.semanas[w]
            self._normalizar_arrays_por_sesion(prev); self._canonizar_orden_hml(prev)
            self._normalizar_arrays_por_sesion(cur);  self._canonizar_orden_hml(cur)
            caps = self._caps_por_fase(w + 1)
            roles = ["H","M"] + (["L"] if len(cur.distrib_series)==3 else [])
            for i, rol in enumerate(roles):
                max_pp = caps["pct"][rol]
                dpp = abs(cur.pct_1rm_sesion[i] - prev.pct_1rm_sesion[i]) - max_pp
                if dpp > 0: pen["trend"] += dpp
                if caps["sets"]:
                    dsets = abs(cur.distrib_series[i] - prev.distrib_series[i]) - caps["sets"]
                    if dsets > 0: pen["trend"] += dsets * 0.5
                drpe = abs(cur.rpe_sesion[i] - prev.rpe_sesion[i]) - caps["rpe"]
                if drpe > 0: pen["trend"] += drpe
        # Roles: gaps mínimos
        for w_idx, s in enumerate(plan.semanas, start=1):
            self._normalizar_arrays_por_sesion(s); self._canonizar_orden_hml(s)
            if len(s.distrib_series) >= 2:
                if (s.pct_1rm_sesion[0] - s.pct_1rm_sesion[1]) < 6.0: pen["roles"] += 1.0
            if len(s.distrib_series) == 3:
                if (s.pct_1rm_sesion[1] - s.pct_1rm_sesion[2]) < 6.0: pen["roles"] += 1.0
            # Singles indebidos M/L o %≥88 en M/L
            if len(s.distrib_series) >= 2:
                if s.reps_sesion[1] == 1 or s.pct_1rm_sesion[1] >= 88.0: pen["singles"] += 2.0
            if len(s.distrib_series) == 3:
                if s.reps_sesion[2] == 1 or s.pct_1rm_sesion[2] >= 88.0: pen["singles"] += 2.0
        return pen

  

    def _normalizar_arrays_por_sesion(self, s: SemanaPlan) -> None:
        """
        Alinea longitudes de distrib_series, pct_1rm_sesion, rpe_sesion y reps_sesion.
        Actualiza s.series = sum(distrib_series).
        """
       
        if not s.distrib_series:
            return
        target = len(s.distrib_series)

        s.pct_1rm_sesion = list(s.pct_1rm_sesion or [])
        s.rpe_sesion = list(s.rpe_sesion or [])
        s.reps_sesion = list(s.reps_sesion or [])

        while len(s.pct_1rm_sesion) < target:
            s.pct_1rm_sesion.append(float(s.pct_1rm))
        while len(s.rpe_sesion) < target:
            s.rpe_sesion.append(self._rpe_discreto(float(s.rpe)))
        while len(s.reps_sesion) < target:
            s.reps_sesion.append(int(s.reps))

        s.pct_1rm_sesion = s.pct_1rm_sesion[:target]
        s.rpe_sesion = s.rpe_sesion[:target]
        s.reps_sesion = s.reps_sesion[:target]
        # >>> NUEVO: dejar todo canónico antes de salir
        self._canonizar_orden_hml(s)
        self._ordenar_rpe_por_rol(s)

        s.series = int(sum(s.distrib_series))
      

    # >>> NUEVO
    def _capar_salto_interbloque_H(self, plan: Plan12Semanas, w_from: int, w_to: int,
                                max_delta_pct_h: float = 8.0,
                                max_delta_rpe_h: float = 0.5) -> None:
        """Limita cuánto puede subir el día H (por sesión) al cruzar de bloque."""
        if not (1 <= w_from <= len(plan.semanas) and 1 <= w_to <= len(plan.semanas)):
            return
        a = plan.semanas[w_from - 1]
        b = plan.semanas[w_to - 1]
        self._normalizar_arrays_por_sesion(a)
        self._normalizar_arrays_por_sesion(b)
        self._canonizar_orden_hml(a)
        self._canonizar_orden_hml(b)
        iH_a = 0
        iH_b = 0
        if (b.pct_1rm_sesion[iH_b] - a.pct_1rm_sesion[iH_a]) > max_delta_pct_h:
            b.pct_1rm_sesion[iH_b] = a.pct_1rm_sesion[iH_a] + max_delta_pct_h
        if (b.rpe_sesion[iH_b] - a.rpe_sesion[iH_a]) > max_delta_rpe_h:
            b.rpe_sesion[iH_b] = self._rpe_discreto(a.rpe_sesion[iH_a] + max_delta_rpe_h)
        self._forzar_minimos_y_gaps_por_rol(b)
        self._validar_coherencia_rir_rpe(b)
        self._canonizar_orden_hml(b)

    def _forzar_minimos_y_gaps_por_rol(self, s: SemanaPlan) -> None:
        """
        Aplica mínimos por sesión y gaps de intensidad.
        - 2 sesiones → roles válidos {H, M}; nunca {H, H} ni 'L'.
        - 3 sesiones → garantizar {H, M, L}.
        - Mínimos de series: L>=3, M>=4, H>=3 (en peaking H>=2).
        - Gaps %: H ≥ M + 5 pp; M ≥ L + 5 pp.
        - Orden por %: 2S: H>M ; 3S: H>M>L.
        - Clamps realistas sin test.
        """
        if not (s.distrib_series and s.series > 0):
            return

        s.pct_1rm_sesion = list(s.pct_1rm_sesion or [])
        s.rpe_sesion = list(s.rpe_sesion or [])
        s.reps_sesion = list(s.reps_sesion or [])

        n = len(s.distrib_series)
        while len(s.pct_1rm_sesion) < n:
            s.pct_1rm_sesion.append(float(s.pct_1rm))
        while len(s.rpe_sesion) < n:
            s.rpe_sesion.append(self._rpe_discreto(float(s.rpe)))
        while len(s.reps_sesion) < n:
            s.reps_sesion.append(int(s.reps))

        s.pct_1rm_sesion = s.pct_1rm_sesion[:n]
        s.rpe_sesion = s.rpe_sesion[:n]
        s.reps_sesion = s.reps_sesion[:n]
        s.series = int(sum(s.distrib_series))

        shares = [x / max(1, s.series) for x in s.distrib_series]
        roles = [self._rol_por_share(sh) for sh in shares]

        if s.sesiones == 2:
            for i, r in enumerate(roles):
                if r == "L":
                    roles[i] = "M"
            if roles.count("H") > 1:
                idxs_h = [i for i, r in enumerate(roles) if r == "H"]
                idx_bajar = min(idxs_h, key=lambda i: s.pct_1rm_sesion[i])
                roles[idx_bajar] = "M"
        elif s.sesiones == 3:
            if "L" not in roles:
                roles[shares.index(min(shares))] = "L"
            if "H" not in roles:
                roles[shares.index(max(shares))] = "H"
            for i in range(3):
                if roles[i] not in ("H", "M", "L"):
                    roles[i] = "M"

        def min_series_por_rol(rol: str) -> int:
            if rol == "L":
                return 3
            if rol == "M":
                return 4
            return 2 if s.pct_1rm >= 85.0 else 3  # H

        for i, rol in enumerate(roles):
            minimo = min(min_series_por_rol(rol), s.series)
            if s.distrib_series[i] < minimo:
                deficit = minimo - s.distrib_series[i]
                donantes = sorted(
                    [j for j in range(n) if j != i],
                    key=lambda j: s.distrib_series[j],
                    reverse=True,
                )
                for j in donantes:
                    if deficit <= 0:
                        break
                    mov = min(deficit, max(0, s.distrib_series[j] - 1))
                    if mov > 0:
                        s.distrib_series[j] -= mov
                        s.distrib_series[i] += mov
                        deficit -= mov

        s.series = int(sum(s.distrib_series))
        shares = [x / max(1, s.series) for x in s.distrib_series]

        val = {"H": [], "M": [], "L": []}
        for i, rol in enumerate(roles):
            val[rol].append((i, s.pct_1rm_sesion[i]))

        def set_pct(idx: int, objetivo: float):
            s.pct_1rm_sesion[idx] = float(min(100.0, max(50.0, objetivo)))

        if val["H"] and val["M"]:
            iH, pctH = max(val["H"], key=lambda x: x[1])
            iM, pctM = max(val["M"], key=lambda x: x[1])
            if pctH < pctM + 5.0:
                set_pct(iH, pctM + 5.0)
        if val["M"] and val["L"]:
            iM2, pctM2 = min(val["M"], key=lambda x: x[1])
            iL, pctL = max(val["L"], key=lambda x: x[1])
            if pctM2 < pctL + 5.0:
                set_pct(iM2, pctL + 5.0)

        idxs_orden_pct = sorted(range(n), key=lambda i: s.pct_1rm_sesion[i], reverse=True)
        if s.sesiones == 2:
            i_alto, i_bajo = idxs_orden_pct[0], idxs_orden_pct[1]
            if roles[i_alto] != "H" or roles[i_bajo] != "M":
                roles[i_alto], roles[i_bajo] = "H", "M"
        elif s.sesiones == 3:
            iH, iM, iL = idxs_orden_pct[0], idxs_orden_pct[1], idxs_orden_pct[2]
            roles[iH], roles[iM], roles[iL] = "H", "M", "L"

        # Clamps realistas: sin test no permitir ≥99% con RPE bajo o múltiples sets
        for i in range(n):
            pct, rpe, reps = s.pct_1rm_sesion[i], s.rpe_sesion[i], s.reps_sesion[i]
            if rpe <= 6.5 and reps <= 2 and pct >= 99.0:
                set_pct(i, 95.0)

            base = s.rpe_sesion[i]
            min_h, min_m, min_l = 6.5, 6.0, 6.0
            if s.pct_1rm >= 75.0:
                min_h, min_m = 7.0, 6.5
            if s.pct_1rm >= 85.0:
                min_h, min_m, min_l = 7.5, 7.0, 6.5

            if roles[i] == "H":
                base = max(base, min_h)
            elif roles[i] == "M":
                base = max(base, min_m)
            else:
                base = max(base, min_l)
            s.rpe_sesion[i] = self._rpe_discreto(base)

        for i in range(n):
            if roles[i] == "H":
                s.reps_sesion[i] = max(1, int(s.reps_sesion[i]) - 1)
            elif roles[i] == "L":
                s.reps_sesion[i] = min(30, int(s.reps_sesion[i]) + 1)

        s.pct_1rm_sesion = [float(min(100.0, max(50.0, x))) for x in s.pct_1rm_sesion]
        s.rpe_sesion = [self._rpe_discreto(float(x)) for x in s.rpe_sesion]
        s.reps_sesion = [int(max(1, min(30, x))) for x in s.reps_sesion]
        s.series = int(sum(s.distrib_series))

    def _distribuir_series(self, total_series: int, numero_sesiones: int) -> List[int]:
        """
        EvolucionEntrenamiento - _distribuir_series:
        Reparte un número total de series entre las sesiones de la semana respetando patrones H/M/L.

        Reglas operativas:
        - Si hay 3 sesiones (H/M/L), se elige aleatoriamente un patrón razonable:
            * 50/33/17 (clásico H/M/L)
            * 40/40/20 (más equilibrado)
            * 34/33/33 (casi igualitario)
            * 45/35/20 (ligeramente cargado al día H)
        - Si hay 2 sesiones, patrones típicos:
            * 60/40, 55/45, 62/38, 58/42 (con pesos de elección sesgados a 60/40 y 55/45)
        - Para otros casos (poco comunes), se reparte de forma equitativa.

        Detalles del algoritmo:
        1) Se calcula la “asignación base” como floor(total_series * proporción).
        2) Se reparte el residuo (las series que faltan para alcanzar total_series)
            priorizando las sesiones con mayor déficit ideal (ideal - base).
        3) Si total_series >= numero_sesiones, se impone mínimo de 1 serie por sesión.
            Si al imponer mínimos nos pasamos, se resta desde las sesiones más cargadas.

        Retorna:
        Lista con las series por sesión en el orden definido por el patrón elegido.
        Para 3 sesiones se asume orden [H, M, L]; para 2 sesiones [H, M].
        """

        

        # 1) Elegir las proporciones objetivo según número de sesiones
        if numero_sesiones == 3:
            proporciones_posibles = [
                [0.50, 0.33, 0.17],  # H/M/L clásico
                [0.40, 0.40, 0.20],  # más equilibrado
                [0.34, 0.33, 0.33],  # casi igualitario
                [0.45, 0.35, 0.20],  # sesgo a H
            ]
            proporciones = random.choice(proporciones_posibles)
        elif numero_sesiones == 2:
            opciones = [[0.60, 0.40], [0.55, 0.45], [0.62, 0.38], [0.58, 0.42]]
            # Pesos priorizan 60/40 y 55/45
            proporciones = random.choices(population=opciones, weights=[0.45, 0.35, 0.10, 0.10], k=1)[0]
        else:
            # Caso genérico: repartir equitativamente
            proporciones = [1.0 / max(1, numero_sesiones)] * max(1, numero_sesiones)

        # 2) Asignación base por redondeo hacia abajo
        series_por_sesion = [int(total_series * prop) for prop in proporciones]
        suma_base = sum(series_por_sesion)
        residuo_por_asignar = total_series - suma_base

        # 3) Cálculo de déficits ideales para decidir dónde colocar el residuo
        #    déficit_i = (ideal_i) - (asignación_base_i), donde ideal_i = total_series * proporción_i
        deficits_ordenados = sorted(
            [(i, (total_series * proporciones[i]) - series_por_sesion[i]) for i in range(numero_sesiones)],
            key=lambda t: t[1],
            reverse=True,
        )

        # 4) Repartir el residuo entregando 1 serie a la vez a quien tenga mayor déficit
        idx_deficit = 0
        while residuo_por_asignar > 0 and deficits_ordenados:
            indice_sesion = deficits_ordenados[idx_deficit][0]
            series_por_sesion[indice_sesion] += 1
            residuo_por_asignar -= 1
            idx_deficit = (idx_deficit + 1) % len(deficits_ordenados)


        # 5) Garantizar mínimo de 1 serie por sesión cuando haya suficientes series totales
        if total_series >= numero_sesiones:
            hubo_ajuste_minimos = False
            for i in range(numero_sesiones):
                if series_por_sesion[i] < 1:
                    series_por_sesion[i] = 1
                    hubo_ajuste_minimos = True
            if hubo_ajuste_minimos:
                # Si imponer mínimos nos pasó del total, recortar desde las sesiones más cargadas
                exceso = sum(series_por_sesion) - total_series
                while exceso > 0:
                    indice_mas_cargado = max(range(numero_sesiones), key=lambda k: series_por_sesion[k])
                    if series_por_sesion[indice_mas_cargado] > 1:
                        series_por_sesion[indice_mas_cargado] -= 1
                        exceso -= 1
                    else:
                        # Si todas están en 1, no se puede recortar más sin romper el mínimo.
                        break

        print(f"EvolucionEntrenamiento - _distribuir_series: distribución final={series_por_sesion}. Devolviendo resultado.")
        return series_por_sesion

    def _reajustar_distrib(self, distrib: List[int], series: int, sesiones: int) -> List[int]:
        """Reescala una distribución existente al nuevo total/num de sesiones manteniendo forma aproximada."""
        if not distrib or len(distrib) != sesiones:
            return self._distribuir_series(series, sesiones)
        total_prev = sum(distrib)
        if total_prev <= 0:
            return self._distribuir_series(series, sesiones)

        proporciones = [x / total_prev for x in distrib]
        base = [int(round(series * p)) for p in proporciones]
        delta = series - sum(base)
        orden = sorted(range(sesiones), key=lambda i: proporciones[i], reverse=True)
        idx = 0
        while delta != 0:
            k = orden[idx % sesiones]
            if delta > 0:
                base[k] += 1
                delta -= 1
            else:
                if base[k] > 0:
                    base[k] -= 1
                    delta += 1
            idx += 1

        if series >= sesiones:
            for j in range(sesiones):
                if base[j] < 1:
                    base[j] = 1
            exceso = sum(base) - series
            while exceso > 0:
                idx_max = max(range(sesiones), key=lambda k: base[k])
                if base[idx_max] > 1:
                    base[idx_max] -= 1
                    exceso -= 1
                else:
                    break
                
        if sesiones == 2 and series > 0:
            a, b = base
            # Si AMBAS cuotas > 45% (ej. 10/10 con 20 sets), fuerza una a 45%
            if (a / series) > 0.45 and (b / series) > 0.45:
                # baja la menor de las dos (o la primera si iguales)
                if a <= b and a > 0:
                    a -= 1; b += 1
                elif b < a and b > 0:
                    b -= 1; a += 1
                base = [a, b]


        return base

    def _semana_aleatoria(self, semana_idx: int) -> SemanaPlan:
        """Genera una semana aleatoria guiada por bloque y H/M/L (rangos actualizados)."""

        if semana_idx in self.BLOQUES[1]["idx"]:
            sesiones = random.choice([2, 3])
            series = random.randint(12, 18)
            pct_1rm = random.uniform(60, 75)
            reps = random.randint(10, 15)
        elif semana_idx in self.BLOQUES[2]["idx"]:
            sesiones = random.choice([2, 3])
            series = random.randint(10, 14)
            pct_1rm = random.uniform(75, 85)
            reps = random.randint(6, 10)
        else:
            sesiones = random.choice([2, 2, 3])  # sesgo a 2 por SNC
            series = random.randint(6, 10)
            pct_1rm = random.uniform(85, 95)
            reps = random.randint(1, 6)

        # Semilla de RPE más real
        rpe = self._rpe_base_semana(semana_idx)

        distrib = self._distribuir_series(series, sesiones)
        sem = SemanaPlan(
            sesiones,
            series,
            round(pct_1rm, 1),
            float(rpe),
            int(reps),
            distrib
        )

       
        self._generar_por_sesion(sem, semana_idx)

        return sem


    def _individuo_aleatorio(self) -> Plan12Semanas:
        semanas = [self._semana_aleatoria(i) for i in range(12)]
        if hasattr(self, "_ajustar_rampa_interbloque"):
            self._ajustar_rampa_interbloque(semanas)
        if hasattr(self, "_ajustar_deload_semana"):
            self._ajustar_deload_semana(semanas)


        return Plan12Semanas(self.patron, semanas)

    def _inicializar_poblacion(self) -> List[Plan12Semanas]:
        pobl: List[Plan12Semanas] = [self._individuo_aleatorio() for _ in range(self.poblacion)]

        # Post: reparaciones por fase/día/DUP antes del primer fitness
        for idx, ind in enumerate(pobl, start=1):
            print(f"EvolucionEntrenamiento - _inicializar_poblacion: reparando individuo {idx}/{self.poblacion}.")
            try:
                # Reparación global (H≥M≥L, gaps, velocidad, taper, etc.)
                self._reparar_y_marcar_bloques(ind)
            except Exception:
                # Fallback mínimo a nivel de semanas
                for w_idx, s in enumerate(ind.semanas, start=1):
                    try:
                        # Alinea arrays por sesión y corrige incoherencias básicas
                        self._normalizar_arrays_por_sesion(s)
                        self._forzar_minimos_y_gaps_por_rol(s)
                        self._validar_coherencia_rir_rpe(s, w_idx - 1)  # usa 0-based

                        
                    except Exception:
                        # Si algo falla en la semana, al menos sincroniza series
                        pass
                    # Sincroniza series con la distribución si hiciera falta
                    if s.distrib_series:
                        s.series = int(sum(s.distrib_series))

            # Asegura consistencia final del plan
            for s in ind.semanas:
                if s.distrib_series and sum(s.distrib_series) != int(s.series):
                    s.series = int(sum(s.distrib_series))

        return pobl

   
    # ─────────────────────────────────────────────────────────────────────────────
    # HIPERTROFIA (W1–W4): +intensidad suave, −reps suave, volumen ligeramente ↓
    # ─────────────────────────────────────────────────────────────────────────────
    def _aplicar_microtrends_hipertrofia(self, plan: Plan12Semanas) -> None:
        semanas = plan.semanas[0:4]
        if len(semanas) < 4:
            return

        # Objetivos por rol a lo largo del bloque (tendencias suaves)
        targets_pct = {
            "H": [70.0, 72.0, 74.0, 75.0],
            "M": [65.0, 66.5, 68.0, 70.0],
            "L": [60.0, 61.5, 63.0, 65.0],
        }
        targets_reps = {
            "H": [8, 8, 7, 6],
            "M": [10, 9, 9, 8],
            "L": [12, 11, 11, 10],
        }
        # Volumen total (series) levemente decreciente
        target_series = [16, 15, 14, 13]

        for i, s in enumerate(semanas):
            self._ajustar_roles_a_objetivos(s, targets_pct, targets_reps, semana_idx=i+1, fase="hipertrofia")
            # Suaviza series hacia target (±1 o ±2 como mucho)
            self._ajustar_series_hacia(s, target_series[i], max_mov=2)

    # ─────────────────────────────────────────────────────────────────────────────
    # FUERZA (W5–W8): DUP garantizado, % ascendente, reps descendentes (L 4–6)
    # ─────────────────────────────────────────────────────────────────────────────
    def _aplicar_microtrends_fuerza(self, plan: Plan12Semanas) -> None:
        semanas = plan.semanas[4:8]
        if len(semanas) < 4:
            return

        # >>> CAMBIO: subir H claramente cada semana; M y L acompañan
        targets_pct = {
            "H": [84.0, 86.0, 88.0, 90.0],
            "M": [78.0, 80.0, 81.0, 82.0],
            "L": [72.0, 74.0, 76.0, 78.0],
        }
        targets_reps = {
            "H": [4, 3, 3, 3],
            "M": [5, 4, 4, 3],
            "L": [6, 5, 5, 4],
        }
        target_series = [13, 12, 12, 11]

        for i, s in enumerate(semanas, start=5):
            self._forzar_minimos_y_gaps_por_rol(s)
            self._ajustar_roles_a_objetivos(s, targets_pct, targets_reps, semana_idx=i, fase="fuerza")
            self._ajustar_series_hacia(s, target_series[i-5], max_mov=2)
            # >>> NUEVO: asegurar orden canónico y rpe por rol
            self._canonizar_orden_hml(s)
            self._ordenar_rpe_por_rol(s)


    def _aplicar_microtrends_pico_y_taper(self, plan: Plan12Semanas) -> None:
        # PICO W9–W11
        semanas_pico = plan.semanas[8:11]
        if len(semanas_pico) == 3:
            targets_pct_pico = {
                "H": [87.0, 89.0, 91.0],
                "M": [85.0, 87.0, 88.5],
                "L": [60.0, 62.0, 65.0],
            }

            targets_reps_pico = {
                "H": [3, 2, 2],   # 4×3→3×2 implícito con series
                "M": [2, 2, 1],   # 5×2→4×1–2
                "L": [2, 2, 2],   # 6×2 @60–65 (se fija abajo)
            }
            target_series_pico = [10, 8, 7]

            for k, s in enumerate(semanas_pico, start=9):
                self._ajustar_roles_a_objetivos(s, targets_pct_pico, targets_reps_pico, semana_idx=k, fase="pico")
                self._fijar_velocidad_L(s, prefer_sets=6)           # L = 6×2 @60–65, RPE 6–7
                self._ajustar_series_hacia(s, target_series_pico[k-9], max_mov=2)
                self._forzar_minimos_y_gaps_por_rol(s)
                self._validar_coherencia_rir_rpe(s, k-1)

        # TAPER W12 con top single
        if len(plan.semanas) >= 12:
            w12 = plan.semanas[11]
            w11 = plan.semanas[10]

            objetivo_series = int(round(w11.series * 0.50))  # dentro de 40–60%
            self._redistribuir_a_total(w12, objetivo_series)
            self._ajustar_sesiones_taper(w12)

            try:
                self._normalizar_arrays_por_sesion(w12)
                n = w12.sesiones
                order = sorted(range(n), key=lambda j: w12.pct_1rm_sesion[j], reverse=True)
                iH = order[0]
                iM = order[1] if n >= 2 else None
                iL = order[2] if n == 3 else (order[-1] if n == 2 else None)

               # >>> CAMBIO: H: top single explícito (2–3 × 1 @90–92, RPE 8.0–8.5)
                w12.pct_1rm_sesion[iH] = float(min(92.0, max(90.0, w12.pct_1rm_sesion[iH])))
                w12.reps_sesion[iH]    = 1
                w12.rpe_sesion[iH]     = self._rpe_discreto(min(8.5, max(8.0, w12.rpe_sesion[iH])))
                w12.distrib_series[iH] = min(3, max(2, int(w12.distrib_series[iH])))


                # M: 0–2 × 1 @85–88 (nada de triples)
                if iM is not None:
                    w12.pct_1rm_sesion[iM] = float(min(88.0, max(85.0, w12.pct_1rm_sesion[iM])))
                    w12.reps_sesion[iM]    = 1
                    w12.rpe_sesion[iM]     = self._rpe_discreto(min(8.5, max(7.5, w12.rpe_sesion[iM])))
                    w12.distrib_series[iM] = min(2, max(0, int(w12.distrib_series[iM])))

                # L opcional: 3×2 @60–65, RPE 6–7
                if iL is not None:
                    w12.pct_1rm_sesion[iL] = float(min(65.0, max(60.0, w12.pct_1rm_sesion[iL])))
                    w12.reps_sesion[iL]    = 2
                    w12.rpe_sesion[iL]     = self._rpe_discreto(min(7.0, max(6.0, w12.rpe_sesion[iL])))
                    w12.distrib_series[iL] = min(3, max(0, int(w12.distrib_series[iL])))

                w12.series = int(sum(w12.distrib_series))
                if w12.distrib_series and w12.pct_1rm_sesion:
                    tot = max(1, sum(w12.distrib_series))
                    w12.pct_1rm = float(sum(p * sets for p, sets in zip(w12.pct_1rm_sesion, w12.distrib_series)) / tot)
                w12.rpe = self._rpe_discreto(
                    sum(r * sets for r, sets in zip(w12.rpe_sesion, w12.distrib_series)) / max(1, w12.series)
                )
                w12.reps = int(round(
                    sum(sets * reps for sets, reps in zip(w12.distrib_series, w12.reps_sesion)) / max(1, w12.series)
                ))
                setattr(w12, "top_single_objetivo", True)
            except Exception:
                pass



    
    def _sync_semana(self, s: SemanaPlan, semana_idx: int | None = None) -> None:

        """
        Sincroniza los agregados semanales con los arrays por sesión.
        - Recalcula s.series = sum(distrib_series)
        - Actualiza %1RM, RPE y reps semanales como medias ponderadas por sets
        """
        # Alinea longitudes y actualiza s.series a la suma de distrib_series
        self._normalizar_arrays_por_sesion(s)
        if s.distrib_series:
            s.series = int(sum(s.distrib_series))

        # % semanal ponderado por sets (si hay arrays)
        if s.distrib_series and s.pct_1rm_sesion:
            s.pct_1rm = float(self._pct_semana_ponderado(s))

        # RPE semanal ponderado por sets (cuantizado)
        if s.distrib_series and s.rpe_sesion:
            s.rpe = self._rpe_discreto(
                sum(r * sets for r, sets in zip(s.rpe_sesion, s.distrib_series)) / max(1, s.series)
            )

        # Reps semanales como media ponderada por sets
        if s.distrib_series and s.reps_sesion:
            s.reps = int(round(
                sum(sets * reps for sets, reps in zip(s.distrib_series, s.reps_sesion)) / max(1, s.series)
            ))
        # >>> NUEVO
        self._canonizar_orden_hml(s)
        self._ordenar_rpe_por_rol(s)



    # ─────────────────────────────────────────────────────────────────────────────
    # Helper: Empuja hacia objetivos por rol, asegura H≥M≥L en % y H≤M≤L en reps.
    # En Pico W9–W10 fuerza “no singles” (reps ≥ 2) en H/M.
    # ─────────────────────────────────────────────────────────────────────────────
    def _ajustar_roles_a_objetivos(self, s, targets_pct, targets_reps, semana_idx: int, fase: str):
        n = int(getattr(s, "sesiones", 0) or 0)
        if n <= 0:
            return

        # Roles por intensidad actual (fallback por cuota si hiciera falta)
        try:
            roles = self._roles_hml_por_pct(list(s.pct_1rm_sesion or []))
            if not roles or len(roles) != n:
                raise ValueError
        except Exception:
            total = max(1, s.series)
            roles = [self._rol_por_share((s.distrib_series[j] if j < len(s.distrib_series) else 0)/total) for j in range(n)]

        pct  = list(s.pct_1rm_sesion or [s.pct_1rm] * n)
        rpe  = list(s.rpe_sesion     or [s.rpe] * n)
        reps = list(s.reps_sesion    or [s.reps] * n)

        pos = (semana_idx - 1) % 4  # 0..3

        for j, rol in enumerate(roles):
            # Día de velocidad real en Pico para L
            if fase == "pico" and self._es_dia_velocidad(fase, rol, s) and rol == "L":
                pct[j]  = min(65.0, max(60.0, pct[j]))
                reps[j] = 2
                rpe[j]  = self._rpe_discreto(min(7.0, max(6.0, rpe[j])))
            else:
                # ========= ASIGNACIÓN DIRECTA A LOS OBJETIVOS =========
                pct_obj  = float(targets_pct[rol][pos])
                reps_obj = int(targets_reps[rol][pos])

                pct[j]  = pct_obj
                reps[j] = reps_obj

                # Antisingles en Pico (W9–W10) para H/M (no velocidad)
                if fase == "pico" and pos in (0, 1) and rol in ("H", "M"):
                    reps[j] = max(2, reps[j])

        # Escribe y normaliza
        s.pct_1rm_sesion = [float(min(100.0, max(50.0, x))) for x in pct]
        s.rpe_sesion     = [self._rpe_discreto(float(x)) for x in rpe]
        s.reps_sesion    = [int(max(1, min(30, x))) for x in reps]

        # Asegura H≥M≥L en % y H≤M≤L en reps
        self._normalizar_arrays_por_sesion(s)
        self._forzar_minimos_y_gaps_por_rol(s)  # % ordenado y gaps

        # Re-impone identidad por reps (H ≤ M ≤ L)
        try:
            roles_now = self._roles_hml_por_pct(s.pct_1rm_sesion)
            if "H" in roles_now and "M" in roles_now:
                iH = roles_now.index("H")
                iM = roles_now.index("M")
                if s.reps_sesion[iH] > s.reps_sesion[iM]:
                    s.reps_sesion[iH] = max(1, s.reps_sesion[iM] - 1)
            if "L" in roles_now and "M" in roles_now:
                iL = roles_now.index("L")
                iM = roles_now.index("M")
                if s.reps_sesion[iM] > s.reps_sesion[iL]:
                    s.reps_sesion[iL] = max(s.reps_sesion[iM], s.reps_sesion[iL])
        except Exception:
            pass

        # Coherencia %–reps–RPE y sincronización
        self._validar_coherencia_rir_rpe(s)
        if s.distrib_series and sum(s.distrib_series) > 0:
            tot = sum(s.distrib_series)
            s.reps = int(round(sum(sets * rr for sets, rr in zip(s.distrib_series, s.reps_sesion)) / tot))



    # ─────────────────────────────────────────────────────────────────────────────
    # Helper: fija día L de velocidad (60–65% x2) y busca 6×2 si cabe en el total.
    # Mantiene al menos 1 set para H y 1 para M.
    # ─────────────────────────────────────────────────────────────────────────────
    def _fijar_velocidad_L(self, s: SemanaPlan, prefer_sets: int = 6) -> None:
        if not (s.distrib_series and s.series > 0 and s.sesiones in (2, 3)):
            return
        self._normalizar_arrays_por_sesion(s)
        n = s.sesiones
        order = sorted(range(n), key=lambda j: s.pct_1rm_sesion[j], reverse=True)
        iL = order[2] if n == 3 else order[-1]

        # Ajustes de velocidad
        s.pct_1rm_sesion[iL] = float(min(65.0, max(60.0, s.pct_1rm_sesion[iL])))
        s.reps_sesion[iL]    = 2
        s.rpe_sesion[iL]     = self._rpe_discreto(6.5)

        # Buscar 6×2 si el total lo permite (mínimo 4×2)
        total = s.series
        min_hm = 2 if n >= 3 else 1  # deja sets para H y M
        max_L_posible = max(0, total - 2 * min_hm)
        objetivo_L = max(4, min(prefer_sets, max_L_posible))
        delta = objetivo_L - s.distrib_series[iL]
        if delta > 0:
            # roba de H y M (del más cargado primero)
            donors = sorted([j for j in order if j != iL], key=lambda j: s.distrib_series[j], reverse=True)
            k = 0
            while delta > 0 and k < len(donors):
                j = donors[k]
                mov = min(delta, max(0, s.distrib_series[j] - 1))
                if mov > 0:
                    s.distrib_series[j] -= mov
                    s.distrib_series[iL] += mov
                    delta -= mov
                k += 1
        elif delta < 0:
            # devuelve exceso a H/M
            receivers = sorted([j for j in order if j != iL], key=lambda j: s.distrib_series[j])
            while delta < 0 and receivers:
                j = receivers.pop(0)
                s.distrib_series[j] += 1
                s.distrib_series[iL] -= 1
                delta += 1

        s.series = int(sum(s.distrib_series))
        self._forzar_minimos_y_gaps_por_rol(s)
        self._validar_coherencia_rir_rpe(s)

    def _reparar_y_marcar_bloques(self, plan: Plan12Semanas) -> None:
        """
        Repara el plan tras generación/cruce/mutación para conservar
        tendencias por bloque y la coherencia H/M/L. Además marca la fase
        de cada semana.
        """
        if not plan or not getattr(plan, "semanas", None):
            return

        # 0) Pase inicial de saneo por semana
        for w_idx, s in enumerate(plan.semanas, start=1):
            try:
                self._normalizar_arrays_por_sesion(s)
                self._forzar_minimos_y_gaps_por_rol(s)

                # >>> NUEVO: brechas H–M–L y rangos por fase antes de validar
                self._enforce_role_separation_strict(s)
                self._enforce_phase_ranges(s, w_idx)

                # Coherencia RIR/RPE y marca de fase
                self._validar_coherencia_rir_rpe(s, w_idx - 1)  # usa 0-based
                fase = "hipertrofia" if w_idx <= 4 else ("fuerza" if w_idx <= 8 else ("pico" if w_idx <= 11 else "taper"))
                setattr(s, "fase", fase)

                # Orden canónico y RPE por rol
                self._canonizar_orden_hml(s)
                self._ordenar_rpe_por_rol(s)
            except Exception:
                pass

        # 1) Microtendencias por bloque
        try:
            self._aplicar_microtrends_hipertrofia(plan)     # W1–W4
        except Exception:
            pass
        try:
            self._aplicar_microtrends_fuerza(plan)          # W5–W8
        except Exception:
            pass
        try:
            self._aplicar_microtrends_pico_y_taper(plan)    # W9–W12
        except Exception:
            pass

        # 2) Capar rampas entre bloques (salto de % medio)
        try:
            self._capar_rampa_interbloque(plan, 4, 5, max_delta_pct=5.0)
        except Exception:
            pass
        try:
            self._capar_rampa_interbloque(plan, 8, 9, max_delta_pct=5.0)
        except Exception:
            pass

        # 3) Suavizar volumen intersemanal (±2 series)
        try:
            self._suavizar_series_intersemanal(plan, max_delta=2)
        except Exception:
            pass

        # 4) Taper final (W12 40–60% de W11 y parámetros)
        try:
            self._enforce_taper(plan)
        except Exception:
            pass

        # 4bis) Limitar salto del día H específico W4→W5
        try:
            self._capar_salto_interbloque_H(plan, 4, 5, max_delta_pct_h=8.0, max_delta_rpe_h=0.5)
        except Exception:
            pass

        # 5) Pase final de coherencia + agregados semanales y marcado de fase
        for w_idx, s in enumerate(plan.semanas, start=1):
            try:
                self._normalizar_arrays_por_sesion(s)
                self._forzar_minimos_y_gaps_por_rol(s)

                # >>> NUEVO: re-imponer separaciones y rangos por fase tras microtrends
                self._enforce_role_separation_strict(s)
                self._enforce_phase_ranges(s, w_idx)

                self._validar_coherencia_rir_rpe(s, w_idx - 1)

                # Sincroniza agregados semanales con las sesiones
                if s.distrib_series:
                    s.series = int(sum(s.distrib_series))

                if s.distrib_series and s.pct_1rm_sesion:
                    s.pct_1rm = float(self._pct_semana_ponderado(s))

                if s.distrib_series and s.rpe_sesion:
                    s.rpe = self._rpe_discreto(
                        sum(r * sets for r, sets in zip(s.rpe_sesion, s.distrib_series)) / max(1, s.series)
                    )

                if s.distrib_series and s.reps_sesion:
                    s.reps = int(round(
                        sum(sets * reps for sets, reps in zip(s.distrib_series, s.reps_sesion)) / max(1, s.series)
                    ))

                # Marca de fase útil para depurar
                fase = "hipertrofia" if w_idx <= 4 else ("fuerza" if w_idx <= 8 else ("pico" if w_idx <= 11 else "taper"))
                setattr(s, "fase", fase)
            except Exception:
                pass

        # 6) Barrido extra de sincronía (mantener orden H/M/L y RPE por rol)
        for w_idx, s in enumerate(plan.semanas, start=1):
            try:
                self._sync_semana(s, w_idx - 1)
                self._canonizar_orden_hml(s)
                self._ordenar_rpe_por_rol(s)
            except Exception:
                pass

        # 7) >>> NUEVO: suavidad intersemanal por rol (deltas por fase)
        try:
            self._limit_weekly_deltas_por_rol(plan)
        except Exception:
            pass

        # (Opcional) metadata de bloques en el plan
        try:
            plan.bloques = {
                1: {"idx": [1, 2, 3, 4], "fase": "hipertrofia"},
                2: {"idx": [5, 6, 7, 8], "fase": "fuerza"},
                3: {"idx": [9, 10, 11],  "fase": "pico"},
                4: {"idx": [12],         "fase": "taper"},
            }
        except Exception:
            pass


       # ─────────────────────────────────────────────────────────────────────────────
    # REGLAS DURAS (config)
    # ─────────────────────────────────────────────────────────────────────────────
    _GAP_PCT_MIN = 6.0   # separaciones mínimas H-M y M-L (p.p.)
    _GAP_RPE_MIN = 0.5   # H ≥ M +0.5 ≥ L +0.5

    def _fase_simple(self, w_idx_1based: int) -> str:
        if w_idx_1based <= 4:  return "acc"   # acumulación
        if w_idx_1based <= 8:  return "int"   # intensificación
        if w_idx_1based <= 11: return "peak"  # peaking
        return "taper"

    _ROLE_DELTA_LIMITS = {
        "acc":  {"H": 3.0, "M": 3.0, "L": 3.0, "rpe": 0.5, "sets": 2},
        "int":  {"H": 4.0, "M": 3.0, "L": 3.0, "rpe": 0.5, "sets": 2},
        "peak": {"H": 3.0, "M": 3.0, "L": 2.0, "rpe": 0.5, "sets": 2},  # 2–3 p.p.
    }

    # Rangos por fase/rol
    _PHASE_BOUNDS = {
        "acc": {
            "H": {"pct": (70, 76), "reps": (6, 8),  "rpe": (6.5, 7.5)},
            "M": {"pct": (68, 72), "reps": (8, 10), "rpe": (6.5, 7.5)},
            "L": {"pct": (60, 68), "reps": (10,12), "rpe": (6.0, 7.0)},
        },
        "int": {
            "H": {"pct": (82, 88), "reps": (2, 4),  "rpe": (7.5, 8.5)},
            "M": {"pct": (76, 84), "reps": (3, 5),  "rpe": (7.0, 8.0)},  # sin singles
            "L": {"pct": (68, 76), "reps": (5, 8),  "rpe": (6.5, 7.5)},  # sin singles
        },
        "peak": {
            "H": {"pct": (88, 92), "reps": (1, 3),  "rpe": (8.0, 9.0)},  # singles OK
            "M": {"pct": (80, 87), "reps": (2, 4),  "rpe": (7.0, 8.5)},  # sin singles
            "L": {"pct": (60, 72), "reps": (2, 3),  "rpe": (6.0, 7.0)},  # velocidad
        },
    }

    def _enforce_role_separation_strict(self, s: SemanaPlan) -> None:
        """H≥M≥L en % con ≥_GAP_PCT_MIN; RPE H≥M≥L con ≥0.5."""
        if not (s.distrib_series and s.pct_1rm_sesion and s.rpe_sesion):
            return

        # usar las constantes de la instancia/clase
        gap_pct = self._GAP_PCT_MIN
        gap_rpe = self._GAP_RPE_MIN

        self._canonizar_orden_hml(s)  # deja 0=H, 1=M, 2=L
        n = len(s.pct_1rm_sesion)

        # %1RM gaps
        if n >= 2 and s.pct_1rm_sesion[0] < s.pct_1rm_sesion[1] + gap_pct:
            s.pct_1rm_sesion[0] = s.pct_1rm_sesion[1] + gap_pct
        if n >= 3 and s.pct_1rm_sesion[1] < s.pct_1rm_sesion[2] + gap_pct:
            s.pct_1rm_sesion[1] = s.pct_1rm_sesion[2] + gap_pct

        # RPE gaps
        s.rpe_sesion[0] = self._rpe_discreto(
            max(
                s.rpe_sesion[0],
                s.rpe_sesion[1] + gap_rpe if n >= 2 else s.rpe_sesion[0]
            )
        )
        if n >= 3:
            s.rpe_sesion[1] = self._rpe_discreto(
                max(s.rpe_sesion[1], s.rpe_sesion[2] + gap_rpe)
            )

        # clamps
        s.pct_1rm_sesion = [
            float(min(100.0, max(50.0, x))) for x in s.pct_1rm_sesion
        ]

    def _enforce_phase_ranges(self, s: SemanaPlan, semana_idx_1b: int) -> None:
        """Clampa %/reps/RPE por FASE y prohíbe singles en M/L fuera de H peaking."""
        fase = self._fase_simple(semana_idx_1b)
        self._canonizar_orden_hml(s)
        roles = ["H", "M"] if s.sesiones == 2 else ["H", "M", "L"]

        for i, rol in enumerate(roles):
            if fase == "taper":
                # aquí no tocamos: lo hace _enforce_taper
                continue

            bounds = self._PHASE_BOUNDS[fase][rol]
            lo_pct, hi_pct = bounds["pct"]
            lo_rep, hi_rep = bounds["reps"]
            lo_rpe, hi_rpe = bounds["rpe"]

            s.pct_1rm_sesion[i] = float(
                min(hi_pct, max(lo_pct, s.pct_1rm_sesion[i]))
            )
            s.reps_sesion[i] = int(
                min(hi_rep, max(lo_rep, int(s.reps_sesion[i])))
            )
            s.rpe_sesion[i] = self._rpe_discreto(
                min(hi_rpe, max(lo_rpe, float(s.rpe_sesion[i])))
            )

            # explícito: sin singles en M/L en intensificación/peaking
            if rol in ("M", "L") and s.reps_sesion[i] < 2:
                s.reps_sesion[i] = 2

        # Reps identidad: L ≥ M +2 y M ≥ H +1 (excepto H en peaking)
        if s.sesiones >= 2:
            if s.reps_sesion[1] < s.reps_sesion[0] + 1 and fase != "taper":
                s.reps_sesion[1] = s.reps_sesion[0] + 1
        if s.sesiones == 3:
            if s.reps_sesion[2] < s.reps_sesion[1] + 2 and fase != "taper":
                s.reps_sesion[2] = s.reps_sesion[1] + 2

        self._validar_coherencia_rir_rpe(s, semana_idx_1b - 1)
        self._enforce_role_separation_strict(s)
        self._sync_semana(s, semana_idx_1b - 1)

    def _limit_weekly_deltas_por_rol(self, plan: Plan12Semanas) -> None:
        """Suaviza semana a semana por ROL (H/M/L) según límites por fase."""
        if not plan or not plan.semanas:
            return

        for w in range(1, len(plan.semanas)):
            prev = plan.semanas[w - 1]
            cur  = plan.semanas[w]

            self._normalizar_arrays_por_sesion(prev)
            self._normalizar_arrays_por_sesion(cur)
            self._canonizar_orden_hml(prev)
            self._canonizar_orden_hml(cur)

            fase = self._fase_simple(w + 1)  # clamp hacia semana actual
            if fase == "peak":
                lims = self._ROLE_DELTA_LIMITS["peak"]
            elif fase == "int":
                lims = self._ROLE_DELTA_LIMITS["int"]
            else:
                lims = self._ROLE_DELTA_LIMITS["acc"]

            n = min(len(prev.pct_1rm_sesion), len(cur.pct_1rm_sesion))
            roles = ["H", "M"] if n == 2 else ["H", "M", "L"][:n]

            for i, rol in enumerate(roles):
                # %1RM
                p0 = float(prev.pct_1rm_sesion[i])
                p1 = float(cur.pct_1rm_sesion[i])
                maxjump = lims[rol]
                if abs(p1 - p0) > maxjump:
                    cur.pct_1rm_sesion[i] = p0 + maxjump * (1 if p1 > p0 else -1)

                # RPE
                r0 = float(prev.rpe_sesion[i])
                r1 = float(cur.rpe_sesion[i])
                if abs(r1 - r0) > lims["rpe"]:
                    cur.rpe_sesion[i] = self._rpe_discreto(
                        r0 + lims["rpe"] * (1 if r1 > r0 else -1)
                    )

                # Sets
                if (cur.distrib_series and prev.distrib_series and
                        i < len(cur.distrib_series) and i < len(prev.distrib_series)):
                    s0 = int(prev.distrib_series[i])
                    s1 = int(cur.distrib_series[i])
                    delta = s1 - s0
                    if abs(delta) > lims["sets"]:
                        cur.distrib_series[i] = s0 + (
                            lims["sets"] if delta > 0 else -lims["sets"]
                        )

            self._forzar_minimos_y_gaps_por_rol(cur)
            self._validar_coherencia_rir_rpe(cur, w)
            self._sync_semana(cur, w)



    def _ajustar_series_hacia(self, s, objetivo: int, max_mov: int = 2) -> None:
        """
        Mueve las series totales de la semana 's' hacia 'objetivo' sin tocar la forma de la distribución.
        Limita el movimiento absoluto a 'max_mov' (a menos que pidas un fijado duro).
        """
        objetivo = int(max(1, objetivo))
        cur = int(s.series)
        if cur == objetivo:
            return
        if max_mov is not None:
            if abs(objetivo - cur) > max_mov:
                objetivo = cur + (max_mov if objetivo > cur else -max_mov)

        # Reescala distribución proporcionalmente y corrige residuo
        if s.distrib_series and sum(s.distrib_series) > 0:
            total = sum(s.distrib_series)
            nueva = [max(0, int(round(objetivo * (x / total)))) for x in s.distrib_series]
            diff = objetivo - sum(nueva)
            # Corrige residuo favoreciendo sesiones con mayor cuota original
            order = sorted(range(len(nueva)), key=lambda j: s.distrib_series[j], reverse=True)
            k = 0
            while diff != 0 and k < len(order):
                j = order[k]
                if diff > 0:
                    nueva[j] += 1; diff -= 1
                else:
                    if nueva[j] > 0:
                        nueva[j] -= 1; diff += 1
                k = (k + 1) % len(order) if len(order) else 0
            s.distrib_series = nueva
            s.series = int(sum(nueva))
        else:
            s.series = objetivo
            s.distrib_series = [objetivo] if s.sesiones == 1 else [objetivo // s.sesiones] * s.sesiones
            rest = objetivo - sum(s.distrib_series)
            for i in range(abs(rest)):
                s.distrib_series[i % len(s.distrib_series)] += 1

    # ─────────────────────────────────────────────────────────────────────────────
    # Ajuste de sesiones en Taper: si hay 2 sesiones, convierte la más ligera en
    # velocidad, y capea ≤3 sets por sesión.
    # ─────────────────────────────────────────────────────────────────────────────
    def _ajustar_sesiones_taper(self, s) -> None:
        n = s.sesiones
        if n <= 0:
            return
        self._normalizar_arrays_por_sesion(s)

        order = sorted(range(n), key=lambda j: s.pct_1rm_sesion[j], reverse=True)
        roles = ["M"] * n
        if n == 1:
            roles[order[0]] = "H"
        elif n == 2:
            roles[order[0]], roles[order[1]] = "H", "M"
        else:
            roles[order[0]], roles[order[1]], roles[order[2]] = "H", "M", "L"

        for j, rol in enumerate(roles):
            if rol in ("H", "M"):
                s.pct_1rm_sesion[j] = float(min(90.0, max(85.0, s.pct_1rm_sesion[j])))
                s.reps_sesion[j]    = int(min(2, max(1, s.reps_sesion[j])))
                s.rpe_sesion[j]     = self._rpe_discreto(min(8.0, max(7.0, s.rpe_sesion[j])))
            else:
                s.pct_1rm_sesion[j] = float(min(65.0, max(60.0, s.pct_1rm_sesion[j])))
                s.reps_sesion[j]    = 2
                s.rpe_sesion[j]     = self._rpe_discreto(min(7.0, max(6.0, s.rpe_sesion[j])))

        # 2 sesiones: fuerza + velocidad
        if n == 2:
            idx_vel = order[-1]  # la más baja en %
            s.pct_1rm_sesion[idx_vel] = float(min(65.0, max(60.0, s.pct_1rm_sesion[idx_vel])))
            s.reps_sesion[idx_vel]    = 2
            s.rpe_sesion[idx_vel]     = self._rpe_discreto(6.5)

        # Cap sets por sesión en taper (≤3)
        if s.distrib_series:
            MAX_SETS_TAPER = 5  
            s.distrib_series = [min(MAX_SETS_TAPER, int(x)) for x in s.distrib_series]

            s.series = int(sum(s.distrib_series))

        self._forzar_minimos_y_gaps_por_rol(s)
        self._validar_coherencia_rir_rpe(s)

    def _suavizar_series_intersemanal(self, plan: Plan12Semanas, max_delta: int = 2) -> None:
        """Evita saltos de volumen bruscos semana a semana (±max_delta)."""
        for i in range(1, len(plan.semanas)):
            prev = plan.semanas[i-1]
            cur  = plan.semanas[i]
            delta = cur.series - prev.series
            if abs(delta) > max_delta:
                objetivo = prev.series + (max_delta if delta > 0 else -max_delta)
                self._ajustar_series_hacia(cur, objetivo, max_mov=None)

    def _capar_rampa_interbloque(self, plan: Plan12Semanas, w_from: int, w_to: int, max_delta_pct: float = 5.0) -> None:
        """
        Reduce saltos de intensidad medio ponderado al pasar de un bloque a otro.
        Ejemplo: W4→W5 (hipertrofia→fuerza) cap en 5 pp.
        """
        if w_from - 1 < 0 or w_to - 1 >= len(plan.semanas):
            return
        a = plan.semanas[w_from - 1]
        b = plan.semanas[w_to - 1]
        # % ponderado por sets (si hay arrays), si no usa % semanal
        def pct_med(s):
            if s.distrib_series and s.pct_1rm_sesion:
                tot = max(1, sum(s.distrib_series))
                return float(sum(p * sets for p, sets in zip(s.pct_1rm_sesion, s.distrib_series)) / tot)
            return float(s.pct_1rm)
        pa = pct_med(a)
        pb = pct_med(b)
        if (pb - pa) > max_delta_pct:
            # Baja las sesiones de b de forma homogénea
            if b.pct_1rm_sesion:
                exceso = (pb - pa) - max_delta_pct
                step = exceso / max(1, len(b.pct_1rm_sesion))
                b.pct_1rm_sesion = [max(50.0, p - step) for p in b.pct_1rm_sesion]
                # Resea gaps y coherencias
                self._forzar_minimos_y_gaps_por_rol(b)
                self._validar_coherencia_rir_rpe(b)
            else:
                b.pct_1rm = max(50.0, b.pct_1rm - ((pb - pa) - max_delta_pct))

    def _pct_semana_ponderado(self, s):
            w = [max(1, x) for x in s.distrib_series]  # evita pesos cero
            num = sum(p * w_i for p, w_i in zip(s.pct_1rm_sesion, w))
            den = sum(w)
            return round(num / den, 1)

    
    def _roles_por_intensidad(self, s: SemanaPlan) -> List[str]:
            """Devuelve ['H','M','L'] indexado por sesión, usando ranking de %; fallback a share si no hay %."""
            n = s.sesiones
            if s.pct_1rm_sesion and len(s.pct_1rm_sesion) == n:
                order = sorted(range(n), key=lambda j: s.pct_1rm_sesion[j], reverse=True)
                roles = ["M"] * n
                if n == 2:
                    roles[order[0]], roles[order[1]] = "H", "M"
                else:
                    roles[order[0]], roles[order[1]], roles[order[2]] = "H", "M", "L"
                return roles
            # fallback por cuota
            total = max(1, s.series)
            return [self._rol_por_share((s.distrib_series[j] if j < len(s.distrib_series) else 0)/total) for j in range(n)]

    def _reparar_semana(self, s, semana_idx, bloque):
            # 1) Forzar DUP (H/M/L) en bloques que lo requieren
            if bloque in (1, 3):  # Hipertrofia y Pico requieren 3 estímulos típicos
                s.sesiones = max(3, s.sesiones)
                self._asegurar_roles_hml(s)  # asigna roles H/M/L y redivide series

            # 2) Gaps e intensidades ordenadas H≥M≥L con ≥5 pp
            self._ordenar_y_gap_intensidades(s, gap_min_pp=5)

            # 3) Reps coherentes con % y RPE por sesión (bajar reps si sube %)
            self._coherencia_pct_reps_rpe(s)

            # 4) Día de velocidad en Pico (W9–11)
            if 8 <= semana_idx <= 10:  # idx 0-based
                self._inyectar_velocidad_en_L(s)

            # 5) Límites de volumen por sesión según rol
            self._limitar_series_por_rol(s, lims={"H": (4, 8), "M": (3, 6), "L": (2, 5)})

    def _asegurar_roles_hml(self, s):
            roles = ["H", "M", "L"]
            # Si faltan sesiones, crea placeholders
            while len(getattr(s, "pct_1rm_sesion", [])) < 3:
                s.pct_1rm_sesion.append(s.pct_1rm)  # base
                s.rpe_sesion.append(s.rpe)
                s.reps_sesion.append(s.reps)
                s.distrib_series.append(2)
            s.sesiones = 3
            s.roles = roles  # guarda el orden
            self._normalizar_arrays_por_sesion(s)

   


    def _ordenar_y_gap_intensidades(self, s, gap_min_pp=5):
            # Asegura que los arrays por sesión estén alineados con distrib_series
            if not s.distrib_series:
                return
            self._normalizar_arrays_por_sesion(s)

            tripletas = list(zip(s.pct_1rm_sesion, s.rpe_sesion, s.reps_sesion, s.distrib_series))
            n = len(tripletas)
            if n < 2:
                # Con 1 sola sesión no hay nada que ordenar ni gappear
                s.pct_1rm_sesion, s.rpe_sesion, s.reps_sesion, s.distrib_series = map(list, zip(*tripletas))
                return

            # Ordena de mayor a menor % (H ≥ M (≥ L))
            tripletas.sort(key=lambda t: t[0], reverse=True)

            # Fuerza gap de intensidad entre vecinos: H-M (y M-L si existe)
            for i in range(n - 1):
                diff = tripletas[i][0] - tripletas[i + 1][0]
                if diff < gap_min_pp:
                    delta = gap_min_pp - diff
                    t = list(tripletas[i + 1])
                    t[0] = max(50.0, t[0] - delta)   # baja el inferior para abrir gap
                    tripletas[i + 1] = tuple(t)

            s.pct_1rm_sesion, s.rpe_sesion, s.reps_sesion, s.distrib_series = map(list, zip(*tripletas))
            s.series = int(sum(s.distrib_series))

    def _coherencia_pct_reps_rpe(self, s):

            # Asegura que las listas por sesión estén alineadas con distrib_series
            self._normalizar_arrays_por_sesion(s)

            for i in range(len(s.pct_1rm_sesion)):
                pct  = float(s.pct_1rm_sesion[i])
                reps = int(s.reps_sesion[i])
                rpe  = float(s.rpe_sesion[i])

                # Reglas simples por rango de %1RM → ajustar reps
                # >80% → reps ≤ 5 ; 70–80% → reps ≤ 10 ; 60–70% → reps ≥ 10
                if pct > 80.0 and reps > 5:
                    reps = 5
                elif 70.0 <= pct <= 80.0:
                    reps = min(reps, 10)
                elif 60.0 <= pct < 70.0:
                    reps = max(reps, 10)

                # Clamps básicos
                reps = max(1, min(30, reps))

                # RPE: clamp 6.0–9.0 y CUANTIZACIÓN al set activo (saltos de 0.5)
                rpe = self._rpe_discreto(min(9.0, max(6.0, rpe)))

                s.reps_sesion[i] = reps
                s.rpe_sesion[i]  = rpe

            # Re-sincroniza los agregados semanales con media ponderada por sets
            if s.distrib_series and sum(s.distrib_series) > 0:
                total_sets = max(1, s.series)
                s.reps = int(round(
                    sum(sets * reps for sets, reps in zip(s.distrib_series, s.reps_sesion)) / total_sets
                ))
                s.rpe = self._rpe_discreto(
                    sum(sets * r for sets, r in zip(s.distrib_series, s.rpe_sesion)) / total_sets
                )

    def _inyectar_velocidad_en_L(self, s):
            # Asume L es el índice 2 tras ordenar (H/M/L)
            idx_L = 2
            s.pct_1rm_sesion[idx_L] = max(60.0, min(65.0, s.pct_1rm_sesion[idx_L]))
            s.reps_sesion[idx_L] = 2
            s.distrib_series[idx_L] = max(4, s.distrib_series[idx_L])  # 4–6 series (6×2 si alcanza)
            s.rpe_sesion[idx_L] = 6.5
            # Marca bandera de velocidad si existe en tu modelo:
            setattr(s, "dia_velocidad", True)

    def _limitar_series_por_rol(self, s, lims):
            if not s.distrib_series:
                return
            n = len(s.distrib_series)
            # Se supone que _ordenar_y_gap_intensidades ya dejó H≥M(≥L) por %.
            roles_en_orden = ["H", "M", "L"][:n]
            for i, rol in enumerate(roles_en_orden):
                lo, hi = lims[rol]
                s.distrib_series[i] = max(lo, min(hi, s.distrib_series[i]))
            s.series = int(sum(s.distrib_series))

    # ────────────────────────────────────────────────────────────────
    # Ajuste final de taper (W12)
    # - Volumen = 50 % de W11 (ya calculado)
    # - top single = 1×1 @ 90-92 %, RPE 8-8.5
    # - Back-offs (M, L) rellenan el resto del volumen
    # ────────────────────────────────────────────────────────────────
    def _enforce_taper(self, plan):
        try:
            w11, w12 = plan.semanas[10], plan.semanas[11]
        except Exception:
            return

        # Volumen objetivo ~50% de W11, pero nunca menos de 2 series
        series11 = sum(w11.distrib_series) if (w11.distrib_series) else 0
        if series11 <= 0:
            return

        ratio = 0.75  # 75 % del volumen de W11
        objetivo = int(round(series11 * ratio))

        # Asegura un mínimo razonable de sets en W12
        objetivo = max(6, objetivo)
        self._redistribuir_a_total(w12, total_objetivo=objetivo)

        # Alinear y ordenar por %
        self._normalizar_arrays_por_sesion(w12)
        self._canonizar_orden_hml(w12)

        n = min(3, len(w12.pct_1rm_sesion or []))
        if n == 0:
            return
        iH = 0
        iM = 1 if n >= 2 else None
        iL = 2 if n == 3 else (1 if n == 2 else None)

        # H: top single 88–90% x1 @ RPE 8.0–8.5 (1×1 real)
        w12.pct_1rm_sesion[iH] = float(max(88.0, min(90.0, w12.pct_1rm_sesion[iH])))
        w12.reps_sesion[iH]    = 1
        w12.rpe_sesion[iH]     = self._rpe_discreto(min(8.5, max(8.0, w12.rpe_sesion[iH])))
        if w12.distrib_series:
            # Forzamos exactamente 1 serie para el top single
            w12.distrib_series[iH] = 1
        setattr(w12, "top_single_objetivo", True)

        # M: ligero 70–75% ×2 @ RPE 6.5–7.5
        if iM is not None:
            w12.pct_1rm_sesion[iM] = float(max(70.0, min(75.0, w12.pct_1rm_sesion[iM])))
            w12.reps_sesion[iM]    = 2
            w12.rpe_sesion[iM]     = self._rpe_discreto(min(7.5, max(6.5, w12.rpe_sesion[iM])))
            if w12.distrib_series:
                w12.distrib_series[iM] = min(3, max(0, int(w12.distrib_series[iM])))

        # L: velocidad 60–70% ×2 @ RPE 6–7 (si existe)
        if iL is not None:
            w12.pct_1rm_sesion[iL] = float(max(60.0, min(70.0, w12.pct_1rm_sesion[iL])))
            w12.reps_sesion[iL]    = 2
            w12.rpe_sesion[iL]     = self._rpe_discreto(min(7.0, max(6.0, w12.rpe_sesion[iL])))
            if w12.distrib_series:
                w12.distrib_series[iL] = min(3, max(0, int(w12.distrib_series[iL])))

        # Ajustar el total de series de W12 al objetivo (taper 40–60% vs W11)
        if w12.distrib_series:
            total_actual = int(sum(w12.distrib_series))
            diff = objetivo - total_actual

            # Si faltan series → añadimos primero a M, luego a L (no tocamos H=1)
            while diff > 0:
                if iM is not None:
                    w12.distrib_series[iM] += 1
                    diff -= 1
                    if diff == 0:
                        break
                if iL is not None:
                    w12.distrib_series[iL] += 1
                    diff -= 1
                    if diff == 0:
                        break
                if iM is None and iL is None:
                    break

            # Si sobran series → quitamos primero de L, luego de M (H se queda en 1)
            while diff < 0:
                if iL is not None and w12.distrib_series[iL] > 0:
                    w12.distrib_series[iL] -= 1
                    diff += 1
                    if diff == 0:
                        break
                if iM is not None and w12.distrib_series[iM] > 0:
                    w12.distrib_series[iM] -= 1
                    diff += 1
                    if diff == 0:
                        break
                # Si ya no se puede ajustar sin tocar H, salimos
                if (iL is None or w12.distrib_series[iL] <= 0) and (iM is None or w12.distrib_series[iM] <= 0):
                    break

            # Recalcular agregados con el nuevo reparto
            w12.series = int(sum(w12.distrib_series))
            w12.pct_1rm = self._pct_semana_ponderado(w12)
            w12.rpe = self._rpe_discreto(
                sum(r * sets for r, sets in zip(w12.rpe_sesion, w12.distrib_series)) / max(1, w12.series)
            )
            w12.reps = int(round(
                sum(sets * reps for sets, reps in zip(w12.distrib_series, w12.reps_sesion)) / max(1, w12.series)
            ))



    def _redistribuir_a_total(self, s: SemanaPlan, total_objetivo: int) -> None:
        if not s.distrib_series or total_objetivo < 0:
            return
        n = len(s.distrib_series)
        if total_objetivo == sum(s.distrib_series):
            return

        suma_actual = sum(s.distrib_series)
        if suma_actual == 0:
            proporciones = [1.0 / max(1, n)] * n
            base = [max(0, int(round(total_objetivo * p))) for p in proporciones]
        else:
            proporciones = [x / suma_actual for x in s.distrib_series]
            base = [max(0, int(round(total_objetivo * p))) for p in proporciones]

        delta = total_objetivo - sum(base)
        orden_add = sorted(range(n), key=lambda i: proporciones[i], reverse=True)
        orden_sub = sorted(range(n), key=lambda i: base[i], reverse=True)

        while delta > 0:
            for i in orden_add:
                base[i] += 1
                delta -= 1
                if delta == 0:
                    break
        while delta < 0:
            for i in orden_sub:
                if base[i] > 0:
                    base[i] -= 1
                    delta += 1
                    if delta == 0:
                        break

        s.distrib_series = base
        s.series = int(sum(base))


    def _ajustar_deload_semana(self, semanas: List[SemanaPlan]) -> None:
            # W4 (idx 3) y W8 (idx 7)
            for w in (3, 7):
                prev, cur = semanas[w - 1], semanas[w]
                # volumen a 65–85% de la semana previa
                # volumen a 65–85% de la semana previa, pero respetando el mínimo ideal de 10
                target_min = max(10, int(round(prev.series * 0.65)))
                target_max = min(20, int(round(prev.series * 0.85)))
                cur.series = int(min(max(cur.series, target_min), target_max))

                if cur.distrib_series:
                    cur.distrib_series = self._reajustar_distrib(cur.distrib_series, cur.series, cur.sesiones)
                # intensidad -7.5 pp (punto medio)
                delta = 7.5
                cur.pct_1rm = max(50.0, prev.pct_1rm - delta)
                if cur.pct_1rm_sesion:
                    for i in range(len(cur.pct_1rm_sesion)):
                        cur.pct_1rm_sesion[i] = max(50.0, cur.pct_1rm_sesion[i] - delta)
                # revalidar coherencia y realinear
                self._validar_coherencia_rir_rpe(cur)
                self._realinear_sets_con_pct(cur)
