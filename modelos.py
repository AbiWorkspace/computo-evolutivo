# modelos.py
"""
Modelos de datos para el plan de entrenamiento evolutivo.
Incluye:
- SemanaPlan: parámetros semanales y desglose por sesión (H/M/L).
- Plan12Semanas: contenedor de 12 semanas para un patrón concreto.
"""

from dataclasses import dataclass, field
from typing import List

__all__ = ["SemanaPlan", "Plan12Semanas"]


@dataclass
class SemanaPlan:
    """
    SemanaPlan
    - sesiones: número de sesiones en la semana (2 o 3).
    - series: total de series semanales (2..20).
    - pct_1rm: %1RM base semanal (50..100).
    - rpe: RPE base semanal (6.0..9.0).
    - reps: repeticiones base semanales (1..30).
    - distrib_series: lista con las series por sesión (len == sesiones).
    - pct_1rm_sesion / rpe_sesion / reps_sesion: valores por sesión, coherentes con H/M/L.
    """
    sesiones: int
    series: int
    pct_1rm: float
    rpe: float
    reps: int
    distrib_series: List[int] = field(default_factory=list)

    # Desglose por sesión (coherente con H/M/L). Se rellenan en la lógica del algoritmo.
    pct_1rm_sesion: List[float] = field(default_factory=list)
    rpe_sesion: List[float] = field(default_factory=list)
    reps_sesion: List[int] = field(default_factory=list)


@dataclass
class Plan12Semanas:
    """
    Plan12Semanas
    - patron: nombre del patrón de movimiento (p. ej., 'empuje', 'tracción', etc.).
    - semanas: lista de 12 objetos SemanaPlan.
    """
    patron: str
    semanas: List[SemanaPlan]
