# Cómputo Evolutivo — Generación de planes de entrenamiento periodizados

Algoritmo genético multiobjetivo (NSGA-II) que diseña, desde cero, un plan de
entrenamiento de **fuerza e hipertrofia periodizado a 12 semanas**. El AG explora
el espacio de combinaciones de volumen, intensidad (%1RM), RPE y repeticiones por
sesión, y devuelve el plan que mejor equilibra estímulo de entrenamiento,
fatiga muscular/SNC y progresión de fuerza, respetando decenas de reglas de
periodización deportiva (bloques, ondulación diaria H/M/L, deloads, taper, etc.).

## Idea general del algoritmo

- **Individuo (cromosoma):** un `Plan12Semanas`, es decir, una lista de 12
  `SemanaPlan` (una por semana). Cada semana define nº de sesiones (2 o 3),
  series totales, %1RM, RPE y reps —a nivel semanal y desglosados por sesión
  según el rol de esa sesión: **H** (pesada), **M** (media), **L** (ligera).
- **Población inicial:** semanas generadas aleatoriamente dentro de rangos
  específicos por bloque (`_semana_aleatoria`), con reparto de series por
  sesión siguiendo patrones realistas (p. ej. 50/33/17 para H/M/L).
- **Selección:** estrategia "Factible-Primero" (`_seleccionar_padres_factible_primero`):
  prioriza individuos que cumplen las restricciones duras y, entre ellos, el
  mejor `score_fp` (escalarización de los objetivos).
- **Cruce (`_cruzar`):** cruce de un punto por semanas completas, seguido de
  reparación del hijo para conservar la coherencia de bloques.
- **Mutación (`_mutar` / `_mutar_semana`):** ruido controlado sobre sesiones,
  series, %1RM, RPE y reps, con *clamps* específicos por bloque y
  reajuste de la distribución de series por sesión.
- **Reemplazo:** NSGA-II clásico (`_reemplazo_nsga2`) con ordenamiento por
  frentes de no-dominación, *crowding distance* y elitismo del primer frente.
- **Reparación:** tras cada cruce/mutación se aplica una batería de funciones
  que fuerzan coherencia física y de diseño (ver más abajo), de modo que la
  mayoría de individuos generados sean válidos o casi válidos.

### Modelo de estímulo y fatiga

El criterio que optimiza el algoritmo es **máximo estímulo por unidad de
fatiga**. Las funciones fisiológicas viven en `epley_utils.py`, separadas del
AG para poder probarlas de forma aislada:

- **Repeticiones efectivas** (`reps_efectivas`): sólo las repeticiones dentro
  de las ~5 previas al fallo generan estímulo. Con `RIR = 10 − RPE`, son
  `clamp(5 − RIR, 0, reps)`. Así una serie pesada de 3 y una moderada de 10
  pueden valer lo mismo, que es lo que hace comparables fuerza e hipertrofia.
- **Estímulo de hipertrofia**: reps efectivas × campana de eficiencia centrada
  en ~75 %1RM.
- **Estímulo de fuerza**: exposición a carga alta (rampa convexa que satura
  hacia el 92 %1RM) × calidad según RIR (óptimo en RIR≈2; llegar al fallo con
  cargas altas dispara la fatiga del SNC sin añadir fuerza).
- **Rendimientos decrecientes** (`saturar`): el estímulo semanal se satura
  (MEV → MRV) mientras la fatiga sigue creciendo. Esto es lo que crea un
  óptimo interior de volumen; sin ello la razón estímulo/fatiga sería casi
  constante y al algoritmo le daría igual hacer 10 series que 40.

### Objetivos multiobjetivo (NSGA-II)

Calculados en `_calcular_metricas_er5`, todos a maximizar:

| Objetivo | Significado |
|---|---|
| `f1` | Estímulo de hipertrofia (`S_hyp`) |
| `f2` | Estímulo de fuerza (`S_str`) |
| `f3` | `−fatiga_muscular` acumulada (minimización real) |
| `f4` | `−fatiga_SNC` acumulada (minimización real) |
| `f5` | `−shock` intersemanal (saltos bruscos de volumen/reps) |
| `f6` | `−(pen_soft + 4·pen_hard)` (violaciones de estructura) |
| `f7` | Progreso de fuerza estimado a lo largo del macrociclo |

`score_fp` combina estos objetivos en un escalar `beneficio − coste` para
ordenar individuos dentro de un mismo frente de Pareto ("factible primero"),
más un muro cuadrático por encima de los techos fisiológicos de fatiga.

### Perfiles de objetivo

Generar el plan es un problema multiobjetivo con un compromiso **medido, no
supuesto**: la fatiga del SNC procede casi por completo del trabajo ≥80 %1RM,
que es exactamente el que desarrolla la fuerza máxima. Son la misma palanca,
así que no existe un plan que maximice la fuerza y minimice esa fatiga a la
vez. Lo que sí mejora es el estímulo obtenido por unidad de fatiga.

Esa decisión se elige de forma explícita con `objetivo=` (o `EE_OBJETIVO`):

| Perfil | Efecto frente al comportamiento anterior |
|---|---|
| `equilibrado` *(defecto)* | Misma fatiga muscular y de SNC, **+10 % de repeticiones efectivas**, +5 % / +11 % de eficiencia, progresión de fuerza mantenida |
| `fuerza` | +9 % de progresión y +5 % de series ≥85 %, a cambio de +6 % de fatiga muscular y +15 % de la del SNC |
| `hipertrofia` | −23 % de fatiga del SNC y +8 % de reps efectivas, pero −37 % de series ≥85 % (no sirve para pico de fuerza) |
| `minima_fatiga` | −33 % de fatiga del SNC, máxima recuperabilidad; cuesta series pesadas y progresión |

Cada perfil ajusta los pesos del escalar (`W_HYP`, `W_STR`, `W_PROG`,
`W_FAT_M`, `W_FAT_SNC`) y la saturación semanal del estímulo. Están definidos
en `EvolucionEntrenamiento.PERFILES_OBJETIVO`.

### Periodización y reglas de dominio

El macrociclo se divide en 3 bloques de 4 semanas cada uno, con fases y rangos
de %1RM / reps / RPE por rol (`_PHASE_BOUNDS`):

| Bloque | Semanas | Fase | Enfoque |
|---|---|---|---|
| 1 | 1–4 | Acumulación (`acc`) | Hipertrofia: mayor volumen, intensidad moderada |
| 2 | 5–8 | Intensificación (`int`) | Transición a fuerza: menos reps, más %1RM |
| 3 | 9–12 | Peaking (`peak`) | Intensificación final / taper en semana 12 |

Reglas y mecanismos clave aplicados durante generación, reparación y
penalización:

- **Ondulación diaria H/M/L:** cada semana con 2–3 sesiones asigna roles
  Heavy/Medium/Light por %1RM, con *gaps* mínimos entre roles (`_GAP_PCT_MIN`,
  `_GAP_RPE_MIN`) para evitar ondulaciones "falsas".
- **Coherencia %1RM–reps–RPE:** vía fórmula de Epley (`epley_utils.py`) y RIR
  objetivo por fase/rol/velocidad (`_rir_objetivo`, `_validar_coherencia_rir_rpe`).
- **Deload intermedio:** reducción de volumen/intensidad en semanas 4 y 8
  (`usar_deload_intermedio`, `_penalizar_deload_intermedio`).
- **Taper final (semana 12):** volumen muy reducido, top singles a alta
  intensidad (`_taper_real_w12`, `_penalizar_taper_final`).
- **Días de velocidad:** en la fase de peaking, la sesión L se trata como día
  de velocidad (menos reps, tolerancia distinta de RIR).
- **Límite de saltos intersemanales:** por rol (`_limitar_salto_intersemanal_por_rol`)
  y por shock de volumen/estímulo agregados (`_shock_intersemanal`).
- **Progresión por nivel del usuario:** rangos de incremento semanal de %1RM
  distintos para novato/principiante/intermedio/avanzado (`NIVELES`).
- Más de 15 funciones de penalización (`penalizaciones.py`,
  `evo_fitness_y_ga.py`) castigan: exceso de días H, singles fuera de sesión
  H, desequilibrio de roles, tendencias erráticas entre semanas, series
  efectivas fuera de rango, incumplimiento de objetivos por bloque, etc.

## Módulos

| Archivo | Función |
|---|---|
| `ejecutar_principal.py` | Punto de entrada. Configura logging/salida (silencio, filtrado, normal), presets de población/generaciones, autoajuste por presupuesto de tiempo (ETA) e imprime el plan final por bloques |
| `evolucion_entrenamiento.py` | Clase `EvolucionEntrenamiento` (fachada): combina los mixins, define rangos por fase/bloque (`_PHASE_BOUNDS`, `BLOQUES`, `IDEAL_BLOQUE`, `NIVELES`, etc.) y utilidades de RIR/RPE |
| `evo_fitness_y_ga.py` | `FitnessGAMixin`: fitness multiobjetivo (NSGA-II), no-dominación, crowding distance, selección "Factible-Primero", cruce, mutación, cálculo de métricas ER5 y diagnóstico del plan |
| `evo_generacion_y_reparaciones.py` | `GeneracionDistribMixin`: generación de individuos/semanas aleatorias, distribución de series por sesión H/M/L, y operadores de reparación (coherencia RIR/RPE, gaps por rol, taper, límites intersemanales) |
| `penalizaciones.py` | `PenalizacionesMixin`: penalizaciones por violación de restricciones blandas (fatiga, roles, tendencias, deload, taper, shock intersemanal) |
| `epley_utils.py` | Conversión %1RM ↔ reps ↔ RIR/RPE mediante la fórmula de Epley |
| `modelos.py` | Estructuras de datos del dominio: `SemanaPlan` (semana individual) y `Plan12Semanas` (plan completo, el "individuo" del AG) |

## Requisitos

- Python 3.11 (usa sintaxis moderna de *type hints*, p. ej. `int | None`)
- `colorama` (opcional; si no está instalada, la salida se degrada sin colores ANSI)

```bash
pip install colorama
```

## Uso

```bash
python ejecutar_principal.py            # modo producción (silencioso)
python ejecutar_principal.py --test     # modo test (logs detallados)
```

Para elegir el compromiso estímulo/fatiga:

```bash
EE_OBJETIVO=fuerza python ejecutar_principal.py
EE_OBJETIVO=minima_fatiga python ejecutar_principal.py
```

Al finalizar, se imprime siempre el plan de 12 semanas resumido por bloques
(series, %1RM, RPE, reps por sesión con su rol H/M/L) junto con el diagnóstico
**estímulo vs fatiga** (estímulo de hipertrofia y de fuerza, reps efectivas,
ambas fatigas frente a su techo, eficiencia, desglose de penalizaciones y
reparto del estímulo por bloque) y una alarma sonora de finalización.

### Presets de tamaño de búsqueda

Controlados por `EE_PRESET` (o forzando `EE_POBLACION` / `EE_GENERACIONES`):

| Preset | Población | Generaciones |
|---|---|---|
| `rapido` (por defecto) | 48 | 77 |
| `medio` | 160 | 240 |
| `exhaustivo` | 4800 | 9600 |
| `auto` | 96 | 144 |

### Autoajuste por tiempo (opcional)

Si se define `EE_MINUTOS_OBJ` o `EE_MAX_MINUTOS`, el script hace una
calibración breve (mini-evolución) para estimar el tiempo por evaluación y
reescala población/generaciones para ajustarse al presupuesto de tiempo dado.

### Variables de entorno

| Variable | Descripción |
|---|---|
| `EE_MODO` | `prod` o `test` |
| `EE_DEBUG` | `1` habilita los prints de depuración |
| `EE_LOGLEVEL` | Nivel de consola (`INFO`, `WARNING`, …) |
| `EE_LOGS_PERMITIR` | Lista de métodos cuyos logs se muestran |
| `EE_LOGS_BLOQUEAR` | Lista de métodos cuyos logs se ocultan |
| `EE_OBJETIVO` | `equilibrado` \| `fuerza` \| `hipertrofia` \| `minima_fatiga` |
| `EE_PRESET` | `rapido` \| `medio` \| `exhaustivo` \| `auto` |
| `EE_POBLACION` / `EE_GENERACIONES` | Fuerzan tamaño de población/generaciones |
| `EE_MINUTOS_OBJ` / `EE_MAX_MINUTOS` | Presupuesto de tiempo objetivo/máximo (activa autoajuste) |
| `EE_MODO_SALIDA` | `auto` \| `silencio` \| `filtrado` \| `normal` |
| `EE_UMBRAL_ESCALA` | Umbral (población × generaciones) para pasar a modo silencio en `auto` |
| `EE_MOSTRAR_FINAL` | `1` (por defecto) muestra el plan final aunque el resto esté en silencio |
| `EE_LOG_ARCHIVO`, `EE_LOG_RUTA`, `EE_LOG_TAM_MB`, `EE_LOG_RESPALDOS` | Configuración del logger rotativo a archivo (`evolucion.log` por defecto) |

## Documentación

La carpeta `X/` contiene la especificación del algoritmo y ejemplos de salida:

- `Informacion para el algoritmo.docx` — reglas de dominio y restricciones
- `Plan de entrenamiento periodizado de 12 semanas para fuerza e hipertrofia.pdf`
- `plan_12_semanas.pdf` — ejemplo de plan generado
