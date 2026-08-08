# Cómputo Evolutivo — Generación de planes de entrenamiento periodizados

Algoritmo genético que genera planes de entrenamiento de fuerza e hipertrofia
periodizados a 12 semanas, optimizando volumen, intensidad y distribución muscular
bajo un conjunto de restricciones y penalizaciones.

## Módulos

| Archivo | Función |
|---|---|
| `ejecutar_principal.py` | Punto de entrada. Configura logging y modos `prod` / `test` |
| `evolucion_entrenamiento.py` | Clase `EvolucionEntrenamiento`: orquesta el ciclo evolutivo |
| `evo_fitness_y_ga.py` | Función de fitness, selección, cruce y mutación |
| `evo_generacion_y_reparaciones.py` | Generación de individuos y operadores de reparación de soluciones inválidas |
| `penalizaciones.py` | Penalizaciones por violación de restricciones (fatiga, frecuencia, balance muscular) |
| `epley_utils.py` | Cálculo de 1RM y cargas de trabajo mediante la fórmula de Epley |
| `modelos.py` | Estructuras de datos del dominio |

## Requisitos

- Python 3.11

## Uso

```bash
python ejecutar_principal.py            # modo producción (silencioso)
python ejecutar_principal.py --test     # modo test (logs detallados)
```

El comportamiento del logging se controla con variables de entorno que el propio
script define según el modo:

| Variable | Descripción |
|---|---|
| `EE_MODO` | `prod` o `test` |
| `EE_DEBUG` | `1` habilita los prints de depuración |
| `EE_LOGLEVEL` | Nivel de consola (`INFO`, `WARNING`, …) |
| `EE_LOGS_PERMITIR` | Lista de métodos cuyos logs se muestran |
| `EE_LOGS_BLOQUEAR` | Lista de métodos cuyos logs se ocultan |

## Documentación

La carpeta `X/` contiene la especificación del algoritmo y ejemplos de salida:

- `Informacion para el algoritmo.docx` — reglas de dominio y restricciones
- `Plan de entrenamiento periodizado de 12 semanas para fuerza e hipertrofia.pdf`
- `plan_12_semanas.pdf` — ejemplo de plan generado
