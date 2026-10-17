# Clasificación estadística (QDA): rápida, liviana y confiable

Trabajo práctico optativo de **Análisis Matemático para Inteligencia Artificial** · Carrera de Especialización en Inteligencia Artificial, FIUBA.

> **Pregunta de apertura:** ¿Se puede decidir a qué categoría pertenece cada caso nuevo (tipo de vino, especie, letra manuscrita) **más rápido, con menos memoria y sin equivocarse por problemas numéricos**, sin cambiar ninguna decisión del modelo?

**Respuesta corta:** sí. Una versión reescrita del clasificador toma **exactamente las mismas decisiones** que la original, es del orden de **150 veces más rápida** sobre un dataset real de 26 categorías, usa **30 veces menos memoria** que la alternativa "vectorizada" obvia, y sigue funcionando en un caso donde la original falla.

---

## Qué problema resuelve

**QDA** (Análisis Discriminante Cuadrático) es un método estadístico clásico para clasificar: aprende cómo se ve cada categoría y, ante un caso nuevo, lo asigna a la categoría a la que más se parece. Es transparente (cada decisión se puede explicar) y no requiere ajustar muchos parámetros.

El problema es que la implementación directa de ese método:

- **Es lenta en producción.** Evalúa un caso y una categoría a la vez, repitiendo cálculos que podrían compartirse.
- **Es frágil con muchas variables.** Un cálculo intermedio (el "determinante") puede salirse del rango numérico de la computadora y producir resultados sin sentido, sin dar ningún error.
- **Las soluciones ingenuas se pagan en memoria.** Procesar todos los casos de una vez parece obvio, pero la memoria crece con el *cuadrado* de la cantidad de casos.

Este repositorio implementa **9 variantes** del mismo clasificador, demuestra que son equivalentes y mide cuánto cuesta cada una.

## Resultados

Medidos sobre datasets reales (*wine*, *letters*, 20.000 casos y 26 categorías) y sintéticos. Las magnitudes dependen de la computadora; los cocientes y las tendencias son lo que se sostiene.

| Qué se midió | Versión original | Versión recomendada (`EfficientChol`) | Mejora |
|---|---|---|---|
| **Tiempo de predicción**, lote de 1.000 casos, 26 categorías | ≈ 220 ms | ≈ 1,3 ms | **≈ 150×** |
| **Tiempo de predicción**, 512 variables, 300 casos | ≈ 2,8 s | ≈ 10 ms | **≈ 300×** |
| **Memoria** de la alternativa "vectorizada simple" (`FasterQDA`) vs. la recomendada, mismo lote | ≈ 200 MiB | ≈ 7 MiB | **≈ 30×** |
| **Tiempo de entrenamiento**, 512 variables | ≈ 48 ms | ≈ 30 ms | ≈ 1,6× |
| **Aciertos con 400 variables** (caso extremo) | **31 %** (≈ azar) | **100 %** | la original falla |
| **Decisiones del modelo** | — | idénticas en las 9 variantes | sin cambios |

### Lo más importante, en tres frases

1. **La mayor ganancia viene de evitar repeticiones, no de matemática sofisticada.** Procesar todos los casos juntos, en lugar de uno por uno, da la mejora principal.
2. **Hacerlo "a lo bruto" tiene un costo oculto.** La variante que procesa todo junto sin cuidado consume memoria proporcional al *cuadrado* de la cantidad de casos: ≈ 7 GiB para un lote de 6.000 casos con 26 categorías.
3. **La factorización de Cholesky vale más por confiabilidad que por velocidad.** Con pocas variables casi no se nota; con muchas, es la diferencia entre acertar y fallar sin aviso.

## ¿Qué versión usar?

| Situación | Variante | Por qué |
|---|---|---|
| **Producción (recomendada)** | `EfficientChol` | La más rápida en todas las pruebas, memoria baja y estable con muchas variables |
| Predecir de a un caso por vez | `TensorizedChol` | Procesa todas las categorías juntas; no necesita lote |
| Quiero la versión de referencia, fácil de leer | `QDA` | La más simple; lenta y frágil con muchas variables |
| Evitar | `FasterQDA` | Rápida con pocos casos, pero la memoria explota al crecer el lote |
| Evitar | `QDA_Chol2` | Con pocas variables es más lenta que la original |

## Cómo sabemos que es correcto

No alcanza con que "dé lo mismo en un ejemplo". Se verificó que:

- **Las 9 variantes coinciden con una referencia independiente** (construida con otra biblioteca, sin compartir código) en 4 datasets y 5 particiones distintas: **20 corridas × 9 variantes, sin una sola diferencia**.
- **Las propiedades matemáticas que justifican cada mejora se cumplen** sobre 50 casos aleatorios.
- **La salida siempre tiene la forma y el tipo esperados.**
- **El caso donde la original falla está documentado como prueba**: si alguien lo corrige, la prueba lo avisa.

Estas verificaciones corren de forma automática (ver "Cómo ejecutar").

## Limitaciones

- **Sin regularización.** Si una categoría tiene menos casos que variables, el método no puede ajustarse. Es una limitación del método, no de la implementación.
- **Mediciones en una sola computadora**, con un solo hilo de cálculo. Con varios hilos se observaron tiempos erráticos de entrenamiento; antes de decidir en un entorno real conviene volver a medir.
- **El objetivo es computacional.** No se busca mejorar la exactitud del modelo (≈ 88 % en *letters*), sino lograr las mismas decisiones a menor costo.

---

## Cómo ejecutar

Requiere [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).

```bash
uv sync                                  # instala dependencias
uv run --with jupyter jupyter lab        # abre el notebook
```

**Notebook principal:** [`TP_QDA_completo.ipynb`](TP_QDA_completo.ipynb) — resuelve los 14 ejercicios de la [consigna](consigna.md) con la teoría, las verificaciones y los experimentos. Tarda ≈ 2 minutos en ejecutarse completo. Para un resumen sin código, ver "Resultados" arriba.

**Pruebas automáticas:**

```bash
uv run python -m unittest discover -s tests -t . -v
```

Son 14 pruebas, ≈ 15 segundos. Cuatro de ellas son "fallas conocidas" a propósito (ver arriba). La prueba sobre el dataset *penguins* se omite sola si no hay internet.

## Contenido del repositorio

| Ruta | Descripción |
|---|---|
| `TP_QDA_completo.ipynb` | Resolución completa: teoría, verificación y mediciones |
| `consigna.md` | Enunciado del trabajo |
| `base/qda.py` | `QDA`, `TensorizedQDA`, `FasterQDA`, `EfficientQDA` |
| `base/cholesky.py` | `QDA_Chol1/2/3`, `TensorizedChol`, `EfficientChol` |
| `base/bayesian.py` | Clase base del clasificador |
| `utils/bench.py` | Herramienta de medición de tiempo y memoria |
| `utils/datasets.py` | Carga de datasets |
| `utils/verification.py` | Pruebas compartidas por el notebook y `tests/` |
| `tests/test_qda.py` | Pruebas automáticas |
| `P_1.ipynb`, `P_2.ipynb`, `P_3.ipynb` | Resolución previa del grupo, por ejercicio |
| `tp.ipynb`, `AMIA_2025_TP1.ipynb` | Plantilla y versión mono-notebook de la cátedra |

## Glosario

| Término | Qué significa |
|---|---|
| **QDA** | Clasificador estadístico que asigna cada caso a la categoría más parecida, cuya "forma" se modela por separado |
| **Tensorización / vectorización** | Calcular todo junto (todas las categorías, todos los casos) en una operación, en lugar de recorrerlos uno por uno |
| **Cholesky** | Una forma de descomponer una matriz que evita invertirla y permite calcular cantidades auxiliares de forma estable y barata |
| **Benchmark** | Medición repetida de tiempo y memoria bajo las mismas condiciones |
| **Oráculo** | Implementación independiente usada como referencia para verificar que los resultados son correctos |
