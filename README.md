# Agente Inteligente para Snake Modificado (DQN)

Proyecto final de curso para la asignatura **SIS-252 Inteligencia Artificial** (UCB Sede Cochabamba).

## Objetivo del Proyecto
Desarrollar un agente inteligente capaz de navegar y sobrevivir de forma autónoma en una variación progresiva del juego "Snake". A diferencia de los entornos estáticos, el agente debe equilibrar la recolección de objetivos (manzanas) con la evasión de obstáculos (muros del nivel y su propio cuerpo en crecimiento) a lo largo de múltiples niveles de dificultad.

El sistema de toma de decisiones está basado en **Aprendizaje por Refuerzo (Deep Q-Learning)** utilizando Redes Neuronales Feedforward, tomando como *baseline* la arquitectura de resolución de laberintos de Zafrany (*Deep reinforcement learning for maze solving*) y expandiéndola para entornos dinámicos, penalizaciones complejas y aprendizaje por currículo (*curriculum learning*).

## Equipo de Desarrollo
* Javier Antezana Vallejos
* Edson Lamas Alcocer
* Mateo Wilson Quispe Aylas
* Isabel Hidalgo Avila

## Estructura del Repositorio
El código está diseñado bajo el Principio de Responsabilidad Única (SRP): el entorno, el agente y el bucle de entrenamiento no dependen unos de otros más allá de sus interfaces.

| Carpeta | Contenido |
|---|---|
| `core/` | Motor del juego (`snake_env.py`), movimientos (`movements.py`), sensores espaciales con BFS (`spatial_sensors.py`), función de recompensa (`rewards.py`) y memoria de experiencias (`experience.py`, replay buffer con muestreo priorizado por torneo). |
| `agent/` | Agente DQN: red neuronal (`network.py`), red online + red objetivo y pasos compilados con `tf.function` (`dqn_model.py`), objetivos de Bellman y pérdida como funciones puras (`q_math.py`), política ε-greedy (`exploration.py`) y lotes de entrenamiento (`batch.py`, `agent_spec.py`). |
| `training/` | Entrenamiento por etapas (`train.py`), elección de nivel por episodio y repaso (`curriculum.py`), evaluación voraz (`evaluation.py`), checkpoints, métricas CSV y salida por consola. |
| `levels/` | Archivos JSON que definen el tamaño del tablero, los obstáculos y las manzanas necesarias para ganar cada nivel. |
| `utils/` | Hiperparámetros (`config.py`), semillas (`seeding.py`), selección de CPU/GPU (`device.py`), gráficas (`plotting.py`) y dibujo del tablero en consola (`console_render.py`). |
| `tools/` | Herramientas de desarrollo: medición de rendimiento (`benchmark_agent.py`) y verificación de estilo (`check_clean_code.py`). |
| `tests/` | Pruebas unitarias con `unittest`, organizadas con la misma estructura que el código. |

## Instalación y Configuración

1. **Clonar el repositorio:**
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd snake_ai_project
   ```

2. **Crear y activar un entorno virtual (recomendado):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instalar las dependencias** (`numpy`, `tensorflow` y `matplotlib`):
   ```bash
   pip install -r requirements.txt
   ```

Todos los comandos se ejecutan desde la raíz del proyecto, porque los imports son absolutos (`from core...`, `from agent...`).

## Uso

`main.py` tiene **tres subcomandos**: `train`, `evaluate` y `play`.

```bash
python main.py train    [--run-name NOMBRE] [--seed N]
python main.py evaluate --weights RUTA [--episodes N] [--seed N]
python main.py play     --weights RUTA --level N [--delay SEGUNDOS] [--seed N]
```

`python main.py --help` o `python main.py <subcomando> --help` muestra todas las opciones.

### ¿Por qué tres formas de ejecutar `main.py`?

Entrenar, medir y observar al agente responden a preguntas distintas y tienen costos muy distintos. Por eso son tres comandos separados en lugar de uno solo:

| Subcomando | Pregunta que responde | Qué hace | Qué produce |
|---|---|---|---|
| `train` | *¿Puede el agente aprender a jugar?* | Entrena desde cero, nivel por nivel, explorando con ε-greedy. Es el único modo que modifica la red. | Una carpeta `runs/<nombre>/` con pesos, métricas y gráficas. |
| `evaluate` | *¿Qué tan bien juega un modelo ya entrenado?* | Carga unos pesos y juega `--episodes` partidas **voraces** (sin exploración) en **todos** los niveles, con semillas fijas. | Una tabla en consola con el win rate y las estadísticas de cada nivel. |
| `play` | *¿Cómo juega, en la práctica?* | Carga unos pesos y juega **una** partida voraz en un nivel, dibujando el tablero cuadro por cuadro. | Una animación en la terminal y el resultado de la partida. |

**Razones de la separación:**

- **El entrenamiento es caro y la evaluación no.** Entrenar las cuatro etapas puede tomar miles de episodios. Con `evaluate` se puede medir cualquier checkpoint guardado (el mejor de una etapa, el último o el final) sin volver a entrenar.
- **Medir sin exploración.** Durante `train` el agente toma acciones aleatorias a propósito, así que el desempeño de esos episodios no refleja lo que aprendió. `evaluate` y `play` usan siempre la mejor acción según la red (política voraz).
- **Resultados comparables y reproducibles.** `evaluate` usa las mismas semillas de evaluación en cada ejecución, de modo que dos modelos (o dos corridas con hiperparámetros distintos) se comparan en exactamente las mismas partidas.
- **Análisis cuantitativo frente a cualitativo.** `evaluate` da números para el informe; `play` permite ver comportamientos que los números no muestran (por ejemplo, si la serpiente da vueltas en círculo hasta agotar el límite de pasos) y sirve para la demostración en vivo.
- **Código simple.** Cada subcomando es una función propia en `main.py` (`command_train`, `command_evaluate`, `command_play`), y todos reutilizan las mismas piezas de `training/`, en lugar de un único `main` lleno de condicionales.

### Flujo de trabajo típico

```bash
# 1. Entrenar (los resultados quedan en runs/experimento_1/)
python main.py train --run-name experimento_1

# 2. Medir el modelo final en los cuatro niveles
python main.py evaluate --weights runs/experimento_1/models/final.weights.h5

# 3. Comparar con el mejor checkpoint de una etapa concreta
python main.py evaluate --weights runs/experimento_1/models/stage_3_best.weights.h5

# 4. Ver una partida en el nivel 4, un cuadro cada 0.1 s
python main.py play --weights runs/experimento_1/models/final.weights.h5 --level 4
```

**El orden importa:** `evaluate` y `play` necesitan un archivo de pesos que ya exista. Si se ejecutan antes de entrenar, fallan con `FileNotFoundError`. Además, `final.weights.h5` se escribe solo cuando `train` termina las cuatro etapas. Mientras el entrenamiento sigue en curso se pueden usar los checkpoints de las etapas ya terminadas, por ejemplo `stage_1_best.weights.h5`.

`evaluate` y `play` construyen la red con la configuración actual de `utils/config.py` (`STATE_SIZE` y `HIDDEN_UNITS`), así que los pesos deben provenir de una corrida con la misma arquitectura. Los pesos entrenados con el estado anterior de 11 valores ya no se pueden cargar.

### Archivos generados por `train`

Si no se indica `--run-name`, la carpeta se nombra con la fecha y hora de inicio.

| Ruta | Contenido |
|---|---|
| `runs/<nombre>/models/` | `stage_<k>_best.weights.h5` (mejor evaluación voraz de la etapa *k*), `stage_<k>_last.weights.h5` (fin de la etapa) y `final.weights.h5`. |
| `runs/<nombre>/metrics/` | `episodes.csv` (un registro por episodio), `evaluations.csv` (evaluaciones periódicas y de fin de etapa) y `curriculum_matrix.csv`. |
| `runs/<nombre>/plots/` | `learning_curve_level_<k>.png`, `eval_win_rate.png` y `curriculum_matrix.png`. |

La **matriz curricular** tiene una fila por etapa terminada y una columna por nivel: muestra el win rate voraz del agente en cada nivel al terminar cada etapa. Permite ver tanto si el agente aprende el nivel nuevo como si **olvida** los anteriores.

## Niveles

| Nivel | Tablero | Obstáculos | Manzanas para ganar |
|---|---|---|---|
| 1 | 8 × 8 | 0 | 1 |
| 2 | 8 × 8 | 7 | 3 |
| 3 | 10 × 10 | 14 | 5 |
| 4 | 12 × 12 | 27 | 8 |

La serpiente empieza con longitud 2. Un episodio termina al ganar, al chocar (con un muro, un obstáculo o su propio cuerpo), al pasar `filas × columnas` pasos sin comer (límite de hambre) o al alcanzar el límite de `MAX_STEPS_PER_EPISODE` pasos.

## Cómo aprende el agente

### Estado, acciones y recompensas

**Estado:** vector de 20 valores entre 0 y 1.

| Valores | Contenido |
|---|---|
| 3 | peligro adelante / a la derecha / a la izquierda |
| 4 | dirección actual |
| 4 | dirección de la manzana en línea recta (arriba / derecha / abajo / izquierda) |
| 9 | **sensores espaciales:** por cada movimiento (recto, derecha, izquierda), la distancia por camino real (BFS) hasta la manzana, el área libre alcanzable (flood-fill) y si la manzana es alcanzable |

Los sensores espaciales existen porque, con solo los 11 primeros valores, el agente no podía distinguir "la manzana está arriba con camino libre" de "la manzana está arriba detrás de un muro". En los niveles 3 y 4, donde casi la mitad de las situaciones exigen rodear un muro, el aprendizaje se estancaba. El BFS **no elige la acción**: solo describe el entorno, y la red sigue aprendiendo qué hacer.

**Acciones:** `UP = 0`, `RIGHT = 1`, `DOWN = 2`, `LEFT = 3`. Una acción opuesta a la dirección actual se ignora.

**Recompensas:**
- `+1` por comer una manzana y `−1` por chocar.
- En los demás pasos: un costo de `−0.01` más un **moldeado basado en potencial** (Ng, Harada y Russell, 1999):

  $$F(s, s') = \gamma\,\Phi(s') - \Phi(s), \qquad \Phi(s) = -\frac{\text{distancia BFS}(\text{cabeza}, \text{manzana})}{\text{casillas libres}}$$

  Acercarse a la manzana **por el camino real** da recompensa positiva, aunque haya que alejarse en línea recta para rodear un muro. Este moldeado no cambia la política óptima y hace que dar vueltas en círculo tenga un costo.
- **Límite de hambre:** si pasan `filas × columnas` pasos sin comer, el episodio se corta (`starvation`).

### Red y objetivo de entrenamiento

La red es densa (20 → 128 → 128 → 4, ReLU) y devuelve un valor Q por acción. Se entrena con **Double DQN**:

$$a^* = \arg\max_{a'} Q_{\text{online}}(s', a'), \qquad y = r + \gamma\,(1 - \text{done})\,Q_{\text{target}}(s', a^*)$$

con $\gamma = 0.95$. `done` solo vale 1 si el episodio terminó por victoria o choque; un corte por límite de pasos **no** se trata como estado terminal. La pérdida es Huber y el optimizador es Adam con recorte de gradiente.

### Qué se conserva del baseline de Zafrany y qué se agregó

| Se conserva | Se agregó | Para qué |
|---|---|---|
| Red feed-forward densa | Red objetivo separada, sincronizada cada `TARGET_UPDATE_EVERY` pasos de gradiente | Estabilizar los objetivos de Bellman |
| Objetivo de Bellman | Double DQN | Reducir la sobreestimación de los valores Q |
| Experience replay | Muestreo priorizado, warmup inicial y entrenamiento cada `TRAIN_EVERY` pasos | Aprovechar mejor las experiencias y acelerar el entrenamiento |
| Política ε-greedy con decaimiento | ε se reabre a `EPSILON_STAGE_START` al comenzar cada etapa | Volver a explorar cuando aparece un nivel nuevo |
| Moldeado de recompensa | Distancia por camino real (BFS) en lugar de Manhattan, costo por paso y límite de hambre | No castigar los desvíos correctos alrededor de los muros y cortar los ciclos |
| Estado mínimo de sensores locales | 9 sensores espaciales (BFS y flood-fill por movimiento) | Ver por dónde se rodea un muro y evitar los callejones sin salida |
| Entrenamiento por etapas | Paso de etapa por evaluación voraz (win rate ≥ `TARGET_WIN_RATE`) y repaso de niveles anteriores con probabilidad `REHEARSAL_PROB` | Avanzar solo cuando el nivel está dominado y evitar el olvido |

### Curriculum learning

El agente entrena una etapa por nivel, en orden. Cada `EVAL_EVERY` episodios se evalúa de forma voraz en el nivel de la etapa; si el win rate alcanza `TARGET_WIN_RATE`, pasa a la siguiente. Si no lo logra en `MAX_EPISODES_PER_STAGE` episodios, avanza de todos modos. Al final de cada etapa se evalúan todos los niveles para construir la matriz curricular.

## Configuración

Todos los hiperparámetros están en `utils/config.py`. Los más relevantes:

| Constante | Valor | Significado |
|---|---|---|
| `STATE_SIZE` | `20` | Tamaño del estado (11 sensores locales + 9 espaciales) |
| `HIDDEN_UNITS` | `(128, 128)` | Neuronas de las capas ocultas |
| `LEARNING_RATE` | `5e-4` | Tasa de aprendizaje de Adam |
| `GAMMA` | `0.95` | Factor de descuento |
| `TARGET_RULE` | `"double"` | `"double"` (Double DQN) o `"dqn"` (DQN con red objetivo) |
| `MAX_MEMORY` / `BATCH_SIZE` | `50 000` / `64` | Capacidad del replay buffer y tamaño del lote |
| `EPSILON_START` / `EPSILON_MIN` / `EPSILON_DECAY` | `1.0` / `0.05` / `0.995` | Exploración |
| `TARGET_WIN_RATE` | `0.90` | Win rate voraz necesario para superar una etapa |
| `MAX_EPISODES_PER_STAGE` | `5000` | Límite de episodios por etapa |
| `MAX_STEPS_PER_EPISODE` | `500` | Límite de pasos por partida |
| `EVAL_EVERY` / `EVAL_EPISODES` | `50` / `50` | Frecuencia y tamaño de las evaluaciones voraces |
| `APPLE_REWARD` / `DEATH_REWARD` / `STEP_REWARD` | `1.0` / `−1.0` / `−0.01` | Recompensas |
| `HUNGER_FACTOR` | `1.0` | Límite de hambre en múltiplos de `filas × columnas` |
| `SEED` | `42` | Semilla por defecto (toda la corrida es reproducible) |
| `FORCE_CPU` | `True` | Oculta la GPU a TensorFlow |

`FORCE_CPU = True` es intencional: con una red de este tamaño y lotes de 64, la CPU es suficiente y evita depender de la configuración de CUDA. Para usar la GPU, cambiarlo a `False`.

Al arrancar, TensorFlow imprime avisos propios (oneDNN, instrucciones de CPU y, si CUDA no está disponible, `failed call to cuInit`). No son errores del proyecto: el programa sigue en CPU con normalidad.

## Interfaz del entorno

`SnakeEnv` utiliza una interfaz de aprendizaje por refuerzo con retornos estilo Gymnasium, sin requerir instalar Gymnasium.

`reset(seed=None, options=None)` inicia un episodio y devuelve `(observation, info)`. `observation` es un vector `float32` de 20 características (ver *Estado, acciones y recompensas*). Los 11 primeros valores son los sensores locales originales y los 9 últimos son los sensores espaciales. `info` incluye el identificador del nivel y el contador de manzanas.

`step(action)` recibe una acción y devuelve `(observation, reward, terminated, truncated, info)`. `terminated` indica una victoria o colisión. `truncated` indica que se alcanzó el límite de pasos (`time_limit`) o el límite de hambre (`starvation`). `info["reason"]` identifica el resultado del paso o el motivo de fin (`moved`, `win`, `time_limit`, `starvation` o el tipo de colisión: `wall`, `obstacle` o `body`).

El constructor es `SnakeEnv(ruta_nivel, max_steps=None, reward_spec=None, hunger_factor=None)`. El límite de pasos y el de hambre se pasan desde `training/factories.py`. Construido directamente, sin `hunger_factor`, el entorno no tiene límite de hambre. Al guardar una transición truncada, el replay buffer conserva `terminated` como máscara terminal para el cálculo de Bellman; el ciclo del episodio termina con `terminated or truncated`.

## Rendimiento

Medido en CPU con `python -m tools.benchmark_agent --cpu`, comparando la versión inicial del agente (`model.predict` + `train_on_batch` en cada paso) con la actual (funciones compiladas con `tf.function`):

| Medición | Versión inicial | Versión actual | Mejora |
|---|---|---|---|
| Elegir una acción | 33.3 ms | 0.096 ms | ≈ 346× |
| Paso de entrenamiento (lote de 64) | 71.5 ms | 0.46 ms | ≈ 155× |
| Paso completo del entorno | 101.9 ms | 0.52 ms | ≈ 195× |

Las dos primeras filas usan la misma red (64 × 64) en ambas versiones. La tercera compara la versión inicial con la configuración actual por defecto (red de 128 × 128, entrenamiento cada 4 pasos). Un episodio de 500 pasos pasó de tardar unos 50 s a menos de medio segundo.

Las mediciones de la tabla son anteriores a los sensores espaciales. Los sensores agregan entre 0.05 ms por paso en el nivel 1 y unos 0.3 ms en el nivel 4, por el BFS. A cambio, los niveles 3 y 4 se vuelven aprendibles: en una corrida corta de 600 episodios por etapa, el win rate voraz del nivel 3 pasó de un máximo de 0.16 a 0.76.

## Desarrollo

```bash
# Ejecutar todas las pruebas
python -m unittest discover -s tests -t .

# Verificar que los archivos cumplan las reglas de estilo del proyecto
python tools/check_clean_code.py <archivos.py>

# Medir el rendimiento del agente (--skip-legacy mide solo la versión actual)
python -m tools.benchmark_agent --cpu --skip-legacy
```

**Reglas de estilo** que verifica `check_clean_code.py`: sin comentarios ni docstrings en los `.py` (el código se explica con nombres descriptivos y type hints) y ninguna función con más de 25 líneas de cuerpo. Además, el proyecto sigue estas convenciones: máximo 4 parámetros por función, sin estado global (el generador aleatorio se inyecta), y todos los hiperparámetros en `utils/config.py`. Identificadores en inglés; mensajes de consola en español.
