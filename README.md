# Agente Inteligente para Snake Modificado (Double DQN)

Proyecto final de curso para la asignatura **SIS-252 Inteligencia Artificial** (UCB Sede Cochabamba).

## Objetivo
Un agente que aprende a jugar una variante de Snake por niveles (tableros de 8×8 a 12×12, con obstáculos fijos y varias manzanas) mediante **Double Deep Q-Learning** con una red neuronal feed-forward en Keras. El *baseline* teórico es Zafrany, *Deep reinforcement learning for maze solving*.

Esta primera entrega usa una física simplificada: la serpiente **no crece** al comer. El crecimiento se activa sin cambiar código (`SNAKE_GROWS` en `config.py`, o el campo `"snake_grows"` en el JSON de un nivel).

## Equipo de Desarrollo
* Javier Antezana Vallejos
* Edson Lamas Alcocer
* Mateo Wilson Quispe Aylas
* Isabel Hidalgo Avila

## Estructura

| Archivo | Contenido |
|---|---|
| `config.py` | Todos los hiperparámetros, las recompensas y las semillas. |
| `snake_env.py` | Física del juego: movimiento, choques, manzanas, victoria, hambre y recompensa. |
| `state.py` | Sensores del agente: el vector de estado de 7 valores. |
| `baselines.py` | Prueba de suficiencia del estado: política heurística y aleatoria, sin red. |
| `agent.py` | Red Keras, política ε-greedy, memoria de repetición y paso de entrenamiento Double DQN. |
| `train.py` | Bucle de entrenamiento con los 4 niveles mezclados; métricas en CSV. |
| `evaluate.py` | Evaluación voraz (ε = 0) con semillas fijas, 100 episodios por nivel. |
| `plots.py` | Gráficas para el informe. |
| `viewer.py` | Visor pygame para ver jugar al agente y analizar sus fallos. |
| `tests/` | Pruebas del entorno, el estado, la memoria y el paso de entrenamiento. |
| `levels/` | Definición de los 4 niveles. |

## Diseño en breve
- **Estado (7 valores):** para cada acción posible (seguir, girar a la derecha, girar a la izquierda), si choca (`blocked`) y si acerca a la manzana por el camino real calculado con BFS (`closer`); además, la distancia BFS normalizada a la manzana.
- **Acciones:** 3 relativas a la dirección actual.
- **Recompensa:** +1 por manzana, −1 por choque, −0.01 por paso. Si pasan filas × columnas pasos sin comer, el episodio se corta (truncamiento, no estado terminal).
- **Red:** `Dense(64, relu) → Dense(64, relu) → Dense(3, linear)`, Adam (lr 5e-4), pérdida Huber.
- **Double DQN:** la red online elige la mejor acción en el estado siguiente y la red objetivo (copiada cada 250 pasos) da su valor: `y = r + γ·(1−done)·Q_target(s', argmax_a Q_online(s', a))`, con γ = 0.95.
- **Memoria de repetición** uniforme de 20 000 transiciones, mini-batch de 64.

## Uso

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python -m unittest discover -s tests -t .
python baselines.py
python train.py --run-name completa
python evaluate.py --weights runs/completa/final.weights.h5
python plots.py --run runs/completa
python viewer.py --level 4 --policy agent --weights runs/completa/final.weights.h5
```

Los resultados se guardan en `runs/`, que git ignora.

**Teclas del visor:** espacio pausa · `n` un paso · `+`/`-` velocidad · `r` siguiente episodio · `esc` salir. Con `--failures` reproduce solo los episodios perdidos en la última evaluación.

## Resultados (sin crecimiento, 100 episodios por nivel, ε = 0, semillas 10000–10099)

| Política | Nivel 1 | Nivel 2 | Nivel 3 | Nivel 4 |
|---|---|---|---|---|
| Aleatoria | 15.5 % | 0 % | 0 % | 0 % |
| Heurística (solo el estado, sin red) | 100 % | 100 % | 100 % | 100 % |
| **Double DQN** | **100 %** | **100 %** | **100 %** | **100 %** |

Las dos primeras filas son de `baselines.py` (200 episodios por nivel). El entrenamiento completo (1500 episodios) tarda unos 3 minutos en CPU.
