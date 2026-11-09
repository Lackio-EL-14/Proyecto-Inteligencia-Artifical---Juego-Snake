# Hiperparámetros Globales del Entrenamiento
LEARNING_RATE = 0.001
GAMMA = 0.95            # Factor de descuento para la Ecuación de Bellman
BATCH_SIZE = 64         # Tamaño del lote para el entrenamiento
MAX_MEMORY = 10000      # Límite del Replay Buffer

# Parámetros de Exploración (Epsilon-Greedy)
EPSILON_START = 1.0     # 100% de exploración al inicio
EPSILON_MIN = 0.05      # Nunca explorar menos del 5%
EPSILON_DECAY = 0.995   # Tasa de decaimiento por episodio

# Condiciones de Victoria Curricular
TARGET_WIN_RATE = 0.90  # El agente debe ganar el 90% de los episodios
WINDOW_SIZE = 100       # Ventana de episodios para evaluar el win rate
