# utils/config.py
# Configuración global del proyecto.
# Todo valor que se quiera cambiar para experimentar se modifica aquí,
# no dentro de train.py ni de main.py.

# --- Rutas ---
LEVEL_PATHS = [
    "levels/level_1.json",
    "levels/level_2.json",
    "levels/level_3.json",
    "levels/level_4.json",
]
MODELS_DIR = "models"
WEIGHTS_FILE = "models/dqn_snake_weights.h5"
PLOTS_DIR = "plots"

# --- Agente y red neuronal ---
STATE_SIZE = 11         # Vector de estado (peligros, dirección, manzana)
ACTION_SIZE = 4         # UP, RIGHT, DOWN, LEFT
LEARNING_RATE = 0.001

# --- Q-Learning ---
GAMMA = 0.95            # Factor de descuento para la Ecuación de Bellman
BATCH_SIZE = 64         # Tamaño del lote para el entrenamiento
MAX_MEMORY = 10000      # Límite del Replay Buffer

# --- Exploración (Epsilon-Greedy) ---
EPSILON_START = 1.0     # 100% de exploración al inicio
EPSILON_MIN = 0.05      # Nunca explorar menos del 5%
EPSILON_DECAY = 0.995   # Tasa de decaimiento por episodio

# --- Currículo ---
TARGET_WIN_RATE = 0.90          # Win rate necesario para pasar de nivel
WINDOW_SIZE = 100               # Ventana de episodios para medir el win rate
MAX_EPISODES_PER_STAGE = 5000   # Límite de episodios por nivel (evita bucles infinitos)