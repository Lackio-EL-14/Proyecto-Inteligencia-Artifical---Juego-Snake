# core/experience.py
import numpy as np


#guardar todas las jugadas de Snake en bloques seguidos en ram, reserva una vez todo el espacio para no ir agrandando en cada paso del juego
class TransitionStorage:
    def __init__(self, capacity, state_size=11):
        self.states = np.empty((capacity, state_size), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty((capacity, state_size), dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.bool_)
        self.weights = np.ones(capacity, dtype=np.float32)

    # Guarda la jugada con pocision idx y da más peso si hubo recompensa fuerte (manzana o choque)
    def save_transition(self, idx, state, action, reward, next_state, done):
        self.states[idx] = state if isinstance(state, np.ndarray) and state.ndim == 1 else np.ravel(state)
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = (
            next_state if isinstance(next_state, np.ndarray) and next_state.ndim == 1 else np.ravel(next_state)
        )
        self.dones[idx] = done
        self.weights[idx] = 1.0 + abs(reward)

    # Saca el lote con .take() para evita copias demas y es mas rápido que indexar con []
    def get_transitions(self, indices):
        return (
            self.states.take(indices, axis=0),
            self.actions.take(indices),
            self.rewards.take(indices),
            self.next_states.take(indices, axis=0),
            self.dones.take(indices),
        )


# Memoria que guarda las partidas recientes y solo saca lotes para entrenar la red neuronal
#por defecto esta el modo torneo  con k=7 y los índices base para lotes
class ExperienceReplay:
    def __init__(self, max_memory=10000, state_size=None, prioritized=True, tournament_k=7, seed=None):
        self.max_memory = max_memory
        self.prioritized = prioritized
        self.tournament_k = tournament_k
        self.write_pos = 0
        self.size = 0
        self.state_dim = state_size or 0
        self.storage = TransitionStorage(max_memory, state_size) if state_size else None
        self.rng = np.random.default_rng(seed)
        self.row_idx = np.arange(64)

    # Guarda una experiencia nueva si cambia la cantidad de sensores entre niveles y reinicia la tabla
    def remember(self, state, action, reward, next_state, done):
        dim = state.size if isinstance(state, np.ndarray) else np.size(state)
        if dim != self.state_dim:
            self.storage = TransitionStorage(self.max_memory, dim)
            self.state_dim = dim
            self.write_pos = 0
            self.size = 0

        self.storage.save_transition(self.write_pos, state, action, reward, next_state, done)
        self.write_pos = (self.write_pos + 1) % self.max_memory
        if self.size < self.max_memory:
            self.size += 1

    # crear un lote de entrenamiento eligiendo al azar simple o por torneo entre k candidatos según su peso
    def sample_batch(self, batch_size):
        if self.size == 0:
            raise ValueError("Memoria vacía.")

        n = batch_size if batch_size < self.size else self.size
        if self.prioritized:
            if n > self.row_idx.size:
                self.row_idx = np.arange(n)
            candidates = self.rng.integers(0, self.size, size=(n, self.tournament_k))
            winners = self.storage.weights.take(candidates).argmax(axis=1)
            indices = candidates[self.row_idx[:n], winners]
        else:
            indices = self.rng.integers(0, self.size, size=n)

        return self.storage.get_transitions(indices)

    # Ajusta los pesos después de entrenar segun cuánto se equivocó en esas jugadas
    def update_weights(self, indices, td_errors):
        if self.storage is not None:
            self.storage.weights[indices] = np.abs(td_errors) + 1e-5

    # telemetria de la cantidad de memoria ram ocupada por la tabla y qué porcentaje de jugadas tienen premio o castigo fuerte
    def get_telemetry(self):
        if self.size == 0 or self.storage is None:
            return {"size": 0, "memory_bytes": 0, "critical_ratio": 0.0}
        s = self.storage
        mem_bytes = (
            s.states.nbytes
            + s.actions.nbytes
            + s.rewards.nbytes
            + s.next_states.nbytes
            + s.dones.nbytes
            + s.weights.nbytes
        )
        critical = float(np.mean(np.abs(s.rewards[: self.size]) >= 1.0))
        return {"size": self.size, "memory_bytes": int(mem_bytes), "critical_ratio": critical}

    #
    def __len__(self):
        return self.size
