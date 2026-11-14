# tests/experience/test_experiencia-media.py
import os
import sys
import time
import random
from collections import deque
import numpy as np

try:
    from core.experience import ExperienceReplay
except ImportError:
    # Permite ejecutarlo directo en local o en Colab tras pegar ExperienceReplay en una celda previa
    if "__file__" in globals():
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from core.experience import ExperienceReplay
    elif "ExperienceReplay" not in globals():
        raise


# Mide en milisegundos cuánto tarda sacar n_batches lotes de una memoria
def medir_extraccion(memoria, es_deque, batch_size, n_batches):
    t0 = time.perf_counter()
    if es_deque:
        for _ in range(n_batches):
            lote = random.sample(memoria, batch_size)
            _ = (
                np.array([x[0] for x in lote]),
                np.array([x[1] for x in lote]),
                np.array([x[2] for x in lote]),
                np.array([x[3] for x in lote]),
                np.array([x[4] for x in lote]),
            )
    else:
        for _ in range(n_batches):
            _ = memoria.sample_batch(batch_size)
    return (time.perf_counter() - t0) * 1000


# Ejecuta pruebas funcionales y comparación de velocidad en 20 corridas (sirve igual en local y en Colab)
def probar_experience_en_colab():
    print("1. Prueba con memoria vacía")
    mem = ExperienceReplay(max_memory=10)
    try:
        mem.sample_batch(4)
    except ValueError as e:
        print(f"   Se lanzó el error esperado: {e}")

    print("\n2. Prueba de memoria circular (capacidad 3, guardando 5 jugadas)")
    mem = ExperienceReplay(max_memory=3, state_size=2, prioritized=False)
    for i in range(5):
        mem.remember([i, i], action=i % 4, reward=float(i), next_state=[i + 1, i + 1], done=False)
    print(f"   Jugadas guardadas (esperado 3): {len(mem)}")
    print(f"   Recompensas actuales (esperado [3. 4. 2.]): {mem.storage.rewards}")

    print("\n3. Prueba de cambio de tamaño entre niveles (11 -> 64 sensores)")
    mem = ExperienceReplay(max_memory=5)
    mem.remember(np.zeros(11), 0, 0.1, np.ones(11), False)
    print(f"   Nivel 1 -> matriz de estados: {mem.storage.states.shape}")
    mem.remember(np.zeros((8, 8)), 1, 1.0, np.ones((8, 8)), True)
    print(f"   Nivel 2 -> matriz adaptada:   {mem.storage.states.shape}")

    print("\n4. Prueba de prioridad hacia jugadas con recompensa alta")
    mem_uniforme = ExperienceReplay(max_memory=100, state_size=4, prioritized=False, seed=42)
    mem_torneo = ExperienceReplay(max_memory=100, state_size=4, prioritized=True, seed=42)
    for m in (mem_uniforme, mem_torneo):
        for _ in range(90):
            m.remember(np.zeros(4), 0, 0.0, np.zeros(4), False)
        for _ in range(10):
            m.remember(np.ones(4), 1, 10.0, np.ones(4), True)

    _, _, rew_u, _, _ = mem_uniforme.sample_batch(100)
    _, _, rew_t, _, _ = mem_torneo.sample_batch(100)
    print("   Manzanas reales en memoria: 10.0%")
    print(f"   Elegidas sin prioridad (uniforme): {np.mean(rew_u == 10.0) * 100:.1f}%")
    print(f"   Elegidas con prioridad (torneo):   {np.mean(rew_t == 10.0) * 100:.1f}%")

    print("\n5. Comparación de velocidad con 10,000 jugadas (20 corridas, 1,000 lotes de 64)")
    n_items, batch_size, n_batches, runs = 10000, 64, 1000, 20
    estado = np.zeros(11, dtype=np.float32)

    tiempos_deque, tiempos_uniforme, tiempos_torneo = [], [], []

    for _ in range(runs):
        legacy_mem = deque(maxlen=n_items)
        mem_u = ExperienceReplay(max_memory=n_items, state_size=11, prioritized=False)
        mem_t = ExperienceReplay(max_memory=n_items, state_size=11, prioritized=True)
        for i in range(n_items):
            legacy_mem.append((estado, i % 4, 0.1, estado, False))
            mem_u.remember(estado, i % 4, 0.1, estado, False)
            mem_t.remember(estado, i % 4, 0.1, estado, False)

        tiempos_deque.append(medir_extraccion(legacy_mem, True, batch_size, n_batches))
        tiempos_uniforme.append(medir_extraccion(mem_u, False, batch_size, n_batches))
        tiempos_torneo.append(medir_extraccion(mem_t, False, batch_size, n_batches))

    t_deq = np.array(tiempos_deque)
    t_uni = np.array(tiempos_uniforme)
    t_tor = np.array(tiempos_torneo)
    mejora_uni = t_deq / t_uni
    mejora_tor = t_deq / t_tor

    print(f"   deque clásico: {t_deq.mean():.2f} ms (desv. {t_deq.std():.2f} ms)")
    print(
        f"   Uniforme:      {t_uni.mean():.2f} ms (desv. {t_uni.std():.2f} ms) "
        f"-> {mejora_uni.mean():.2f}x más rápido (mín {mejora_uni.min():.2f}x, máx {mejora_uni.max():.2f}x)"
    )
    print(
        f"   Torneo (k=7):  {t_tor.mean():.2f} ms (desv. {t_tor.std():.2f} ms) "
        f"-> {mejora_tor.mean():.2f}x más rápido (mín {mejora_tor.min():.2f}x, máx {mejora_tor.max():.2f}x)"
    )

    stats = mem_t.get_telemetry()
    print(f"\n   RAM fija reservada para 10,000 jugadas: {stats['memory_bytes'] / 1024:.1f} KB")


if __name__ == "__main__":
    probar_experience_en_colab()
