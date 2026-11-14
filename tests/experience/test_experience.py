# tests/experience/test_experience.py
import os
import sys
import time
import unittest
from collections import deque
import random
import numpy as np

try:
    from core.experience import ExperienceReplay
except ImportError:
    if "__file__" in globals():
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from core.experience import ExperienceReplay
    elif "ExperienceReplay" not in globals():
        raise


# Suite de pruebas unitarias y rendimiento para ExperienceReplay.
class ExperienceReplayTests(unittest.TestCase):
    # Comprueba que pedir un lote con la memoria vacía lance ValueError.
    def test_empty_memory_raises_value_error(self):
        memory = ExperienceReplay(max_memory=10)
        with self.assertRaises(ValueError):
            memory.sample_batch(4)

    # Comprueba que la memoria circular sobrescriba las jugadas más viejas al llenarse.
    def test_ring_buffer_overwrites_oldest_transitions(self):
        memory = ExperienceReplay(max_memory=3, state_size=2, prioritized=False)
        for i in range(5):
            memory.remember([i, i], action=i % 4, reward=float(i), next_state=[i + 1, i + 1], done=False)

        self.assertEqual(len(memory), 3)
        np.testing.assert_array_equal(memory.storage.rewards, np.array([3.0, 4.0, 2.0], dtype=np.float32))

    # Verifica que si se pide un lote mayor a las jugadas guardadas, devuelva solo las disponibles.
    def test_sample_batch_clamps_to_available_size(self):
        for prioritized in (False, True):
            memory = ExperienceReplay(max_memory=20, state_size=4, prioritized=prioritized, seed=42)
            for i in range(5):
                memory.remember(np.zeros(4), i % 4, float(i), np.ones(4), False)

            states, actions, rewards, next_states, dones = memory.sample_batch(64)
            self.assertEqual(states.shape, (5, 4))
            self.assertEqual(actions.shape, (5,))
            self.assertEqual(rewards.shape, (5,))
            self.assertEqual(next_states.shape, (5, 4))
            self.assertEqual(dones.shape, (5,))

    # Verifica que la memoria se reinicie sin fallar si cambia el tamaño de entrada entre niveles.
    def test_auto_resizes_when_state_dimension_changes(self):
        memory = ExperienceReplay(max_memory=5)
        memory.remember(np.zeros(11), 0, 0.1, np.ones(11), False)
        self.assertEqual(memory.storage.states.shape, (5, 11))

        memory.remember(np.zeros((8, 8)), 1, 1.0, np.ones((8, 8)), True)
        self.assertEqual(memory.storage.states.shape, (5, 64))
        self.assertEqual(len(memory), 1)

    # Comprueba que el modo torneo elija jugadas críticas mucho más seguido que el modo uniforme.
    def test_tournament_sampling_favors_critical_rewards(self):
        uniform_mem = ExperienceReplay(max_memory=100, state_size=4, prioritized=False, seed=42)
        prioritized_mem = ExperienceReplay(max_memory=100, state_size=4, prioritized=True, seed=42)

        for mem in (uniform_mem, prioritized_mem):
            for _ in range(90):
                mem.remember(np.zeros(4), 0, 0.0, np.zeros(4), False)
            for _ in range(10):
                mem.remember(np.ones(4), 1, 10.0, np.ones(4), True)

        _, _, uniform_rewards, _, _ = uniform_mem.sample_batch(100)
        _, _, prioritized_rewards, _, _ = prioritized_mem.sample_batch(100)

        self.assertLess(np.mean(uniform_rewards == 10.0), 0.20)
        self.assertGreater(np.mean(prioritized_rewards == 10.0), 0.35)

    # Verifica la actualización de pesos y el reporte de telemetría vacía y con datos.
    def test_update_weights_and_telemetry(self):
        memory = ExperienceReplay(max_memory=10, state_size=4)
        self.assertEqual(memory.get_telemetry(), {"size": 0, "memory_bytes": 0, "critical_ratio": 0.0})

        memory.remember([1, 2, 3, 4], 2, 1.0, [4, 3, 2, 1], True)
        memory.update_weights(np.array([0]), np.array([-2.5]))

        self.assertAlmostEqual(float(memory.storage.weights[0]), 2.50001, places=4)
        stats = memory.get_telemetry()
        self.assertEqual(stats["size"], 1)
        self.assertEqual(stats["critical_ratio"], 1.0)
        self.assertEqual(stats["memory_bytes"], 10 * (4 * 4 + 4 + 4 + 4 * 4 + 1 + 4))

    # Compara el tiempo de muestreo frente al deque tradicional en modo uniforme y por torneo.
    def test_sampling_speed_vs_deque(self):
        n_items = 10000
        batch_size = 64
        n_batches = 500
        sample_state = np.zeros(11, dtype=np.float32)

        legacy_mem = deque(maxlen=n_items)
        replay_mem = ExperienceReplay(max_memory=n_items, state_size=11, prioritized=False, seed=42)
        for i in range(n_items):
            legacy_mem.append((sample_state, i % 4, 0.1, sample_state, False))
            replay_mem.remember(sample_state, i % 4, 0.1, sample_state, False)

        t0 = time.perf_counter()
        for _ in range(n_batches):
            batch = random.sample(legacy_mem, batch_size)
            _ = (
                np.array([x[0] for x in batch]),
                np.array([x[1] for x in batch]),
                np.array([x[2] for x in batch]),
                np.array([x[3] for x in batch]),
                np.array([x[4] for x in batch]),
            )
        legacy_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for _ in range(n_batches):
            _ = replay_mem.sample_batch(batch_size)
        uniform_ms = (time.perf_counter() - t0) * 1000

        replay_mem.prioritized = True
        t0 = time.perf_counter()
        for _ in range(n_batches):
            _ = replay_mem.sample_batch(batch_size)
        tournament_ms = (time.perf_counter() - t0) * 1000

        print(
            f"\nComparación (500 lotes de 64) -> "
            f"deque: {legacy_ms:.2f} ms | "
            f"uniforme: {uniform_ms:.2f} ms ({legacy_ms / uniform_ms:.1f}x) | "
            f"torneo: {tournament_ms:.2f} ms ({legacy_ms / tournament_ms:.1f}x)"
        )
        self.assertLess(uniform_ms, legacy_ms)
        self.assertLess(tournament_ms, legacy_ms)


if __name__ == "__main__":
    unittest.main()
