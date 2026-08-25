"""Trajetorias de ground truth simples (Constant Velocity) para o
simulador. Objetivo do simulador: testar software e contrato, nao
produzir resultados cientificos - por isso nao precisa reproduzir
manobras complexas."""

from __future__ import annotations

import numpy as np


def straight_line(
    initial_state: np.ndarray, duration_s: float, dt: float, t0: float = 0.0
) -> list[tuple[float, np.ndarray]]:
    samples: list[tuple[float, np.ndarray]] = []
    state = np.asarray(initial_state, dtype=float).copy()
    t = t0
    samples.append((t, state.copy()))

    steps = int(round(duration_s / dt))
    for _ in range(steps):
        state = state.copy()
        state[:3] += state[3:6] * dt
        t += dt
        samples.append((t, state.copy()))

    return samples
