"""Ruido gaussiano + geracao de covariancia valida para o simulador."""

from __future__ import annotations

import numpy as np


def add_gaussian_noise(
    state: np.ndarray, position_std: float, velocity_std: float, rng: np.random.Generator
) -> np.ndarray:
    noisy = state.copy()
    noisy[:3] += rng.normal(0.0, position_std, size=3)
    noisy[3:6] += rng.normal(0.0, velocity_std, size=3)
    return noisy


def diagonal_covariance(position_std: float, velocity_std: float) -> np.ndarray:
    std = np.array([position_std] * 3 + [velocity_std] * 3)
    return np.diag(std**2)
