"""Metricas de erro de trajetoria (RMSE) e continuidade/disponibilidade,
usando o `ground_truth` embutido nos LocalTracklet e um mapeamento de
identidade majoritaria (ver association_metrics.majority_label_per_global_id).
"""

from __future__ import annotations

import numpy as np


def position_velocity_rmse(
    state_snapshots: list[tuple[float, int, np.ndarray]],
    ground_truth_trajectories: dict[str, list[tuple[float, np.ndarray]]],
    majority_labels: dict[int, str],
) -> tuple[float, float]:
    truth_lookup: dict[tuple[str, float], np.ndarray] = {}
    for uav_name, trajectory in ground_truth_trajectories.items():
        for t, state in trajectory:
            truth_lookup[(uav_name, round(t, 6))] = state

    position_errors: list[float] = []
    velocity_errors: list[float] = []
    for t, global_track_id, state in state_snapshots:
        uav_name = majority_labels.get(global_track_id)
        if uav_name is None:
            continue
        true_state = truth_lookup.get((uav_name, round(t, 6)))
        if true_state is None:
            continue
        position_errors.append(float(np.linalg.norm(state[:3] - true_state[:3])))
        velocity_errors.append(float(np.linalg.norm(state[3:6] - true_state[3:6])))

    position_rmse = float(np.sqrt(np.mean(np.square(position_errors)))) if position_errors else float("nan")
    velocity_rmse = float(np.sqrt(np.mean(np.square(velocity_errors)))) if velocity_errors else float("nan")
    return position_rmse, velocity_rmse


def track_availability(
    state_snapshots: list[tuple[float, int, np.ndarray]],
    ground_truth_trajectories: dict[str, list[tuple[float, np.ndarray]]],
    majority_labels: dict[int, str],
) -> dict[str, float]:
    """Fracao dos instantes da trajetoria real em que existia um
    GlobalTrack correspondente ativo (independente do status - CONFIRMED
    ou COASTING/LOST ainda contam como "disponivel", so DELETED nao)."""
    observed: dict[str, set[float]] = {}
    for t, global_track_id, _ in state_snapshots:
        uav_name = majority_labels.get(global_track_id)
        if uav_name is None:
            continue
        observed.setdefault(uav_name, set()).add(round(t, 6))

    availability: dict[str, float] = {}
    for uav_name, trajectory in ground_truth_trajectories.items():
        total = len(trajectory)
        seen = len(observed.get(uav_name, set()))
        availability[uav_name] = seen / total if total else float("nan")
    return availability
