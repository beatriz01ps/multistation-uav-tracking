"""Matriz de custo (Mahalanobis, apos gating) + assignment via
scipy.optimize.linear_sum_assignment.

Pares rejeitados pelo gate recebem um custo sentinela alto (nao literalmente
infinito - mantem o solver numericamente estavel) e sao descartados do
resultado mesmo que o solver "escolha" um deles por falta de alternativa.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from association.gating import gate
from models.enums import AssociationMode

REJECTED_COST = 1.0e6


def build_cost_matrix(
    track_states: list[tuple[np.ndarray, np.ndarray]],
    tracklet_states: list[tuple[np.ndarray, np.ndarray]],
    mode: AssociationMode,
    chi_square_probability: float,
) -> tuple[np.ndarray, np.ndarray]:
    """track_states / tracklet_states: lista de (state, covariance).
    Retorna (cost_matrix, valid_mask), ambas (n_tracks, n_tracklets)."""
    n_tracks = len(track_states)
    n_tracklets = len(tracklet_states)
    cost = np.full((n_tracks, n_tracklets), REJECTED_COST)
    valid = np.zeros((n_tracks, n_tracklets), dtype=bool)

    for i, (track_state, track_cov) in enumerate(track_states):
        for j, (meas_state, meas_cov) in enumerate(tracklet_states):
            passed, d2 = gate(track_state, track_cov, meas_state, meas_cov, mode, chi_square_probability)
            if passed:
                cost[i, j] = d2
                valid[i, j] = True

    return cost, valid


def solve_assignment(cost: np.ndarray, valid: np.ndarray) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Retorna (matches, unmatched_track_idx, unmatched_tracklet_idx)."""
    n_tracks, n_tracklets = cost.shape
    if n_tracks == 0 or n_tracklets == 0:
        return [], list(range(n_tracks)), list(range(n_tracklets))

    row_idx, col_idx = linear_sum_assignment(cost)
    matches = [(int(r), int(c)) for r, c in zip(row_idx, col_idx) if valid[r, c]]

    matched_tracks = {m[0] for m in matches}
    matched_tracklets = {m[1] for m in matches}
    unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]
    unmatched_tracklets = [j for j in range(n_tracklets) if j not in matched_tracklets]

    return matches, unmatched_tracks, unmatched_tracklets
