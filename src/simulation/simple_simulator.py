"""Simulador simples para desenvolvimento e testes. Gera dados sinteticos
com ground truth conhecido, mas o
algoritmo de rastreamento NUNCA tem acesso a essa equivalencia - so os
`LocalTracklet` (sem `uav_name`) chegam ao Tracker; `truth_labels` e
`ground_truth_trajectories` existem so para avaliacao, depois da execucao.

Objetivo explicito (nao mudar): testar software e contrato, nao produzir
resultados cientificos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from models.local_tracklet import LocalTracklet
from simulation.virtual_station import VirtualStation


@dataclass
class UavScenario:
    name: str  # nunca enviado ao tracker - so para avaliacao
    trajectory: list[tuple[float, np.ndarray]]
    dropout_windows: list[tuple[float, float]] = field(default_factory=list)
    id_switch_after_dropout: bool = False


@dataclass
class SimulationResult:
    tracklets_by_timestamp: dict[float, list[LocalTracklet]]
    # (station_id, local_track_id, timestamp) -> nome do UAV real (SO avaliacao)
    truth_labels: dict[tuple[str, str, float], str]
    # nome do UAV -> trajetoria real (SO avaliacao)
    ground_truth_trajectories: dict[str, list[tuple[float, np.ndarray]]]


def _in_any_window(t: float, windows: list[tuple[float, float]]) -> bool:
    return any(start <= t <= end for start, end in windows)


def run_simulation(
    uavs: list[UavScenario], stations: list[VirtualStation], scenario_id: Optional[str] = None
) -> SimulationResult:
    tracklets_by_timestamp: dict[float, list[LocalTracklet]] = {}
    truth_labels: dict[tuple[str, str, float], str] = {}

    for uav in uavs:
        was_visible = True
        for timestamp, true_state in uav.trajectory:
            visible = not _in_any_window(timestamp, uav.dropout_windows)
            if not visible:
                was_visible = False
                continue

            if not was_visible and uav.id_switch_after_dropout:
                for station in stations:
                    station.force_new_local_id(uav.name)
            was_visible = True

            for station in stations:
                tracklet = station.observe(uav.name, true_state, timestamp, scenario_id=scenario_id)
                tracklets_by_timestamp.setdefault(timestamp, []).append(tracklet)
                truth_labels[(tracklet.station_id, tracklet.local_track_id, timestamp)] = uav.name

    ground_truth_trajectories = {uav.name: uav.trajectory for uav in uavs}
    return SimulationResult(tracklets_by_timestamp, truth_labels, ground_truth_trajectories)
