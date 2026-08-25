"""Agrega association_metrics + trajectory_metrics num relatorio unico, e
oferece `run_and_evaluate` para rodar uma SimulationResult inteira contra
um Tracker e calcular tudo de uma vez - usado pelos testes obrigatorios e
por scripts de experimento.
"""

from __future__ import annotations

from config.models import AppConfig
from evaluation.association_metrics import (
    AssociationEvent,
    association_accuracy,
    id_switches,
    majority_label_per_global_id,
    track_fragmentation,
)
from evaluation.trajectory_metrics import position_velocity_rmse, track_availability
from simulation.simple_simulator import SimulationResult
from tracking.tracker import Tracker


def run_and_evaluate(simulation: SimulationResult, config: AppConfig) -> dict:
    tracker = Tracker(config)

    events: list[AssociationEvent] = []
    state_snapshots: list[tuple[float, int, "object"]] = []

    for timestamp in sorted(simulation.tracklets_by_timestamp.keys()):
        tracklets = simulation.tracklets_by_timestamp[timestamp]
        cycle_events = tracker.process_batch(tracklets, timestamp)

        for event in cycle_events:
            events.append(
                AssociationEvent(event.station_id, event.local_track_id, event.global_track_id, event.timestamp)
            )

        for track in tracker.track_manager.active_tracks():
            state_snapshots.append((timestamp, track.global_track_id, track.state.copy()))

    majority_labels = majority_label_per_global_id(events, simulation.truth_labels)
    fragmentation = track_fragmentation(majority_labels)
    pos_rmse, vel_rmse = position_velocity_rmse(state_snapshots, simulation.ground_truth_trajectories, majority_labels)

    return {
        "association_accuracy": association_accuracy(events, simulation.truth_labels, majority_labels),
        "num_global_tracks_created": len(majority_labels),
        "num_real_uavs": len(simulation.ground_truth_trajectories),
        "track_fragmentation": fragmentation,
        "id_switches": id_switches(events, simulation.truth_labels),
        "track_availability": track_availability(state_snapshots, simulation.ground_truth_trajectories, majority_labels),
        "position_rmse_m": pos_rmse,
        "velocity_rmse_mps": vel_rmse,
        "tracker": tracker,
    }
