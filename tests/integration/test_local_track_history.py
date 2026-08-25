"""Testes de como `GlobalTrack.local_track_history` e POPULADO pelo
Tracker real - complementar a tests/unit/test_duplicate_merger.py, que
testa como o DuplicateTrackMerger CONSOME esse historico. Ver
models/global_track.py::local_track_history pro raciocinio completo."""

from __future__ import annotations

import numpy as np
import pytest

from config.models import AppConfig
from models.local_tracklet import LocalTracklet
from tracking.tracker import Tracker


def _tracklet(station_id, local_id, t, state):
    return LocalTracklet(station_id=station_id, local_track_id=local_id, timestamp=t, state=state, covariance=np.eye(6))


def _step(state: np.ndarray, dt: float = 1.0) -> np.ndarray:
    state = state.copy()
    state[:3] += state[3:6] * dt
    return state


@pytest.fixture
def tracker() -> Tracker:
    return Tracker(AppConfig())


def test_history_survives_a_local_id_switch_while_associated_local_tracks_shows_only_the_latest(tracker: Tracker):
    state = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0])
    tracker.process_batch([_tracklet("A", "A001", 0.0, state)], 0.0)
    tracker.process_batch([_tracklet("A", "A002", 1.0, _step(state))], 1.0)

    (track,) = tracker.track_manager.tracks.values()

    assert track.associated_local_tracks["A"] == "A002"
    assert track.local_track_history == {("A", "A001"), ("A", "A002")}


def test_new_track_history_is_initialized_with_every_contributing_station(tracker: Tracker):
    state = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0])
    tracker.process_batch(
        [
            _tracklet("A", "A001", 0.0, state),
            _tracklet("B", "B004", 0.0, state + np.array([0.5, 0.5, 0, 0, 0, 0])),
            _tracklet("C", "C007", 0.0, state + np.array([-0.5, 0.3, 0, 0, 0, 0])),
        ],
        0.0,
    )

    (track,) = tracker.track_manager.tracks.values()
    assert track.local_track_history == {("A", "A001"), ("B", "B004"), ("C", "C007")}
