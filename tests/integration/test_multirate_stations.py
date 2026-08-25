"""Regressao: estacoes com frequencias diferentes (nao assumir
timestamps/frequencia iguais entre estacoes)."""

import numpy as np
import pytest

from config.models import AppConfig
from models.local_tracklet import LocalTracklet
from tracking.tracker import Tracker


def _tracklet(station_id, local_id, t, x):
    return LocalTracklet(
        station_id=station_id,
        local_track_id=local_id,
        timestamp=t,
        state=[x, 0.0, 100.0, 10.0, 0.0, 0.0],
        covariance=np.eye(6),
    )


def test_stations_with_different_rates_converge_to_one_track():
    tracker = Tracker(AppConfig())

    # station_a: 10 Hz (dt=0.1s), station_b: 5 Hz (dt=0.2s), station_c: 2 Hz (dt=0.5s)
    # voo reto a 10 m/s ao longo de 2s: x(t) = 10*t
    for step in range(20):
        t = round(step * 0.1, 2)
        tracker.process_batch([_tracklet("station_a", "001", t, x=10 * t)], t)

        if step % 2 == 0:
            tracker.process_batch([_tracklet("station_b", "001", t, x=10 * t)], t)

        if step % 5 == 0:
            tracker.process_batch([_tracklet("station_c", "001", t, x=10 * t)], t)

    assert len(tracker.track_manager.tracks) == 1
    (track,) = tracker.track_manager.tracks.values()
    assert track.associated_local_tracks == {"station_a": "001", "station_b": "001", "station_c": "001"}
    assert track.state[0] == pytest.approx(19.0, abs=1.0)  # x(1.9s) = 19m
