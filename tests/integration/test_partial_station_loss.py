"""Perda gradual de estacoes. A+B+C observando -> C some -> A+B continuam
(CONFIRMED) -> A some -> so B (ainda pode atualizar) -> nenhuma ->
COASTING."""

import numpy as np

from config.models import AppConfig
from models.enums import TrackStatus
from models.local_tracklet import LocalTracklet
from tracking.tracker import Tracker


def _tracklet(station_id, t, x):
    return LocalTracklet(
        station_id=station_id, local_track_id="001", timestamp=t, state=[x, 0.0, 100.0, 10.0, 0.0, 0.0], covariance=np.eye(6)
    )


def test_gradual_station_loss_keeps_track_alive_until_all_are_gone():
    tracker = Tracker(AppConfig())

    t = 0.0
    # fase 1: A + B + C observando, confirma o track
    for _ in range(3):
        tracker.process_batch(
            [_tracklet("station_a", t, 10 * t), _tracklet("station_b", t, 10 * t), _tracklet("station_c", t, 10 * t)],
            t,
        )
        t += 1.0

    (track,) = tracker.track_manager.tracks.values()
    assert track.status is TrackStatus.CONFIRMED

    # fase 2: C some, A + B continuam
    for _ in range(3):
        tracker.process_batch([_tracklet("station_a", t, 10 * t), _tracklet("station_b", t, 10 * t)], t)
        t += 1.0
    assert track.status is TrackStatus.CONFIRMED
    assert "station_c" not in {"station_a", "station_b"}  # sanidade: so A/B mandaram

    # fase 3: A tambem some, so B
    for _ in range(3):
        tracker.process_batch([_tracklet("station_b", t, 10 * t)], t)
        t += 1.0
    assert track.status is TrackStatus.CONFIRMED

    # fase 4: nenhuma estacao observa
    tracker.process_batch([], t)
    assert track.status is TrackStatus.COASTING
