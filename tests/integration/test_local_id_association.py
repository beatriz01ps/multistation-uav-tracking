"""IDs locais nao tem nenhuma relacao entre estacoes. O sistema deve
descobrir a equivalencia
fisica mesmo quando os numeros dos IDs locais sao completamente diferentes
entre estacoes - nunca usar igualdade de local_track_id como regra."""

import numpy as np

from config.models import AppConfig
from models.local_tracklet import LocalTracklet
from tracking.tracker import Tracker


def _tracklet(station_id, local_id, t, state):
    return LocalTracklet(station_id=station_id, local_track_id=local_id, timestamp=t, state=state, covariance=np.eye(6))


def test_discovers_equivalence_despite_unrelated_local_ids():
    tracker = Tracker(AppConfig())

    uav1 = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0])
    uav2 = np.array([1000.0, 1000.0, 100.0, -10.0, 0.0, 0.0])

    def step(state):
        state = state.copy()
        state[:3] += state[3:6]
        return state

    t = 0.0
    for _ in range(3):
        tracker.process_batch(
            [
                _tracklet("station_a", "001", t, uav1),
                _tracklet("station_b", "054", t, uav1 + np.array([1, 1, 0, 0, 0, 0])),
                _tracklet("station_a", "002", t, uav2),
                _tracklet("station_b", "012", t, uav2 + np.array([-1, 1, 0, 0, 0, 0])),
            ],
            t,
        )
        uav1, uav2 = step(uav1), step(uav2)
        t += 1.0

    tracks = list(tracker.track_manager.tracks.values())
    assert len(tracks) == 2

    track_by_a001 = next(tr for tr in tracks if tr.associated_local_tracks.get("station_a") == "001")
    track_by_a002 = next(tr for tr in tracks if tr.associated_local_tracks.get("station_a") == "002")

    assert track_by_a001.associated_local_tracks["station_b"] == "054"
    assert track_by_a002.associated_local_tracks["station_b"] == "012"
    assert track_by_a001.global_track_id != track_by_a002.global_track_id
