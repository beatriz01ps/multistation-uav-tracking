"""Duas estacoes observando o MESMO instante fisico, mas chegando pela
rede com atraso relativo entre si (ainda dentro da janela de
sincronizacao) - devem ser processadas no MESMO lote/ciclo, nao em lotes
separados."""

import numpy as np

from config.models import AppConfig
from models.local_tracklet import LocalTracklet
from tests.helpers import FakeClock
from tracking.tracker import Tracker


def _tracklet(station_id, local_id, t):
    return LocalTracklet(
        station_id=station_id, local_track_id=local_id, timestamp=t, state=[0.0, 0.0, 100.0, 10.0, 0.0, 0.0], covariance=np.eye(6)
    )


def test_same_instant_different_arrival_times_join_the_same_batch():
    clock = FakeClock()
    tracker = Tracker(AppConfig(), clock=clock)  # synchronization_window_ms=100 default

    tracker.ingest(_tracklet("station_a", "001", 1.0))  # chega em arrival=0ms
    clock.advance(0.09)
    tracker.ingest(_tracklet("station_b", "007", 1.0))  # chega em arrival=90ms, mesmo instante fisico

    clock.advance(0.02)  # 110ms desde a abertura do cluster (arrival de A) -> fecha
    events = tracker.tick()

    # as duas contribuiram para o MESMO GlobalTrack, no mesmo ciclo
    (track,) = tracker.track_manager.tracks.values()
    assert track.current_contributors == [("station_a", "001"), ("station_b", "007")]
    assert {e.global_track_id for e in events} == {track.global_track_id}
