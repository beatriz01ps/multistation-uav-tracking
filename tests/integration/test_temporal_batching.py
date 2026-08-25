"""Mensagens da MESMA fonte, chegando dentro da mesma janela de flush do
buffer, nao podem ser tratadas como simultaneas. Testa o caminho
completo via ingest()+tick(), nao so o buffer isolado."""

import numpy as np

from config.models import AppConfig
from models.local_tracklet import LocalTracklet
from tests.helpers import FakeClock
from tracking.tracker import Tracker


def _tracklet(t, x):
    return LocalTracklet(
        station_id="station_a", local_track_id="001", timestamp=t, state=[x, 0.0, 100.0, 10.0, 0.0, 0.0], covariance=np.eye(6)
    )


def test_rapid_same_source_readings_produce_one_evolving_track_not_five():
    clock = FakeClock()
    tracker = Tracker(AppConfig(), clock=clock)

    for t in [0.0, 0.1, 0.2, 0.3, 0.4]:
        tracker.ingest(_tracklet(t, x=t * 10))

    clock.advance(0.2)  # passa a janela de sincronizacao (100ms default)
    tracker.tick()

    assert len(tracker.track_manager.tracks) == 1
    (track,) = tracker.track_manager.tracks.values()
    assert track.hit_count == 5
