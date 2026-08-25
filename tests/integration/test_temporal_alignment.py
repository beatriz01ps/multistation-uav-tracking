"""Tracklets dentro da mesma janela de sincronizacao, mas com timestamps
de medicao diferentes (alvo em movimento rapido, estacoes em frequencias
diferentes) - devem ser alinhados a um timestamp comum antes do gating, e
resultar no MESMO GlobalTrack."""

import numpy as np

from config.models import AppConfig
from models.local_tracklet import LocalTracklet
from synchronization.temporal_alignment import align_tracklet_to_timestamp
from tracking.tracker import Tracker


def test_close_but_different_timestamps_in_same_batch_produce_one_track():
    tracker = Tracker(AppConfig())

    # UAV a 100 m/s: em t=0.00 esta em x=0; em t=0.10 esta em x=10 - mesma
    # trajetoria, leituras de DUAS estacoes em instantes levemente
    # diferentes, dentro da janela de sincronizacao (100ms default).
    tracklet_a = LocalTracklet(
        station_id="station_a", local_track_id="001", timestamp=0.00, state=[0.0, 0.0, 100.0, 100.0, 0.0, 0.0], covariance=np.eye(6) * 0.5
    )
    tracklet_b = LocalTracklet(
        station_id="station_b", local_track_id="001", timestamp=0.10, state=[10.0, 0.0, 100.0, 100.0, 0.0, 0.0], covariance=np.eye(6) * 0.5
    )

    tracker.process_batch([tracklet_a, tracklet_b], timestamp=0.10)

    assert len(tracker.track_manager.tracks) == 1
    (track,) = tracker.track_manager.tracks.values()
    assert track.associated_local_tracks == {"station_a": "001", "station_b": "001"}


def test_align_tracklet_to_timestamp_propagates_state_and_covariance():
    tracklet = LocalTracklet(
        station_id="a", local_track_id="1", timestamp=0.0, state=[0.0, 0.0, 100.0, 10.0, 0.0, 0.0], covariance=np.eye(6)
    )
    aligned = align_tracklet_to_timestamp(tracklet, reference_timestamp=1.0, process_noise_acceleration_std=2.0)

    assert aligned.timestamp == 1.0
    assert aligned.state[0] == 10.0  # x + vx*dt = 0 + 10*1
    assert aligned.state[3] == 10.0  # velocidade inalterada
    assert np.trace(aligned.covariance) > np.trace(tracklet.covariance)  # incerteza cresceu


def test_align_tracklet_is_noop_when_already_at_reference():
    tracklet = LocalTracklet(
        station_id="a", local_track_id="1", timestamp=1.0, state=[0.0] * 6, covariance=np.eye(6)
    )
    aligned = align_tracklet_to_timestamp(tracklet, reference_timestamp=1.0, process_noise_acceleration_std=2.0)
    assert aligned is tracklet
