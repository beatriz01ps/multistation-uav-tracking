"""Um UAV observado normalmente, depois nenhuma estacao o observa por um
intervalo, depois volta a ser observado. O sistema falha se criar um novo
global_track_id para o UAV apos uma perda curta e previsivel."""

import numpy as np
import pytest

from config.models import AppConfig
from models.enums import TrackStatus
from models.local_tracklet import LocalTracklet
from tracking.tracker import Tracker


def _tracklet(t, state):
    return LocalTracklet(station_id="station_a", local_track_id="001", timestamp=t, state=state, covariance=np.eye(6))


def test_short_dropout_preserves_global_track_id():
    config = AppConfig()  # default: coasting_timeout=3.0s, deletion_timeout=10.0s
    tracker = Tracker(config)

    state = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0])

    # t = 0..10s: observado normalmente
    for t in range(0, 11):
        tracker.process_batch([_tracklet(float(t), state)], float(t))
        state = state.copy()
        state[:3] += state[3:6]

    (track,) = tracker.track_manager.tracks.values()
    original_global_id = track.global_track_id
    assert track.status is TrackStatus.CONFIRMED

    # t = 11..13s: nenhuma estacao observa (o alvo real continua se movendo)
    for t in range(11, 14):
        tracker.process_batch([], float(t))
        state = state.copy()
        state[:3] += state[3:6]

    assert track.global_track_id == original_global_id
    assert track.status in (TrackStatus.COASTING, TrackStatus.LOST)
    assert track.last_update_kind.value == "prediction_only"

    # t = 14+: observacoes retornam
    tracker.process_batch([_tracklet(14.0, state)], 14.0)

    assert len(tracker.track_manager.tracks) == 1
    (recovered_track,) = tracker.track_manager.tracks.values()
    assert recovered_track.global_track_id == original_global_id
    assert recovered_track.status is TrackStatus.CONFIRMED


def test_state_keeps_evolving_during_dropout_never_freezes():
    config = AppConfig()
    tracker = Tracker(config)
    state = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0])

    for t in range(0, 4):
        tracker.process_batch([_tracklet(float(t), state)], float(t))
        state = state.copy()
        state[:3] += state[3:6]

    (track,) = tracker.track_manager.tracks.values()
    x_at_dropout_start = track.state[0]

    tracker.process_batch([], 4.0)
    tracker.process_batch([], 5.0)

    assert track.state[0] > x_at_dropout_start  # continuou evoluindo, nao congelou
    # tolerancia mais larga que "exato": com Coordinated Turn (default),
    # ha um pequeno efeito de Jensen mesmo sem manobra real (ver
    # CoordinatedTurnModel) - a media ponderada pelos sigma points de uma
    # funcao nao-linear nao coincide bit a bit com a formula CV fechada.
    assert track.state[0] == pytest.approx(x_at_dropout_start + 20.0, abs=1.0)  # v=10, 2s
