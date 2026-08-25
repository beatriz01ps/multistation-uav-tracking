"""Regressao do bug real (critico) onde `io_/receiver.py` passava
`time.time()` (epoch Unix, ~1.7e9) direto para `Tracker.tick()`, que
comparava isso com `last_prediction_timestamp` no dominio de
`measurement_timestamp` (que pode ser 0.0, 0.1, ...) - o `dt` resultante
podia ser da ordem de bilhoes de segundos. Ver
synchronization/tracking_clock.py para o design da correcao (ancora entre
os dois dominios + extrapolacao por tempo monotonic DECORRIDO, nunca por
valor absoluto)."""

import numpy as np
import pytest

from config.models import AppConfig
from models.enums import TrackStatus, UpdateKind
from models.local_tracklet import LocalTracklet
from tests.helpers import FakeClock
from tracking.tracker import Tracker


def _tracklet(t, x):
    return LocalTracklet(
        station_id="station_a", local_track_id="001", timestamp=t, state=[x, 0.0, 100.0, 10.0, 0.0, 0.0], covariance=np.eye(6)
    )


def test_huge_monotonic_clock_value_never_leaks_into_measurement_domain():
    # o relogio monotonic aqui parte de um valor gigante, simulando o que
    # `time.time()` (epoch Unix) daria no bug antigo - so a DIFERENCA entre
    # leituras dele pode importar, nunca seu valor absoluto.
    clock = FakeClock(start=1_700_000_000.0)
    tracker = Tracker(AppConfig(), clock=clock)

    tracker.ingest(_tracklet(0.0, x=0.0))
    clock.advance(0.2)  # passa a janela de sincronizacao (100ms default)
    tracker.tick()

    tracker.ingest(_tracklet(0.1, x=1.0))
    clock.advance(0.2)
    tracker.tick()

    (track,) = tracker.track_manager.tracks.values()
    # dt entre os dois lotes deve estar no dominio de measurement_timestamp
    # (~0.1s) - nunca bilhoes de segundos.
    assert track.last_prediction_timestamp == pytest.approx(0.1)
    assert np.trace(track.covariance) < 1000.0  # nao explodiu por causa de um dt absurdo

    # catch-up (sem nenhuma medicao nova) tambem so deve avancar pelo tempo
    # monotonic REALMENTE decorrido, nao pelo valor absoluto do relogio.
    clock.advance(0.05)
    tracker.tick()

    (track,) = tracker.track_manager.tracks.values()
    assert track.last_prediction_timestamp == pytest.approx(0.15)
    assert np.trace(track.covariance) < 1000.0


def test_coasting_track_is_extrapolated_from_elapsed_monotonic_time():
    clock = FakeClock()
    tracker = Tracker(AppConfig(), clock=clock)

    for t in [8.0, 9.0, 10.0]:
        tracker.ingest(_tracklet(t, x=t * 10))
        clock.advance(0.2)  # passa a janela de sincronizacao -> lote fica pronto
        tracker.tick()

    (track,) = tracker.track_manager.tracks.values()
    assert track.status is TrackStatus.CONFIRMED
    global_track_id = track.global_track_id
    covariance_at_t10 = track.covariance.copy()

    clock.advance(2.0)  # 2s de relogio monotonic decorridos, sem nenhuma medicao nova
    tracker.tick()

    (track,) = tracker.track_manager.tracks.values()
    assert track.global_track_id == global_track_id
    assert track.last_prediction_timestamp == pytest.approx(12.0)
    assert track.status is TrackStatus.COASTING
    assert track.last_update_kind is UpdateKind.PREDICTION_ONLY
    assert np.trace(track.covariance) > np.trace(covariance_at_t10)
