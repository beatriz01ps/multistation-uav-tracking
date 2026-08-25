import numpy as np
import pytest

from config.models import AppConfig
from models.enums import TrackStatus
from models.local_tracklet import LocalTracklet
from tracking.tracker import Tracker


def _step(state: np.ndarray, dt: float = 1.0) -> np.ndarray:
    state = state.copy()
    state[:3] += state[3:6] * dt
    return state


def _tracklet(station_id, local_id, t, state, covariance=None):
    return LocalTracklet(
        station_id=station_id, local_track_id=local_id, timestamp=t, state=state, covariance=covariance or np.eye(6)
    )


@pytest.fixture
def tracker() -> Tracker:
    return Tracker(AppConfig())


def test_two_stations_first_sighting_fuse_into_one_track(tracker: Tracker):
    """Sem isso, cada estacao nasceria com seu proprio GlobalTrack na
    primeira observacao simultanea do mesmo alvo."""
    uav = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0])
    tracker.process_batch(
        [
            _tracklet("station_a", "001", 0.0, uav),
            _tracklet("station_b", "007", 0.0, uav + np.array([1.0, 1.0, 0, 0, 0, 0])),
        ],
        0.0,
    )
    assert len(tracker.track_manager.tracks) == 1
    (track,) = tracker.track_manager.tracks.values()
    assert track.associated_local_tracks == {"station_a": "001", "station_b": "007"}


def test_two_uavs_stay_separate_tracks(tracker: Tracker):
    uav1 = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0])
    uav2 = np.array([1000.0, 1000.0, 100.0, 0.0, 10.0, 0.0])
    tracker.process_batch(
        [_tracklet("station_a", "001", 0.0, uav1), _tracklet("station_a", "002", 0.0, uav2)], 0.0
    )
    assert len(tracker.track_manager.tracks) == 2


def test_coasting_predict_does_not_accumulate_dt(tracker: Tracker):
    """Regressao de um bug critico de um projeto anterior: dt do predict()
    calculado desde a ultima medicao (em vez da ultima predicao) fazia a
    propagacao acumular incorretamente ciclo a ciclo durante COASTING."""
    state = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0])
    t = 0.0
    for _ in range(3):
        tracker.process_batch([_tracklet("station_a", "001", t, state)], t)
        state = _step(state)
        t += 1.0

    (track,) = tracker.track_manager.tracks.values()
    state_at_coasting_start = track.state.copy()

    for _ in range(5):
        tracker.process_batch([], t)
        t += 1.0

    expected_x = state_at_coasting_start[0] + state_at_coasting_start[3] * 5.0
    # tolerancia larga o suficiente pra absorver o efeito de Jensen do
    # Coordinated Turn (default - ver CoordinatedTurnModel; medido
    # empiricamente aqui em ~3.4m apos 3 updates + 5 predicts), mas MUITO
    # menor que o erro que a acumulacao de dt produziria: dt acumulado
    # daria 1+2+3+4+5=15s de propagacao em vez de 5s - um erro de DEZENAS
    # de metros (v=10 m/s => ~100m de diferenca), nao de poucos metros.
    assert track.state[0] == pytest.approx(expected_x, abs=5.0)


def test_lost_track_does_not_steal_a_nearby_active_tracks_observation(tracker: Tracker):
    """Regressao: um track LOST (covariancia inflada) nao pode vencer, no
    assignment, a observacao que pertence a um track ativo proximo -
    associacao roda em dois estagios (ver association_manager.py)."""
    uav_a = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0])
    uav_b = np.array([50.0, 0.0, 100.0, 10.0, 0.0, 0.0])

    t = 0.0
    for _ in range(3):
        tracker.process_batch(
            [_tracklet("station_a", "A", t, uav_a), _tracklet("station_a", "B", t, uav_b)], t
        )
        uav_a, uav_b = _step(uav_a), _step(uav_b)
        t += 1.0

    tracks_by_local = {tr.associated_local_tracks.get("station_a"): tr for tr in tracker.track_manager.tracks.values()}
    track_a, track_b = tracks_by_local["A"], tracks_by_local["B"]
    assert track_a.status is TrackStatus.CONFIRMED

    # default: coasting_timeout=10.0s (COASTING->LOST), deletion_timeout=32.0s
    # absoluto (safety cap) - 12 ciclos (12s sem medicao) deixa track_a em
    # LOST sem chegar la.
    for _ in range(12):
        tracker.process_batch([_tracklet("station_a", "B", t, uav_b)], t)
        uav_b = _step(uav_b)
        t += 1.0
    assert track_a.status is TrackStatus.LOST
    assert track_b.status is TrackStatus.CONFIRMED

    t += 1.0
    tracker.process_batch([_tracklet("station_a", "B", t, uav_b)], t)
    assert track_b.status is TrackStatus.CONFIRMED
    assert track_b.associated_local_tracks["station_a"] == "B"
