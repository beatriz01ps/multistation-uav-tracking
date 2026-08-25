"""Teste de integracao: quando uma estacao troca de local_track_id (ex.:
reinicia seu proprio rastreador local - ver VirtualStation.force_new_local_id),
o Tracker precisa continuar associando o LocalTracklet novo ao MESMO
GlobalTrack ja existente, pela compatibilidade cinematica (estado +
covariancia), nunca pela igualdade do local_track_id.

Existia uma lacuna real aqui: o cenario `dropout_and_id_switch.yaml` foi
criado especificamente pra testar isso, mas nenhum teste automatizado
checava que o ID realmente MUDAVA apos force_new_local_id - um bug real
em VirtualStation (ver tests/unit/test_virtual_station.py) fazia o "novo"
ID sair identico ao antigo nesse cenario (UAV unico), entao o teste do
cenario passava sem nunca exercitar a troca de verdade."""

from __future__ import annotations

import numpy as np
import pytest

from config.models import AppConfig
from simulation.virtual_station import VirtualStation
from tracking.tracker import Tracker


def _step(state: np.ndarray, dt: float = 1.0) -> np.ndarray:
    state = state.copy()
    state[:3] += state[3:6] * dt
    return state


@pytest.fixture
def tracker() -> Tracker:
    return Tracker(AppConfig())


def test_new_local_id_after_switch_reacquires_the_same_global_track(tracker: Tracker):
    station = VirtualStation("B", position_std=0.5, velocity_std=0.1, rng=np.random.default_rng(0))
    state = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0])
    t = 0.0

    tracklet = station.observe("uav_1", state, t)
    id_before_switch = tracklet.local_track_id
    tracker.process_batch([tracklet], t)
    (track,) = tracker.track_manager.tracks.values()
    global_id = track.global_track_id

    for _ in range(4):
        state = _step(state)
        t += 1.0
        tracker.process_batch([station.observe("uav_1", state, t)], t)

    station.force_new_local_id("uav_1")
    state = _step(state)
    t += 1.0
    tracklet_after_switch = station.observe("uav_1", state, t)

    # a propria correcao do bug: o ID precisa ter mudado de verdade, senao
    # este teste nao estaria exercitando reaquisicao nenhuma.
    assert tracklet_after_switch.local_track_id != id_before_switch

    tracker.process_batch([tracklet_after_switch], t)

    assert len(tracker.track_manager.tracks) == 1
    (track,) = tracker.track_manager.tracks.values()
    assert track.global_track_id == global_id
    assert track.associated_local_tracks["B"] == tracklet_after_switch.local_track_id
