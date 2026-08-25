"""Testes de VirtualStation - foco no contrato de nao-reutilizacao de
local_track_id (ver docstring de simulation/virtual_station.py), do qual
DuplicateTrackMerger depende (tracking/duplicate_merger.py, criterio de
"mesma fonte historica"). Bug real corrigido aqui: o sufixo numerico do ID
vinha de `len(self._local_ids)` (quantos alvos a estacao rastreia AGORA),
nao de um contador monotonico - `force_new_local_id` num cenario de UAV
unico fazia o "novo" ID sair identico ao antigo."""

import numpy as np

from simulation.virtual_station import VirtualStation

STATE = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0])


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_single_uav_gets_a_new_id_after_force_new_local_id():
    station = VirtualStation("B", rng=_rng())
    first = station.observe("uav_1", STATE, 0.0).local_track_id

    station.force_new_local_id("uav_1")
    second = station.observe("uav_1", STATE, 1.0).local_track_id

    assert first != second


def test_multiple_uavs_never_collide():
    station = VirtualStation("B", rng=_rng())
    ids = {station.observe(f"uav_{i}", STATE, 0.0).local_track_id for i in range(5)}
    assert len(ids) == 5


def test_repeated_consecutive_switches_never_repeat_an_id():
    station = VirtualStation("B", rng=_rng())
    seen = {station.observe("uav_1", STATE, 0.0).local_track_id}
    for i in range(1, 10):
        station.force_new_local_id("uav_1")
        new_id = station.observe("uav_1", STATE, float(i)).local_track_id
        assert new_id not in seen
        seen.add(new_id)
    assert len(seen) == 10


def test_no_reuse_across_the_whole_session_even_with_multiple_uavs_switching():
    station = VirtualStation("B", rng=_rng())
    seen = set()
    for cycle in range(5):
        for uav_name in ("uav_1", "uav_2", "uav_3"):
            seen.add(station.observe(uav_name, STATE, float(cycle)).local_track_id)
        station.force_new_local_id("uav_2")  # so uav_2 troca a cada ciclo
    # 3 IDs no ciclo 0 + 1 ID novo por ciclo subsequente pro uav_2 (4 ciclos) = 7
    assert len(seen) == 3 + 4


def test_no_collision_between_uavs_after_a_switch():
    """Caso especifico do bug antigo: forcar novo ID pro PRIMEIRO uav
    observado nao pode gerar um ID que colida com um uav JA ativo."""
    station = VirtualStation("B", rng=_rng())
    id_a = station.observe("uav_a", STATE, 0.0).local_track_id
    id_b = station.observe("uav_b", STATE, 0.0).local_track_id

    station.force_new_local_id("uav_a")
    new_id_a = station.observe("uav_a", STATE, 1.0).local_track_id

    assert new_id_a != id_a
    assert new_id_a != id_b


def test_local_id_format_is_station_letter_plus_zero_padded_number():
    station = VirtualStation("north", rng=_rng())
    tracklet = station.observe("uav_1", STATE, 0.0)
    assert tracklet.local_track_id == "H001"  # ultima letra de "north", maiuscula
