"""Testes da fusao de GlobalTracks duplicados (tracking/duplicate_merger.py)
- isolados do resto do Tracker, direto contra um TrackManager real."""

import numpy as np

from association.gating import gate
from config.models import AssociationConfig
from filtering.motion_models import ConstantVelocityModel
from filtering.ukf import UkfTracker
from fusion.information_fusion import InformationFusion
from models.enums import AssociationMode
from models.global_track import GlobalTrack
from tracking.duplicate_merger import DuplicateTrackMerger
from tracking.track_manager import TrackManager


def _add_track(
    track_manager, global_track_id, state, covariance,
    associated_local_tracks=None, current_contributors=None, local_track_history=None,
    associated_local_track_timestamps=None,
):
    associated_local_tracks = associated_local_tracks or {}
    track = GlobalTrack(
        global_track_id=global_track_id,
        state=state,
        covariance=covariance,
        last_prediction_timestamp=0.0,
        last_measurement_timestamp=0.0,
        created_at=0.0,
        associated_local_tracks=associated_local_tracks,
        associated_local_track_timestamps=associated_local_track_timestamps or {},
        current_contributors=current_contributors or [],
        # invariante real do sistema: tudo que esta em associated_local_tracks
        # JA passou por uma associacao aceita, entao ja esta no historico
        # tambem - a menos que o teste passe um historico proprio (pra
        # simular explicitamente "ID atual difere do historico").
        local_track_history=local_track_history if local_track_history is not None else set(associated_local_tracks.items()),
    )
    tracker = UkfTracker(
        initial_state=state, initial_covariance=covariance, motion_model=ConstantVelocityModel(), process_noise_acceleration_std=2.0
    )
    track_manager.add(track, tracker)
    return track


def _merger(chi_square_probability=0.99, mode=AssociationMode.FULL_STATE):
    config = AssociationConfig(chi_square_probability=chi_square_probability, mode=mode)
    return DuplicateTrackMerger(config, InformationFusion())


def test_two_statistically_close_tracks_get_merged():
    track_manager = TrackManager()
    _add_track(track_manager, 1, [0.0, 0.0, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0)
    _add_track(track_manager, 2, [0.5, 0.2, 100.0, 10.1, 0.0, 0.0], np.eye(6) * 4.0)

    events = _merger().merge_duplicates(track_manager)

    assert len(events) == 1
    assert events[0].survivor_global_track_id == 1
    assert events[0].merged_global_track_id == 2
    assert len(track_manager.active_tracks()) == 1
    assert 1 in track_manager.tracks
    assert 2 not in track_manager.tracks


def test_lower_global_track_id_always_survives_regardless_of_creation_order():
    track_manager = TrackManager()
    # o track de ID MAIOR e adicionado primeiro - garante que a sobrevivencia
    # e decidida pelo ID, nao pela ordem de insercao no TrackManager.
    _add_track(track_manager, 9, [0.0, 0.0, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0)
    _add_track(track_manager, 3, [0.1, 0.1, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0)

    events = _merger().merge_duplicates(track_manager)

    assert events[0].survivor_global_track_id == 3
    assert 3 in track_manager.tracks
    assert 9 not in track_manager.tracks


def test_far_apart_tracks_are_not_merged():
    track_manager = TrackManager()
    _add_track(track_manager, 1, [0.0, 0.0, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0)
    _add_track(track_manager, 2, [500.0, 500.0, 100.0, -10.0, -10.0, 0.0], np.eye(6) * 4.0)

    events = _merger().merge_duplicates(track_manager)

    assert events == []
    assert len(track_manager.active_tracks()) == 2


def test_same_position_but_different_velocity_is_not_merged_even_in_position_only_mode():
    """Salvaguarda central: fundir tracks SEMPRE usa FULL_STATE, mesmo que
    o sistema esteja configurado com association.mode=position_only - dois
    UAVs cruzando trajetorias podem compartilhar posicao por um instante
    (test_crossing.py), mas praticamente nunca posicao E velocidade."""
    track_manager = TrackManager()
    _add_track(track_manager, 1, [0.0, 0.0, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0)
    _add_track(track_manager, 2, [0.1, 0.1, 100.0, -10.0, 0.0, 0.0], np.eye(6) * 4.0)  # mesma posicao, velocidade oposta

    events = _merger(mode=AssociationMode.POSITION_ONLY).merge_duplicates(track_manager)

    assert events == []
    assert len(track_manager.active_tracks()) == 2


def test_merged_track_state_matches_the_configured_fusion_strategy():
    track_manager = TrackManager()
    state_a = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0])
    state_b = np.array([1.0, 0.0, 100.0, 10.0, 0.0, 0.0])
    covariance = np.eye(6) * 4.0
    _add_track(track_manager, 1, state_a, covariance)
    _add_track(track_manager, 2, state_b, covariance)

    expected_state, expected_covariance = InformationFusion().fuse_states([(state_a, covariance), (state_b, covariance)])

    _merger().merge_duplicates(track_manager)

    survivor = track_manager.tracks[1]
    assert np.allclose(survivor.state, expected_state)
    assert np.allclose(survivor.covariance, expected_covariance)


def test_associated_local_tracks_and_contributors_are_merged_by_union():
    track_manager = TrackManager()
    _add_track(
        track_manager, 1, [0.0, 0.0, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0,
        associated_local_tracks={"A": "A001"}, current_contributors=[("A", "A001")],
    )
    _add_track(
        track_manager, 2, [0.1, 0.1, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0,
        associated_local_tracks={"B": "B004"}, current_contributors=[("B", "B004")],
    )

    _merger().merge_duplicates(track_manager)

    survivor = track_manager.tracks[1]
    assert survivor.associated_local_tracks == {"A": "A001", "B": "B004"}
    assert set(survivor.current_contributors) == {("A", "A001"), ("B", "B004")}


def test_merge_keeps_the_more_recent_local_track_id_when_merged_side_is_newer():
    track_manager = TrackManager()
    _add_track(
        track_manager, 1, [0.0, 0.0, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0,
        associated_local_tracks={"A": "A001"}, associated_local_track_timestamps={"A": 10.0},
    )
    _add_track(
        track_manager, 2, [0.1, 0.1, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0,
        associated_local_tracks={"A": "A002"}, associated_local_track_timestamps={"A": 12.0},
    )

    _merger().merge_duplicates(track_manager)

    survivor = track_manager.tracks[1]
    assert survivor.associated_local_tracks["A"] == "A002"
    assert survivor.associated_local_track_timestamps["A"] == 12.0


def test_merge_keeps_the_more_recent_local_track_id_when_survivor_side_is_newer():
    track_manager = TrackManager()
    _add_track(
        track_manager, 1, [0.0, 0.0, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0,
        associated_local_tracks={"A": "A002"}, associated_local_track_timestamps={"A": 12.0},
    )
    _add_track(
        track_manager, 2, [0.1, 0.1, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0,
        associated_local_tracks={"A": "A001"}, associated_local_track_timestamps={"A": 10.0},
    )

    _merger().merge_duplicates(track_manager)

    survivor = track_manager.tracks[1]
    assert survivor.associated_local_tracks["A"] == "A002"
    assert survivor.associated_local_track_timestamps["A"] == 12.0


def test_merge_with_disjoint_stations_keeps_both_unaffected_by_recency():
    track_manager = TrackManager()
    _add_track(
        track_manager, 1, [0.0, 0.0, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0,
        associated_local_tracks={"A": "A001"}, associated_local_track_timestamps={"A": 5.0},
    )
    _add_track(
        track_manager, 2, [0.1, 0.1, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0,
        associated_local_tracks={"B": "B004"}, associated_local_track_timestamps={"B": 99.0},
    )

    _merger().merge_duplicates(track_manager)

    survivor = track_manager.tracks[1]
    assert survivor.associated_local_tracks == {"A": "A001", "B": "B004"}


def test_recency_based_merge_still_preserves_the_full_local_track_history():
    track_manager = TrackManager()
    _add_track(
        track_manager, 1, [0.0, 0.0, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0,
        associated_local_tracks={"A": "A001"}, associated_local_track_timestamps={"A": 10.0},
    )
    _add_track(
        track_manager, 2, [0.1, 0.1, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0,
        associated_local_tracks={"A": "A002"}, associated_local_track_timestamps={"A": 12.0},
    )

    _merger().merge_duplicates(track_manager)

    survivor = track_manager.tracks[1]
    # o mapa "atual" so guarda A002 (mais recente), mas o historico
    # continua com os dois - nunca perde informacao, so o snapshot muda.
    assert survivor.associated_local_tracks == {"A": "A002"}
    assert survivor.local_track_history == {("A", "A001"), ("A", "A002")}


def test_tie_in_timestamp_keeps_the_survivor_value_deterministically():
    track_manager = TrackManager()
    _add_track(
        track_manager, 1, [0.0, 0.0, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0,
        associated_local_tracks={"A": "A001"}, associated_local_track_timestamps={"A": 10.0},
    )
    _add_track(
        track_manager, 2, [0.1, 0.1, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0,
        associated_local_tracks={"A": "A999"}, associated_local_track_timestamps={"A": 10.0},  # mesmo timestamp
    )

    _merger().merge_duplicates(track_manager)

    survivor = track_manager.tracks[1]
    # empate: fica com o valor do sobrevivente (decisao documentada em
    # duplicate_merger.py::_merge_into), nao com o do track absorvido.
    assert survivor.associated_local_tracks["A"] == "A001"


def test_three_way_duplicate_collapses_into_a_single_survivor():
    track_manager = TrackManager()
    _add_track(track_manager, 5, [0.0, 0.0, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0)
    _add_track(track_manager, 2, [0.1, 0.1, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0)
    _add_track(track_manager, 8, [0.2, -0.1, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0)

    events = _merger().merge_duplicates(track_manager)

    assert len(track_manager.active_tracks()) == 1
    assert 2 in track_manager.tracks
    assert len(events) == 2  # duas fusoes ate sobrar 1 track so


def test_shared_local_track_id_merges_even_when_the_statistical_gate_fails():
    """O caso real que motivou o segundo criterio: dois tracks bem
    confiantes (covariancia pequena), estatisticamente afastados demais
    pro gate passar, mas que ja compartilharam a mesma fonte
    (station_id, local_track_id) em algum ciclo passado - sinal
    praticamente livre de falso positivo, independente de covariancia."""
    track_manager = TrackManager()
    tight_covariance = np.eye(6) * 0.05  # bem confiante - gate normal falharia aqui
    _add_track(
        track_manager, 1, [0.0, 0.0, 100.0, 10.0, 0.0, 0.0], tight_covariance,
        associated_local_tracks={"A": "A003"},
    )
    _add_track(
        track_manager, 2, [3.0, 3.0, 100.0, 10.0, 0.0, 0.0], tight_covariance,
        associated_local_tracks={"A": "A003"},  # MESMA fonte que o track 1 ja teve
    )

    # confirma que, SEM o criterio novo, o gate estatistico sozinho realmente rejeitaria este par
    passed, _ = gate(
        np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0]), tight_covariance,
        np.array([3.0, 3.0, 100.0, 10.0, 0.0, 0.0]), tight_covariance,
        AssociationMode.FULL_STATE, 0.99,
    )
    assert not passed

    events = _merger().merge_duplicates(track_manager)

    assert len(events) == 1
    assert events[0].reason == "shared_local_track_id"
    assert len(track_manager.active_tracks()) == 1
    assert 1 in track_manager.tracks


def test_shared_history_matches_even_when_current_associated_local_tracks_have_diverged():
    """O caso que motivou trocar o criterio de `associated_local_tracks`
    (so o mais recente) para `local_track_history` (tudo que ja passou):
    track #1 recebeu A001 e depois trocou pra A002 (ex.: a estacao
    reiniciou - VirtualStation.force_new_local_id); track #2 tem A001 no
    HISTORICO. Mesmo com os IDs ATUAIS diferentes (A002 vs A001), o
    historico compartilhado (A001, presente nos dois) precisa continuar
    disparando o criterio - com a implementacao antiga (comparar so
    associated_local_tracks atuais), esse par NUNCA se reconheceria como
    da mesma fonte."""
    track_manager = TrackManager()
    tight_covariance = np.eye(6) * 0.05  # gate normal falharia aqui, igual ao teste acima
    _add_track(
        track_manager, 1, [0.0, 0.0, 100.0, 10.0, 0.0, 0.0], tight_covariance,
        associated_local_tracks={"A": "A002"},  # ID ATUAL (depois da troca)
        local_track_history={("A", "A001"), ("A", "A002")},  # mas o historico lembra do A001
    )
    _add_track(
        track_manager, 2, [3.0, 3.0, 100.0, 10.0, 0.0, 0.0], tight_covariance,
        associated_local_tracks={"A": "A001"},  # nunca trocou
        local_track_history={("A", "A001")},
    )

    events = _merger().merge_duplicates(track_manager)

    assert len(events) == 1
    assert events[0].reason == "shared_local_track_id"
    assert len(track_manager.active_tracks()) == 1


def test_same_local_id_string_from_different_stations_never_counts_as_shared():
    """Guarda contra o par errado: o par tem que ser (station_id,
    local_track_id) INTEIRO - "001" da estacao A e "001" da estacao B sao
    fontes fisicas DIFERENTES, mesmo com o mesmo texto de ID."""
    track_manager = TrackManager()
    tight_covariance = np.eye(6) * 0.05
    _add_track(
        track_manager, 1, [0.0, 0.0, 100.0, 10.0, 0.0, 0.0], tight_covariance,
        associated_local_tracks={"A": "001"},
    )
    _add_track(
        track_manager, 2, [3.0, 3.0, 100.0, 10.0, 0.0, 0.0], tight_covariance,
        associated_local_tracks={"B": "001"},
    )

    events = _merger().merge_duplicates(track_manager)

    assert events == []
    assert len(track_manager.active_tracks()) == 2


def test_merge_unions_the_local_track_history_of_both_tracks():
    track_manager = TrackManager()
    _add_track(
        track_manager, 1, [0.0, 0.0, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0,
        associated_local_tracks={"A": "A001", "B": "B002"},
    )
    _add_track(
        track_manager, 2, [0.1, 0.1, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0,
        associated_local_tracks={"A": "A003", "C": "C004"},
    )

    _merger().merge_duplicates(track_manager)

    survivor = track_manager.tracks[1]
    assert survivor.local_track_history == {("A", "A001"), ("B", "B002"), ("A", "A003"), ("C", "C004")}


def test_merge_does_not_change_the_meaning_of_associated_local_tracks():
    """`associated_local_tracks` continua sendo "mais recente conhecido
    por estacao" apos o merge - a introducao de `local_track_history` nao
    pode mudar essa semantica (ver models/global_track.py)."""
    track_manager = TrackManager()
    _add_track(
        track_manager, 1, [0.0, 0.0, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0,
        associated_local_tracks={"A": "A001"},
    )
    _add_track(
        track_manager, 2, [0.1, 0.1, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0,
        associated_local_tracks={"B": "B004"},
    )

    _merger().merge_duplicates(track_manager)

    survivor = track_manager.tracks[1]
    # continua um mapa station_id -> ultimo local_track_id conhecido, nao
    # uma lista/historico - cada estacao aparece uma unica vez.
    assert survivor.associated_local_tracks == {"A": "A001", "B": "B004"}


def test_merge_reason_is_reported_correctly_for_gate_only_and_for_both():
    # so gate: proximos estatisticamente, sem historico de fonte compartilhada
    track_manager_gate_only = TrackManager()
    _add_track(track_manager_gate_only, 1, [0.0, 0.0, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0)
    _add_track(track_manager_gate_only, 2, [0.1, 0.1, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0)
    events = _merger().merge_duplicates(track_manager_gate_only)
    assert events[0].reason == "gate"

    # os dois criterios batem ao mesmo tempo
    track_manager_both = TrackManager()
    _add_track(
        track_manager_both, 1, [0.0, 0.0, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0,
        associated_local_tracks={"A": "A001"},
    )
    _add_track(
        track_manager_both, 2, [0.1, 0.1, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0,
        associated_local_tracks={"A": "A001"},
    )
    events = _merger().merge_duplicates(track_manager_both)
    assert events[0].reason == "gate+shared_local_track_id"


def test_merger_disabled_by_config_never_merges_even_identical_tracks():
    """`AssociationConfig.duplicate_merge_enabled=False` existe so para
    ablacao cientifica (ver docstring de tracking/duplicate_merger.py) -
    com ele desligado, nem um par estatisticamente identico pode fundir."""
    track_manager = TrackManager()
    _add_track(track_manager, 1, [0.0, 0.0, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0)
    _add_track(track_manager, 2, [0.0, 0.0, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0)  # identico

    config = AssociationConfig(duplicate_merge_enabled=False)
    events = DuplicateTrackMerger(config, InformationFusion()).merge_duplicates(track_manager)

    assert events == []
    assert len(track_manager.active_tracks()) == 2
    assert 1 in track_manager.tracks and 2 in track_manager.tracks


def test_different_local_track_id_from_the_same_station_does_not_trigger_the_shared_id_signal():
    # mesma estacao, mas IDs locais DIFERENTES - nao e o mesmo sinal (e o
    # comportamento correto: cada local_track_id e um alvo distinto do
    # ponto de vista daquela estacao). Sem proximidade estatistica
    # tambem, entao nao deve fundir.
    track_manager = TrackManager()
    tight_covariance = np.eye(6) * 0.05
    _add_track(
        track_manager, 1, [0.0, 0.0, 100.0, 10.0, 0.0, 0.0], tight_covariance,
        associated_local_tracks={"A": "A001"},
    )
    _add_track(
        track_manager, 2, [3.0, 3.0, 100.0, 10.0, 0.0, 0.0], tight_covariance,
        associated_local_tracks={"A": "A002"},
    )

    events = _merger().merge_duplicates(track_manager)

    assert events == []
    assert len(track_manager.active_tracks()) == 2
