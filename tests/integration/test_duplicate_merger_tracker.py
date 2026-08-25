"""Integracao ponta a ponta da fusao de tracks duplicados atraves do
Tracker real: reproduz o cenario do v0 (fragmentacao
estatistica isolada cria um GlobalTrack novo representando o mesmo alvo)
de forma deterministica - forca dois tracks duplicados diretamente no
TrackManager (sem depender de sorte de RNG) e confirma que um ciclo real
de process_batch() os funde de volta em um so, inclusive corrigindo
qualquer TrackUpdateEvent deste MESMO ciclo que apontasse para o ID
absorvido."""

import numpy as np

from config.models import AppConfig
from filtering.motion_models import ConstantVelocityModel
from filtering.ukf import UkfTracker
from models.global_track import GlobalTrack
from models.local_tracklet import LocalTracklet
from tracking.tracker import Tracker


def _inject_track(tracker, global_track_id, state, covariance, associated_local_tracks=None):
    associated_local_tracks = associated_local_tracks or {}
    track = GlobalTrack(
        global_track_id=global_track_id,
        state=state,
        covariance=covariance,
        last_prediction_timestamp=0.0,
        last_measurement_timestamp=0.0,
        created_at=0.0,
        associated_local_tracks=associated_local_tracks,
        # invariante real: quem ja esta em associated_local_tracks ja
        # passou por uma associacao aceita, entao ja esta no historico.
        local_track_history=set(associated_local_tracks.items()),
    )
    filter_ = UkfTracker(
        initial_state=state, initial_covariance=covariance, motion_model=ConstantVelocityModel(), process_noise_acceleration_std=2.0
    )
    tracker.track_manager.add(track, filter_)


def test_process_batch_merges_pre_existing_duplicates_and_remaps_this_cycle_event():
    tracker = Tracker(AppConfig())

    # dois tracks duplicados, ja no TrackManager antes deste ciclo (o
    # ponto e testar a FUSAO em si, nao como eles surgiram).
    _inject_track(tracker, 1, [0.0, 0.0, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0, {"A": "A001"})
    _inject_track(tracker, 2, [0.3, 0.1, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0, {"B": "B004"})

    # timestamp=0.0 (igual ao last_prediction_timestamp injetado) para
    # nenhum dos dois tracks se deslocar por predicao CV antes da
    # associacao - assim a distancia comparada e exatamente a das posicoes
    # injetadas. Tracklet deliberadamente mais perto do track 2 (o que
    # devera "perder" a fusao, ja que sobrevive sempre o ID menor) -
    # garante que o evento gerado neste ciclo aponta pro ID que sera
    # absorvido (confirmado por reproducao: sem isso, associa direto ao
    # track 1 e o teste passaria sem de fato exercitar o remap).
    tracklet = LocalTracklet(
        station_id="C", local_track_id="C009", timestamp=0.0,
        state=[0.32, 0.12, 100.0, 10.0, 0.0, 0.0], covariance=np.eye(6) * 1.0,
    )

    events = tracker.process_batch([tracklet], 0.0)

    # so sobrou 1 track, com o ID menor (1) - o outro foi absorvido.
    assert set(tracker.track_manager.tracks.keys()) == {1}

    # o evento deste ciclo, mesmo tendo sido originalmente associado ao
    # track 2, saiu com o ID do sobrevivente (1) - nunca informa a estacao
    # de um global_track_id que ja nao existe mais.
    assert len(events) == 1
    assert events[0].global_track_id == 1
    assert events[0].station_id == "C"
    assert events[0].local_track_id == "C009"

    # a identidade conhecida dos DOIS tracks originais foi reunificada sob
    # o sobrevivente.
    survivor = tracker.track_manager.tracks[1]
    assert survivor.associated_local_tracks["A"] == "A001"
    assert survivor.associated_local_tracks["B"] == "B004"
    assert survivor.associated_local_tracks["C"] == "C009"


def test_process_batch_with_no_tracklets_still_merges_pre_existing_duplicates():
    # o merge tambem roda no ramo de lote VAZIO (so predict/catch-up) - nao
    # so quando ha uma medicao real neste ciclo especifico.
    tracker = Tracker(AppConfig())
    _inject_track(tracker, 4, [0.0, 0.0, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0)
    _inject_track(tracker, 7, [0.2, 0.1, 100.0, 10.0, 0.0, 0.0], np.eye(6) * 4.0)

    tracker.process_batch([], 0.5)

    assert set(tracker.track_manager.tracks.keys()) == {4}
