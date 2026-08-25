"""Cria GlobalTracks novos a partir de candidatos formados pela associacao
(grupos de tracklets nao associados a nenhum track existente, ja agrupados
por compatibilidade cinematica entre estacoes pela AssociationStrategy -
ver association/association_manager.py)."""

from __future__ import annotations

import itertools

from config.models import AppConfig
from filtering.base import FilterTracker
from filtering.factory import create_filter_tracker
from filtering.motion_models import create_motion_model
from fusion.base import create_fusion_strategy
from models.enums import UpdateKind
from models.global_track import GlobalTrack
from models.local_tracklet import LocalTracklet


class TrackInitializer:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._fusion_strategy = create_fusion_strategy(config.fusion.strategy)
        self._id_counter = itertools.count(1)

    def create_track(self, tracklets: list[LocalTracklet], timestamp: float) -> tuple[GlobalTrack, FilterTracker]:
        fused = self._fusion_strategy.fuse(tracklets)

        motion_model = create_motion_model(
            self._config.filter.motion_model,
            turn_rate_process_noise_std=self._config.filter.turn_rate_process_noise_std,
            initial_turn_rate_std=self._config.filter.initial_turn_rate_std,
        )
        tracker = create_filter_tracker(
            self._config.filter.type,
            initial_state=fused.state,
            initial_covariance=fused.covariance,
            motion_model=motion_model,
            process_noise_acceleration_std=self._config.filter.process_noise_acceleration_std,
        )

        track = GlobalTrack(
            global_track_id=next(self._id_counter),
            state=fused.state,
            covariance=fused.covariance,
            last_prediction_timestamp=timestamp,
            last_measurement_timestamp=timestamp,
            created_at=timestamp,
            hit_count=1,
            # a criacao SEMPRE parte de uma medicao real (association so cria
            # um candidato a partir de tracklets de verdade) - sem isso, o
            # default do modelo (PREDICTION_ONLY) mentiria sobre um track
            # que acabou de nascer de uma observacao.
            last_update_kind=UpdateKind.MEASUREMENT_UPDATED,
        )
        track.current_contributors = [(t.station_id, t.local_track_id) for t in tracklets]
        for tracklet in tracklets:
            track.associated_local_tracks[tracklet.station_id] = tracklet.local_track_id
            track.associated_local_track_timestamps[tracklet.station_id] = tracklet.timestamp
            track.local_track_history.add((tracklet.station_id, tracklet.local_track_id))

        return track, tracker
