"""Orquestrador principal: liga buffer de sincronizacao, associacao,
fusao, filtro UKF e ciclo de vida no pipeline central do tracker.

Pipeline por ciclo (process_batch):
  predict de todos os GlobalTracks ativos ate o timestamp do ciclo
        |
  Mahalanobis gating + Hungarian assignment (por estacao, 2 estagios)
        |
  agrupar tracklets associados por GlobalTrack
        |
  track-to-track fusion (estrategia configurada)
        |
  UKF update (medicao = fusao) + ciclo de vida
        |
  tracklets sem par viram candidatos -> podem virar GlobalTrack novo
        |
  fusao de GlobalTracks duplicados (tracking/duplicate_merger.py) - so
  roda no final, depois que candidatos novos ja foram criados

Importante: `process_batch` roda o predict de TODOS os tracks ativos
mesmo quando `tracklets` estiver vazio - e assim que o requisito "o
tracker precisa continuar prevendo mesmo sem nenhum pacote chegar" e
satisfeito. Quem decide QUANDO chamar process_batch/tick (a cada pacote
recebido ou num timer proprio independente) e responsabilidade de quem usa
esta classe (ver simulation/ e main.py) - n deste orquestrador.
"""

from __future__ import annotations

import logging
import time as time_module
from dataclasses import dataclass

from association.association_manager import MahalanobisHungarianAssociation
from association.gating import dims_for_mode
from association.mahalanobis import mahalanobis_squared
from config.models import AppConfig
from fusion.base import create_fusion_strategy
from models.enums import TrackStatus, UpdateKind
from models.global_track import GlobalTrack
from models.local_tracklet import LocalTracklet
from synchronization.temporal_alignment import align_batch_to_timestamp
from synchronization.tracklet_buffer import TrackletBuffer
from synchronization.tracking_clock import TrackingClock
from tracking import lifecycle
from tracking.duplicate_merger import DuplicateTrackMerger
from tracking.track_initializer import TrackInitializer
from tracking.track_manager import TrackManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackUpdateEvent:
    """Uma associacao local<->global efetivada num ciclo (sempre
    MEASUREMENT_UPDATED - e o retorno que as estacoes usam para aprender o
    global_track_id do que mandaram). Para o snapshot completo de estado de
    cada GlobalTrack por ciclo, incluindo os que ficaram PREDICTION_ONLY,
    ver io_/serializer.py, que le tracker.track_manager.tracks diretamente."""

    station_id: str
    local_track_id: str
    global_track_id: int
    timestamp: float
    update_kind: UpdateKind = UpdateKind.MEASUREMENT_UPDATED


class Tracker:
    def __init__(self, config: AppConfig, *, clock=None) -> None:
        self._config = config
        self.track_manager = TrackManager()
        self._association = MahalanobisHungarianAssociation(config.association)
        self._fusion_strategy = create_fusion_strategy(config.fusion.strategy)
        self._initializer = TrackInitializer(config)
        self._duplicate_merger = DuplicateTrackMerger(config.association, self._fusion_strategy)

        # `clock` injetavel para testes deterministicos (default:
        # time.monotonic) - usado tanto pelo buffer de sincronizacao
        # (watermark de chegada) quanto pelo TrackingClock (extrapolacao
        # de "agora" durante COASTING). Nunca e o measurement_timestamp -
        # ver synchronization/tracking_clock.py.
        monotonic = clock if clock is not None else time_module.monotonic
        self._buffer = TrackletBuffer(config.association, clock=monotonic)
        self._tracking_clock = TrackingClock(time_scale=config.time.time_scale, monotonic=monotonic)

    def ingest(self, tracklet: LocalTracklet) -> bool:
        """Poe um tracklet no buffer de sincronizacao. Retorna False se foi
        descartado por chegar atrasado (politica DROP)."""
        return self._buffer.add(tracklet)

    def tick(self, *, flush_all: bool = False) -> list[TrackUpdateEvent]:
        """Processa os lotes prontos no buffer, em ORDEM CRONOLOGICA, cada
        um com o timestamp de MEDICAO real daquele lote - tracklets de
        instantes diferentes, liberados juntos pelo buffer, nunca podem
        ser tratados como simultaneos (ver
        synchronization/tracklet_buffer.py).

        NAO recebe um `timestamp` de fora: o "agora" para o passo de
        catch-up (trazer os tracks ate o presente mesmo sem medicao nova)
        vem do `TrackingClock` interno, extrapolado a partir da ultima
        medicao real - nunca de `time.time()` (misturar Unix epoch com o
        dominio de measurement_timestamp faria o filtro tentar propagar
        por bilhoes de segundos).

        `flush_all=True` ignora a janela de sincronizacao e pula o
        catch-up (replay/simulacao offline, onde quem dirige o tempo e o
        proprio chamador, chamando process_batch/tick em sequencia)."""
        batches = self._buffer.flush_all_batches() if flush_all else self._buffer.pop_ready_batches()

        events: list[TrackUpdateEvent] = []

        # Um track que participou de um lote real mas NAO conseguiu
        # associacao ja teve on_no_measurement() aplicado dentro de
        # process_batch() (via _apply_misses) - ele precisa entrar aqui
        # tambem, nao so os tracks ASSOCIADOS, ou o catch-up no final do
        # MESMO tick() o veria como "nao processado" e aplicaria
        # on_no_measurement() de novo, dobrando miss_count num unico ciclo
        # temporal. `processed_this_tick` marca TODO GlobalTrack que ja
        # teve sua chance de associacao/miss avaliada por um lote real
        # neste tick, associado ou nao.
        processed_this_tick: set[int] = set()
        for batch in batches:
            batch_timestamp = max(tracklet.timestamp for tracklet in batch)
            batch_events, processed_ids = self._process_batch_tracked(batch, batch_timestamp)
            events.extend(batch_events)
            processed_this_tick.update(processed_ids)

        if not flush_all and self._tracking_clock.is_anchored:

            # NAO reusar process_batch([], timestamp) aqui. Esse catch-up
            # so precisa trazer o relogio ate "agora" - ele NUNCA pode
            # reinterpretar isso como "nenhuma estacao observou este track
            # agora", ou um track que acabou de ser confirmado ha poucos
            # ms, DENTRO DESTE MESMO tick(), viraria COASTING so porque o
            # relogio avancou um pouco mais depois.
            self._catch_up(self._tracking_clock.now(), already_handled_this_tick=processed_this_tick)

        return events

    def _catch_up(self, timestamp: float, already_handled_this_tick: set[int]) -> None:
        """So avanca o estado no tempo (predict) para TODOS os tracks
        ativos - nunca aplica a logica de miss/lifecycle para um track que
        ja foi tratado (associado OU avaliado e deixado sem associacao) por
        um lote real dentro deste MESMO tick(). So quem nao participou de
        NENHUM lote deste tick recebe um miss aqui, pelo tempo que realmente
        passou sem observacao."""
        active = self.track_manager.active_tracks()

        for track in active:
            dt = timestamp - track.last_prediction_timestamp
            if dt > 0:
                self.track_manager.filter_for(track.global_track_id).predict(dt)
                track.last_prediction_timestamp = timestamp
                self.track_manager.sync_track_from_filter(track.global_track_id)
            track.age += 1

        untouched = [t for t in active if t.global_track_id not in already_handled_this_tick]
        self._apply_misses(untouched, timestamp)

    def process_batch(self, tracklets: list[LocalTracklet], timestamp: float) -> list[TrackUpdateEvent]:
        events, _ = self._process_batch_tracked(tracklets, timestamp)
        return events

    def _process_batch_tracked(
        self, tracklets: list[LocalTracklet], timestamp: float
    ) -> tuple[list[TrackUpdateEvent], set[int]]:
        """Igual a `process_batch`, mas tambem retorna o conjunto de
        `global_track_id` que ja tiveram sua chance de associacao/miss
        avaliada neste lote (associados ou nao) - usado por `tick()` para
        que o catch-up nao aplique um segundo miss sobre eles (ver
        `tick()`)."""
        active = self.track_manager.active_tracks()

        for track in active:
            dt = timestamp - track.last_prediction_timestamp
            if dt > 0:
                self.track_manager.filter_for(track.global_track_id).predict(dt)
                track.last_prediction_timestamp = timestamp
                self.track_manager.sync_track_from_filter(track.global_track_id)
            track.age += 1

        if not tracklets:
            self._apply_misses(active, timestamp)
            processed_ids = {t.global_track_id for t in active}
            return self._merge_duplicates_and_remap_events([]), processed_ids

        # Reancora o dominio de measurement_timestamp <-> relogio local:
        # esta e uma medicao REAL, entao "agora" (para catch-up futuro, ate
        # a proxima medicao) passa a ser extrapolado a partir dela.
        self._tracking_clock.anchor(timestamp)

        # Tracklets do mesmo lote podem ter timestamps de MEDICAO
        # diferentes entre si (estacoes em frequencias diferentes dentro
        # da mesma janela de sincronizacao). Alinhar todos a `timestamp`
        # (a referencia deste lote) ANTES do gating evita comparar um
        # estado de t=0.00 com um de t=0.10 como se fossem do mesmo
        # instante.
        tracklets = align_batch_to_timestamp(
            tracklets, timestamp, self._config.filter.process_noise_acceleration_std
        )

        events: list[TrackUpdateEvent] = []
        result = self._association.associate(active, tracklets, timestamp)

        dims = dims_for_mode(self._config.association.mode)
        matched_track_ids: set[int] = set()
        for position, tracklet_indices in result.matched_tracklets_by_track.items():
            track = active[position]
            group = [tracklets[i] for i in tracklet_indices]

            for tracklet in group:
                d2 = mahalanobis_squared(track.state, track.covariance, tracklet.state, tracklet.covariance, dims)
                logger.info(
                    "LOCAL_TRACK_ASSOCIATED station=%s local=%s -> global=%s d2=%.2f",
                    tracklet.station_id,
                    tracklet.local_track_id,
                    track.global_track_id,
                    d2,
                )

            fused = self._fusion_strategy.fuse(group)
            self.track_manager.filter_for(track.global_track_id).update(fused.state, fused.covariance)
            self.track_manager.sync_track_from_filter(track.global_track_id)

            previous_status = track.status
            lifecycle.on_measurement(track, timestamp, self._config.tracking)
            if track.status is TrackStatus.CONFIRMED and previous_status is not TrackStatus.CONFIRMED:
                event_name = "GLOBAL_TRACK_RECOVERED" if previous_status in (
                    TrackStatus.COASTING, TrackStatus.LOST
                ) else "GLOBAL_TRACK_CONFIRMED"
                logger.info("%s global=%s", event_name, track.global_track_id)

            track.current_contributors = [(t.station_id, t.local_track_id) for t in group]
            for tracklet in group:
                track.associated_local_tracks[tracklet.station_id] = tracklet.local_track_id
                track.associated_local_track_timestamps[tracklet.station_id] = tracklet.timestamp
                track.local_track_history.add((tracklet.station_id, tracklet.local_track_id))
                events.append(
                    TrackUpdateEvent(tracklet.station_id, tracklet.local_track_id, track.global_track_id, timestamp)
                )
            matched_track_ids.add(track.global_track_id)

        unmatched_active = [t for t in active if t.global_track_id not in matched_track_ids]
        self._apply_misses(unmatched_active, timestamp)

        # Todo track ativo ja teve sua chance de associacao avaliada neste
        # lote real - associado (matched_track_ids) ou nao (unmatched_active,
        # que ja recebeu seu unico miss deste ciclo acima). Nenhum dos dois
        # grupos pode receber outro miss do catch-up neste mesmo tick().
        processed_ids = {t.global_track_id for t in active}

        new_tracks = self._create_new_tracks(result.new_track_candidates, tracklets, timestamp)
        for track in new_tracks:
            for station_id, local_track_id in track.associated_local_tracks.items():
                events.append(TrackUpdateEvent(station_id, local_track_id, track.global_track_id, timestamp))

        # Um track recem-criado neste lote (hit_count=1, TENTATIVE) tambem
        # nao pode levar um miss do catch-up no mesmo tick em que nasceu.
        processed_ids.update(t.global_track_id for t in new_tracks)

        return self._merge_duplicates_and_remap_events(events), processed_ids

    def _merge_duplicates_and_remap_events(self, events: list[TrackUpdateEvent]) -> list[TrackUpdateEvent]:
        """Roda a fusao de tracks duplicados (tracking/duplicate_merger.py)
        ao final de todo ciclo real - e o unico lugar onde um GlobalTrack
        novo pode ter nascido (candidato de tracklets sem par), entao e
        aqui que uma fusao mal-sucedida vira, na pratica, um candidato
        duplicado do que ja existia. Se algum `TrackUpdateEvent` deste
        MESMO ciclo apontava para o global_track_id que acabou de ser
        absorvido, reescreve para o sobrevivente - sem isso, a estacao
        aprenderia um global_track_id que ja nao existe mais."""
        merge_events = self._duplicate_merger.merge_duplicates(self.track_manager)
        if not merge_events:
            return events

        resolution: dict[int, int] = {}
        for merge_event in merge_events:
            resolution[merge_event.merged_global_track_id] = merge_event.survivor_global_track_id

        def resolve(global_track_id: int) -> int:
            while global_track_id in resolution:
                global_track_id = resolution[global_track_id]
            return global_track_id

        return [
            event
            if resolve(event.global_track_id) == event.global_track_id
            else TrackUpdateEvent(
                event.station_id, event.local_track_id, resolve(event.global_track_id), event.timestamp, event.update_kind
            )
            for event in events
        ]

    def _create_new_tracks(
        self, candidate_groups: list[list[int]], tracklets: list[LocalTracklet], timestamp: float
    ) -> list[GlobalTrack]:
        created = []
        for group_indices in candidate_groups:
            group = [tracklets[i] for i in group_indices]
            track, tracker = self._initializer.create_track(group, timestamp)
            self.track_manager.add(track, tracker)
            created.append(track)
            logger.info(
                "GLOBAL_TRACK_CREATED global=%s stations=%s",
                track.global_track_id,
                [t.station_id for t in group],
            )
        return created

    def _apply_misses(self, tracks: list[GlobalTrack], timestamp: float) -> None:
        to_delete = []
        for track in tracks:
            track.current_contributors = []  # PREDICTION_ONLY: ninguem contribuiu neste ciclo
            previous_status = track.status
            deleted = lifecycle.on_no_measurement(track, timestamp, self._config.tracking)
            if track.status != previous_status:
                logger.info("GLOBAL_TRACK_%s global=%s", track.status.value.upper(), track.global_track_id)
            if deleted:
                to_delete.append(track.global_track_id)
        for global_track_id in to_delete:
            self.track_manager.remove(global_track_id)
