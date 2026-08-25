"""Armazenamento e operacoes de baixo nivel sobre o conjunto de
GlobalTracks ativos e seus filtros associados.

GlobalTrack (dado) e o filtro (UkfTracker OU EkfTracker - mecanica do
filtro) sao mantidos em dicts paralelos, nao um dentro do outro -
GlobalTrack fica facil de validar e serializar (Pydantic puro), e a
mecanica do filtro fica isolada em filtering/, atras de
filtering/base.py::FilterTracker (o TrackManager nunca precisa saber qual
dos dois esta em uso). `sync_track_from_filter` e o unico lugar que copia
o resultado do filtro de volta para o snapshot de dados.
"""

from __future__ import annotations

from filtering.base import FilterTracker
from models.enums import TrackStatus
from models.global_track import GlobalTrack


class TrackManager:
    def __init__(self) -> None:
        self._tracks: dict[int, GlobalTrack] = {}
        self._filters: dict[int, FilterTracker] = {}

    @property
    def tracks(self) -> dict[int, GlobalTrack]:
        return self._tracks

    def active_tracks(self) -> list[GlobalTrack]:
        return [t for t in self._tracks.values() if t.status is not TrackStatus.DELETED]

    def filter_for(self, global_track_id: int) -> FilterTracker:
        return self._filters[global_track_id]

    def add(self, track: GlobalTrack, tracker: FilterTracker) -> None:
        self._tracks[track.global_track_id] = track
        self._filters[track.global_track_id] = tracker

    def remove(self, global_track_id: int) -> None:
        del self._tracks[global_track_id]
        del self._filters[global_track_id]

    def sync_track_from_filter(self, global_track_id: int) -> None:
        track = self._tracks[global_track_id]
        tracker = self._filters[global_track_id]
        track.state = tracker.state
        track.covariance = tracker.covariance
