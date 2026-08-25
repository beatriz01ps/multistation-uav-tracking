"""Interface comum entre `UkfTracker` e `EkfTracker` - o resto do sistema
(TrackManager, TrackInitializer, DuplicateTrackMerger) usa um `GlobalTrack`
via esse contrato, sem saber (nem precisar saber) qual dos dois filtros
esta por baixo. Ver filtering/factory.py para a escolha por config."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class FilterTracker(Protocol):
    def predict(self, dt: float) -> None: ...

    def update(self, measurement: np.ndarray, measurement_covariance: np.ndarray) -> None: ...

    def set_state(self, state: np.ndarray, covariance: np.ndarray) -> None: ...

    @property
    def state(self) -> np.ndarray: ...

    @property
    def covariance(self) -> np.ndarray: ...
