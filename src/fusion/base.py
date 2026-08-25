"""Interface comum das estrategias de fusao de tracklets associados ao
mesmo GlobalTrack no mesmo ciclo. Nao acoplar o restante do rastreador a
uma formula de fusao especifica - permite trocar/comparar estrategias sem
mudar quem as chama.

`fuse_states` e o nucleo matematico de cada estrategia, operando direto em
pares (state, covariance) - sem exigir um `LocalTracklet` (com
station_id/local_track_id/timestamp, que nao fazem sentido fora do
contexto de tracklets). `fuse()` (tracklets -> FusedMeasurement) e uma
casca fina por cima de `fuse_states()`. Isso e o que permite
`tracking/duplicate_merger.py` reusar a MESMA formula de fusao configurada
para combinar dois GlobalTracks duplicados, sem duplicar a matematica nem
inventar um LocalTracklet falso so para caber na assinatura antiga."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from models.enums import FusionStrategyName
from models.fused_measurement import FusedMeasurement
from models.local_tracklet import LocalTracklet


class TrackFusionStrategy(Protocol):
    def fuse(self, tracklets: list[LocalTracklet]) -> FusedMeasurement: ...

    def fuse_states(self, entries: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]: ...


def create_fusion_strategy(name: FusionStrategyName) -> TrackFusionStrategy:
    from fusion.covariance_intersection import CovarianceIntersectionFusion
    from fusion.information_fusion import InformationFusion

    if name is FusionStrategyName.INFORMATION:
        return InformationFusion()
    if name is FusionStrategyName.COVARIANCE_INTERSECTION:
        return CovarianceIntersectionFusion()
    raise ValueError(f"estrategia de fusao desconhecida: {name}")
