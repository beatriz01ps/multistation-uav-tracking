"""Fusao em espaco de informacao (covariance-weighted average) - estrategia
default da V1 (config: fusion.strategy = "information").

Sob a hipotese de independencia entre as estimativas locais:

    P_fused = inv(sum(inv(P_i)))
    x_fused = P_fused @ sum(inv(P_i) @ x_i)

Atencao metodologica: essa hipotese de independencia e otimista quando as
estimativas locais tem correlacoes
desconhecidas entre si (cada estacao roda seu proprio filtro local,
potencialmente sujeito a fontes de erro correlacionadas - ex.: mesma
trajetoria real "vista" de formas parecidas). Por isso Covariance
Intersection (fusion/covariance_intersection.py) fica disponivel atras da
mesma interface como estrategia alternativa, sem exigir mudanca em nenhum
outro modulo.
"""

from __future__ import annotations

import numpy as np

from models.fused_measurement import FusedMeasurement
from models.local_tracklet import LocalTracklet
from models.validation import regularize_for_inversion


class InformationFusion:
    def fuse_states(self, entries: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
        if not entries:
            raise ValueError("fuse_states() precisa de pelo menos uma entrada")

        if len(entries) == 1:
            return entries[0]

        info_matrix = np.zeros((6, 6))
        info_vector = np.zeros(6)

        for state, covariance in entries:
            # covariancia PSD (aceita pela validacao de entrada) pode ser
            # SINGULAR - regulariza antes de inverter (ver
            # models.validation.regularize_for_inversion para a politica).
            covariance = regularize_for_inversion(covariance)
            precision = np.linalg.inv(covariance)
            info_matrix += precision
            info_vector += precision @ state

        fused_covariance = np.linalg.inv(regularize_for_inversion(info_matrix))
        fused_state = fused_covariance @ info_vector
        return fused_state, fused_covariance

    def fuse(self, tracklets: list[LocalTracklet]) -> FusedMeasurement:
        if not tracklets:
            raise ValueError("fuse() precisa de pelo menos um tracklet")

        fused_state, fused_covariance = self.fuse_states([(t.state, t.covariance) for t in tracklets])

        return FusedMeasurement(
            state=fused_state,
            covariance=fused_covariance,
            timestamp=max(t.timestamp for t in tracklets),
            contributing_tracklets=[(t.station_id, t.local_track_id) for t in tracklets],
        )
