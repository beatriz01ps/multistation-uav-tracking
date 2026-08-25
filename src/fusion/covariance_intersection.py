"""Covariance Intersection (CI) - estrategia alternativa de fusao,
consistente mesmo quando as correlacoes entre as estimativas locais sao
desconhecidas (ao contrario de InformationFusion, que assume
independencia). Mesma interface (TrackFusionStrategy), trocavel via
config (fusion.strategy = "covariance_intersection") sem mudar mais nada.

Portado do proj em rust, onde ja foi validado numericamente (testes de order-dependence entre permutacoes de fontes).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from models.fused_measurement import FusedMeasurement
from models.local_tracklet import LocalTracklet
from models.validation import regularize_for_inversion


def _pairwise_ci(
    x1: np.ndarray, p1: np.ndarray, x2: np.ndarray, p2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # p1/p2 sao PSD (aceitas pela validacao de entrada) mas podem ser
    # singulares - regulariza antes de inverter (models.validation.regularize_for_inversion).
    p1_inv = np.linalg.inv(regularize_for_inversion(p1))
    p2_inv = np.linalg.inv(regularize_for_inversion(p2))

    def fused_trace(omega: float) -> float:
        p_inv = omega * p1_inv + (1 - omega) * p2_inv
        return float(np.trace(np.linalg.inv(p_inv)))

    result = minimize_scalar(fused_trace, bounds=(1e-6, 1 - 1e-6), method="bounded")
    omega = float(result.x)

    p_fused_inv = omega * p1_inv + (1 - omega) * p2_inv
    p_fused = np.linalg.inv(p_fused_inv)
    x_fused = p_fused @ (omega * p1_inv @ x1 + (1 - omega) * p2_inv @ x2)
    return x_fused, p_fused


class CovarianceIntersectionFusion:
    """Encadeia CI par a par (A,B -> AB; AB,C -> ABC; ...).

    NOTA: essa forma sequencial NAO e order-independent em geral para 3+
    fontes - e uma otimizacao gulosa local a cada par, nao uma otimizacao
    conjunta de N pesos. Verificado empiricamente (projeto anterior): com
    covariancias diagonais similares, a diferenca entre ordens fica na
    casa de 1e-5 (ruido numerico); com covariancias correlacionadas e
    disparidade de ~30x entre incertezas, a diferenca chega a ~2% no traco
    da covariancia fundida. Aceitavel para V1 porque a ordem de
    processamento (por station_id, alfabetica) e deterministica; uma fusao
    N-aria conjunta fica como extensao futura se o efeito se mostrar
    relevante na pratica.
    """

    def fuse_states(self, entries: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
        if not entries:
            raise ValueError("fuse_states() precisa de pelo menos uma entrada")

        x, p = entries[0]
        for state, covariance in entries[1:]:
            x, p = _pairwise_ci(x, p, state, covariance)
        return x, p

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
