"""Distancia de Mahalanobis entre um GlobalTrack previsto e um tracklet
local. Evita inversao explicita da matriz de inovacao: resolve o sistema
linear S @ y = innovation via numpy.linalg.solve (mais estavel
numericamente do que calcular inv(S) e multiplicar).

`predicted_covariance` normalmente ja vem regularizada (e a P interna do
UKF, sempre positiva definida - ver filtering/ukf.py). Mas para um
candidato de track novo com uma unica fonte, "predicted_covariance" pode
ser a covariancia BRUTA de um tracklet (possivelmente singular - PSD e
aceito na entrada, nao exige ser definida). `solve()` quebra em matriz
singular do mesmo jeito que `inv()` quebraria - regulariza aqui tambem,
como ultima linha de defesa independente de quem chamou.
"""

from __future__ import annotations

import numpy as np

from models.validation import regularize_for_inversion


def mahalanobis_squared(
    predicted_state: np.ndarray,
    predicted_covariance: np.ndarray,
    measurement: np.ndarray,
    measurement_covariance: np.ndarray,
    dims: tuple[int, ...],
) -> float:
    idx = list(dims)
    innovation = (measurement - predicted_state)[idx]
    innovation_covariance = (predicted_covariance + measurement_covariance)[np.ix_(idx, idx)]
    innovation_covariance = regularize_for_inversion(innovation_covariance)
    y = np.linalg.solve(innovation_covariance, innovation)
    return float(innovation @ y)
