"""Chi-square gating sobre a distancia de Mahalanobis.

Os graus de liberdade do teste correspondem exatamente as dimensoes
efetivamente usadas na comparacao (3 para position_only, 6 para
full_state) - nunca um valor fixo independente do modo, para o gate
continuar estatisticamente correto nos dois modos configuraveis.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from association.mahalanobis import mahalanobis_squared
from models.enums import AssociationMode

POSITION_DIMS: tuple[int, ...] = (0, 1, 2)
FULL_STATE_DIMS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)


def dims_for_mode(mode: AssociationMode) -> tuple[int, ...]:
    return POSITION_DIMS if mode is AssociationMode.POSITION_ONLY else FULL_STATE_DIMS


def chi_square_threshold(probability: float, degrees_of_freedom: int) -> float:
    return float(chi2.ppf(probability, df=degrees_of_freedom))


def gate(
    predicted_state: np.ndarray,
    predicted_covariance: np.ndarray,
    measurement: np.ndarray,
    measurement_covariance: np.ndarray,
    mode: AssociationMode,
    chi_square_probability: float,
) -> tuple[bool, float]:
    """Retorna (passou_no_gate, d2). Se d2 > threshold, o par e
    estatisticamente incompativel e nao deve ser considerado pelo
    Hungarian."""
    dims = dims_for_mode(mode)
    d2 = mahalanobis_squared(predicted_state, predicted_covariance, measurement, measurement_covariance, dims)
    threshold = chi_square_threshold(chi_square_probability, degrees_of_freedom=len(dims))
    return d2 <= threshold, d2
