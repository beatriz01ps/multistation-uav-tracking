"""Validacao de vetores/matrizes numpy usados nos modelos Pydantic.

Centralizado aqui para nao duplicar a mesma logica em LocalTracklet,
GlobalTrack e FusedMeasurement (todos usam o mesmo contrato de estado
[x,y,z,vx,vy,vz] + covariancia 6x6).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

STATE_DIM = 6 # tem q ser 6


def coerce_state_vector(value) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (STATE_DIM,):
        raise ValueError(f"state deve ter shape ({STATE_DIM},), recebeu {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("state contem valores nao finitos (nan/inf)")
    return array


def coerce_optional_state_vector(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    return coerce_state_vector(value)


# tolerancia numerica para a checagem de semidefinida positiva: autovalores
# levemente negativos por erro de ponto flutuante (ex.: -1e-10) sao
# aceitos; qualquer coisa mais negativa que isso indica uma matriz
# matematicamente invalida como covariancia (nao so "feia numericamente").
_PSD_EIGENVALUE_TOLERANCE = 1e-8


def coerce_covariance_matrix(value) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (STATE_DIM, STATE_DIM):
        raise ValueError(f"covariance deve ter shape ({STATE_DIM},{STATE_DIM}), recebeu {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("covariance contem valores nao finitos (nan/inf)")
    if not np.allclose(array, array.T, atol=1e-6):
        raise ValueError("covariance deve ser simetrica")
    if np.any(np.diag(array) < 0):
        raise ValueError("covariance tem variancia negativa na diagonal")

    # simetria + diagonal nao-negativa NAO garantem positiva semidefinida -
    # uma matriz pode passar nos dois testes acima e ainda ter autovalor
    # negativo (ex.: covariancia "inventada" com correlacao inconsistente).
    # Isso quebraria Mahalanobis/fusao/UKF silenciosamente mais tarde -
    # melhor rejeitar aqui, na fronteira de entrada.
    eigenvalues = np.linalg.eigvalsh(array)
    smallest = float(eigenvalues.min())
    if smallest < -_PSD_EIGENVALUE_TOLERANCE:
        raise ValueError(
            f"covariance nao e positiva semidefinida (menor autovalor={smallest:.3e}); "
            "matrizes de covariancia validas tem todos os autovalores >= 0"
        )
    return array


def regularize_for_inversion(matrix: np.ndarray, min_eigenvalue: float = 1e-9) -> np.ndarray:
    """Simetriza e garante um piso minimo de autovalor.

    Politica adotada (decisao explicita, nao escolha silenciosa): uma
    covariancia positiva semidefinida (aceita por
    `coerce_covariance_matrix`) pode ser SINGULAR (autovalor exatamente 0)
    - matematicamente valida, mas nao invertivel. Em vez de rejeitar essa
    covariancia na entrada (Opcao A: exigir positiva DEFINIDA, mais
    restritivo do que o contrato pede), regularizamos aqui, no PONTO DE
    USO que precisa de inversao (Opcao B) - fusao, Mahalanobis, UKF -
    somando um epsilon minusculo a diagonal (`P + eps*I`). O efeito no
    resultado numerico e desprezivel (eps=1e-9 << qualquer variancia real
    de posicao/velocidade), mas evita crash de `LinAlgError: Singular
    matrix` sem alterar silenciosamente o SIGNIFICADO da covariancia."""
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(matrix)
    smallest = float(eigenvalues.min())
    if smallest < min_eigenvalue:
        matrix = matrix + (min_eigenvalue - smallest) * np.eye(matrix.shape[0])
    return matrix
