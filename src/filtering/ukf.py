"""UkfTracker: camada propria sobre o FilterPy, para que o resto do sistema
nao dependa diretamente da API do FilterPy (troca de biblioteca no futuro
fica isolada aqui). Encapsula o UKF para um unico GlobalTrack.

A medicao, nesta primeira versao, e tratada como observacao direta do
estado CINEMATICO completo [x,y,z,vx,vy,vz] (H = projecao das 6 primeiras
dimensoes) - os tracklets locais ja sao estimativas filtradas, nao leituras
brutas de sensor. O estado INTERNO do filtro pode ter mais dimensoes que
isso (ver filtering/motion_models.py::CoordinatedTurnModel, que acrescenta
uma taxa de giro nao observada diretamente) - `state`/`covariance` sempre
projetam de volta para as 6 dimensoes do contrato externo antes de sair
desta classe, entao quem usa `UkfTracker` de fora nunca precisa saber
quantas dimensoes o modelo de movimento escolhido usa por baixo.
"""

from __future__ import annotations

import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

from filtering.motion_models import MotionModel
from models.validation import regularize_for_inversion as regularize_covariance

MEASUREMENT_DIM = 6  # dimensao do contrato EXTERNO - sempre fixa, independente do modelo de movimento


class UkfTracker:
    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        motion_model: MotionModel,
        process_noise_acceleration_std: float,
    ) -> None:
        state_dim = motion_model.state_dim
        augmented_state, augmented_covariance = motion_model.augment_initial_state(
            np.asarray(initial_state, dtype=float).copy(),
            np.asarray(initial_covariance, dtype=float).copy(),
        )

        points = MerweScaledSigmaPoints(n=state_dim, alpha=0.1, beta=2.0, kappa=3 - state_dim)
        self._ukf = UnscentedKalmanFilter(
            dim_x=state_dim,
            dim_z=MEASUREMENT_DIM,
            dt=1.0,
            fx=lambda x, dt: motion_model.step(x, dt),
            hx=lambda x: x[:MEASUREMENT_DIM],
            points=points,
        )
        self._ukf.x = augmented_state
        self._ukf.P = regularize_covariance(augmented_covariance)
        self._motion_model = motion_model
        self._process_noise_acceleration_std = process_noise_acceleration_std

    def predict(self, dt: float) -> None:
        if dt <= 0:
            return
        self._ukf.Q = self._motion_model.process_noise(dt, self._process_noise_acceleration_std)
        # P precisa estar bem-condicionada ANTES do predict: o proprio
        # predict ja gera sigma points (Cholesky) a partir dela.
        self._ukf.P = regularize_covariance(self._ukf.P)
        self._ukf.predict(dt=dt)
        self._ukf.P = regularize_covariance(self._ukf.P)

    def update(self, measurement: np.ndarray, measurement_covariance: np.ndarray) -> None:
        self._ukf.update(
            np.asarray(measurement, dtype=float), R=np.asarray(measurement_covariance, dtype=float)
        )
        self._ukf.P = regularize_covariance(self._ukf.P)

    def set_state(self, state: np.ndarray, covariance: np.ndarray) -> None:
        """Sobrescreve a parte CINEMATICA (as 6 dimensoes do contrato
        externo) do estado interno do filtro - usado pela fusao de
        GlobalTracks duplicados (tracking/duplicate_merger.py), nao pelo
        predict/update normais. Dimensoes internas extras (ex.: omega do
        Coordinated Turn) ficam INALTERADAS: nao ha um jeito natural de
        "fundir" a taxa de giro estimada de dois tracks que deveriam ser o
        mesmo alvo - a estimativa que este filtro ja tinha continua sendo a
        melhor informacao disponivel sobre ela."""
        self._ukf.x[:MEASUREMENT_DIM] = np.asarray(state, dtype=float)
        self._ukf.P[:MEASUREMENT_DIM, :MEASUREMENT_DIM] = regularize_covariance(np.asarray(covariance, dtype=float))
        self._ukf.P = regularize_covariance(self._ukf.P)

    @property
    def state(self) -> np.ndarray:
        return self._ukf.x[:MEASUREMENT_DIM].copy()

    @property
    def covariance(self) -> np.ndarray:
        return self._ukf.P[:MEASUREMENT_DIM, :MEASUREMENT_DIM].copy()
