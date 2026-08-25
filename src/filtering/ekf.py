"""EkfTracker: Extended Kalman Filter - existe para comparacao cientifica
com o UKF (ver docs/arquitetura.md e filtering/ukf.py), nao substitui o
UKF como default. Mesma interface publica que `UkfTracker`
(filtering/base.py::FilterTracker) - trocavel via config
(`filter.type: ekf`), sem mudar mais nada no resto do sistema.

Diferenca central pro UKF: em vez de propagar sigma points pela funcao de
transicao (nao-linear), o EKF LINEARIZA a transicao em torno do estado
atual via o Jacobiano (`MotionModel.jacobian`) e propaga a covariancia com
a formula linear classica `P' = F P F^T + Q`. Para um modelo LINEAR
(Constant Velocity), o Jacobiano E a propria matriz de transicao - entao o
PREDICT de EKF e UKF e bit a bit identico sempre (verificado em
tests/unit/test_ekf.py).

O UPDATE e uma historia mais sutil: so bate bit a bit quando o ruido de
processo Q e zero. Com Q > 0 (o caso real), o UKF do FilterPy calcula a
covariancia cruzada do update() reusando os sigma points do predict()
anterior - computados ANTES de Q ser somado a P (formulacao "additive
noise" padrao da biblioteca, ver filterpy/kalman/UKF.py: predict() chama
unscented_transform(sigmas_f, ..., Q) pra obter self.P, mas update() usa
os MESMOS self.sigmas_f, nao regenerados a partir do P ja inflado por Q).
Isso faz o ganho de Kalman do UKF diferir um pouco do EKF mesmo em CV,
numa magnitude que cresce com Q (verificado empiricamente: nao e ruido de
ponto flutuante, e uma diferenca estrutural real, ainda que geralmente
pequena perto da diferenca que aparece em Coordinated Turn). Nao e um bug
nem no UKF nem no EKF - e uma caracteristica conhecida da formulacao
aditiva do UKF, e faz parte do que estamos comparando cientificamente
aqui: mesmo em CV, a escolha de biblioteca/formulacao do UKF tem um custo
mensuravel, nao so teorico.

Para Coordinated Turn (nao-linear) os dois divergem de verdade, ja no
predict - e exatamente o que motiva ter os dois implementados: medir se a
abordagem mais sofisticada do UKF (sem aproximacao de primeira ordem)
realmente compra alguma coisa na pratica, ou se o UKF so "faz sentido no
papel" pra este problema especifico.

Implementacao das equacoes do EKF diretamente (sem depender da classe
`ExtendedKalmanFilter` do FilterPy, cuja API assume `predict()` linear por
default e exigiria contornar isso) - mesma politica de
`association/mahalanobis.py` e `models/validation.py`: sempre regularizar
a covariancia no ponto de uso que precisa dela bem-condicionada.
"""

from __future__ import annotations

import numpy as np

from filtering.motion_models import MotionModel
from models.validation import regularize_for_inversion as regularize_covariance

MEASUREMENT_DIM = 6  # dimensao do contrato EXTERNO - sempre fixa, independente do modelo de movimento


class EkfTracker:
    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        motion_model: MotionModel,
        process_noise_acceleration_std: float,
    ) -> None:
        augmented_state, augmented_covariance = motion_model.augment_initial_state(
            np.asarray(initial_state, dtype=float).copy(),
            np.asarray(initial_covariance, dtype=float).copy(),
        )
        self._x = augmented_state
        self._P = regularize_covariance(augmented_covariance)
        self._motion_model = motion_model
        self._process_noise_acceleration_std = process_noise_acceleration_std

    def predict(self, dt: float) -> None:
        if dt <= 0:
            return
        # Jacobiano AVALIADO NO ESTADO ATUAL (antes de propagar o estado em
        # si) - e essa a definicao de linearizacao do EKF: aproxima a
        # funcao nao-linear pela sua tangente no ponto onde estamos agora,
        # nao no ponto de destino.
        transition_jacobian = self._motion_model.jacobian(self._x, dt)
        self._x = self._motion_model.step(self._x, dt)
        process_noise = self._motion_model.process_noise(dt, self._process_noise_acceleration_std)
        self._P = regularize_covariance(
            transition_jacobian @ self._P @ transition_jacobian.T + process_noise
        )

    def update(self, measurement: np.ndarray, measurement_covariance: np.ndarray) -> None:
        # H = projecao das MEASUREMENT_DIM primeiras dimensoes - a mesma
        # observacao direta do estado cinematico completo que o UKF usa
        # (hx=lambda x: x[:MEASUREMENT_DIM] em filtering/ukf.py), aqui como
        # matriz explicita (o EKF exige uma matriz H linear de verdade,
        # nao uma funcao hx generica como o UKF).
        state_dim = self._x.shape[0]
        observation_matrix = np.zeros((MEASUREMENT_DIM, state_dim))
        observation_matrix[:, :MEASUREMENT_DIM] = np.eye(MEASUREMENT_DIM)

        measurement = np.asarray(measurement, dtype=float)
        measurement_covariance = np.asarray(measurement_covariance, dtype=float)

        innovation = measurement - observation_matrix @ self._x
        innovation_covariance = regularize_covariance(
            observation_matrix @ self._P @ observation_matrix.T + measurement_covariance
        )
        # K = P H^T S^-1, resolvido via solve() sobre S^T (evita inv()
        # explicita) - mesma politica de estabilidade numerica ja usada em
        # association/mahalanobis.py.
        kalman_gain = np.linalg.solve(innovation_covariance.T, observation_matrix @ self._P.T).T

        self._x = self._x + kalman_gain @ innovation
        identity = np.eye(state_dim)
        # forma de Joseph (numericamente mais estavel que (I-KH)P simples,
        # e continua simetrica por construcao mesmo com erro de ponto
        # flutuante) - mesmo cuidado que o resto do projeto ja tem com
        # covariancia mal-condicionada.
        gain_term = identity - kalman_gain @ observation_matrix
        self._P = regularize_covariance(
            gain_term @ self._P @ gain_term.T
            + kalman_gain @ measurement_covariance @ kalman_gain.T
        )

    def set_state(self, state: np.ndarray, covariance: np.ndarray) -> None:
        """Ver docstring identica em UkfTracker.set_state - dimensoes
        internas extras (omega do Coordinated Turn) ficam intocadas."""
        self._x[:MEASUREMENT_DIM] = np.asarray(state, dtype=float)
        self._P[:MEASUREMENT_DIM, :MEASUREMENT_DIM] = regularize_covariance(np.asarray(covariance, dtype=float))
        self._P = regularize_covariance(self._P)

    @property
    def state(self) -> np.ndarray:
        return self._x[:MEASUREMENT_DIM].copy()

    @property
    def covariance(self) -> np.ndarray:
        return self._P[:MEASUREMENT_DIM, :MEASUREMENT_DIM].copy()
