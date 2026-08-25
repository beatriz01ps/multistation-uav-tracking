"""Testes do modelo Coordinated Turn em si (filtering/motion_models.py),
isolado do UKF - a transicao de estado, o ruido de processo e a
inicializacao do estado aumentado."""

import numpy as np
import pytest

from filtering.motion_models import (
    STATE_DIM,
    CoordinatedTurnModel,
    constant_velocity_transition_matrix,
    coordinated_turn_jacobian,
    coordinated_turn_process_noise,
    coordinated_turn_step,
)


def _numeric_jacobian(state: np.ndarray, dt: float, eps: float = 1e-6) -> np.ndarray:
    n = len(state)
    jacobian = np.zeros((n, n))
    for i in range(n):
        plus = state.copy()
        plus[i] += eps
        minus = state.copy()
        minus[i] -= eps
        jacobian[:, i] = (coordinated_turn_step(plus, dt) - coordinated_turn_step(minus, dt)) / (2 * eps)
    return jacobian


def test_zero_turn_rate_reduces_to_constant_velocity():
    state = np.array([0.0, 0.0, 100.0, 10.0, 3.0, -1.0, 0.0])  # omega=0
    dt = 1.5

    ct_result = coordinated_turn_step(state, dt)
    cv_result = constant_velocity_transition_matrix(dt) @ state[:STATE_DIM]

    assert np.allclose(ct_result[:STATE_DIM], cv_result)
    assert ct_result[STATE_DIM] == 0.0  # omega permanece 0


def test_nonzero_turn_rate_curves_away_from_the_straight_line():
    state = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0, 0.5])  # omega=0.5 rad/s
    dt = 1.0

    curved = coordinated_turn_step(state, dt)
    straight = constant_velocity_transition_matrix(dt) @ state[:STATE_DIM]

    # com omega != 0, a posicao NAO pode coincidir com a extrapolacao reta
    assert not np.allclose(curved[:2], straight[:2])
    # mas o modulo da velocidade se preserva (giro nao acelera nem freia)
    assert np.linalg.norm(curved[3:5]) == pytest.approx(np.linalg.norm(state[3:5]), abs=1e-9)


def test_full_period_returns_to_the_starting_point():
    # um giro completo (dt = 2*pi/omega) tem que devolver posicao E
    # velocidade ao ponto de partida - e um circulo fechado.
    omega = 0.5
    state = np.array([10.0, -5.0, 100.0, 8.0, -2.0, 0.0, omega])
    period = 2 * np.pi / omega

    result = coordinated_turn_step(state, period)

    assert np.allclose(result[:6], state[:6], atol=1e-6)
    assert result[6] == omega


def test_altitude_axis_is_unaffected_by_the_turn():
    state = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 5.0, 0.3])
    result = coordinated_turn_step(state, dt=2.0)
    assert result[2] == pytest.approx(110.0)  # z + vz*dt, igual ao CV
    assert result[5] == pytest.approx(5.0)  # vz inalterado


def test_process_noise_has_the_extra_turn_rate_dimension():
    q = coordinated_turn_process_noise(dt=1.0, acceleration_std=2.0, turn_rate_process_noise_std=0.05)
    assert q.shape == (STATE_DIM + 1, STATE_DIM + 1)
    assert np.allclose(q, q.T)  # simetrica
    eigenvalues = np.linalg.eigvalsh(q)
    assert eigenvalues.min() >= -1e-9  # positiva semidefinida

    # variancia de omega cresce com dt (passeio aleatorio) - checagem direta da formula
    assert q[STATE_DIM, STATE_DIM] == pytest.approx(0.05**2 * 1.0)


def test_jacobian_matches_finite_difference_away_from_the_omega_epsilon_boundary():
    """So longe da fronteira |omega| ~ _OMEGA_EPSILON: perto dela,
    `coordinated_turn_step` tem um corte artificial (estabilidade
    numerica) que deixa a saida localmente constante em omega - diferenca
    finita ali mediria uma derivada falsa, nao um erro no Jacobiano (ver
    docstring de coordinated_turn_jacobian)."""
    rng = np.random.default_rng(1)
    max_error = 0.0
    for _ in range(300):
        state = rng.normal(0, 10, size=7)
        state[6] = rng.choice([-1, 1]) * rng.uniform(1e-3, 2.0)  # nunca perto da fronteira 1e-6
        dt = rng.uniform(0.05, 3.0)
        analytic = coordinated_turn_jacobian(state, dt)
        numeric = _numeric_jacobian(state, dt)
        max_error = max(max_error, float(np.max(np.abs(analytic - numeric))))
    assert max_error < 1e-6


def test_jacobian_at_zero_omega_matches_constant_velocity_jacobian():
    state = np.array([0.0, 0.0, 100.0, 10.0, 3.0, -1.0, 0.0])
    dt = 1.5
    jacobian = coordinated_turn_jacobian(state, dt)
    assert np.allclose(jacobian[:STATE_DIM, :STATE_DIM], constant_velocity_transition_matrix(dt))
    assert np.allclose(jacobian[STATE_DIM, :], [0, 0, 0, 0, 0, 0, 1])  # d(omega')/d(*) = so ele mesmo


def test_omega_limit_formula_is_the_true_continuous_limit_of_the_general_formula():
    """Confirma que a formula fechada usada perto de omega=0 e mesmo o
    limite matematico da formula geral - calculando a formula GERAL (nao o
    atalho do limite) em omegas cada vez menores, sem nunca usar o
    proprio atalho, e checando que ela converge suavemente para o valor
    que o Jacobiano devolve em omega=0 exato."""
    state = np.array([0.0, 0.0, 0.0, 3.0, -2.0, 0.0, 0.0])
    dt = 1.7
    limit_jacobian = coordinated_turn_jacobian(state, dt)

    previous_error = None
    for omega_small in [1e-2, 1e-3, 1e-4, 1e-5]:
        phi = omega_small * dt
        s, c = np.sin(phi), np.cos(phi)
        a, b = s / omega_small, (1.0 - c) / omega_small
        da_dw = (dt * c * omega_small - s) / omega_small**2
        db_dw = (dt * s * omega_small - (1.0 - c)) / omega_small**2
        general_dx_domega = da_dw * state[3] - db_dw * state[4]
        general_dy_domega = db_dw * state[3] + da_dw * state[4]

        error = abs(general_dx_domega - limit_jacobian[0, 6]) + abs(general_dy_domega - limit_jacobian[1, 6])
        if previous_error is not None:
            assert error < previous_error  # converge monotonicamente conforme omega->0
        previous_error = error

    assert previous_error < 1e-3


def test_augment_initial_state_adds_omega_zero_with_the_configured_prior():
    model = CoordinatedTurnModel(turn_rate_process_noise_std=0.05, initial_turn_rate_std=0.3)
    state = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0])
    covariance = np.eye(6) * 4.0

    augmented_state, augmented_covariance = model.augment_initial_state(state, covariance)

    assert augmented_state.shape == (7,)
    assert augmented_state[6] == 0.0  # sem giro conhecido a priori
    assert augmented_covariance.shape == (7, 7)
    assert np.allclose(augmented_covariance[:6, :6], covariance)  # bloco cinematico original preservado
    assert augmented_covariance[6, 6] == pytest.approx(0.3**2)
    assert np.allclose(augmented_covariance[6, :6], 0.0)  # sem correlacao a priori
    assert np.allclose(augmented_covariance[:6, 6], 0.0)
