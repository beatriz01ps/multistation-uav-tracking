"""Testes do EkfTracker - espelham test_ukf.py de proposito, pra deixar
claro que os dois tem a MESMA interface/comportamento esperado com
Constant Velocity. Os dois testes mais importantes deste arquivo:

- `test_ekf_matches_ukf_bit_for_bit_for_constant_velocity_with_zero_process_noise`:
  prova formal de que, pra um modelo linear SEM ruido de processo (Q=0), o
  PREDICT e o UPDATE de EKF e UKF sao o mesmo calculo matematico (Jacobiano
  = matriz de transicao, sigma points de funcao linear reproduzem a formula
  fechada exata).
- `test_ekf_and_ukf_diverge_slightly_for_constant_velocity_with_real_process_noise`:
  mostra que essa identidade NAO se estende ao caso real (Q > 0) - o UKF do
  FilterPy reusa, no update(), os sigma points computados no predict()
  ANTES de Q ser somado a P (formulacao "additive noise" da biblioteca,
  ver filterpy/kalman/UKF.py). O predict continua batendo bit a bit; so o
  update diverge, numa magnitude pequena mas real e mensuravel. Nao e bug -
  e uma caracteristica conhecida da implementacao, documentada tambem em
  filtering/ekf.py e config/models.py::FilterConfig.type."""

import numpy as np
import pytest

from filtering.ekf import EkfTracker
from filtering.motion_models import ConstantVelocityModel, CoordinatedTurnModel, coordinated_turn_step
from filtering.ukf import UkfTracker


@pytest.fixture
def tracker() -> EkfTracker:
    return EkfTracker(
        initial_state=np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0]),
        initial_covariance=np.eye(6) * 4.0,
        motion_model=ConstantVelocityModel(),
        process_noise_acceleration_std=2.0,
    )


def test_predict_moves_position_by_velocity_times_dt(tracker: EkfTracker):
    tracker.predict(1.0)
    assert tracker.state[0] == pytest.approx(10.0, abs=1e-6)
    assert tracker.state[3] == pytest.approx(10.0, abs=1e-6)


def test_predict_is_exact_for_constant_velocity_model(tracker: EkfTracker):
    tracker.predict(2.5)
    assert tracker.state[0] == pytest.approx(25.0, abs=1e-6)


def test_prediction_only_increases_covariance():
    tracker = EkfTracker(np.zeros(6), np.eye(6), ConstantVelocityModel(), process_noise_acceleration_std=2.0)
    trace_before = np.trace(tracker.covariance)
    tracker.predict(1.0)
    assert np.trace(tracker.covariance) > trace_before


def test_update_pulls_state_toward_measurement(tracker: EkfTracker):
    tracker.predict(1.0)
    tracker.update(np.array([10.0, 5.0, 100.0, 10.0, 0.0, 0.0]), np.eye(6) * 1.0)
    assert 0.0 < tracker.state[1] < 5.0


def test_update_reduces_uncertainty(tracker: EkfTracker):
    tracker.predict(1.0)
    trace_before = np.trace(tracker.covariance)
    tracker.update(np.array([10.0, 0.0, 100.0, 10.0, 0.0, 0.0]), np.eye(6) * 1.0)
    assert np.trace(tracker.covariance) < trace_before


def test_coasting_state_keeps_evolving_without_measurement(tracker: EkfTracker):
    positions = [tracker.state[0]]
    for _ in range(5):
        tracker.predict(1.0)
        positions.append(tracker.state[0])

    assert positions == sorted(positions)
    assert positions[-1] == pytest.approx(50.0, abs=1e-6)
    assert np.trace(tracker.covariance) > 4.0 * 6


def test_predict_with_non_positive_dt_is_a_noop(tracker: EkfTracker):
    state_before = tracker.state.copy()
    tracker.predict(0.0)
    assert np.array_equal(tracker.state, state_before)


def test_set_state_overwrites_the_kinematic_state_and_covariance(tracker: EkfTracker):
    tracker.set_state(np.array([50.0, 60.0, 100.0, 1.0, 2.0, 3.0]), np.eye(6) * 0.5)
    assert np.allclose(tracker.state, [50.0, 60.0, 100.0, 1.0, 2.0, 3.0])
    assert np.allclose(tracker.covariance, np.eye(6) * 0.5)


def test_set_state_preserves_the_extra_internal_dimension_of_coordinated_turn():
    tracker = EkfTracker(
        initial_state=np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0]),
        initial_covariance=np.eye(6) * 4.0,
        motion_model=CoordinatedTurnModel(turn_rate_process_noise_std=0.05, initial_turn_rate_std=0.3),
        process_noise_acceleration_std=2.0,
    )
    true_state = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0, 0.3])
    for _ in range(5):
        true_state = coordinated_turn_step(true_state, 0.5)
        tracker.predict(0.5)
        tracker.update(true_state[:6], np.eye(6) * 0.1)

    omega_before = tracker._x[6]
    assert abs(omega_before) > 1e-3

    tracker.set_state(np.array([1.0, 1.0, 100.0, 5.0, 5.0, 0.0]), np.eye(6) * 2.0)

    assert tracker._x[6] == omega_before
    assert np.allclose(tracker.state, [1.0, 1.0, 100.0, 5.0, 5.0, 0.0])


# --- O teste que importa: EKF vs UKF ---


def test_ekf_matches_ukf_bit_for_bit_for_constant_velocity_with_zero_process_noise():
    """Prova formal (nao so 'parece igual'): Constant Velocity e um
    modelo LINEAR - o Jacobiano do EKF E a propria matriz de transicao, e
    os sigma points do UKF propagados por uma funcao linear reproduzem
    exatamente a mesma media/covariancia que a formula linear fechada. Com
    Q=0 isso vale tanto pro predict quanto pro update - mesmo estado
    inicial, mesmas medicoes, os dois terminam bit a bit iguais (tolerancia
    numerica de ponto flutuante, nao de metodo). Ver o teste seguinte pro
    que muda quando Q > 0."""
    initial_state = np.array([0.0, 0.0, 100.0, 10.0, 3.0, 0.0])
    initial_covariance = np.eye(6) * 4.0

    ekf = EkfTracker(initial_state, initial_covariance, ConstantVelocityModel(), process_noise_acceleration_std=0.0)
    ukf = UkfTracker(initial_state, initial_covariance, ConstantVelocityModel(), process_noise_acceleration_std=0.0)

    rng = np.random.default_rng(0)
    for _ in range(8):
        dt = float(rng.uniform(0.2, 1.5))
        ekf.predict(dt)
        ukf.predict(dt)
        assert np.allclose(ekf.state, ukf.state, atol=1e-9)
        assert np.allclose(ekf.covariance, ukf.covariance, atol=1e-9)

        measurement = rng.normal(0, 3, size=6)
        measurement_covariance = np.eye(6) * float(rng.uniform(0.5, 3.0))
        ekf.update(measurement, measurement_covariance)
        ukf.update(measurement, measurement_covariance)
        assert np.allclose(ekf.state, ukf.state, atol=1e-9)
        assert np.allclose(ekf.covariance, ukf.covariance, atol=1e-9)


def test_ekf_and_ukf_diverge_slightly_for_constant_velocity_with_real_process_noise():
    """Com Q > 0 (o caso real, process_noise_acceleration_std=2.0 como no
    default do projeto), o predict continua batendo bit a bit - mas o
    update passa a divergir, porque o UKF do FilterPy calcula a covariancia
    cruzada do update() reusando os sigma points do predict() anterior,
    computados ANTES de Q ser somado a P (ver filtering/ekf.py para a
    explicacao completa). Este teste documenta e QUANTIFICA essa
    divergencia, em vez de fingir que ela nao existe: pequena (a covariancia
    dos dois filtros fica na mesma ordem de grandeza), mas real - nao e
    ruido de ponto flutuante (ordem de 1e-13), e uma diferenca estrutural
    (tipicamente > 1e-3 em relacao ao valor absoluto)."""
    initial_state = np.array([0.0, 0.0, 100.0, 10.0, 3.0, 0.0])
    initial_covariance = np.eye(6) * 4.0

    ekf = EkfTracker(initial_state, initial_covariance, ConstantVelocityModel(), process_noise_acceleration_std=2.0)
    ukf = UkfTracker(initial_state, initial_covariance, ConstantVelocityModel(), process_noise_acceleration_std=2.0)

    rng = np.random.default_rng(0)

    # antes do primeiro update, o predict ainda bate bit a bit (Q e somado
    # identicamente nos dois: P' = F P F^T + Q) - a partir do primeiro
    # update() os ESTADOS ja divergiram, entao predicts seguintes partem de
    # pontos de partida diferentes e nao ha por que continuarem batendo.
    dt_first = float(rng.uniform(0.2, 1.5))
    ekf.predict(dt_first)
    ukf.predict(dt_first)
    assert np.allclose(ekf.state, ukf.state, atol=1e-9)
    assert np.allclose(ekf.covariance, ukf.covariance, atol=1e-9)

    max_relative_covariance_diff = 0.0
    for i in range(8):
        if i > 0:
            dt = float(rng.uniform(0.2, 1.5))
            ekf.predict(dt)
            ukf.predict(dt)

        measurement = rng.normal(0, 3, size=6)
        measurement_covariance = np.eye(6) * float(rng.uniform(0.5, 3.0))
        ekf.update(measurement, measurement_covariance)
        ukf.update(measurement, measurement_covariance)

        # so a DIAGONAL (variancias) - termos fora da diagonal podem ficar
        # perto de zero e explodir uma diferenca relativa sem significado
        # pratico nenhum.
        ekf_diag = np.diag(ekf.covariance)
        ukf_diag = np.diag(ukf.covariance)
        diff = np.abs(ekf_diag - ukf_diag)
        scale = np.abs(ekf_diag) + 1e-9
        max_relative_covariance_diff = max(max_relative_covariance_diff, float(np.max(diff / scale)))

    # a divergencia e real (nao ponto flutuante) - mas os dois filtros
    # continuam na mesma ordem de grandeza, nao "explodem" um em relacao ao
    # outro.
    assert max_relative_covariance_diff > 1e-3
    assert max_relative_covariance_diff < 5.0


def test_ekf_coordinated_turn_also_extrapolates_the_curve_after_learning_omega():
    """Mesma logica de test_ukf_coordinated_turn.py: o EKF tambem deve
    aprender omega a partir de medicoes reais de uma curva e continuar
    extrapolando ela durante coasting - nao e uma capacidade exclusiva do
    UKF, so a QUALIDADE da extrapolacao que pode diferir (proximo teste)."""
    omega_true = 0.3
    dt = 0.5
    rng = np.random.default_rng(0)
    measurement_covariance = np.eye(6) * 0.25

    true_state = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0, omega_true])
    tracker = EkfTracker(
        true_state[:6], np.eye(6) * 4.0,
        CoordinatedTurnModel(turn_rate_process_noise_std=0.05, initial_turn_rate_std=0.3),
        process_noise_acceleration_std=1.0,
    )

    for _ in range(10):
        true_state = coordinated_turn_step(true_state, dt)
        tracker.predict(dt)
        tracker.update(true_state[:6] + rng.normal(0.0, 0.5, size=6), measurement_covariance)

    last_position = tracker.state[:3].copy()
    last_velocity = tracker.state[3:6].copy()

    coast_steps = 4
    for _ in range(coast_steps):
        true_state = coordinated_turn_step(true_state, dt)
        tracker.predict(dt)

    error_ct = np.linalg.norm(tracker.state[:3] - true_state[:3])
    naive_straight_line = last_position + last_velocity * (dt * coast_steps)
    error_naive = np.linalg.norm(naive_straight_line - true_state[:3])

    assert error_ct < error_naive


def test_ukf_and_ekf_diverge_for_coordinated_turn_under_a_real_maneuver():
    """AQUI a diferenca entre UKF e EKF deveria aparecer de verdade: CT e
    nao-linear, entao o EKF (linearizacao de 1a ordem) e o UKF (sigma
    points, sem aproximacao linear) NAO tem por que dar o mesmo resultado
    quando omega realmente se move para longe de 0. Nao afirma qual e
    "melhor" aqui (isso fica pra comparacao com verdade em cenario
    completo, via network_simulator) - so confirma que divergem, provando
    que a comparacao cientifica pedida faz sentido (se fossem sempre
    iguais, a pergunta "o UKF faz diferenca?" nao teria como ter resposta
    interessante)."""
    omega_true = 0.4
    dt = 0.5
    rng = np.random.default_rng(3)
    measurement_covariance = np.eye(6) * 0.25
    motion_model_kwargs = dict(turn_rate_process_noise_std=0.05, initial_turn_rate_std=0.3)

    true_state = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0, omega_true])
    initial_state = true_state[:6].copy()
    initial_covariance = np.eye(6) * 4.0

    ekf = EkfTracker(initial_state, initial_covariance, CoordinatedTurnModel(**motion_model_kwargs), 1.0)
    ukf = UkfTracker(initial_state, initial_covariance, CoordinatedTurnModel(**motion_model_kwargs), 1.0)

    state_for_measurements = true_state.copy()
    for _ in range(10):
        state_for_measurements = coordinated_turn_step(state_for_measurements, dt)
        noisy_measurement = state_for_measurements[:6] + rng.normal(0.0, 0.5, size=6)
        ekf.predict(dt)
        ukf.predict(dt)
        ekf.update(noisy_measurement, measurement_covariance)
        ukf.update(noisy_measurement, measurement_covariance)

    assert not np.allclose(ekf.state, ukf.state, atol=1e-6)
