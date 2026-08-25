import numpy as np
import pytest

from filtering.motion_models import ConstantVelocityModel, CoordinatedTurnModel, create_motion_model
from filtering.ukf import UkfTracker
from models.enums import MotionModelName


@pytest.fixture
def tracker() -> UkfTracker:
    return UkfTracker(
        initial_state=np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0]),
        initial_covariance=np.eye(6) * 4.0,
        motion_model=ConstantVelocityModel(),
        process_noise_acceleration_std=2.0,
    )


def test_predict_moves_position_by_velocity_times_dt(tracker: UkfTracker):
    tracker.predict(1.0)
    assert tracker.state[0] == pytest.approx(10.0, abs=1e-6)
    assert tracker.state[3] == pytest.approx(10.0, abs=1e-6)  # CV: velocidade nao muda


def test_predict_is_exact_for_constant_velocity_model(tracker: UkfTracker):
    # modelo CV e linear -> UKF deve propagar exatamente, sem aproximacao
    tracker.predict(2.5)
    assert tracker.state[0] == pytest.approx(25.0, abs=1e-6)


def test_prediction_only_increases_covariance():
    tracker = UkfTracker(np.zeros(6), np.eye(6), ConstantVelocityModel(), process_noise_acceleration_std=2.0)
    trace_before = np.trace(tracker.covariance)
    tracker.predict(1.0)
    assert np.trace(tracker.covariance) > trace_before


def test_update_pulls_state_toward_measurement(tracker: UkfTracker):
    tracker.predict(1.0)
    tracker.update(np.array([10.0, 5.0, 100.0, 10.0, 0.0, 0.0]), np.eye(6) * 1.0)
    assert 0.0 < tracker.state[1] < 5.0


def test_update_reduces_uncertainty(tracker: UkfTracker):
    tracker.predict(1.0)
    trace_before = np.trace(tracker.covariance)
    tracker.update(np.array([10.0, 0.0, 100.0, 10.0, 0.0, 0.0]), np.eye(6) * 1.0)
    assert np.trace(tracker.covariance) < trace_before


def test_coasting_state_keeps_evolving_without_measurement(tracker: UkfTracker):
    """Requisito central do rastreador: durante ausencia de observacao, o
    estado deve continuar evoluindo segundo o modelo dinamico - nunca
    congelar na ultima posicao conhecida."""
    positions = [tracker.state[0]]
    for _ in range(5):
        tracker.predict(1.0)
        positions.append(tracker.state[0])

    assert positions == sorted(positions)  # x cresce monotonicamente (v=10>0)
    assert positions[-1] == pytest.approx(50.0, abs=1e-6)
    # covariancia deve ter crescido em relacao ao inicio, refletindo perda de confianca
    assert np.trace(tracker.covariance) > 4.0 * 6


def test_predict_with_non_positive_dt_is_a_noop(tracker: UkfTracker):
    state_before = tracker.state.copy()
    tracker.predict(0.0)
    assert np.array_equal(tracker.state, state_before)


def test_set_state_overwrites_the_kinematic_state_and_covariance(tracker: UkfTracker):
    tracker.set_state(np.array([50.0, 60.0, 100.0, 1.0, 2.0, 3.0]), np.eye(6) * 0.5)
    assert np.allclose(tracker.state, [50.0, 60.0, 100.0, 1.0, 2.0, 3.0])
    assert np.allclose(tracker.covariance, np.eye(6) * 0.5)


def test_set_state_preserves_the_extra_internal_dimension_of_coordinated_turn():
    # a taxa de giro estimada (omega) nao faz parte do contrato externo de
    # 6 dimensoes - set_state() nao deve zera-la nem mexer nela.
    tracker = UkfTracker(
        initial_state=np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0]),
        initial_covariance=np.eye(6) * 4.0,
        motion_model=CoordinatedTurnModel(turn_rate_process_noise_std=0.05, initial_turn_rate_std=0.3),
        process_noise_acceleration_std=2.0,
    )
    # forca uma estimativa de omega diferente de zero via alguns updates ao
    # longo de uma curva real, exatamente como no teste de coasting em curva.
    from filtering.motion_models import coordinated_turn_step

    true_state = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0, 0.3])
    for _ in range(5):
        true_state = coordinated_turn_step(true_state, 0.5)
        tracker.predict(0.5)
        tracker.update(true_state[:6], np.eye(6) * 0.1)

    omega_before = tracker._ukf.x[6]
    assert abs(omega_before) > 1e-3  # confirma que a estimativa realmente se moveu do prior (0.0)

    tracker.set_state(np.array([1.0, 1.0, 100.0, 5.0, 5.0, 0.0]), np.eye(6) * 2.0)

    assert tracker._ukf.x[6] == omega_before  # omega intocado
    assert np.allclose(tracker.state, [1.0, 1.0, 100.0, 5.0, 5.0, 0.0])


def test_create_motion_model_constant_velocity():
    model = create_motion_model(MotionModelName.CONSTANT_VELOCITY)
    assert isinstance(model, ConstantVelocityModel)


def test_create_motion_model_coordinated_turn():
    model = create_motion_model(MotionModelName.COORDINATED_TURN)
    assert isinstance(model, CoordinatedTurnModel)
    assert model.state_dim == 7


def test_create_motion_model_raises_for_unimplemented_extensions():
    # constant_acceleration continua reservado, ainda nao implementado -
    # so coordinated_turn deixou de ser NotImplementedError.
    with pytest.raises(NotImplementedError):
        create_motion_model(MotionModelName.CONSTANT_ACCELERATION)
