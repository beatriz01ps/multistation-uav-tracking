import numpy as np
import pytest

from filtering.process_noise import constant_velocity_process_noise


def test_matches_piecewise_constant_acceleration_formula():
    """Q_axis = sigma^2 * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]] (modelo
    "piecewise constant white acceleration" - ver docstring do modulo para
    a justificativa de por que esta formula, e nao "continuous white
    noise acceleration", e a consistente com um parametro chamado
    `acceleration_std` em m/s^2)."""
    dt, sigma = 1.5, 2.0
    q = constant_velocity_process_noise(dt, sigma)

    expected_pos_var = sigma**2 * dt**4 / 4
    expected_cross = sigma**2 * dt**3 / 2
    expected_vel_var = sigma**2 * dt**2

    assert q[0, 0] == pytest.approx(expected_pos_var)
    assert q[0, 3] == pytest.approx(expected_cross)
    assert q[3, 0] == pytest.approx(expected_cross)
    assert q[3, 3] == pytest.approx(expected_vel_var)


def test_axes_are_independent():
    q = constant_velocity_process_noise(dt=1.0, acceleration_std=2.0)
    assert q[0, 1] == 0.0  # x nao correlaciona com y
    assert q[0, 4] == 0.0  # x nao correlaciona com vy


def test_zero_dt_gives_zero_process_noise():
    q = constant_velocity_process_noise(dt=0.0, acceleration_std=2.0)
    assert np.allclose(q, 0.0)


def test_larger_dt_means_more_uncertainty():
    q_short = constant_velocity_process_noise(dt=0.5, acceleration_std=2.0)
    q_long = constant_velocity_process_noise(dt=2.0, acceleration_std=2.0)
    assert np.trace(q_long) > np.trace(q_short)


def test_larger_acceleration_std_means_more_uncertainty():
    q_confident = constant_velocity_process_noise(dt=1.0, acceleration_std=0.5)
    q_unsure = constant_velocity_process_noise(dt=1.0, acceleration_std=5.0)
    assert np.trace(q_unsure) > np.trace(q_confident)
