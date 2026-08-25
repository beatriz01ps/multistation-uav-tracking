import numpy as np

from association.gating import chi_square_threshold, dims_for_mode, gate
from models.enums import AssociationMode


def test_dims_for_mode():
    assert dims_for_mode(AssociationMode.POSITION_ONLY) == (0, 1, 2)
    assert dims_for_mode(AssociationMode.FULL_STATE) == (0, 1, 2, 3, 4, 5)


def test_chi_square_threshold_increases_with_probability():
    low = chi_square_threshold(0.90, degrees_of_freedom=6)
    high = chi_square_threshold(0.99, degrees_of_freedom=6)
    assert high > low


def test_gate_accepts_close_measurement_full_state():
    predicted_state = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0])
    predicted_cov = np.eye(6) * 4.0
    measurement = predicted_state + 0.5
    measurement_cov = np.eye(6) * 2.0

    passed, d2 = gate(predicted_state, predicted_cov, measurement, measurement_cov, AssociationMode.FULL_STATE, 0.99)
    assert passed
    assert d2 >= 0.0


def test_gate_rejects_far_measurement():
    predicted_state = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0])
    predicted_cov = np.eye(6) * 1.0
    measurement = predicted_state + np.array([500.0, 500.0, 0.0, 0.0, 0.0, 0.0])
    measurement_cov = np.eye(6) * 1.0

    passed, _ = gate(predicted_state, predicted_cov, measurement, measurement_cov, AssociationMode.FULL_STATE, 0.99)
    assert not passed


def test_gate_position_only_ignores_velocity_mismatch():
    predicted_state = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0])
    predicted_cov = np.eye(6) * 4.0
    measurement = np.array([0.5, 0.5, 100.0, 999.0, 999.0, 999.0])
    measurement_cov = np.eye(6) * 2.0

    passed_position_only, _ = gate(
        predicted_state, predicted_cov, measurement, measurement_cov, AssociationMode.POSITION_ONLY, 0.99
    )
    passed_full_state, _ = gate(
        predicted_state, predicted_cov, measurement, measurement_cov, AssociationMode.FULL_STATE, 0.99
    )

    assert passed_position_only
    assert not passed_full_state
