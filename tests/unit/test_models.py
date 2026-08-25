import numpy as np
import pytest
from pydantic import ValidationError

from models.enums import TrackStatus
from models.fused_measurement import FusedMeasurement
from models.global_track import GlobalTrack
from models.local_tracklet import LocalTracklet


def test_local_tracklet_accepts_valid_state_and_covariance():
    tracklet = LocalTracklet(
        station_id="station_a",
        local_track_id="001",
        timestamp=0.0,
        state=[0, 0, 100, 10, 0, 0],
        covariance=np.eye(6).tolist(),
    )
    assert np.array_equal(tracklet.position, [0, 0, 100])
    assert np.array_equal(tracklet.velocity, [10, 0, 0])


@pytest.mark.parametrize("bad_state", [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
def test_local_tracklet_rejects_wrong_state_shape(bad_state):
    with pytest.raises(ValidationError):
        LocalTracklet(station_id="a", local_track_id="1", timestamp=0.0, state=bad_state, covariance=np.eye(6))


def test_local_tracklet_rejects_non_square_covariance():
    with pytest.raises(ValidationError):
        LocalTracklet(
            station_id="a", local_track_id="1", timestamp=0.0, state=[0] * 6, covariance=np.eye(5)
        )


def test_local_tracklet_rejects_asymmetric_covariance():
    cov = np.eye(6)
    cov[0, 1] = 999.0
    with pytest.raises(ValidationError):
        LocalTracklet(station_id="a", local_track_id="1", timestamp=0.0, state=[0] * 6, covariance=cov)


def test_local_tracklet_rejects_negative_diagonal_covariance():
    cov = np.eye(6)
    cov[2, 2] = -1.0
    with pytest.raises(ValidationError):
        LocalTracklet(station_id="a", local_track_id="1", timestamp=0.0, state=[0] * 6, covariance=cov)


def test_local_tracklet_rejects_covariance_with_negative_eigenvalue():
    """Simetrica + diagonal nao-negativa NAO garante positiva semidefinida -
    esta matriz passa nos dois testes mas tem autovalor -1 (invalido)."""
    cov = np.eye(6)
    cov[0, 1] = cov[1, 0] = 2.0
    with pytest.raises(ValidationError):
        LocalTracklet(station_id="a", local_track_id="1", timestamp=0.0, state=[0] * 6, covariance=cov)


def test_local_tracklet_rejects_non_finite_state():
    with pytest.raises(ValidationError):
        LocalTracklet(
            station_id="a", local_track_id="1", timestamp=0.0, state=[0, 0, 0, 0, 0, float("nan")], covariance=np.eye(6)
        )


def test_local_tracklet_ground_truth_is_optional_and_not_required_for_tracking():
    tracklet = LocalTracklet(station_id="a", local_track_id="1", timestamp=0.0, state=[0] * 6, covariance=np.eye(6))
    assert tracklet.ground_truth is None


def test_global_track_defaults_to_tentative():
    track = GlobalTrack(
        global_track_id=1,
        state=[0] * 6,
        covariance=np.eye(6),
        last_prediction_timestamp=0.0,
        last_measurement_timestamp=0.0,
        created_at=0.0,
    )
    assert track.status is TrackStatus.TENTATIVE
    assert track.time_since_measurement(5.0) == 5.0


def test_global_track_assignment_is_revalidated():
    track = GlobalTrack(
        global_track_id=1,
        state=[0] * 6,
        covariance=np.eye(6),
        last_prediction_timestamp=0.0,
        last_measurement_timestamp=0.0,
        created_at=0.0,
    )
    with pytest.raises(ValidationError):
        track.state = [0, 0, 0]  # shape errada


def test_fused_measurement_tracks_contributing_tracklets():
    fused = FusedMeasurement(
        state=[0] * 6, covariance=np.eye(6), timestamp=1.0, contributing_tracklets=[("a", "1"), ("b", "2")]
    )
    assert len(fused.contributing_tracklets) == 2
