import numpy as np

from io_.parser import parse_local_tracklet


def test_parse_local_tracklet_from_flat_dict():
    payload = {
        "station_id": "station_a",
        "local_track_id": "001",
        "timestamp": 1.5,
        "x": 1.0,
        "y": 2.0,
        "z": 3.0,
        "vx": 4.0,
        "vy": 5.0,
        "vz": 6.0,
        "covariance": np.eye(6).tolist(),
    }
    tracklet = parse_local_tracklet(payload)
    assert tracklet.station_id == "station_a"
    assert np.array_equal(tracklet.state, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_parse_local_tracklet_from_state_field():
    payload = {
        "station_id": "station_a",
        "local_track_id": "001",
        "timestamp": 0.0,
        "state": [0, 0, 100, 10, 0, 0],
        "covariance": np.eye(6).tolist(),
    }
    tracklet = parse_local_tracklet(payload)
    assert np.array_equal(tracklet.state, [0, 0, 100, 10, 0, 0])


def test_parse_local_tracklet_extracts_ground_truth_fields():
    payload = {
        "station_id": "station_a",
        "local_track_id": "001",
        "timestamp": 0.0,
        "state": [0] * 6,
        "covariance": np.eye(6).tolist(),
        "ground_truth_x": 1.0,
        "ground_truth_y": 2.0,
        "ground_truth_z": 3.0,
        "ground_truth_vx": 4.0,
        "ground_truth_vy": 5.0,
        "ground_truth_vz": 6.0,
    }
    tracklet = parse_local_tracklet(payload)
    assert np.array_equal(tracklet.ground_truth, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_parse_local_tracklet_without_ground_truth():
    payload = {
        "station_id": "station_a",
        "local_track_id": "001",
        "timestamp": 0.0,
        "state": [0] * 6,
        "covariance": np.eye(6).tolist(),
    }
    tracklet = parse_local_tracklet(payload)
    assert tracklet.ground_truth is None
