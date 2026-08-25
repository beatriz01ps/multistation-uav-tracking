import numpy as np

from association.hungarian import build_cost_matrix, solve_assignment
from models.enums import AssociationMode


def test_build_cost_matrix_and_solve_matches_closest_pairs():
    tracks = [
        (np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0]), np.eye(6) * 4.0),
        (np.array([500.0, 500.0, 100.0, 0.0, 10.0, 0.0]), np.eye(6) * 4.0),
    ]
    tracklets = [
        (np.array([9.0, 0.5, 100.0, 10.0, 0.0, 0.0]), np.eye(6) * 2.0),
        (np.array([501.0, 505.0, 100.0, 0.0, 10.0, 0.0]), np.eye(6) * 2.0),
    ]

    cost, valid = build_cost_matrix(tracks, tracklets, AssociationMode.FULL_STATE, 0.99)
    matches, unmatched_tracks, unmatched_tracklets = solve_assignment(cost, valid)

    assert set(matches) == {(0, 0), (1, 1)}
    assert unmatched_tracks == []
    assert unmatched_tracklets == []


def test_solve_assignment_leaves_far_tracklet_unmatched():
    tracks = [(np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0]), np.eye(6))]
    tracklets = [
        (np.array([0.5, 0.0, 100.0, 10.0, 0.0, 0.0]), np.eye(6)),
        (np.array([9999.0, 0.0, 100.0, 0.0, 0.0, 0.0]), np.eye(6)),
    ]

    cost, valid = build_cost_matrix(tracks, tracklets, AssociationMode.FULL_STATE, 0.99)
    matches, _, unmatched_tracklets = solve_assignment(cost, valid)

    assert matches == [(0, 0)]
    assert unmatched_tracklets == [1]


def test_no_tracks_leaves_everything_unmatched():
    cost, valid = build_cost_matrix([], [(np.zeros(6), np.eye(6))], AssociationMode.FULL_STATE, 0.99)
    matches, _, unmatched_tracklets = solve_assignment(cost, valid)
    assert matches == []
    assert unmatched_tracklets == [0]
