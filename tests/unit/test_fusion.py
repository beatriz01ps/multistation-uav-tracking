import numpy as np
import pytest

from fusion.base import create_fusion_strategy
from fusion.covariance_intersection import CovarianceIntersectionFusion
from fusion.information_fusion import InformationFusion
from models.enums import FusionStrategyName
from models.local_tracklet import LocalTracklet


def _tracklet(station_id, state, covariance, timestamp=0.0):
    return LocalTracklet(station_id=station_id, local_track_id="1", timestamp=timestamp, state=state, covariance=covariance)


def test_information_fusion_single_tracklet_is_identity():
    tracklet = _tracklet("a", [1.0] * 6, np.eye(6) * 3.0)
    fused = InformationFusion().fuse([tracklet])
    assert np.allclose(fused.state, tracklet.state)
    assert np.allclose(fused.covariance, tracklet.covariance)


def test_information_fusion_reduces_uncertainty_with_more_sources():
    t1 = _tracklet("a", [0.0] * 6, np.eye(6) * 4.0)
    t2 = _tracklet("b", [1.0] * 6, np.eye(6) * 4.0)
    fused = InformationFusion().fuse([t1, t2])
    assert np.trace(fused.covariance) < np.trace(t1.covariance)
    assert len(fused.contributing_tracklets) == 2


def test_information_fusion_weights_toward_more_confident_source():
    confident = _tracklet("a", [0.0] * 6, np.eye(6) * 0.1)
    unsure = _tracklet("b", [100.0] * 6, np.eye(6) * 1000.0)
    fused = InformationFusion().fuse([confident, unsure])
    assert fused.state[0] == pytest.approx(0.0, abs=1.0)


def test_covariance_intersection_never_more_confident_than_best_input():
    t1 = _tracklet("a", [10.0] * 6, np.eye(6) * 2.0)
    t2 = _tracklet("b", [10.5] * 6, np.eye(6) * 3.0)
    fused = CovarianceIntersectionFusion().fuse([t1, t2])
    assert np.trace(fused.covariance) <= np.trace(t1.covariance) + 1e-2


def test_create_fusion_strategy_factory():
    assert isinstance(create_fusion_strategy(FusionStrategyName.INFORMATION), InformationFusion)
    assert isinstance(
        create_fusion_strategy(FusionStrategyName.COVARIANCE_INTERSECTION), CovarianceIntersectionFusion
    )
