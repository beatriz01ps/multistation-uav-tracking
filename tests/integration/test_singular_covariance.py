"""Uma covariancia positiva semidefinida mas SINGULAR (autovalor
exatamente 0) e matematicamente valida e deve ser aceita na entrada - mas
nunca pode causar um crash de `LinAlgError: Singular matrix` mais tarde em
fusao/associacao/UKF."""

import numpy as np
import pytest

from association.mahalanobis import mahalanobis_squared
from config.models import AppConfig
from fusion.covariance_intersection import CovarianceIntersectionFusion
from fusion.information_fusion import InformationFusion
from models.local_tracklet import LocalTracklet
from tracking.tracker import Tracker


def _singular_covariance() -> np.ndarray:
    cov = np.eye(6)
    cov[0, 0] = 0.0  # variancia exatamente zero em x -> singular, mas PSD valida
    return cov


def test_local_tracklet_accepts_singular_psd_covariance():
    tracklet = LocalTracklet(
        station_id="a", local_track_id="1", timestamp=0.0, state=[0.0] * 6, covariance=_singular_covariance()
    )
    assert np.linalg.eigvalsh(tracklet.covariance).min() == pytest.approx(0.0, abs=1e-9)


def test_information_fusion_does_not_crash_on_singular_covariance():
    t1 = LocalTracklet(station_id="a", local_track_id="1", timestamp=0.0, state=[0.0] * 6, covariance=_singular_covariance())
    t2 = LocalTracklet(station_id="b", local_track_id="1", timestamp=0.0, state=[1.0] * 6, covariance=np.eye(6))
    fused = InformationFusion().fuse([t1, t2])
    assert np.all(np.isfinite(fused.state))
    assert np.all(np.isfinite(fused.covariance))


def test_covariance_intersection_does_not_crash_on_singular_covariance():
    t1 = LocalTracklet(station_id="a", local_track_id="1", timestamp=0.0, state=[0.0] * 6, covariance=_singular_covariance())
    t2 = LocalTracklet(station_id="b", local_track_id="1", timestamp=0.0, state=[1.0] * 6, covariance=np.eye(6))
    fused = CovarianceIntersectionFusion().fuse([t1, t2])
    assert np.all(np.isfinite(fused.state))
    assert np.all(np.isfinite(fused.covariance))


def test_mahalanobis_does_not_crash_on_singular_covariance():
    d2 = mahalanobis_squared(
        predicted_state=np.zeros(6),
        predicted_covariance=_singular_covariance(),
        measurement=np.ones(6),
        measurement_covariance=_singular_covariance(),
        dims=(0, 1, 2, 3, 4, 5),
    )
    assert np.isfinite(d2)


def test_tracker_pipeline_does_not_crash_on_singular_covariance():
    tracker = Tracker(AppConfig())
    tracklet = LocalTracklet(
        station_id="station_a", local_track_id="001", timestamp=0.0, state=[0.0, 0.0, 100.0, 10.0, 0.0, 0.0], covariance=_singular_covariance()
    )
    tracker.process_batch([tracklet], timestamp=0.0)
    assert len(tracker.track_manager.tracks) == 1
