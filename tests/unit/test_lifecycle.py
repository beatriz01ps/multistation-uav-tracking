import numpy as np

from config.models import TrackingConfig
from models.enums import TrackStatus, UpdateKind
from models.global_track import GlobalTrack
from tracking.lifecycle import on_measurement, on_no_measurement


def _make_track(status=TrackStatus.TENTATIVE, hit_count=0, miss_count=0, last_measurement=0.0):
    return GlobalTrack(
        global_track_id=1,
        state=np.zeros(6),
        covariance=np.eye(6),
        last_prediction_timestamp=last_measurement,
        last_measurement_timestamp=last_measurement,
        created_at=last_measurement,
        status=status,
        hit_count=hit_count,
        miss_count=miss_count,
    )


def test_tentative_confirms_after_configured_hits():
    config = TrackingConfig(tentative_confirmation_hits=3)
    track = _make_track(status=TrackStatus.TENTATIVE, hit_count=2)
    on_measurement(track, timestamp=1.0, config=config)
    assert track.status is TrackStatus.CONFIRMED
    assert track.hit_count == 3


def test_tentative_not_confirmed_before_enough_hits():
    config = TrackingConfig(tentative_confirmation_hits=3)
    track = _make_track(status=TrackStatus.TENTATIVE, hit_count=0)
    on_measurement(track, timestamp=1.0, config=config)
    assert track.status is TrackStatus.TENTATIVE


def test_measurement_marks_update_kind():
    config = TrackingConfig()
    track = _make_track()
    on_measurement(track, timestamp=1.0, config=config)
    assert track.last_update_kind is UpdateKind.MEASUREMENT_UPDATED


def test_no_measurement_marks_update_kind():
    config = TrackingConfig()
    track = _make_track(status=TrackStatus.CONFIRMED)
    on_no_measurement(track, timestamp=1.0, config=config)
    assert track.last_update_kind is UpdateKind.PREDICTION_ONLY


def test_tentative_deleted_after_timeout():
    config = TrackingConfig(tentative_timeout_seconds=5.0)
    track = _make_track(status=TrackStatus.TENTATIVE, last_measurement=0.0)
    deleted = on_no_measurement(track, timestamp=5.0, config=config)
    assert deleted
    assert track.status is TrackStatus.DELETED


def test_tentative_not_deleted_before_timeout_even_with_multiple_misses():
    """Contar misses por CICLO do tracker (em vez de tempo) deletaria um
    track TENTATIVE antes mesmo dele ter chance de acumular as medicoes
    de confirmacao, se o tracker rodasse mais rapido que a estacao manda
    dado."""
    config = TrackingConfig(tentative_timeout_seconds=5.0)
    track = _make_track(status=TrackStatus.TENTATIVE, last_measurement=0.0)
    for t in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        deleted = on_no_measurement(track, timestamp=t, config=config)
        assert not deleted
        assert track.status is TrackStatus.TENTATIVE


def test_confirmed_goes_to_coasting_on_miss():
    config = TrackingConfig()
    track = _make_track(status=TrackStatus.CONFIRMED)
    deleted = on_no_measurement(track, timestamp=1.0, config=config)
    assert not deleted
    assert track.status is TrackStatus.COASTING


def test_coasting_becomes_lost_after_timeout():
    config = TrackingConfig(coasting_timeout_seconds=3.0, deletion_timeout_seconds=100.0)
    track = _make_track(status=TrackStatus.COASTING, last_measurement=0.0)
    on_no_measurement(track, timestamp=3.0, config=config)
    assert track.status is TrackStatus.LOST


def test_coasting_stays_coasting_before_timeout():
    config = TrackingConfig(coasting_timeout_seconds=3.0)
    track = _make_track(status=TrackStatus.COASTING, last_measurement=0.0)
    on_no_measurement(track, timestamp=1.0, config=config)
    assert track.status is TrackStatus.COASTING


def test_lost_deleted_after_lost_timeout():
    config = TrackingConfig(coasting_timeout_seconds=3.0, lost_timeout_seconds=6.0, deletion_timeout_seconds=100.0)
    track = _make_track(status=TrackStatus.LOST, last_measurement=0.0)
    deleted = on_no_measurement(track, timestamp=9.0, config=config)
    assert deleted
    assert track.status is TrackStatus.DELETED


def test_deletion_timeout_is_an_absolute_safety_cap():
    # mesmo com coasting/lost timeouts folgados, deletion_timeout_seconds forca DELETED
    config = TrackingConfig(coasting_timeout_seconds=100.0, lost_timeout_seconds=100.0, deletion_timeout_seconds=5.0)
    track = _make_track(status=TrackStatus.COASTING, last_measurement=0.0)
    deleted = on_no_measurement(track, timestamp=5.0, config=config)
    assert deleted
    assert track.status is TrackStatus.DELETED


def test_reacquisition_from_coasting_goes_straight_to_confirmed():
    config = TrackingConfig()
    track = _make_track(status=TrackStatus.COASTING)
    on_measurement(track, timestamp=5.0, config=config)
    assert track.status is TrackStatus.CONFIRMED


def test_reacquisition_from_lost_goes_straight_to_confirmed_and_resets_misses():
    config = TrackingConfig()
    track = _make_track(status=TrackStatus.LOST, miss_count=10)
    on_measurement(track, timestamp=5.0, config=config)
    assert track.status is TrackStatus.CONFIRMED
    assert track.miss_count == 0
