"""Um GlobalTrack recem-criado a partir de uma medicao real deve sair como
MEASUREMENT_UPDATED, nunca herdar o default PREDICTION_ONLY."""

import numpy as np

from config.models import AppConfig
from models.enums import UpdateKind
from models.local_tracklet import LocalTracklet
from tracking.tracker import Tracker


def test_newly_created_track_is_measurement_updated_not_prediction_only():
    tracker = Tracker(AppConfig())
    tracklet = LocalTracklet(
        station_id="station_a", local_track_id="001", timestamp=0.0, state=[0.0, 0.0, 100.0, 10.0, 0.0, 0.0], covariance=np.eye(6)
    )
    tracker.process_batch([tracklet], timestamp=0.0)

    (track,) = tracker.track_manager.tracks.values()
    assert track.last_update_kind is UpdateKind.MEASUREMENT_UPDATED
