import numpy as np

from io_.serializer import serialize_global_track
from models.enums import UpdateKind
from models.global_track import GlobalTrack


def _track(**overrides):
    defaults = dict(
        global_track_id=4,
        state=[105.2, 46.8, 31.1, 4.2, -0.3, 0.1],
        covariance=np.eye(6),
        last_prediction_timestamp=123.45,
        last_measurement_timestamp=123.45,
        created_at=0.0,
        associated_local_tracks={"A": "001", "B": "007"},
        current_contributors=[("A", "001"), ("B", "007")],
    )
    defaults.update(overrides)
    return GlobalTrack(**defaults)


def test_serialize_includes_state_and_source_tracks():
    track = _track()
    payload = serialize_global_track(track, timestamp=123.45)

    assert payload["global_track_id"] == 4
    assert payload["state"] == {"x": 105.2, "y": 46.8, "z": 31.1, "vx": 4.2, "vy": -0.3, "vz": 0.1}
    assert payload["source_tracks"] == [
        {"station_id": "A", "local_track_id": "001"},
        {"station_id": "B", "local_track_id": "007"},
    ]


def test_serialize_source_tracks_differs_from_known_local_tracks_when_stale():
    """B esta em associated_local_tracks (ja contribuiu no passado) mas
    NAO em current_contributors (nao contribuiu neste ciclo) - source_tracks
    nao pode sugerir que B participou desta atualizacao."""
    track = _track(
        associated_local_tracks={"A": "001", "B": "007"},
        current_contributors=[("A", "001")],
    )
    payload = serialize_global_track(track, timestamp=123.45)

    assert payload["source_tracks"] == [{"station_id": "A", "local_track_id": "001"}]
    assert payload["known_local_tracks"] == [
        {"station_id": "A", "local_track_id": "001"},
        {"station_id": "B", "local_track_id": "007"},
    ]


def test_serialize_marks_prediction_only():
    track = _track(last_update_kind=UpdateKind.PREDICTION_ONLY)
    payload = serialize_global_track(track, timestamp=1.0)
    assert payload["prediction_only"] is True


def test_serialize_always_includes_covariance_even_during_coasting():
    from models.enums import TrackStatus

    track = _track(status=TrackStatus.COASTING, last_update_kind=UpdateKind.PREDICTION_ONLY)
    payload = serialize_global_track(track, timestamp=1.0)
    assert payload["status"] == "COASTING"
    assert len(payload["covariance"]) == 6
