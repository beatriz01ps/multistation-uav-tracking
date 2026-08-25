"""Testes minimos do gerador: campos obrigatorios, dimensoes, simetria da
covariancia, e que ground_truth NUNCA aparece na mensagem enviada ao
tracker."""

import json

import numpy as np

from models.local_tracklet import LocalTracklet
from network_simulator.message_builder import build_tracklet_message


def _tracklet():
    return LocalTracklet(
        station_id="A",
        local_track_id="A001",
        timestamp=1.25,
        state=[105.2, 49.7, 21.0, 10.1, 0.2, 0.0],
        covariance=np.eye(6) * 4.0,
        ground_truth=[100.0, 50.0, 20.0, 10.0, 0.0, 0.0],
    )


def test_message_has_all_required_fields():
    message = build_tracklet_message(_tracklet())
    for key in ("station_id", "local_track_id", "timestamp", "state", "covariance"):
        assert key in message


def test_state_has_dimension_6():
    message = build_tracklet_message(_tracklet())
    assert len(message["state"]) == 6


def test_covariance_has_dimension_6x6():
    message = build_tracklet_message(_tracklet())
    assert len(message["covariance"]) == 6
    assert all(len(row) == 6 for row in message["covariance"])


def test_covariance_is_symmetric():
    message = build_tracklet_message(_tracklet())
    cov = np.array(message["covariance"])
    assert np.allclose(cov, cov.T)


def test_ground_truth_never_appears_in_the_wire_message():
    message = build_tracklet_message(_tracklet())
    assert "ground_truth" not in message
    # nem escondido em algum lugar do JSON serializado - a mensagem inteira,
    # nao so as chaves de primeiro nivel.
    assert "ground_truth" not in json.dumps(message)
