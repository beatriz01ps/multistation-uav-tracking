"""Testes do TrackHistoryLogger: CSV com campos leves (posicao, velocidade,
status), JSONL com o registro completo (incluindo a covariancia 6x6, que
nao caberia de forma legivel numa planilha)."""

import csv
import json

import numpy as np

from io_.track_history_logger import TrackHistoryLogger
from models.enums import TrackStatus, UpdateKind
from models.global_track import GlobalTrack


def _track(global_track_id=1, x=10.0, status=TrackStatus.CONFIRMED, prediction_only=False):
    return GlobalTrack(
        global_track_id=global_track_id,
        state=[x, 20.0, 100.0, 5.0, 0.0, 0.0],
        covariance=np.eye(6) * 3.0,
        last_prediction_timestamp=1.5,
        last_measurement_timestamp=1.5,
        created_at=0.0,
        status=status,
        last_update_kind=UpdateKind.PREDICTION_ONLY if prediction_only else UpdateKind.MEASUREMENT_UPDATED,
        hit_count=4,
        miss_count=1,
        current_contributors=[("station_a", "A001")],
        associated_local_tracks={"station_a": "A001", "station_b": "B004"},
    )


def test_log_snapshot_writes_one_csv_row_and_one_jsonl_line_per_track(tmp_path):
    logger = TrackHistoryLogger(tmp_path)
    logger.log_snapshot([_track(global_track_id=1), _track(global_track_id=2, x=99.0)])
    logger.close()

    with open(tmp_path / "track_history.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["global_track_id"] == "1"
    assert rows[1]["global_track_id"] == "2"
    assert float(rows[1]["x"]) == 99.0

    lines = (tmp_path / "track_history.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["global_track_id"] == 1


def test_csv_row_has_flat_kinematic_fields_but_no_covariance_matrix(tmp_path):
    logger = TrackHistoryLogger(tmp_path)
    logger.log_snapshot([_track()])
    logger.close()

    with open(tmp_path / "track_history.csv", newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    assert float(row["x"]) == 10.0
    assert float(row["y"]) == 20.0
    assert row["status"] == "CONFIRMED"
    assert row["hit_count"] == "4"
    assert row["miss_count"] == "1"
    assert "covariance" not in row


def test_jsonl_record_carries_the_full_covariance_matrix(tmp_path):
    logger = TrackHistoryLogger(tmp_path)
    logger.log_snapshot([_track()])
    logger.close()

    record = json.loads((tmp_path / "track_history.jsonl").read_text(encoding="utf-8").strip())
    covariance = np.array(record["covariance"])
    assert covariance.shape == (6, 6)
    assert np.allclose(covariance, np.eye(6) * 3.0)


def test_source_tracks_and_known_local_tracks_are_joined_readably_in_csv(tmp_path):
    logger = TrackHistoryLogger(tmp_path)
    logger.log_snapshot([_track()])
    logger.close()

    with open(tmp_path / "track_history.csv", newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    assert row["source_tracks"] == "station_a:A001"
    assert row["known_local_tracks"] in ("station_a:A001|station_b:B004", "station_b:B004|station_a:A001")


def test_prediction_only_flag_is_visible_in_both_formats(tmp_path):
    logger = TrackHistoryLogger(tmp_path)
    logger.log_snapshot([_track(status=TrackStatus.COASTING, prediction_only=True)])
    logger.close()

    with open(tmp_path / "track_history.csv", newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["prediction_only"] == "True"
    assert row["status"] == "COASTING"

    record = json.loads((tmp_path / "track_history.jsonl").read_text(encoding="utf-8").strip())
    assert record["prediction_only"] is True
