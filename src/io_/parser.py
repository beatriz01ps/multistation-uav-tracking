"""Converte uma mensagem crua (dict, tipicamente vinda de JSON) em um
LocalTracklet validado. Unico ponto de entrada de dados externos no
dominio de tracking - qualquer erro de formato falha aqui, nao depois."""

from __future__ import annotations

from typing import Any

from models.local_tracklet import LocalTracklet


def parse_local_tracklet(payload: dict[str, Any]) -> LocalTracklet:
    state = payload.get("state")
    if state is None:
        state = [payload[k] for k in ("x", "y", "z", "vx", "vy", "vz")]

    ground_truth = payload.get("ground_truth")
    if ground_truth is None and "ground_truth_x" in payload:
        ground_truth = [
            payload["ground_truth_x"],
            payload["ground_truth_y"],
            payload["ground_truth_z"],
            payload["ground_truth_vx"],
            payload["ground_truth_vy"],
            payload["ground_truth_vz"],
        ]

    return LocalTracklet(
        station_id=payload["station_id"],
        local_track_id=payload["local_track_id"],
        timestamp=float(payload["timestamp"]),
        state=state,
        covariance=payload["covariance"],
        scenario_id=payload.get("scenario_id"),
        sensor_mode=payload.get("sensor_mode"),
        pos_uncertainty=payload.get("pos_uncertainty"),
        vel_uncertainty=payload.get("vel_uncertainty"),
        ground_truth=ground_truth,
    )
