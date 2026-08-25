"""Historico continuo de GlobalTracks - a saida de trajetoria/posicionamento
que faltava (o mapeamento local->global que ja volta por UDP em cada tick,
em io_/receiver.py::_send_responses, e deliberadamente enxuto: so
{station_id, local_track_id, global_track_id, timestamp}, sem estado).

Este logger grava o ESTADO COMPLETO de cada GlobalTrack ativo, a cada ciclo
do tracker - reusa `serializer.py::serialize_global_track` (que ja existia,
pronta, mas sem nada que a chamasse em producao) para nao ter duas formas
de "o que e a saida completa de um GlobalTrack" no projeto.

Dois formatos, propositalmente separados (mesmo padrao ja usado em
network_simulator/scenario_logger.py):

  track_history.csv    - campos leves (posicao, velocidade, status,
                          hit/miss, IDs locais conhecidos) - facil de abrir
                          em Excel/pandas, para inspecao humana rapida.
  track_history.jsonl   - registro completo, incluindo a matriz de
                          covariancia 6x6 (nao caberia de forma legivel
                          numa planilha) - para quem precisar da incerteza
                          completa depois (ex.: reprocessar metricas).
"""

from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from typing import Any, Union

from io_.serializer import serialize_global_track
from models.global_track import GlobalTrack

_CSV_FIELDS = [
    "global_track_id",
    "timestamp",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "status",
    "prediction_only",
    "hit_count",
    "miss_count",
    "source_tracks",
    "known_local_tracks",
]


class TrackHistoryLogger:
    def __init__(self, out_dir: Union[str, Path]) -> None:
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        self._csv_file = (self._out_dir / "track_history.csv").open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=_CSV_FIELDS)
        self._csv_writer.writeheader()

        self._jsonl_file = (self._out_dir / "track_history.jsonl").open("w", encoding="utf-8")

    def log_snapshot(self, tracks: list[GlobalTrack]) -> None:
        """Grava uma linha por track ATIVO passado. O timestamp usado e o
        do PROPRIO track (`last_prediction_timestamp` - sempre no dominio
        de measurement_timestamp, nunca um relogio de parede compartilhado
        entre tracks - ver synchronization/tracking_clock.py), nao um
        timestamp de ciclo externo: tracks diferentes podem estar "como de"
        instantes ligeiramente diferentes dentro do mesmo tick."""
        with self._lock:
            for track in tracks:
                record = serialize_global_track(track, timestamp=track.last_prediction_timestamp)
                self._jsonl_file.write(json.dumps(record) + "\n")
                self._csv_writer.writerow(self._to_csv_row(record))
            self._jsonl_file.flush()
            self._csv_file.flush()

    @staticmethod
    def _to_csv_row(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "global_track_id": record["global_track_id"],
            "timestamp": record["timestamp"],
            "x": record["state"]["x"],
            "y": record["state"]["y"],
            "z": record["state"]["z"],
            "vx": record["state"]["vx"],
            "vy": record["state"]["vy"],
            "vz": record["state"]["vz"],
            "status": record["status"],
            "prediction_only": record["prediction_only"],
            "hit_count": record["hit_count"],
            "miss_count": record["miss_count"],
            "source_tracks": _join_local_ids(record["source_tracks"]),
            "known_local_tracks": _join_local_ids(record["known_local_tracks"]),
        }

    def close(self) -> None:
        with self._lock:
            self._csv_file.close()
            self._jsonl_file.close()


def _join_local_ids(entries: list[dict[str, str]]) -> str:
    return "|".join(f"{entry['station_id']}:{entry['local_track_id']}" for entry in entries)
