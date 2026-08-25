"""Serializa um LocalTracklet gerado pelo simulador no MESMO contrato JSON
que `io_/parser.py::parse_local_tracklet` ja aceita (verificado lendo o
parser real antes de escrever este modulo - nao um segundo protocolo
inventado).

NUNCA inclui `ground_truth`: esse campo existe em `LocalTracklet` so para
avaliacao offline do simulador cientifico (`simulation/`); a interface
operacional real (estacao -> tracker) nunca deveria carregar a identidade
verdadeira do alvo. Omitir aqui, na fronteira de saida do simulador, e o
que garante isso - nao alterar o parser para "aceitar mas ignorar"."""

from __future__ import annotations

from typing import Any

from models.local_tracklet import LocalTracklet


def build_tracklet_message(tracklet: LocalTracklet) -> dict[str, Any]:
    return {
        "station_id": tracklet.station_id,
        "local_track_id": tracklet.local_track_id,
        "timestamp": tracklet.timestamp,
        "state": tracklet.state.tolist(),
        "covariance": tracklet.covariance.tolist(),
    }
