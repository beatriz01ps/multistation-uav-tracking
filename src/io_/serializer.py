"""Serializa a saida do sistema - sempre inclui estado e covariancia,
mesmo durante COASTING/LOST (nunca omitir so porque esta em
prediction_only: quem consome a saida precisa poder distinguir os dois
casos, nao so receber menos dado quando a confianca cai).

A saida NUNCA apresenta uma predicao como se fosse medicao:
`prediction_only` deixa isso explicito para quem consome a saida.

`source_tracks` reflete `current_contributors` (quem participou DESTA
atualizacao especifica) - nao `associated_local_tracks` (o historico de
identidade, que pode incluir uma estacao que nao manda nada ha varios
ciclos). Consumir `source_tracks` como "quem contribuiu agora" e
`known_local_tracks` como "identidade conhecida ate agora" evita confundir
os dois conceitos.
"""

from __future__ import annotations

from typing import Any

from models.enums import UpdateKind
from models.global_track import GlobalTrack


def serialize_global_track(track: GlobalTrack, timestamp: float) -> dict[str, Any]:
    x, y, z, vx, vy, vz = track.state.tolist()
    return {
        "global_track_id": track.global_track_id,
        "timestamp": timestamp,
        "state": {"x": x, "y": y, "z": z, "vx": vx, "vy": vy, "vz": vz},
        "covariance": track.covariance.tolist(),
        "status": track.status.value.upper(),
        "prediction_only": track.last_update_kind is UpdateKind.PREDICTION_ONLY,
        "hit_count": track.hit_count,
        "miss_count": track.miss_count,
        "source_tracks": [
            {"station_id": station_id, "local_track_id": local_id}
            for station_id, local_id in track.current_contributors
        ],
        "known_local_tracks": [
            {"station_id": station_id, "local_track_id": local_id}
            for station_id, local_id in track.associated_local_tracks.items()
        ],
    }


def serialize_all_tracks(tracks: list[GlobalTrack], timestamp: float) -> list[dict[str, Any]]:
    return [serialize_global_track(track, timestamp) for track in tracks]
