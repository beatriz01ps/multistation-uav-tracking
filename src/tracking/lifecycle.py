"""Transicoes de estado de um GlobalTrack.

TENTATIVE -> CONFIRMED -> COASTING -> LOST -> DELETED, com reacquisition
(COASTING ou LOST -> CONFIRMED direto) quando uma medicao volta a ser
associada. Nenhum limiar fica hardcoded - tudo vem de TrackingConfig.

Semantica adotada para os tres timeouts de COASTING/LOST/DELETED (decisao
de interpretacao documentada aqui explicitamente, nao escolhida
silenciosamente - qual e incremental vs. absoluto):
  - coasting_timeout_seconds: tempo em COASTING antes de LOST. Como
    COASTING comeca no mesmo instante da ultima medicao, isso equivale a
    "tempo desde a ultima medicao" enquanto o track esta em COASTING.
  - lost_timeout_seconds: tempo ADICIONAL em LOST (apos coasting_timeout_seconds
    ja ter passado) antes de DELETED.
  - deletion_timeout_seconds: teto de seguranca ABSOLUTO, medido desde a
    ultima medicao real, independente do estado atual - forca DELETED
    mesmo que os dois incrementais acima ainda nao tenham somado esse
    valor (protege contra configuracoes inconsistentes entre os tres).

TENTATIVE usa um timeout proprio, baseado em TEMPO desde a ultima medicao
(`tentative_timeout_seconds`) - nao em contagem de misses por ciclo do
tracker: contar "misses" por chamada de process_batch e fragil a
diferenca entre a cadencia interna do tracker e a taxa de envio da
estacao (um track TENTATIVE poderia morrer antes de acumular
`tentative_confirmation_hits` so porque o tracker rodou mais vezes do
que a estacao mandou dado nesse intervalo).
"""

from __future__ import annotations

from config.models import TrackingConfig
from models.enums import TrackStatus, UpdateKind
from models.global_track import GlobalTrack


def on_measurement(track: GlobalTrack, timestamp: float, config: TrackingConfig) -> None:
    track.hit_count += 1
    track.miss_count = 0
    track.last_measurement_timestamp = timestamp
    track.last_update_kind = UpdateKind.MEASUREMENT_UPDATED

    if track.status is TrackStatus.TENTATIVE:
        if track.hit_count >= config.tentative_confirmation_hits:
            track.status = TrackStatus.CONFIRMED
    elif track.status in (TrackStatus.COASTING, TrackStatus.LOST):
        track.status = TrackStatus.CONFIRMED  # reacquisition


def on_no_measurement(track: GlobalTrack, timestamp: float, config: TrackingConfig) -> bool:
    """Chamado apos o predict() de um ciclo sem medicao associada a este
    track. Retorna True se o track deve ser removido (DELETED)."""
    track.miss_count += 1
    track.last_update_kind = UpdateKind.PREDICTION_ONLY

    time_since_measurement = track.time_since_measurement(timestamp)

    if time_since_measurement >= config.deletion_timeout_seconds:
        track.status = TrackStatus.DELETED
        return True

    if track.status is TrackStatus.TENTATIVE:
        if time_since_measurement >= config.tentative_timeout_seconds:
            track.status = TrackStatus.DELETED
            return True
        return False

    if track.status is TrackStatus.CONFIRMED:
        track.status = TrackStatus.COASTING
        return False

    if track.status is TrackStatus.COASTING:
        if time_since_measurement >= config.coasting_timeout_seconds:
            track.status = TrackStatus.LOST
        return False

    if track.status is TrackStatus.LOST:
        if time_since_measurement >= config.coasting_timeout_seconds + config.lost_timeout_seconds:
            track.status = TrackStatus.DELETED
            return True
        return False

    return False  # ja DELETED
