"""Alinha tracklets do mesmo lote a um timestamp de referencia comum,
ANTES de gating/association/fusion.

Mesmo depois do fechamento de lote por watermark (tracklet_buffer.py),
um lote pode legitimamente conter tracklets com timestamps de MEDICAO
diferentes entre si (ex.:
estacoes em frequencias diferentes - 10 Hz vs 5 Hz vs 2 Hz - observando o
mesmo alvo em instantes proximos mas nao identicos, dentro da janela de
sincronizacao). Comparar os estados brutos desses tracklets diretamente no
gating equivale a tratar leituras de instantes diferentes como se fossem
do mesmo instante - com covariancia pequena, isso pode fazer o Mahalanobis
rejeitar uma associacao valida (ou, em casos com covariancia grande
demais, aceitar uma invalida).

Usa o MESMO modelo Constant Velocity do filtro (filtering/motion_models.py)
para propagar [x,y,z,vx,vy,vz] do timestamp original ate a referencia:

    x_ref = x + vx*dt   (idem y, z)
    vx_ref = vx         (inalterado)

E propaga a covariancia com a formula padrao de propagacao de incerteza:

    P_ref = F @ P @ F.T [+ Q(dt)]

O termo de ruido de processo Q e OPCIONAL e ligado por default
(`include_process_noise=True`): ele representa a incerteza adicional de
assumir CV perfeito durante o `dt` do alinhamento (o alvo pode ter
acelerado/manobrado nesse intervalo curto) - sem ele, a propagacao seria
otimista demais (so herda a incerteza original, sem crescer). Usa o mesmo
`process_noise_acceleration_std` do filtro principal, por consistencia.
"""

from __future__ import annotations

from filtering.motion_models import constant_velocity_transition_matrix
from filtering.process_noise import constant_velocity_process_noise
from models.local_tracklet import LocalTracklet


def align_tracklet_to_timestamp(
    tracklet: LocalTracklet,
    reference_timestamp: float,
    process_noise_acceleration_std: float = 0.0,
    include_process_noise: bool = True,
) -> LocalTracklet:
    dt = reference_timestamp - tracklet.timestamp
    if dt == 0:
        return tracklet

    transition = constant_velocity_transition_matrix(dt)
    aligned_state = transition @ tracklet.state
    aligned_covariance = transition @ tracklet.covariance @ transition.T

    if include_process_noise and process_noise_acceleration_std > 0:
        aligned_covariance = aligned_covariance + constant_velocity_process_noise(
            abs(dt), process_noise_acceleration_std
        )

    return tracklet.model_copy(
        update={"state": aligned_state, "covariance": aligned_covariance, "timestamp": reference_timestamp}
    )


def align_batch_to_timestamp(
    tracklets: list[LocalTracklet],
    reference_timestamp: float,
    process_noise_acceleration_std: float = 0.0,
    include_process_noise: bool = True,
) -> list[LocalTracklet]:
    return [
        align_tracklet_to_timestamp(t, reference_timestamp, process_noise_acceleration_std, include_process_noise)
        for t in tracklets
    ]
