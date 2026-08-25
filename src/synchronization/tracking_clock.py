"""Separa o dominio de tempo da MEDICAO do relogio LOCAL de espera/rede.
Misturar os dois e um risco real e concreto: `time.time()` (~1.7e9, epoch
Unix) comparado diretamente com `last_prediction_timestamp` no dominio de
`measurement_timestamp` (que pode ser 0.0, 0.1, 0.2... num
cenario/simulacao) produziria um `dt` da ordem de bilhoes de segundos.

Papeis, nunca misturados:

  measurement_timestamp  -> dominio fisico/logico do tracking: dt do
                             filtro, temporal alignment, data association,
                             lifecycle, timestamps do GlobalTrack.

  time.monotonic()        -> dominio local de espera/processamento:
                             fechamento de clusters (watermark), janela de
                             sincronizacao, timeout de buffer. NUNCA usado
                             diretamente pelo filtro nem comparado
                             numericamente com measurement_timestamp.

  time.time()              -> NAO participa da matematica do tracker.

Continuidade durante COASTING sem nenhuma medicao nova: mantemos uma
ANCORA entre os dois dominios, atualizada toda vez que um lote real e
processado. Enquanto nada de novo chega, o "agora" do tracking e
extrapolado a partir dessa ancora usando o tempo monotonic local
decorrido (escalado por `time_scale`, para permitir replay acelerado no
futuro sem mudar a logica):

    tracking_now = measurement_anchor + (monotonic() - monotonic_anchor) * time_scale
"""

from __future__ import annotations

import time as time_module
from typing import Optional


class TrackingClock:
    def __init__(self, time_scale: float = 1.0, monotonic=time_module.monotonic) -> None:
        self._time_scale = time_scale
        self._monotonic = monotonic
        self._measurement_anchor: Optional[float] = None
        self._monotonic_anchor: Optional[float] = None

    def anchor(self, measurement_timestamp: float) -> None:
        """Chamado toda vez que um lote REAL (com medicao) e processado -
        reancora os dois dominios de tempo a partir dele."""
        self._measurement_anchor = measurement_timestamp
        self._monotonic_anchor = self._monotonic()

    @property
    def is_anchored(self) -> bool:
        return self._measurement_anchor is not None

    def now(self) -> float:
        """"Agora" no dominio de measurement_timestamp, extrapolado a
        partir da ultima ancora real. So chamar se `is_anchored`."""
        if self._measurement_anchor is None or self._monotonic_anchor is None:
            raise RuntimeError("TrackingClock.now() chamado antes de qualquer anchor() (nenhuma medicao real ainda)")
        elapsed_monotonic = self._monotonic() - self._monotonic_anchor
        return self._measurement_anchor + elapsed_monotonic * self._time_scale
