"""Simula uma estacao terrestre: observa UAVs do ground truth do
simulador, atribui IDs locais proprios (nunca o nome real do UAV), e
adiciona ruido. `force_new_local_id` simula a estacao "esquecendo" o alvo
e recriando o ID local (ex.: apos reiniciar o rastreador local dela).

CONTRATO (do qual `tracking/duplicate_merger.py` depende - ver
DuplicateTrackMerger, criterio de "mesma fonte historica"): dentro de uma
mesma sessao de execucao desta estacao, um `local_track_id` NUNCA e
reutilizado para representar outro alvo fisico - nem apos
`force_new_local_id`, nem entre UAVs diferentes. Por isso o sufixo
numerico vem de um contador MONOTONICO proprio (`_next_local_id_number`),
nunca de `len(self._local_ids)`: um dict que encolhe ao esquecer um alvo
faria o proximo ID reusar um numero ja usado (ver
tests/unit/test_virtual_station.py). Se no futuro uma estacao
puder reiniciar sua PROPRIA sessao (reiniciar o processo, nao so
"esquecer" um alvo), a identidade da fonte precisaria virar
`(station_id, station_session_id, local_track_id)` - nao implementado
agora porque o contrato atual (uma sessao = um VirtualStation, nunca
reiniciado) ja garante nao-reuso sem precisar disso."""

from __future__ import annotations

from typing import Optional

import numpy as np

from models.local_tracklet import LocalTracklet
from simulation.noise import add_gaussian_noise, diagonal_covariance


class VirtualStation:
    def __init__(
        self,
        station_id: str,
        position_std: float = 2.0,
        velocity_std: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.station_id = station_id
        self._position_std = position_std
        self._velocity_std = velocity_std
        self._rng = rng if rng is not None else np.random.default_rng()
        self._local_ids: dict[str, str] = {}
        # contador PROPRIO, nunca derivado do tamanho de `_local_ids` -
        # ver docstring do modulo pro contrato de nao-reutilizacao.
        self._next_local_id_number = 1

    def _local_id_for(self, uav_name: str) -> str:
        if uav_name not in self._local_ids:
            self._local_ids[uav_name] = f"{self.station_id[-1].upper()}{self._next_local_id_number:03d}"
            self._next_local_id_number += 1
        return self._local_ids[uav_name]

    def force_new_local_id(self, uav_name: str) -> None:
        self._local_ids.pop(uav_name, None)

    def observe(
        self, uav_name: str, true_state: np.ndarray, timestamp: float, scenario_id: Optional[str] = None
    ) -> LocalTracklet:
        noisy_state = add_gaussian_noise(true_state, self._position_std, self._velocity_std, self._rng)
        covariance = diagonal_covariance(self._position_std, self._velocity_std)
        return LocalTracklet(
            station_id=self.station_id,
            local_track_id=self._local_id_for(uav_name),
            timestamp=timestamp,
            state=noisy_state,
            covariance=covariance,
            scenario_id=scenario_id,
            ground_truth=true_state,
        )
