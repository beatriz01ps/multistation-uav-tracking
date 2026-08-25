"""Registro de ground truth - loop PROPRIO, independente de qualquer
estacao (ver network_simulator/station_runner.py e
trajectories.py::true_state_for_uav pro raciocinio completo).

Antes, `ScenarioLogger.log_truth` era chamado de dentro do loop de CADA
`StationRunner`, uma vez por UAV que aquela estacao especifica conseguia
observar - ou seja, o registro de "o que realmente aconteceu" ficava
acoplado a frequencia E a visibilidade de uma estacao particular (se
todas as estacoes estivessem ocluidas num instante, esse instante nao
teria verdade nenhuma registrada; se so uma estacao via o alvo, a
densidade de amostragem da verdade dependia de QUAL estacao). Isso nunca
corrompeu valores (a formula e deterministica, toda estacao calculava o
mesmo estado), mas era um acoplamento indevido - `TruthRunner` existe
para que ground truth represente o estado do alvo, ponto, sem depender
de sensor nenhum.

Frequencia do loop: a MAIOR `frequency_hz` entre as estacoes do cenario -
preserva a densidade de amostragem que `ground_truth.jsonl` ja tinha
antes (dominada pela estacao mais rapida), sem precisar de um novo
parametro de configuracao."""

from __future__ import annotations

import logging
import threading
import time as time_module

from network_simulator.scenario import ScenarioConfig
from network_simulator.scenario_logger import ScenarioLogger
from network_simulator.trajectories import true_state_for_uav

logger = logging.getLogger(__name__)


class TruthRunner:
    def __init__(self, scenario: ScenarioConfig, scenario_logger: ScenarioLogger, realtime: bool = True) -> None:
        self._scenario = scenario
        self._scenario_logger = scenario_logger
        self._realtime = realtime
        self._frequency_hz = max(station.frequency_hz for station in scenario.stations)

    def run(self, stop_event: threading.Event) -> None:
        period = 1.0 / self._frequency_hz
        start_monotonic = time_module.monotonic()
        tick = 0

        while not stop_event.is_set():
            t = tick * period
            if t > self._scenario.duration_s:
                break

            for uav in self._scenario.uavs:
                self._scenario_logger.log_truth(uav.name, t, true_state_for_uav(uav, t))

            tick += 1
            if self._realtime:
                target = start_monotonic + tick * period
                remaining = target - time_module.monotonic()
                if remaining > 0:
                    stop_event.wait(remaining)

        logger.info("truth runner finalizado")
