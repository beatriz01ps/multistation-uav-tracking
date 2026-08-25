"""Uma estacao virtual rodando no SEU proprio ritmo: cada `StationRunner`
e um loop independente, na sua propria
`frequency_hz`, sem sincronizacao artificial com as outras estacoes -
mensagens de estacoes diferentes chegam ao tracker fora de ordem/entrelacadas
de proposito; e o `TrackletBuffer` quem resolve agrupamento e ordem
temporal, nao o simulador.

Reusa `simulation.virtual_station.VirtualStation` (ruido + atribuicao de ID
local + `force_new_local_id`) - a unica coisa nova aqui e o AGENDAMENTO
(quando observar) e o ENVIO (UDP), nao a logica de observacao em si.

NAO registra ground truth (`ScenarioLogger.log_truth`) - isso e
responsabilidade exclusiva de `main.py::_truth_loop`, deliberadamente
independente de qualquer estacao (ver docstring de
network_simulator/trajectories.py::true_state_for_uav). Um StationRunner
so sabe da verdade o suficiente pra gerar SUA PROPRIA observacao ruidosa
e checar visibilidade - registrar isso como "o" ground truth acoplaria a
verdade a frequencia/visibilidade de UMA estacao especifica."""

from __future__ import annotations

import logging
import threading
import time as time_module

from network_simulator.message_builder import build_tracklet_message
from network_simulator.scenario import ScenarioConfig, StationSpec
from network_simulator.scenario_logger import ScenarioLogger
from network_simulator.trajectories import true_state_for_uav
from network_simulator.udp_sender import UdpSender
from network_simulator.visibility import line_of_sight_blocked
from simulation.virtual_station import VirtualStation

logger = logging.getLogger(__name__)


class StationRunner:
    def __init__(
        self,
        station_spec: StationSpec,
        scenario: ScenarioConfig,
        sender: UdpSender,
        scenario_logger: ScenarioLogger,
        rng,
        realtime: bool = True,
    ) -> None:
        self._spec = station_spec
        self._scenario = scenario
        self._sender = sender
        self._scenario_logger = scenario_logger
        self._realtime = realtime
        self._station = VirtualStation(
            station_id=station_spec.station_id,
            position_std=station_spec.position_std,
            velocity_std=station_spec.velocity_std,
            rng=rng,
        )
        self._already_switched: set = set()

    def _is_visible(self, uav, t: float, target_xy) -> bool:
        if any(window.covers(self._spec.station_id, t) for window in uav.dropout_windows):
            return False
        return not any(
            line_of_sight_blocked(self._spec.position, target_xy, obstacle.center, obstacle.radius)
            for obstacle in self._scenario.obstacles
        )

    def _apply_pending_id_switches(self, uav, t: float) -> None:
        for event in uav.id_switch_events:
            if event.station_id != self._spec.station_id:
                continue
            marker = (uav.name, event.at_s)
            if t >= event.at_s and marker not in self._already_switched:
                self._station.force_new_local_id(uav.name)
                self._already_switched.add(marker)

    def run(self, stop_event: threading.Event) -> None:
        period = 1.0 / self._spec.frequency_hz
        start_monotonic = time_module.monotonic()
        tick = 0

        while not stop_event.is_set():
            t = tick * period
            if t > self._scenario.duration_s:
                break

            for uav in self._scenario.uavs:
                self._apply_pending_id_switches(uav, t)

                true_state = true_state_for_uav(uav, t)
                if not self._is_visible(uav, t, target_xy=true_state[:2]):
                    continue

                tracklet = self._station.observe(uav.name, true_state, t, scenario_id=self._scenario.scenario_id)
                message = build_tracklet_message(tracklet)

                self._sender.send(message)
                self._scenario_logger.log_sent(message)
                logger.info(
                    "SENT station=%s local=%s t=%.3f", tracklet.station_id, tracklet.local_track_id, t
                )

            tick += 1
            if self._realtime:
                target = start_monotonic + tick * period
                remaining = target - time_module.monotonic()
                if remaining > 0:
                    stop_event.wait(remaining)

        logger.info("estacao %s finalizada", self._spec.station_id)
