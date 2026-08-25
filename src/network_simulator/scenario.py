"""Configuracao de cenario para o gerador de mensagens: UAVs (trajetoria
Constant Velocity, ou por trechos de taxa de giro via `turn_schedule` - ver
trajectories.py::true_state_at_turning), estacoes (posicao + frequencia +
ruido proprios), obstaculos (oclusao geometrica de linha de visada - ver
visibility.py), dropout por janela de tempo e troca de ID local por
estacao. Carregavel de YAML (mesmo padrao de `config/loader.py`) para nao
exigir editar codigo Python a cada cenario novo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import yaml

from network_simulator.trajectories import TurnSegment


@dataclass(frozen=True)
class DropoutWindow:
    """Uma estacao especifica para de observar um UAV especifico durante
    [start_s, end_s] (tempo de CENARIO, dominio de measurement_timestamp -
    nunca relogio local/epoch)."""

    station_id: str
    start_s: float
    end_s: float

    def covers(self, station_id: str, t: float) -> bool:
        return station_id == self.station_id and self.start_s <= t <= self.end_s


@dataclass(frozen=True)
class IdSwitchEvent:
    """Uma estacao especifica "esquece" o ID local que vinha usando para um
    UAV a partir de `at_s` (tempo de cenario) - simula, por ex., a estacao
    reiniciando seu proprio rastreador local. O tracker central NUNCA sabe
    disso de antemao; precisa reassociar pelo estado/covariancia."""

    station_id: str
    at_s: float


@dataclass
class UavSpec:
    name: str  # identidade de VERDADE - nunca enviada ao tracker
    initial_state: list  # [x, y, z, vx, vy, vz] em t=0
    dropout_windows: list = field(default_factory=list)  # list[DropoutWindow]
    id_switch_events: list = field(default_factory=list)  # list[IdSwitchEvent]
    # vazio (default) = Constant Velocity puro, `trajectories.py::true_state_at`.
    # nao-vazio = trajetoria por trechos de taxa de giro,
    # `trajectories.py::true_state_at_turning` - ver TurnSegment.
    turn_schedule: list = field(default_factory=list)  # list[TurnSegment]


@dataclass(frozen=True)
class StationSpec:
    station_id: str
    frequency_hz: float
    position_std: float
    velocity_std: float
    # posicao fisica da estacao no plano XY - so metadado geometrico (nao
    # entra no modelo de ruido, que continua isotropico independente de
    # distancia): usada para checar oclusao de linha de visada contra
    # `ScenarioConfig.obstacles` e para visualizacao/relatorio. Default
    # (0, 0) para cenarios que nao se importam com geometria.
    position: tuple = (0.0, 0.0)


@dataclass(frozen=True)
class Obstacle:
    """Obstaculo circular no plano XY que bloqueia linha de visada entre
    uma estacao e um UAV - ver visibility.py::line_of_sight_blocked.
    Diferente de `DropoutWindow`: aqui a estacao perde visibilidade por
    GEOMETRIA (onde o alvo esta em relacao a estacao/obstaculo a cada
    instante), nao por um intervalo de tempo fixo cravado no cenario."""

    center: tuple
    radius: float


@dataclass
class ScenarioConfig:
    duration_s: float
    seed: int
    uavs: list  # list[UavSpec]
    stations: list  # list[StationSpec]
    obstacles: list = field(default_factory=list)  # list[Obstacle]
    scenario_id: Optional[str] = None


def _parse_uav(raw: dict) -> UavSpec:
    dropout_windows = [
        DropoutWindow(station_id=w["station_id"], start_s=float(w["start_s"]), end_s=float(w["end_s"]))
        for w in raw.get("dropout_windows", [])
    ]
    id_switch_events = [
        IdSwitchEvent(station_id=e["station_id"], at_s=float(e["at_s"])) for e in raw.get("id_switch_events", [])
    ]
    turn_schedule = [
        TurnSegment(duration_s=float(s["duration_s"]), omega=float(s["omega"]))
        for s in raw.get("turn_schedule", [])
    ]
    return UavSpec(
        name=raw["name"],
        initial_state=[float(v) for v in raw["initial_state"]],
        dropout_windows=dropout_windows,
        id_switch_events=id_switch_events,
        turn_schedule=turn_schedule,
    )


def _parse_station(raw: dict) -> StationSpec:
    position = raw.get("position", [0.0, 0.0])
    return StationSpec(
        station_id=str(raw["id"]),
        frequency_hz=float(raw["frequency_hz"]),
        position_std=float(raw["position_std"]),
        velocity_std=float(raw["velocity_std"]),
        position=(float(position[0]), float(position[1])),
    )


def _parse_obstacle(raw: dict) -> Obstacle:
    center = raw["center"]
    return Obstacle(center=(float(center[0]), float(center[1])), radius=float(raw["radius"]))


def load_scenario(path: Union[str, Path]) -> ScenarioConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    sim = raw.get("simulation", {})
    return ScenarioConfig(
        duration_s=float(sim.get("duration", 30.0)),
        seed=int(sim.get("seed", 0)),
        uavs=[_parse_uav(u) for u in raw["uavs"]],
        stations=[_parse_station(s) for s in raw["stations"]],
        obstacles=[_parse_obstacle(o) for o in raw.get("obstacles", [])],
        scenario_id=sim.get("scenario_id"),
    )


def station_rngs(scenario: ScenarioConfig) -> dict:
    """Um `np.random.Generator` independente por estacao, derivado
    deterministicamente de `scenario.seed` via `SeedSequence.spawn` -
    mesma seed => mesmos streams de ruido por estacao, em qualquer ordem de
    execucao - garante reprodutibilidade do cenario inteiro."""
    seed_sequence = np.random.SeedSequence(scenario.seed)
    child_seeds = seed_sequence.spawn(len(scenario.stations))
    return {
        station.station_id: np.random.default_rng(child_seed)
        for station, child_seed in zip(scenario.stations, child_seeds)
    }
