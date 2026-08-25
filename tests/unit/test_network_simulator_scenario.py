"""Testes minimos do gerador: reprodutibilidade por seed, IDs locais
independentes por estacao, frequencia por estacao respeitada, e geometria
(posicao de estacao + oclusao por obstaculo)."""

import threading
from pathlib import Path

import numpy as np

from network_simulator.scenario import Obstacle, ScenarioConfig, StationSpec, UavSpec, load_scenario, station_rngs
from network_simulator.scenario_logger import ScenarioLogger
from network_simulator.station_runner import StationRunner

SCENARIOS_DIR = Path(__file__).resolve().parents[2] / "src" / "network_simulator" / "scenarios"


def test_same_seed_produces_the_same_rng_streams_per_station():
    scenario = load_scenario(SCENARIOS_DIR / "one_uav_three_stations.yaml")

    rngs_a = station_rngs(scenario)
    rngs_b = station_rngs(scenario)

    for station_id in rngs_a:
        drawn_a = rngs_a[station_id].normal(size=5)
        drawn_b = rngs_b[station_id].normal(size=5)
        assert np.allclose(drawn_a, drawn_b)


def test_different_seeds_produce_different_streams():
    scenario_1 = load_scenario(SCENARIOS_DIR / "one_uav_three_stations.yaml")
    scenario_2 = load_scenario(SCENARIOS_DIR / "one_uav_three_stations.yaml")
    scenario_2.seed = scenario_1.seed + 1

    rngs_1 = station_rngs(scenario_1)
    rngs_2 = station_rngs(scenario_2)

    station_id = scenario_1.stations[0].station_id
    drawn_1 = rngs_1[station_id].normal(size=5)
    drawn_2 = rngs_2[station_id].normal(size=5)
    assert not np.allclose(drawn_1, drawn_2)


class _RecordingSender:
    def __init__(self):
        self.sent = []

    def send(self, message):
        self.sent.append(message)


def _run_station(frequency_hz: float, duration_s: float, tmp_path) -> list:
    scenario = ScenarioConfig(
        duration_s=duration_s,
        seed=1,
        uavs=[UavSpec(name="uav_1", initial_state=[0.0, 0.0, 100.0, 10.0, 0.0, 0.0])],
        stations=[StationSpec(station_id="A", frequency_hz=frequency_hz, position_std=1.0, velocity_std=0.1)],
    )
    sender = _RecordingSender()
    scenario_logger = ScenarioLogger(tmp_path)
    runner = StationRunner(
        scenario.stations[0], scenario, sender, scenario_logger, np.random.default_rng(0), realtime=False
    )
    runner.run(threading.Event())
    scenario_logger.close()
    return sender.sent


def test_station_local_ids_are_independent_per_station(tmp_path):
    scenario = ScenarioConfig(
        duration_s=0.2,
        seed=7,
        uavs=[UavSpec(name="uav_1", initial_state=[0.0, 0.0, 100.0, 10.0, 0.0, 0.0])],
        stations=[
            StationSpec(station_id="A", frequency_hz=10, position_std=1.0, velocity_std=0.1),
            StationSpec(station_id="B", frequency_hz=10, position_std=1.0, velocity_std=0.1),
        ],
    )
    scenario_logger = ScenarioLogger(tmp_path)
    rngs = station_rngs(scenario)

    local_ids_by_station = {}

    for station_spec in scenario.stations:
        sender = _RecordingSender()
        runner = StationRunner(station_spec, scenario, sender, scenario_logger, rngs[station_spec.station_id], realtime=False)
        runner.run(threading.Event())
        local_ids_by_station[station_spec.station_id] = {m["local_track_id"] for m in sender.sent}
    scenario_logger.close()

    # cada estacao atribuiu seu proprio ID local ao mesmo UAV - nada exige
    # (nem deveria) que sejam iguais entre estacoes.
    assert local_ids_by_station["A"] != set()
    assert local_ids_by_station["B"] != set()
    # e nenhuma mensagem carrega qualquer coisa que amarre A e B ao mesmo alvo.
    for messages in local_ids_by_station.values():
        assert all(isinstance(local_id, str) for local_id in messages)


def test_station_frequency_is_respected(tmp_path):
    # 10 Hz por 1.0s (nao-realtime, ticks em t=0.0,0.1,...,1.0) -> 11 envios.
    sent = _run_station(frequency_hz=10.0, duration_s=1.0, tmp_path=tmp_path)
    assert len(sent) == 11

    # 2 Hz por 1.0s (t=0.0,0.5,1.0) -> 3 envios.
    sent_slow = _run_station(frequency_hz=2.0, duration_s=1.0, tmp_path=tmp_path / "slow")
    assert len(sent_slow) == 3


def test_load_scenario_parses_station_position_and_obstacles():
    scenario = load_scenario(SCENARIOS_DIR / "visual_occlusion.yaml")

    stations_by_id = {s.station_id: s for s in scenario.stations}
    assert stations_by_id["close"].position == (80.0, 90.0)
    assert stations_by_id["far"].position == (95.0, 10.0)

    assert len(scenario.obstacles) == 1
    assert scenario.obstacles[0].center == (55.0, 35.0)
    assert scenario.obstacles[0].radius == 10.0


def test_load_scenario_parses_turn_schedule():
    scenario = load_scenario(SCENARIOS_DIR / "multi_uav_nonlinear.yaml")

    uavs_by_name = {u.name: u for u in scenario.uavs}
    assert uavs_by_name["uav_straight_1"].turn_schedule == []

    schedule = uavs_by_name["uav_curve"].turn_schedule
    assert len(schedule) == 2
    assert schedule[0].duration_s == 3.0
    assert schedule[0].omega == 0.3
    assert schedule[1].duration_s == 3.0
    assert schedule[1].omega == -0.3


def test_station_stops_sending_only_while_its_line_of_sight_is_blocked(tmp_path):
    # trajetoria diagonal (10t, 10t); station "far" tem a visada bloqueada
    # pelo obstaculo entre t=3.4 e t=5.0 (verificado numericamente ao
    # desenhar o cenario); station "close" nunca e bloqueada.
    scenario = ScenarioConfig(
        duration_s=10.0,
        seed=1,
        uavs=[UavSpec(name="uav_1", initial_state=[0.0, 0.0, 100.0, 10.0, 10.0, 0.0])],
        stations=[
            StationSpec(station_id="close", frequency_hz=5, position_std=1.0, velocity_std=0.1, position=(80.0, 90.0)),
            StationSpec(station_id="far", frequency_hz=5, position_std=1.0, velocity_std=0.1, position=(95.0, 10.0)),
        ],
        obstacles=[Obstacle(center=(55.0, 35.0), radius=10.0)],
    )
    scenario_logger = ScenarioLogger(tmp_path)
    rngs = station_rngs(scenario)

    sent_by_station = {}
    for station_spec in scenario.stations:
        sender = _RecordingSender()
        runner = StationRunner(station_spec, scenario, sender, scenario_logger, rngs[station_spec.station_id], realtime=False)
        runner.run(threading.Event())
        sent_by_station[station_spec.station_id] = sender.sent
    scenario_logger.close()

    close_timestamps = {round(m["timestamp"], 2) for m in sent_by_station["close"]}
    far_timestamps = {round(m["timestamp"], 2) for m in sent_by_station["far"]}

    # "close" nunca perde a visada: manda em todos os 51 ticks (0.0..10.0, 5Hz)
    assert len(close_timestamps) == 51

    # "far" pula exatamente os ticks dentro da janela bloqueada
    blocked_ticks = {3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0}
    assert len(far_timestamps) == 51 - len(blocked_ticks)
    assert far_timestamps.isdisjoint(blocked_ticks)
    assert 3.2 in far_timestamps  # imediatamente antes de bloquear, ainda manda
    assert 5.2 in far_timestamps  # imediatamente depois, ja volta a mandar


def test_load_scenario_parses_loop_crossing_scenarios():
    loop = load_scenario(SCENARIOS_DIR / "loop_crossing.yaml")
    uavs_by_name = {u.name: u for u in loop.uavs}
    assert uavs_by_name["uav_straight"].turn_schedule == []
    assert len(uavs_by_name["uav_loop"].turn_schedule) == 2
    assert [s.station_id for s in loop.stations] == ["N", "SW", "E"]

    loop_obstacle = load_scenario(SCENARIOS_DIR / "loop_crossing_obstacle.yaml")
    assert {u.name for u in loop_obstacle.uavs} == {"uav_straight_1", "uav_straight_2", "uav_loop"}
    assert len(loop_obstacle.obstacles) == 1
