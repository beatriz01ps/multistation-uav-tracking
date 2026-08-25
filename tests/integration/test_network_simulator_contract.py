"""Teste de contrato - o mais importante desta bateria: prova que o
simulador produz EXATAMENTE o formato que o parser
REAL do tracker consegue consumir, passando pela mesma serializacao JSON
que a rede faria (ida e volta), sem nenhum atalho. Evita divergencia entre
"contrato documentado" e "contrato implementado"."""

import json

import numpy as np

from io_.parser import parse_local_tracklet
from network_simulator.message_builder import build_tracklet_message
from network_simulator.trajectories import true_state_at
from simulation.virtual_station import VirtualStation


def test_simulator_message_round_trips_through_the_real_parser():
    station = VirtualStation("A", position_std=1.0, velocity_std=0.5, rng=np.random.default_rng(0))
    true_state = true_state_at([0.0, 0.0, 100.0, 10.0, 3.0, 0.0], t=1.25)
    tracklet = station.observe("uav_1", true_state, timestamp=1.25)

    message = build_tracklet_message(tracklet)
    wire_payload = json.loads(json.dumps(message))  # exatamente o que passaria pela rede

    parsed = parse_local_tracklet(wire_payload)

    assert parsed.station_id == tracklet.station_id
    assert parsed.local_track_id == tracklet.local_track_id
    assert parsed.timestamp == tracklet.timestamp
    assert np.allclose(parsed.state, tracklet.state)
    assert np.allclose(parsed.covariance, tracklet.covariance)
    assert parsed.ground_truth is None  # nunca trafega pela interface real


def test_simulator_messages_are_valid_json_for_every_station_tick():
    station = VirtualStation("B", position_std=2.0, velocity_std=0.3, rng=np.random.default_rng(1))
    for t in [0.0, 0.2, 0.4, 0.6]:
        true_state = true_state_at([0.0, 0.0, 100.0, 10.0, 0.0, 0.0], t)
        tracklet = station.observe("uav_1", true_state, timestamp=t)
        message = build_tracklet_message(tracklet)

        parsed = parse_local_tracklet(json.loads(json.dumps(message)))
        assert parsed.timestamp == t
