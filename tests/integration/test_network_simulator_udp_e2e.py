"""Teste de integracao UDP real, ponta a ponta: UdpSender real -> socket
real -> UdpReceiver real -> Tracker real, e a resposta local->global real
de volta. Usa `port=0` (o SO escolhe uma
porta livre) para nao colidir com nada em uso e nao ser flaky em CI; usa
polling com timeout em vez de `sleep` fixo pelo mesmo motivo."""

from __future__ import annotations

import time

import numpy as np

from config.models import AppConfig
from io_.receiver import UdpReceiver
from network_simulator.message_builder import build_tracklet_message
from network_simulator.udp_sender import UdpSender
from simulation.virtual_station import VirtualStation
from tracking.tracker import Tracker


def test_message_sent_by_simulator_flows_through_udp_into_a_global_track():
    tracker = Tracker(AppConfig())
    receiver = UdpReceiver(tracker, host="127.0.0.1", port=0, process_interval_s=0.05)
    receiver.start()
    sender = None
    try:
        real_port = receiver._socket.getsockname()[1]
        sender = UdpSender("127.0.0.1", real_port)

        station = VirtualStation("A", position_std=0.5, velocity_std=0.1, rng=np.random.default_rng(0))
        true_state = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0])
        tracklet = station.observe("uav_1", true_state, timestamp=0.0)
        sender.send(build_tracklet_message(tracklet))

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not tracker.track_manager.tracks:
            time.sleep(0.05)

        assert len(tracker.track_manager.tracks) == 1

        # a resposta local->global tambem deve chegar de volta pela mesma via.
        response = None
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            response = sender.try_receive()
            if response is not None:
                break

        assert response is not None
        assert response["station_id"] == "A"
        assert response["local_track_id"] == tracklet.local_track_id
        assert response["global_track_id"] in tracker.track_manager.tracks
    finally:
        receiver.stop()
        if sender is not None:
            sender.close()
