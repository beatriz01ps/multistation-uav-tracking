"""Confirma que o UdpReceiver, quando configurado com um track_logger, grava
o historico completo em disco durante a operacao real (rede + tracker),
sem depender de ninguem chamar io_/serializer.py manualmente."""

from __future__ import annotations

import csv
import time

import numpy as np

from config.models import AppConfig
from io_.receiver import UdpReceiver
from io_.track_history_logger import TrackHistoryLogger
from network_simulator.message_builder import build_tracklet_message
from network_simulator.udp_sender import UdpSender
from simulation.virtual_station import VirtualStation
from tracking.tracker import Tracker


def test_receiver_with_track_logger_writes_history_to_disk(tmp_path):
    tracker = Tracker(AppConfig())
    track_logger = TrackHistoryLogger(tmp_path)
    receiver = UdpReceiver(tracker, host="127.0.0.1", port=0, process_interval_s=0.05, track_logger=track_logger)
    receiver.start()
    sender = None
    try:
        real_port = receiver._socket.getsockname()[1]
        sender = UdpSender("127.0.0.1", real_port)

        station = VirtualStation("A", position_std=0.5, velocity_std=0.1, rng=np.random.default_rng(0))
        true_state = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0])
        tracklet = station.observe("uav_1", true_state, timestamp=0.0)
        sender.send(build_tracklet_message(tracklet))

        csv_path = tmp_path / "track_history.csv"
        deadline = time.monotonic() + 3.0
        rows: list[dict] = []
        while time.monotonic() < deadline:
            if csv_path.exists():
                with open(csv_path, newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    break
            time.sleep(0.05)

        assert rows, "esperava pelo menos uma linha gravada no historico"
        assert rows[0]["global_track_id"] == "1"
        assert rows[0]["status"] == "TENTATIVE"
    finally:
        receiver.stop()
        if sender is not None:
            sender.close()
