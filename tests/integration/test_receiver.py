import json
import socket
import time

import numpy as np

from config.models import AppConfig
from io_.receiver import UdpReceiver
from tracking.tracker import Tracker


def test_udp_receiver_ingests_ticks_and_returns_global_id():
    """Ponta a ponta: manda um pacote, espera o tick automatico processar
    (nao so quando ha pacote), e confirma que a equivalencia
    local<->global volta pela rede."""
    tracker = Tracker(AppConfig())
    receiver = UdpReceiver(tracker, host="127.0.0.1", port=9977, process_interval_s=0.1)
    receiver.start()
    try:
        time.sleep(0.2)
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.bind(("127.0.0.1", 0))
        client.settimeout(3.0)
        message = {
            "station_id": "station_a",
            "local_track_id": "001",
            "timestamp": time.time(),
            "state": [0.0, 0.0, 100.0, 10.0, 0.0, 0.0],
            "covariance": np.eye(6).tolist(),
        }
        client.sendto(json.dumps(message).encode("utf-8"), ("127.0.0.1", 9977))

        raw, _ = client.recvfrom(65536)
        response = json.loads(raw.decode("utf-8"))

        assert response["station_id"] == "station_a"
        assert response["local_track_id"] == "001"
        assert isinstance(response["global_track_id"], int)
        assert len(tracker.track_manager.tracks) == 1
    finally:
        receiver.stop()


def test_udp_receiver_ignores_malformed_packet_without_crashing():
    tracker = Tracker(AppConfig())
    receiver = UdpReceiver(tracker, host="127.0.0.1", port=9978, process_interval_s=0.1)
    receiver.start()
    try:
        time.sleep(0.2)
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.bind(("127.0.0.1", 0))
        client.settimeout(3.0)
        client.sendto(b"isso nao e json valido", ("127.0.0.1", 9978))
        time.sleep(0.3)

        message = {
            "station_id": "station_a",
            "local_track_id": "001",
            "timestamp": time.time(),
            "state": [0.0, 0.0, 100.0, 10.0, 0.0, 0.0],
            "covariance": np.eye(6).tolist(),
        }
        client.sendto(json.dumps(message).encode("utf-8"), ("127.0.0.1", 9978))

        raw, _ = client.recvfrom(65536)
        response = json.loads(raw.decode("utf-8"))
        assert response["local_track_id"] == "001"
        assert len(tracker.track_manager.tracks) == 1
    finally:
        receiver.stop()
