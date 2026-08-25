"""Ponto de entrada CLI: sobe o tracker central escutando UDP/JSON.

Uso:
    python src/main.py --port 9999
    python src/main.py --port 9999 --config src/config/default.yaml
    python src/main.py --port 9999 --track-log-dir data/track_history
"""

from __future__ import annotations

import argparse
import logging
import time

from config.loader import load_config
from io_.receiver import UdpReceiver
from io_.track_history_logger import TrackHistoryLogger
from tracking.tracker import Tracker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--config", default=None, help="caminho para um YAML de configuracao")
    parser.add_argument("--process-interval", type=float, default=0.5, help="segundos entre ciclos do tracker")
    parser.add_argument(
        "--track-log-dir",
        default=None,
        help="se definido, grava o estado completo (posicao, velocidade, covariancia, status) "
        "de cada GlobalTrack a cada ciclo, em CSV + JSONL, nesse diretorio",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("main")

    config = load_config(args.config)
    tracker = Tracker(config)
    track_logger = TrackHistoryLogger(args.track_log_dir) if args.track_log_dir else None
    receiver = UdpReceiver(
        tracker,
        host=args.host,
        port=args.port,
        process_interval_s=args.process_interval,
        track_logger=track_logger,
    )

    receiver.start()
    logger.info(
        "tracker central escutando em %s:%s (fusao=%s, filtro=%s, track_log_dir=%s)",
        args.host,
        args.port,
        config.fusion.strategy.value,
        config.filter.motion_model.value,
        args.track_log_dir or "desabilitado",
    )

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("encerrando...")
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
