"""CLI: envia um cenario sintetico pela interface UDP/JSON REAL do tracker -
representa as estacoes externas, uma por thread,
cada uma na sua propria frequencia. NAO e o simulador cientifico definitivo
dos experimentos (isso continua sendo `simulation/`); serve para validar o
contrato de interface e testar o pipeline completo end-to-end sem depender
das estacoes reais.

Uso:
    python src/main.py --port 9999                              # terminal 1: tracker
    python src/network_simulator/main.py --port 9999             # terminal 2: simulador
    python src/network_simulator/main.py --scenario src/network_simulator/scenarios/two_uavs_three_stations.yaml
    python src/network_simulator/main.py --no-realtime --duration 5 --out-dir data/synthetic/smoke_test
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path

# `network_simulator/main.py` esta um nivel mais fundo que `src/main.py` -
# rodar `python src/network_simulator/main.py` diretamente so poe
# `src/network_simulator/` no sys.path (a pasta do proprio script), nao
# `src/`, entao `import network_simulator...`/`import models...` etc.
# falhariam. Este bootstrap garante que funciona do mesmo jeito que
# `python src/main.py` funciona, sem exigir `python -m` ou PYTHONPATH manual.
_SRC_ROOT = str(Path(__file__).resolve().parents[1])
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from network_simulator.response_listener import listen_for_responses
from network_simulator.scenario import load_scenario, station_rngs
from network_simulator.scenario_logger import ScenarioLogger
from network_simulator.station_runner import StationRunner
from network_simulator.truth_runner import TruthRunner
from network_simulator.udp_sender import UdpSender

DEFAULT_SCENARIO = Path(__file__).parent / "scenarios" / "one_uav_three_stations.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", default=str(DEFAULT_SCENARIO), help="YAML do cenario (ver scenario.py)")
    parser.add_argument("--host", default="127.0.0.1", help="endereco do tracker (receiver UDP)")
    parser.add_argument("--port", type=int, default=9999, help="porta do tracker (mesmo default de src/main.py)")
    parser.add_argument("--duration", type=float, default=None, help="sobrescreve simulation.duration do cenario (s)")
    parser.add_argument("--seed", type=int, default=None, help="sobrescreve simulation.seed do cenario")
    parser.add_argument("--realtime", dest="realtime", action="store_true", default=True, help="respeita o tempo real do cenario (default)")
    parser.add_argument("--no-realtime", dest="realtime", action="store_false", help="envia o mais rapido possivel, sem esperar")
    parser.add_argument("--out-dir", default="data/synthetic/network_simulator_run", help="onde salvar sent_messages/ground_truth/responses .jsonl")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("network_simulator")

    scenario = load_scenario(args.scenario)
    if args.duration is not None:
        scenario.duration_s = args.duration
    if args.seed is not None:
        scenario.seed = args.seed

    sender = UdpSender(args.host, args.port)
    scenario_logger = ScenarioLogger(args.out_dir)
    stop_event = threading.Event()

    rngs = station_rngs(scenario)
    runners = [
        StationRunner(station_spec, scenario, sender, scenario_logger, rngs[station_spec.station_id], realtime=args.realtime)
        for station_spec in scenario.stations
    ]

    listener_thread = threading.Thread(
        target=listen_for_responses, args=(sender, scenario_logger, stop_event), name="response-listener", daemon=True
    )
    # thread PROPRIA de ground truth, independente de qualquer estacao -
    # ver network_simulator/truth_runner.py.
    truth_runner = TruthRunner(scenario, scenario_logger, realtime=args.realtime)
    truth_thread = threading.Thread(target=truth_runner.run, args=(stop_event,), name="truth-runner")
    station_threads = [
        threading.Thread(target=runner.run, args=(stop_event,), name=f"station-{spec.station_id}")
        for runner, spec in zip(runners, scenario.stations)
    ]

    logger.info(
        "cenario=%s duration=%.1fs seed=%d stations=%s -> %s:%s (realtime=%s, out_dir=%s)",
        args.scenario,
        scenario.duration_s,
        scenario.seed,
        [s.station_id for s in scenario.stations],
        args.host,
        args.port,
        args.realtime,
        args.out_dir,
    )

    listener_thread.start()
    truth_thread.start()
    for thread in station_threads:
        thread.start()
    try:
        for thread in station_threads:
            thread.join()
        truth_thread.join()
    finally:
        stop_event.set()
        listener_thread.join(timeout=2.0)
        sender.close()
        scenario_logger.close()

    logger.info("cenario finalizado. logs em %s", args.out_dir)


if __name__ == "__main__":
    main()
