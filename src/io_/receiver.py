"""Receptor UDP/JSON: substitui o simulador pelos tracklets reais das
estacoes.

Threads de rede SEPARADAS da thread de processamento (mesma razao de
sempre: uma extracao/computacao lenta no processamento nunca pode atrasar
a leitura de novos pacotes). A thread de processamento chama
`tracker.tick()` numa cadencia FIXA e propria (`process_interval_s`),
independente de pacotes terem chegado ou nao - e assim que o requisito "o
tracker precisa continuar prevendo mesmo sem nenhum pacote chegar" fica
satisfeito tambem no caminho de rede ao vivo (em simulacao/replay isso e
feito chamando `tick(..., flush_all=True)` manualmente a cada passo, sem
essa thread).

`receive_threads` (default 2, nao 1): numa frequencia de estacao alta
(algumas centenas de Hz, bem acima dos 2-10Hz usados nos cenarios de
teste), UMA thread so chamando `recvfrom()` num loop perde uma fracao
grande dos pacotes por nao conseguir drenar o socket rapido o suficiente
(`recvfrom()` libera o GIL durante o syscall bloqueante, entao MULTIPLAS
threads conseguem paralelizar essa espera de verdade). Medido
isoladamente (receptor sem nenhum processamento) num cenario de stress:
a taxa de captura mais que dobra de 1 para 2 threads, mas piora de 2 para
4 (contencao entre threads competindo pelo mesmo socket, nao falta de
CPU) - por isso o default e 2, nao um numero maior. Aumentar `SO_RCVBUF`
nao ajudou de forma consistente nesse mesmo teste, sozinho ou combinado
com threads. As threads compartilham o MESMO socket e despacham para a
MESMA `TrackletBuffer`/`Tracker.ingest()`, que ja e thread-safe (lock
proprio) independente de quantas threads de recepcao chamam - nenhuma
mudanca de sincronizacao adicional foi necessaria aqui.

Depois de cada tick, a equivalencia local<->global e devolvida por UDP ao
remetente original de cada tracklet - o sistema so cumpre a
responsabilidade de "fornecer a equivalencia" se isso realmente sair pela
rede, nao so existir em memoria. Essa resposta e deliberadamente enxuta
(so os 4 campos de equivalencia, nunca o estado completo) - quem precisar
do estado completo (posicao, velocidade, covariancia, status) de cada GlobalTrack consulta o
`track_logger` opcional (ver io_/track_history_logger.py), nao a resposta
de rede - decisao explicita para nao inflar todo pacote UDP de resposta
so porque uma parte dos consumidores quer mais detalhe.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from typing import TYPE_CHECKING, Optional

from io_.parser import parse_local_tracklet
from tracking.tracker import Tracker

if TYPE_CHECKING:
    from io_.track_history_logger import TrackHistoryLogger

logger = logging.getLogger(__name__)

Address = tuple[str, int]


class UdpReceiver:
    def __init__(
        self,
        tracker: Tracker, # valores default (se n receber nd)
        host: str = "0.0.0.0",
        port: int = 9999,
        process_interval_s: float = 0.5,
        max_packet_size: int = 65536,
        track_logger: Optional["TrackHistoryLogger"] = None,
        receive_threads: int = 2,
    ) -> None:
        self._tracker = tracker
        self._host = host
        self._port = port
        self._process_interval_s = process_interval_s
        self._max_packet_size = max_packet_size
        # opcional: grava o ESTADO COMPLETO de cada GlobalTrack a cada tick
        # (CSV + JSONL) - ver io_/track_history_logger.py. Separado de
        # proposito da resposta UDP simples abaixo (_send_responses).
        self._track_logger = track_logger
        self._receive_thread_count = receive_threads

        self._socket: Optional[socket.socket] = None
        self._receive_threads: list[threading.Thread] = []
        self._process_thread: Optional[threading.Thread] = None
        self._running = threading.Event()

        self._address_lock = threading.Lock()
        self._last_address_by_source: dict[tuple[str, str], Address] = {}

    def start(self) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((self._host, self._port))
        self._socket.settimeout(0.5)
        self._running.set()

        # Todas as threads de recepcao leem do MESMO socket, concorrentemente
        # - o SO despacha cada datagrama para UMA delas (ver docstring do
        # modulo pro numero medido que justifica o default de 2). Compartilham
        # o mesmo `_receive_loop`, ja escrito sem estado por-thread.
        self._receive_threads = [
            threading.Thread(target=self._receive_loop, name=f"udp-receive-{i}", daemon=True)
            for i in range(self._receive_thread_count)
        ]
        self._process_thread = threading.Thread(target=self._process_loop, name="tracker-tick", daemon=True)
        for thread in self._receive_threads:
            thread.start()
        self._process_thread.start()

    def stop(self) -> None:
        self._running.clear()
        for thread in self._receive_threads:
            thread.join(timeout=2.0)
        if self._process_thread is not None:
            self._process_thread.join(timeout=2.0)
        if self._socket is not None:
            self._socket.close()
        if self._track_logger is not None:
            self._track_logger.close()

    def _receive_loop(self) -> None:
        assert self._socket is not None
        while self._running.is_set():
            try:
                raw, addr = self._socket.recvfrom(self._max_packet_size)
            except socket.timeout:
                continue
            except OSError:
                break  # socket fechado por stop()

            try:
                payload = json.loads(raw.decode("utf-8"))
                tracklet = parse_local_tracklet(payload)
            except Exception:
                logger.exception("pacote invalido descartado")
                continue

            with self._address_lock:
                self._last_address_by_source[(tracklet.station_id, tracklet.local_track_id)] = addr

            self._tracker.ingest(tracklet)

    def _process_loop(self) -> None:
        while self._running.is_set():
            try:
                # NUNCA passar time.time() aqui - Tracker.tick() deriva o
                # "agora" internamente a partir do TrackingClock, ancorado
                # no dominio de measurement_timestamp (ver
                # synchronization/tracking_clock.py). Misturar Unix epoch
                # com esse dominio quebraria o calculo de dt do filtro.
                events = self._tracker.tick()
                self._send_responses(events)
                if self._track_logger is not None:
                    self._track_logger.log_snapshot(self._tracker.track_manager.active_tracks())
            except Exception:
                # um erro num ciclo n pode derrubar a thread de processamento
                # pro resto da execucao.
                logger.exception("erro processando ciclo; tracker continua rodando")
            time.sleep(self._process_interval_s)

    def _send_responses(self, events) -> None:
        if not events or self._socket is None:
            return
        with self._address_lock:
            addresses = dict(self._last_address_by_source)

        for event in events:
            addr = addresses.get((event.station_id, event.local_track_id))
            if addr is None:
                continue
            response = {
                "station_id": event.station_id,
                "local_track_id": event.local_track_id,
                "global_track_id": event.global_track_id,
                "timestamp": event.timestamp,
            }
            try:
                self._socket.sendto(json.dumps(response).encode("utf-8"), addr)
            except OSError:
                logger.exception("falha ao enviar resposta para %s", addr)
