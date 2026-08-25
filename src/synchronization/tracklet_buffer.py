"""Buffer de sincronizacao temporal entre estacoes - nunca assume que
Station A/B/C mandam mensagens exatamente ao mesmo tempo, e nunca assume
que mensagens que CHEGARAM juntas representam o MESMO instante de medicao.

Modelo: "cluster" = grupo de tracklets que representam o mesmo instante
fisico (proximos em timestamp de MEDICAO). Cada cluster tem um watermark
proprio - `opened_at`, o horario de CHEGADA do seu primeiro membro - e so
fecha quando `synchronization_window_ms` tiver se passado DESDE ENTAO, nao
desde a chegada de cada item individualmente.

Por que isso importa: liberar cada tracklet com base na sua PROPRIA
idade de chegada faria com que duas estacoes observando o MESMO
instante, mas que chegam pela rede com poucas dezenas de ms de
diferenca, acabassem em lotes SEPARADOS - a mais "velha" seria liberada
sozinha antes da mais nova terminar sua propria espera. Isso perderia a
chance de fusao conjunta e poderia contar duas confirmacoes sequenciais
(TENTATIVE) para o que era, na verdade, uma unica observacao conjunta do
mesmo instante. Com o watermark por cluster, a segunda estacao
que chega DENTRO da janela do cluster já aberto entra nele, e as duas saem
juntas quando o cluster fecha.

Uma mesma fonte (station_id, local_track_id) nunca aparece duas vezes no
mesmo cluster - representaria uma sequencia temporal do mesmo alvo, nao
duas fontes simultaneas; forca a criacao de outro cluster.

Thread-safe: `add()` roda na thread de rede, `pop_ready_batches()`/
`flush_all_batches()` rodam na thread de processamento - protegido por
`threading.Lock`.

Mensagens atrasadas: um tracklet com timestamp anterior ao
ultimo lote ja liberado NUNCA é processado, em nenhuma das duas politicas
- aplicar uma medicao antiga sobre um estado que ja avancou para um
instante posterior corrompe o filtro (nao é OOSM de verdade). `DROP` e
`LOG_ONLY` diferem so no nivel de detalhe do log, nunca em aplicar ou nao
a medicao - "logar e aplicar mesmo assim" nao e uma politica segura aqui.
`DROP` e o default.
"""

from __future__ import annotations

import logging
import threading
import time as time_module

from config.models import AssociationConfig
from models.enums import LateMessagePolicy
from models.local_tracklet import LocalTracklet

logger = logging.getLogger(__name__)


class _Cluster:
    __slots__ = ("opened_at", "tracklets")

    def __init__(self, opened_at: float, first_tracklet: LocalTracklet) -> None:
        self.opened_at = opened_at
        self.tracklets: list[LocalTracklet] = [first_tracklet]

    @property
    def reference_timestamp(self) -> float:
        return self.tracklets[0].timestamp

    @property
    def processing_timestamp(self) -> float:
        """Timestamp que o Tracker efetivamente usa ao processar este lote
        (`batch_timestamp = max(tracklet.timestamp for tracklet in batch)`
        em `Tracker.tick()`): um cluster novo membro pode ter
        measurement_timestamp MAIOR que o do primeiro membro
        (`reference_timestamp`), entao dois clusters ordenados por
        `reference_timestamp` ainda assim poderiam ser entregues ao
        Tracker fora de ordem pelo timestamp que ele realmente usa, se
        essa propriedade usasse `reference_timestamp` em vez de
        recalcular sobre todos os membros. `reference_timestamp` continua
        servindo para decidir
        proximidade/compatibilidade na hora de INSERIR um tracklet
        (`accepts`, cluster mais proximo em `add`) - so a ordem de
        LIBERACAO dos lotes precisa deste valor, calculado sobre TODOS os
        membros atuais do cluster."""
        return max(t.timestamp for t in self.tracklets)

    def accepts(self, tracklet: LocalTracklet, window_s: float) -> bool:
        within_window = abs(tracklet.timestamp - self.reference_timestamp) <= window_s
        key = (tracklet.station_id, tracklet.local_track_id)
        key_already_present = any((t.station_id, t.local_track_id) == key for t in self.tracklets)
        return within_window and not key_already_present


class TrackletBuffer:
    def __init__(self, config: AssociationConfig, clock=time_module.monotonic) -> None:
        self._config = config
        self._clock = clock
        self._lock = threading.Lock()
        self._clusters: list[_Cluster] = []
        self._last_released_timestamp: float = float("-inf")

    def add(self, tracklet: LocalTracklet) -> bool:
        """Retorna True se aceito no buffer, False se descartado (atrasado
        demais - nunca processado, em nenhuma politica)."""
        window_s = self._config.synchronization_window_ms / 1000.0

        with self._lock:
            if tracklet.timestamp < self._last_released_timestamp:
                if self._config.late_message_policy is LateMessagePolicy.LOG_ONLY:
                    logger.warning(
                        "tracklet atrasado descartado (policy=log_only, diagnostico): "
                        "station=%s local=%s timestamp=%.3f < ultimo_liberado=%.3f",
                        tracklet.station_id,
                        tracklet.local_track_id,
                        tracklet.timestamp,
                        self._last_released_timestamp,
                    )
                else:
                    logger.warning(
                        "tracklet atrasado descartado (policy=drop): station=%s local=%s timestamp=%.3f < ultimo_liberado=%.3f",
                        tracklet.station_id,
                        tracklet.local_track_id,
                        tracklet.timestamp,
                        self._last_released_timestamp,
                    )
                return False

            now = self._clock()
            # Escolhe o cluster MAIS PROXIMO por timestamp de medicao, nao
            # o primeiro que aceita - com janela=100ms, cluster 1 em
            # t=0.00 e cluster 2 em t=0.10, um tracklet chegando em t=0.09
            # esta a 9ms do cluster 1 mas so 1ms do cluster 2. O cluster
            # fisicamente correto e sempre o mais proximo em timestamp de
            # MEDICAO, nao o primeiro da lista.
            candidates = [c for c in self._clusters if c.accepts(tracklet, window_s)]
            if candidates:
                # Desempate deterministico para distancia empatada entre
                # dois ou mais clusters compativeis: o `reference_timestamp`
                # menor vence. Escolhido em vez de "ordem de insercao"
                # (que seria implicito e fragil - dependeria da ordem de
                # chegada dos pacotes, nao de uma propriedade temporal) ou
                # "ordem de criacao do cluster" (mesmo problema). Precisa
                # ser a SEGUNDA chave de `min`, nao so a ordem de `candidates`,
                # porque `candidates` segue `self._clusters`, cuja ordem e
                # de insercao - sem isso, o desempate mudaria conforme qual
                # cluster foi criado primeiro, nao conforme o timestamp.
                nearest = min(
                    candidates,
                    key=lambda c: (abs(tracklet.timestamp - c.reference_timestamp), c.reference_timestamp),
                )
                nearest.tracklets.append(tracklet)
                return True

            self._clusters.append(_Cluster(now, tracklet))
            return True

    def pop_ready_batches(self) -> list[list[LocalTracklet]]:
        """Retorna lotes cujo cluster ja fechou (watermark de CHEGADA do
        primeiro membro + janela), em ordem cronologica de timestamp de
        medicao.

        Um cluster mais NOVO (timestamp de medicao maior) pode ter seu
        watermark de chegada completar ANTES de um cluster mais ANTIGO
        que ainda esta esperando (chegou depois na rede, mas representa
        um instante fisico anterior). Liberar o mais novo primeiro faria
        o tracker processar t=10.0 antes de t=5.0 - quebraria a ordem
        cronologica que o resto do pipeline (dt do filtro, anchor() do
        TrackingClock) assume. Por isso: percorremos os clusters ordenados
        por `processing_timestamp` (o valor que o Tracker realmente usa
        como timestamp do lote, nao `reference_timestamp` - ver a
        propriedade em `_Cluster`) e paramos no primeiro que ainda nao esta
        pronto - nenhum cluster posterior a ele pode ser liberado ainda,
        mesmo que o SEU proprio watermark ja tenha passado."""
        window_s = self._config.synchronization_window_ms / 1000.0

        with self._lock:
            now = self._clock()
            ordered = sorted(self._clusters, key=lambda c: c.processing_timestamp)
            ready: list[_Cluster] = []
            for cluster in ordered:
                if now - cluster.opened_at >= window_s:
                    ready.append(cluster)
                else:
                    break  # cluster temporalmente anterior ainda pendente -> nada depois dele libera

            if not ready:
                return []

            ready_ids = {id(c) for c in ready}
            self._clusters = [c for c in self._clusters if id(c) not in ready_ids]

        return self._finalize(ready)

    def flush_all_batches(self) -> list[list[LocalTracklet]]:
        """Como pop_ready_batches, mas ignora o watermark - usado em
        replay/simulacao offline, onde nao ha "tempo real" esperando."""
        with self._lock:
            ready = self._clusters
            self._clusters = []

        return self._finalize(ready)

    def _finalize(self, clusters: list[_Cluster]) -> list[list[LocalTracklet]]:
        # `processing_timestamp`, nao `reference_timestamp`: e a ordem que
        # importa para quem recebe estes lotes (Tracker.tick() usa
        # max(tracklet.timestamp) como o timestamp do lote).
        clusters.sort(key=lambda c: c.processing_timestamp)
        batches = [sorted(c.tracklets, key=lambda t: t.timestamp) for c in clusters]

        if batches:
            newest = max(t.timestamp for batch in batches for t in batch)
            with self._lock:
                self._last_released_timestamp = max(self._last_released_timestamp, newest)

        return batches
