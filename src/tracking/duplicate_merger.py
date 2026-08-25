"""Deteccao e fusao de GlobalTracks duplicados: dois tracks que passam a
representar o MESMO alvo fisico sem que o resto do pipeline perceba,
porque associacao/gating so compara TRACK vs TRACKLET, nunca TRACK vs
TRACK. Isso acontece na pratica (testado com oclusao geometrica, ver
docs/arquitetura.md): um gate de 99% rejeita ~1% das medicoes legitimas
por acaso estatistico; num momento de baixa redundancia (uma estacao
ocluida), essa rejeicao isolada bastava para nascer um candidato novo
representando o mesmo alvo que um track ja confirmado.

Reusa a MESMA distancia de Mahalanobis + gate qui-quadrado ja usados para
associacao tracklet<->track (association/gating.py) - dois tracks sao
duplicados quando a distancia entre suas estimativas, medida pela soma das
covariancias (P_a + P_b, exatamente a formula que association/mahalanobis.py
ja calcula), passa no teste estatistico. E reusa a MESMA FusionStrategy
configurada (fusion.strategy) para fundir as duas estimativas numa so -
nunca uma segunda formula de "como combinar duas estimativas" no projeto
(ver fusion/base.py::fuse_states).

Sempre FULL_STATE, nunca herda `association.mode`: fundir
dois tracks e uma acao muito mais consequente e dificil de reverter que
gatear um unico tracklet (destroi um dos dois IDs permanentemente). Dois
UAVs DIFERENTES podem compartilhar POSICAO por um instante (cruzamento de
trajetorias - ver tests/integration/test_crossing.py), mas praticamente
nunca compartilham posicao E velocidade ao mesmo tempo (isso exigiria uma
colisao de verdade). Por isso a checagem de duplicidade aqui sempre usa as
6 dimensoes (posicao+velocidade), mesmo que o sistema esteja configurado
com `association.mode: position_only` - e a salvaguarda que evita fundir
dois alvos reais que so cruzaram caminho.

Segundo criterio, COMPLEMENTAR ao gate estatistico (nao um substituto):
dois tracks tambem sao considerados duplicados se ja compartilharam, em
qualquer ciclo passado, a MESMA fonte - a intersecao de
`local_track_history` dos dois e nao-vazia (ver models/global_track.py
pro raciocinio completo de por que e o historico, nao so o mais recente
por estacao). Isso importa na pratica (testado com cenarios de multiplos
UAVs, ver docs/arquitetura.md): conforme um par de tracks duplicados vai ficando
mais confiante (covariancia menor a cada update), a MESMA probabilidade
que gateia bem tracklet<->track fica progressivamente mais rigorosa para
track<->track - um par podia ficar alternando COASTING/RECOVERED por
varios segundos sem nunca fundir, porque a distancia ESTATISTICA
(normalizada pela covariancia, cada vez menor) ficava acima do limiar,
mesmo a distancia ABSOLUTA sendo pequena. Uma estacao nunca reusa o mesmo
`local_track_id` para dois alvos fisicos diferentes (nem no simulador,
nem na convencao assumida para estacoes reais) - entao esse sinal e
praticamente livre de falso positivo, e nao depende de covariancia
nenhuma, so de historico. Corrige exatamente o caso acima, sem afrouxar a
salvaguarda contra cruzamento (a condicao FULL_STATE continua valendo do
mesmo jeito para o outro criterio).

NAO corrige (limitacao conhecida, registrada aqui pra nao superestimar o
que este criterio cobre): uma estacao trocar de local_track_id
(VirtualStation.force_new_local_id) exatamente no instante em que o gate
estatistico rejeita a reassociacao por acaso. Nesse caso o track antigo
nunca chega a receber o ID novo (foi rejeitado), entao seu historico so
tem o ID velho, e o candidato novo nasce (ver track_initializer.py) so
com o ID novo no historico - intersecao vazia, este criterio nao
reconhece o par. Esse cenario especifico continua dependendo so do gate
estatistico Mahalanobis (pode ou nao pegar, dependendo da covariancia no
momento) - nao foi criada nenhuma heuristica nova pra cobri-lo (decisao
deliberada, nao lacuna esquecida).

Convencao de sobrevivencia: o track com `global_track_id` MENOR (o mais
antigo, ja que os IDs sao atribuidos por um contador crescente) sobrevive
- minimiza troca de ID do ponto de vista de quem consome a saida.
`hit_count`/`miss_count`/`status` do sobrevivente NAO sao alterados (sao
historico do proprio ciclo de vida dele, nao algo que faca sentido somar
entre dois tracks). `current_contributors` e `local_track_history` sao a
UNIAO dos dois - reunificar o que as estacoes sabiam sobre o mesmo alvo
sob um unico ID. `associated_local_tracks` (mais recente conhecido POR
ESTACAO) e diferente: quando as duas metades tem um valor pra MESMA
estacao, NAO da pra saber qual e o mais recente so pelo `global_track_id`
(menor ID = track mais antigo, nao quer dizer que TODAS as suas
associacoes por estacao sejam mais recentes que as do outro) - por isso
`associated_local_track_timestamps` guarda o `measurement_timestamp` de
cada associacao, e o merge usa isso pra decidir, por estacao, qual valor
manter (ver `_merge_into`).

LIMITACAO CONHECIDA, documentada e nao corrigida ainda (nao exige mudanca
imediata): a distancia de Mahalanobis do gate estatistico usa P_a + P_b
como covariancia da inovacao - isso trata os erros dos dois GlobalTracks
como INDEPENDENTES. Na pratica, dois tracks duplicados podem ter recebido
informacao parcialmente vinda das MESMAS estacoes ao longo do tempo (ou
ainda compartilhar parte da mesma fusao original que os originou), entao
pode existir correlacao real entre os erros que este teste nao modela
explicitamente. O efeito pratico seria o gate ficar um pouco mais
conservador ou mais permissivo do que o ideal em alguns casos - nao
invalida o criterio (que ja tem o segundo sinal complementar de historico
de fonte compartilhada como rede de seguranca), mas fica registrado aqui
como hipotese simplificadora conhecida, nao como comportamento acidental.

`AssociationConfig.duplicate_merge_enabled` (default True) permite
desligar esta etapa inteira - existe so para ABLACAO CIENTIFICA (medir o
efeito real da fusao de duplicados comparando execucoes com/sem ela:
fragmentacao, trocas de ID, GlobalTracks espurios), nunca pensado para
ficar desligado em producao.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from association.gating import gate
from config.models import AssociationConfig
from fusion.base import TrackFusionStrategy
from models.enums import AssociationMode, TrackStatus
from models.global_track import GlobalTrack
from tracking.track_manager import TrackManager

logger = logging.getLogger(__name__)

_DUPLICATE_CHECK_MODE = AssociationMode.FULL_STATE


@dataclass(frozen=True)
class MergeEvent:
    survivor_global_track_id: int
    merged_global_track_id: int
    d2: float
    reason: str  # "gate" | "shared_local_track_id" | "gate+shared_local_track_id"


class DuplicateTrackMerger:
    def __init__(self, association_config: AssociationConfig, fusion_strategy: TrackFusionStrategy) -> None:
        self._chi_square_probability = association_config.chi_square_probability
        self._fusion_strategy = fusion_strategy
        self._enabled = association_config.duplicate_merge_enabled

    def merge_duplicates(self, track_manager: TrackManager) -> list[MergeEvent]:
        """Funde pares duplicados repetidamente ate nao sobrar nenhum par
        duplicado (gate estatistico OU historico de local_track_id
        compartilhado) entre os tracks ativos (cobre o caso raro de 3+
        candidatos fragmentados do mesmo alvo, nao so 2). A cada fusao, os
        tracks ativos sao reavaliados do zero - uma fusao pode mudar qual e
        o par mais proximo restante.

        Se `association_config.duplicate_merge_enabled` for False, retorna
        sempre `[]` sem avaliar nenhum par - existe so para ablacao
        cientifica (ver docstring do modulo), nunca para producao."""
        if not self._enabled:
            return []

        events: list[MergeEvent] = []

        while True:
            pair = self._closest_duplicate_pair(track_manager.active_tracks())
            if pair is None:
                break

            track_a, track_b, d2, reason = pair
            if track_a.global_track_id < track_b.global_track_id:
                survivor, merged = track_a, track_b
            else:
                survivor, merged = track_b, track_a

            self._merge_into(track_manager, survivor, merged)
            events.append(MergeEvent(survivor.global_track_id, merged.global_track_id, d2, reason))
            logger.info(
                "GLOBAL_TRACK_MERGED survivor=%s merged=%s d2=%.2f reason=%s",
                survivor.global_track_id,
                merged.global_track_id,
                d2,
                reason,
            )

        return events

    def _closest_duplicate_pair(
        self, active: list[GlobalTrack]
    ) -> Optional[tuple[GlobalTrack, GlobalTrack, float, str]]:
        best: Optional[tuple[GlobalTrack, GlobalTrack, float, str]] = None
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                track_a, track_b = active[i], active[j]
                passed, d2 = gate(
                    track_a.state,
                    track_a.covariance,
                    track_b.state,
                    track_b.covariance,
                    _DUPLICATE_CHECK_MODE,
                    self._chi_square_probability,
                )
                shares_local_id = self._shares_local_track_id(track_a, track_b)
                if passed and not shares_local_id and self._is_stale_covariance_pair(track_a, track_b):
                    passed = False
                if not (passed or shares_local_id):
                    continue

                if passed and shares_local_id:
                    reason = "gate+shared_local_track_id"
                elif passed:
                    reason = "gate"
                else:
                    reason = "shared_local_track_id"

                if best is None or d2 < best[2]:
                    best = (track_a, track_b, d2, reason)
        return best

    @staticmethod
    def _is_stale_covariance_pair(track_a: GlobalTrack, track_b: GlobalTrack) -> bool:
        """True se exatamente um dos dois esta CONFIRMED (recebendo
        medicao real) e o outro esta COASTING/LOST (sem medicao real ha um
        tempo) - mesmo raciocinio do estagio de prioridade em
        association/association_manager.py, aplicado aqui: a covariancia
        de um track cresce sem limite (chega a dobrar por segundo, medido
        empiricamente com o Coordinated Turn default) enquanto ele fica
        sem medicao, entao P_a + P_b (a covariancia combinada que o gate
        usa) fica dominada pelo lado inflado - o gate estatistico "passa"
        so porque aquele track parou de saber onde esta, nao porque os
        dois sao de fato o mesmo alvo. Isso pode apagar permanentemente a
        identidade de um track CONFIRMED que esta sendo alimentado com
        medicoes reais o tempo todo (regressao coberta por
        tests/integration/test_tracker_pipeline.py::
        test_lost_track_does_not_steal_a_nearby_active_tracks_observation).

        So neutraliza o criterio de GATE (estatistico) - o criterio de
        `_shares_local_track_id` continua valendo do mesmo jeito para este
        par, ja que aquele sinal nao depende de covariancia nenhuma."""
        statuses = {track_a.status, track_b.status}
        return TrackStatus.CONFIRMED in statuses and bool(statuses & {TrackStatus.COASTING, TrackStatus.LOST})

    @staticmethod
    def _shares_local_track_id(track_a: GlobalTrack, track_b: GlobalTrack) -> bool:
        """True se, em qualquer ciclo passado, os dois tracks ja foram
        alimentados pela MESMA fonte (station_id, local_track_id) - uma
        estacao nunca reusa o mesmo local_track_id para dois alvos fisicos
        diferentes, entao esse sinal e praticamente livre de falso
        positivo, mesmo quando a distancia estatistica (covariancia
        pequena demais) diria "nao sao duplicados".

        Compara `local_track_history` (tudo que cada track JA recebeu,
        nunca sobrescrito - ver models/global_track.py), NAO
        `associated_local_tracks` (so o mais recente por estacao): pega o
        caso de um par que compartilhou uma fonte no passado mas cujos
        IDs ATUAIS ja divergiram desde entao (ver docstring do modulo e
        de models/global_track.py::local_track_history pro exemplo
        completo, e pra ressalva do que isto NAO cobre - troca de ID
        exatamente durante uma rejeicao de gate)."""
        return bool(track_a.local_track_history & track_b.local_track_history)

    def _merge_into(self, track_manager: TrackManager, survivor: GlobalTrack, merged: GlobalTrack) -> None:
        fused_state, fused_covariance = self._fusion_strategy.fuse_states(
            [(survivor.state, survivor.covariance), (merged.state, merged.covariance)]
        )

        track_manager.filter_for(survivor.global_track_id).set_state(fused_state, fused_covariance)
        track_manager.sync_track_from_filter(survivor.global_track_id)

        for station_id, local_track_id in merged.associated_local_tracks.items():
            merged_timestamp = merged.associated_local_track_timestamps.get(station_id, float("-inf"))
            survivor_timestamp = survivor.associated_local_track_timestamps.get(station_id, float("-inf"))
            # so troca se o lado MERGED for estritamente mais recente -
            # empate (ou ausencia de timestamp dos dois lados) fica com o
            # sobrevivente de proposito (decisao documentada, nao acidente:
            # evita trocar a identidade local sem evidencia real de que o
            # outro lado e mais novo).
            if merged_timestamp > survivor_timestamp:
                survivor.associated_local_tracks[station_id] = local_track_id
                survivor.associated_local_track_timestamps[station_id] = merged_timestamp
            elif station_id not in survivor.associated_local_tracks:
                survivor.associated_local_tracks[station_id] = local_track_id
                survivor.associated_local_track_timestamps[station_id] = merged_timestamp
        survivor.current_contributors = list(set(survivor.current_contributors) | set(merged.current_contributors))
        survivor.local_track_history = survivor.local_track_history | merged.local_track_history

        track_manager.remove(merged.global_track_id)
