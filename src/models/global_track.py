"""Identidade global mantida pela camada central.

GlobalTrack e um modelo de dados "puro" (snapshot do estado atual) - a
mecanica do filtro (predict/update) vive em filtering/ukf.py e e mantida
separadamente pelo TrackManager. Isso evita misturar logica de tracking
com o objeto de dados, e mantem GlobalTrack facil de serializar.

Dois timestamps sao mantidos deliberadamente separados (nao um so
"last_update_timestamp" ambiguo):
  - last_prediction_timestamp: atualizado a cada predict(), com ou sem
    medicao no ciclo. E o que deve ser usado para calcular dt do proximo
    predict().
  - last_measurement_timestamp: atualizado SO quando uma medicao real e
    incorporada (update()). Usado apenas para as transicoes de ciclo de
    vida (COASTING/LOST/DELETED).
Herança do problema q deu do projeto em rust com um unico contador (tipo o link)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from models.enums import TrackStatus, UpdateKind
from models.validation import coerce_covariance_matrix, coerce_state_vector


class GlobalTrack(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    global_track_id: int

    state: np.ndarray
    covariance: np.ndarray

    last_prediction_timestamp: float
    last_measurement_timestamp: float
    created_at: float

    status: TrackStatus = TrackStatus.TENTATIVE
    last_update_kind: UpdateKind = UpdateKind.PREDICTION_ONLY

    age: int = 0
    hit_count: int = 0
    miss_count: int = 0

    # station_id -> local_track_id mais recente conhecido - HISTORICO de
    # identidade, sobrescrito a cada match mas nunca limpo (nao e permanente
    # como identidade fisica, mas persiste entre ciclos mesmo quando aquela
    # estacao nao contribuiu no ciclo atual).
    associated_local_tracks: dict[str, str] = Field(default_factory=dict)

    # station_id -> measurement_timestamp da associacao que gerou o valor
    # ATUAL de `associated_local_tracks[station_id]` - existe SO para
    # `tracking/duplicate_merger.py::_merge_into` decidir, quando os dois
    # tracks fundidos tem um valor diferente pra MESMA estacao, qual dos
    # dois e realmente o mais recente (nao dava pra saber isso antes -
    # nem `global_track_id` nem qualquer outro campo indicava recencia
    # POR ESTACAO, so a idade de criacao do track inteiro, que nao e a
    # mesma coisa).
    associated_local_track_timestamps: dict[str, float] = Field(default_factory=dict)

    # (station_id, local_track_id) que contribuiram para a fusao/update
    # DESTE ciclo especificamente - vazio quando o ciclo foi PREDICTION_ONLY.
    # Separado de `associated_local_tracks` de proposito, para evitar risco
    # real de confusao: uma estacao pode aparecer em
    # associated_local_tracks por ter contribuido ha varios ciclos, mesmo
    # sem ter mandado nada agora - current_contributors nunca mente sobre
    # "quem participou desta atualizacao especifica".
    current_contributors: list[tuple[str, str]] = Field(default_factory=list)

    # (station_id, local_track_id) que JA contribuiram para este
    # GlobalTrack em QUALQUER ciclo passado - nunca sobrescrito, so cresce
    # (diferente de `associated_local_tracks`, que guarda so o mais
    # recente por estacao). Existe para
    # `tracking/duplicate_merger.py::_shares_local_track_id` reconhecer
    # dois tracks como duplicados quando eles JA COMPARTILHARAM, em
    # algum momento passado, o mesmo (station_id, local_track_id) - MESMO
    # que os IDs ATUAIS de cada um ja tenham divergido depois disso (ex.:
    # Track A recebeu A001 e depois A002; Track B recebeu A001 antes; os
    # dois tem A001 no historico, mesmo que hoje Track A mostre A002 em
    # `associated_local_tracks` - comparar so os dicts atuais, como a
    # implementacao antiga fazia, nunca acharia essa intersecao).
    #
    # Ressalva (nao e comportamento acidental, e uma limitacao conhecida):
    # isto NAO resolve o caso de uma estacao trocar de ID
    # (VirtualStation.force_new_local_id) exatamente no
    # instante em que o gate estatistico rejeita a reassociacao por
    # acaso. Nesse cenario especifico, o track ANTIGO nunca chega a
    # receber o ID NOVO (a reassociacao foi rejeitada, e disso que se
    # trata), entao seu historico so tem o ID velho; o track NOVO que
    # nasce a partir da medicao rejeitada tem, na criacao, so o ID novo
    # (ver track_initializer.py::create_track, que inicializa o
    # historico so com os tracklets que formaram o track). Os dois
    # historicos nao tem nenhum elemento em comum - a interseccao e
    # vazia, e este criterio NAO os reconhece como duplicados. Esse caso
    # especifico continua dependendo so do gate estatistico Mahalanobis
    # (que pode ou nao pegar, dependendo da covariancia no momento) -
    # nenhuma heuristica nova foi criada pra cobri-lo nesta versao (par
    # so entra aqui numa associacao de verdade aceita, nunca em um
    # candidato so avaliado/rejeitado).
    local_track_history: set[tuple[str, str]] = Field(default_factory=set)

    @field_validator("state", mode="before")
    @classmethod
    def _validate_state(cls, v):
        return coerce_state_vector(v)

    @field_validator("covariance", mode="before")
    @classmethod
    def _validate_covariance(cls, v):
        return coerce_covariance_matrix(v)

    @property
    def position(self) -> np.ndarray:
        return self.state[:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:6]

    def time_since_measurement(self, timestamp: float) -> float:
        return timestamp - self.last_measurement_timestamp
