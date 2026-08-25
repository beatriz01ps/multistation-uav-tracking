"""Configuracao do sistema, centralizada - nenhum threshold deve ficar
hardcoded no algoritmo.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from models.enums import AssociationMode, FilterType, FusionStrategyName, LateMessagePolicy, MotionModelName


class TrackingConfig(BaseModel):
    state_dimension: int = 6
    tentative_confirmation_hits: int = 3
    # BASEADO EM TEMPO, nao em contagem de ciclos do tracker: um contador
    # de misses por tick e fragil a diferenca entre a cadencia interna do
    # tracker e a taxa de envio real da estacao - um track TENTATIVE podia
    # ser apagado antes de acumular tentative_confirmation_hits so porque
    # o tracker rodou mais rapido que a estacao mandou dado. Um candidato
    # TENTATIVE sem nenhuma medicao por mais que este tempo e deletado.
    tentative_timeout_seconds: float = 5.0

    # Desvio deliberado do exemplo da especificacao (que sugeria
    # 3.0/6.0/10.0): esses tres campos sao explicitamente parametros de
    # configuracao a ajustar experimentalmente, nao valores fixos. Os
    # atuais cobrem oclusoes/dropouts bem mais longos que os 9s totais do
    # exemplo original, mantendo a identidade do track por mais tempo sem
    # perder continuidade durante a perda de observacao.
    coasting_timeout_seconds: float = 10.0
    lost_timeout_seconds: float = 20.0
    deletion_timeout_seconds: float = 32.0


class AssociationConfig(BaseModel):
    mode: AssociationMode = AssociationMode.FULL_STATE
    # 0.9983 (nao 0.99): correcao tipo Bonferroni pro aumento de frequencia
    # das estacoes (2-10Hz -> 20-60Hz, fator ~6x, ver frequency_hz nos
    # cenarios). A taxa de falsa rejeicao do gate (1 - chi_square_probability)
    # e POR TENTATIVA de associacao - com 6x mais tentativas por segundo, a
    # taxa de fragmentacao espuria ABSOLUTA multiplicava pelo mesmo fator (a
    # 0.99, loop_crossing chegava a 25 GlobalTracks criados para 2 UAVs
    # reais). Validado nos 14 cenarios de src/network_simulator/scenarios/:
    # numero de tracks confirmados bate exatamente com o numero real de
    # UAVs em todos, com esta config completa (ver docs/arquitetura.md).
    chi_square_probability: float = 0.9983
    synchronization_window_ms: float = 100.0
    # DROP por padrao: sem suporte real a Out-of-Sequence Measurements
    # (OOSM), aplicar uma medicao atrasada sobre um estado que ja avancou
    # para um instante posterior corrompe o filtro. LOG_ONLY existe so
    # para diagnostico com mais detalhe - NUNCA aplica a medicao atrasada
    # tambem (ver synchronization/tracklet_buffer.py).
    late_message_policy: LateMessagePolicy = LateMessagePolicy.DROP
    # True por padrao - desligar so serve para ABLACAO CIENTIFICA (medir
    # fragmentacao/troca de ID/GlobalTracks espurios COM vs SEM a fusao de
    # duplicados, ver tracking/duplicate_merger.py). Nunca desligar em
    # produção: sem isso, uma rejeicao legitima pelo gate estatistico (~1%
    # de chance, mesmo com o par certo) pode fragmentar um alvo confirmado
    # em dois GlobalTracks permanentemente.
    duplicate_merge_enabled: bool = True


class FilterConfig(BaseModel):
    # UKF (Unscented) e o default de PRODUCAO. EKF (Extended, linearizacao
    # via Jacobiano) fica disponivel como BASELINE CIENTIFICA
    # (`filter.type: ekf`, ver filtering/ekf.py e filtering/factory.py) -
    # nao trocar o default sem reabrir essa decisao (ver docs/arquitetura.md,
    # secao "EKF vs UKF", para a comparacao empirica completa).
    #
    # Por que UKF, apesar do EKF ter precisao media (RMSE) melhor na
    # maioria dos cenarios testados: para Coordinated Turn, o EKF estima
    # uma covariancia sistematicamente mais apertada (mais confiante) que
    # o UKF para o mesmo historico de medicoes - propriedade conhecida da
    # linearizacao do EKF em sistemas nao-lineares (subestima incerteza),
    # nao um bug pontual. Isso pode fazer o EKF rejeitar estatisticamente
    # uma reassociacao legitima que o UKF aceitaria com a mesma medicao
    # (covariancia mais realista absorve o erro de predicao). RMSE sozinha
    # nao captura esse risco: mede precisao media, nao a chance de
    # rejeitar reassociacoes legitimas em cenarios com multiplos alvos
    # competindo por observacoes - exatamente o tipo de cenario que este
    # projeto precisa suportar. Para um tracker onde perder ou confundir a
    # identidade de um alvo tem consequencia real, essa robustez pesa mais
    # que o ganho medio de precisao do EKF.
    #
    # Pra Constant Velocity (modelo linear), o PREDICT dos dois e sempre
    # bit a bit identico (o Jacobiano E a propria matriz de transicao, e
    # sigma points propagados por funcao linear reproduzem o resultado
    # exato) - mas o UPDATE so bate bit a bit quando
    # process_noise_acceleration_std == 0. Com ruido de processo > 0 (o
    # caso real), o UKF do FilterPy reusa os sigma points calculados ANTES
    # de somar Q no predict() para o proprio update() - entao a covariancia
    # cruzada usada no update nao reflete o Q recem somado, e o ganho de
    # Kalman diverge um pouco do EKF mesmo em CV. Diferenca documentada e
    # quantificada em tests/unit/test_ekf.py, nao e bug. Pra Coordinated
    # Turn (nao-linear) os dois divergem de verdade, ja no predict.
    type: FilterType = FilterType.UKF
    # Default e coordinated_turn, nao constant_velocity: nao da pra saber
    # de antemao se um alvo vai manobrar ou nao, entao o tracker fica
    # SEMPRE pronto pra extrapolar uma curva durante COASTING, nao so uma
    # reta - ver filtering/motion_models.py::CoordinatedTurnModel. Um
    # track que nunca manobra continua se comportando essencialmente como
    # CV (omega fica perto de 0, nunca e estimado pra longe disso sem
    # medicao real de curva) - o custo e so o efeito de Jensen documentado
    # em CoordinatedTurnModel (pequeno desvio, nao identidade bit a bit,
    # ver tests/unit/test_ukf_coordinated_turn.py).
    motion_model: MotionModelName = MotionModelName.COORDINATED_TURN
    process_noise_acceleration_std: float = 2.0

    # Usados so quando motion_model == coordinated_turn (ver
    # filtering/motion_models.py::CoordinatedTurnModel). Nao calibrados
    # empiricamente - ordem de grandeza fisicamente plausivel para manobra
    # de UAV, servem de ponto de partida configuravel, nunca hardcoded no
    # algoritmo.
    turn_rate_process_noise_std: float = 0.05  # rad/s por sqrt(s): quanto a taxa de giro ESTIMADA pode variar
    initial_turn_rate_std: float = 0.3  # rad/s: incerteza inicial (prior) sobre a taxa de giro de um track novo


class FusionConfig(BaseModel):
    strategy: FusionStrategyName = FusionStrategyName.INFORMATION


class TimeConfig(BaseModel):
    # 1.0 = tempo real (o relogio de tracking extrapolado durante COASTING
    # anda no mesmo ritmo do relogio monotonic local). Preparado para
    # replay acelerado no futuro (ex.: 10.0) sem exigir mudanca na logica
    # do tracker - ver synchronization/tracking_clock.py.
    time_scale: float = 1.0


class AppConfig(BaseModel):
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    association: AssociationConfig = Field(default_factory=AssociationConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    time: TimeConfig = Field(default_factory=TimeConfig)
