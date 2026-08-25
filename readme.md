# Tracker Global Multiestação de UAVs

Camada central de rastreamento que recebe estimativas de estado locais
(`tracklets`) produzidas de forma independente por várias estações
terrestres, associa quais leituras pertencem a qual alvo, funde as
estimativas de estações que observam o mesmo UAV e mantém uma
identidade global (`GlobalTrack`) persistente por alvo — inclusive
durante perdas temporárias de observação (oclusão, estação fora de
alcance, dropout de rede).

A camada central não depende de como cada estação obtém sua estimativa
local (RF, câmera, fusão local própria): cada estação é tratada como uma
caixa-preta que fornece `[x, y, z, vx, vy, vz]` + covariância.

## Pipeline

LocalTracklets (UDP/JSON)
    ↓
buffer de sincronização temporal (agrupa leituras de estações
diferentes que representam o mesmo instante físico)
    ↓
predição do estado de cada GlobalTrack ativo até o instante do lote
    ↓
gating estatístico (Mahalanobis + qui-quadrado) + assignment húngaro,
por estação, em dois estágios de prioridade
    ↓
fusão track-to-track dos tracklets associados a um mesmo GlobalTrack
    ↓
atualização do filtro (measurement update)
    ↓
ciclo de vida (TENTATIVE → CONFIRMED → COASTING → LOST → DELETED)
    ↓
fusão de GlobalTracks duplicados
    ↓
saída: equivalência local ↔ global (UDP) + histórico de trajetória (opcional)

## Estado atual do sistema

- **Filtros**: UKF (Unscented Kalman Filter, default de produção) e EKF
  (Extended Kalman Filter, baseline científica) — escolhidos por config
  (`filter.type`), mesma interface pública para os dois.
- **Modelos de movimento**: Constant Velocity e Coordinated Turn
  (default). Constant Acceleration está reservado no código, mas ainda
  não implementado.
- **Fusão track-to-track**: Information Fusion (default) e Covariance
  Intersection, ambas por trás da mesma interface (`fusion.strategy`).
- **Associação**: gating por distância de Mahalanobis + teste
  qui-quadrado, assignment húngaro por estação, em dois estágios de
  prioridade (tracks ativos competem primeiro; tracks `LOST` só disputam
  o que sobra).
- **Ciclo de vida**: `TENTATIVE → CONFIRMED → COASTING → LOST →
  DELETED`, com reaquisição direta para `CONFIRMED` quando uma medição
  volta a ser associada a um track em `COASTING`/`LOST`.
- **Fusão de tracks duplicados**: configurável
  (`association.duplicate_merge_enabled`, default habilitado) —
  reconcilia `GlobalTrack`s que fragmentaram a identidade do mesmo alvo.
- **Gerador de tráfego sintético** (`network_simulator/`): fala o
  protocolo UDP/JSON real, usado para testar o pipeline completo fim a
  fim sem depender de estações físicas.

## Contrato de entrada (UDP/JSON)

Cada mensagem operacional enviada por uma estação ao tracker central
tem o formato:

```json
{
  "station_id": "A",
  "local_track_id": "A001",
  "timestamp": 12.4,
  "state": [x, y, z, vx, vy, vz],
  "covariance": [[...], ...]
}
```

`covariance` é a matriz 6×6 de incerteza do estado, em ordem
`[x, y, z, vx, vy, vz]`. `timestamp` é o instante de MEDIÇÃO (não o
horário de chegada do pacote). Este é o payload real aceito pelo
receiver (`io_/parser.py`) e produzido pelo gerador de tráfego
(`network_simulator/message_builder.py`) — campos de ground truth
existem no modelo interno apenas para avaliação offline do simulador e
nunca trafegam nesta interface.

**Regra de identidade dos IDs locais**: `local_track_id` é único dentro
da sessão de uma estação (`station_id`) e nunca deve ser reutilizado
para representar outro alvo físico durante essa sessão. Os IDs são
sempre *station-scoped* — `("A", "001")` e `("B", "001")` são fontes
diferentes por definição, mesmo com o mesmo texto de ID.

A resposta que o tracker devolve por UDP a cada ciclo (equivalência
local↔global) é enxuta:

```json
{"station_id": "A", "local_track_id": "A001", "global_track_id": 7, "timestamp": 12.4}
```

## Requisitos

- Python ≥ 3.10 (desenvolvido e testado em 3.14)

## Instalação

```bash
python -m venv .venv
```

Ativar o ambiente virtual:

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

Instalar dependências:

```bash
pip install -r requirements.txt
```

## Rodando

**Terminal 1 — sobe o tracker central:**

```bash
python src/main.py --port 9999
```

**Terminal 2 — gerador de tráfego sintético** (simula estações reais
enviando dados via UDP; use isso se ainda não tiver estações de verdade
pra conectar):

```bash
python src/network_simulator/main.py --port 9999
```

Outros cenários prontos em `src/network_simulator/scenarios/`:

```bash
python src/network_simulator/main.py --scenario src/network_simulator/scenarios/two_uavs_three_stations.yaml
```

**Salvando o histórico de trajetória** (posição, velocidade, status por
ciclo, em CSV + JSONL) — opcional, soma `--track-log-dir`:

```bash
python src/main.py --port 9999 --track-log-dir data/track_history
```

## Testes

```bash
python -m pytest tests/ -q
```

## Configuração

`src/config/default.yaml` tem os parâmetros default: janela de
sincronização temporal, timeouts de ciclo de vida, filtro (tipo +
modelo de movimento + ruído de processo), estratégia de fusão, e o
toggle de fusão de duplicados. Nenhum limiar fica hardcoded no
algoritmo — tudo passa por essa configuração. Para usar um arquivo
próprio:

```bash
python src/main.py --config caminho/para/config.yaml
```

## Estrutura do projeto

```
src/
  association/       - gating estatístico + assignment húngaro
  config/            - modelos de configuração + default.yaml
  filtering/         - UKF, EKF, modelos de movimento
  fusion/            - estratégias de fusão track-to-track
  io_/               - receiver UDP, parser, serializer, histórico
  models/            - modelos de dados (GlobalTrack, LocalTracklet, enums)
  network_simulator/ - gerador de tráfego UDP sintético + cenários
  simulation/        - estação virtual e utilitários de simulação offline
  synchronization/   - buffer temporal, relógio de tracking
  tracking/          - orquestração central, ciclo de vida, fusão de duplicados
  evaluation/        - métricas de avaliação offline

tests/
  unit/
  integration/
```
