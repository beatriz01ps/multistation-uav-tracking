import numpy as np

from config.models import AssociationConfig
from models.local_tracklet import LocalTracklet
from synchronization.tracklet_buffer import TrackletBuffer
from tests.helpers import FakeClock


def _tracklet(station_id, local_id, t):
    return LocalTracklet(
        station_id=station_id, local_track_id=local_id, timestamp=t, state=[0.0] * 6, covariance=np.eye(6)
    )


def test_same_source_rapid_readings_split_into_separate_batches():
    """5 leituras da MESMA fonte, chegando dentro da mesma janela de flush,
    nao podem virar 1 lote so (o Hungarian trataria como 5 candidatos
    concorrentes) nem viram 5 GlobalTracks - devem virar 5 lotes
    sequenciais, um por leitura."""
    config = AssociationConfig(synchronization_window_ms=100)
    clock = FakeClock()
    buffer = TrackletBuffer(config, clock=clock)

    for t in [0.0, 0.1, 0.2, 0.3, 0.4]:
        buffer.add(_tracklet("station_a", "001", t))

    clock.advance(0.2)  # passa da janela de sincronizacao (100ms)
    batches = buffer.pop_ready_batches()

    assert len(batches) == 5
    for batch in batches:
        assert len(batch) == 1
    assert [b[0].timestamp for b in batches] == [0.0, 0.1, 0.2, 0.3, 0.4]


def test_different_stations_close_in_time_merge_into_one_batch():
    config = AssociationConfig(synchronization_window_ms=100)
    clock = FakeClock()
    buffer = TrackletBuffer(config, clock=clock)

    buffer.add(_tracklet("station_a", "001", 5.0))
    buffer.add(_tracklet("station_b", "007", 5.05))

    clock.advance(0.2)
    batches = buffer.pop_ready_batches()

    assert len(batches) == 1
    assert len(batches[0]) == 2


def test_batches_released_in_chronological_order():
    config = AssociationConfig(synchronization_window_ms=50)
    clock = FakeClock()
    buffer = TrackletBuffer(config, clock=clock)

    buffer.add(_tracklet("station_a", "001", 10.0))
    buffer.add(_tracklet("station_a", "001", 5.0))  # timestamp de medicao menor, chegou depois

    clock.advance(0.2)
    batches = buffer.pop_ready_batches()

    timestamps = [b[0].timestamp for b in batches]
    assert timestamps == sorted(timestamps)


def test_pop_ready_batches_waits_for_window_before_releasing():
    config = AssociationConfig(synchronization_window_ms=200)
    clock = FakeClock()
    buffer = TrackletBuffer(config, clock=clock)

    buffer.add(_tracklet("station_a", "001", 0.0))
    clock.advance(0.05)
    assert buffer.pop_ready_batches() == []  # ainda nao passou a janela de chegada

    clock.advance(0.2)
    batches = buffer.pop_ready_batches()
    assert len(batches) == 1


def test_flush_all_batches_ignores_arrival_window():
    config = AssociationConfig(synchronization_window_ms=100)
    buffer = TrackletBuffer(config, clock=FakeClock())

    buffer.add(_tracklet("station_a", "001", 0.0))
    buffer.add(_tracklet("station_a", "001", 1.0))

    batches = buffer.flush_all_batches()
    assert len(batches) == 2  # mesma fonte, timestamps distantes -> lotes separados


def test_newer_cluster_never_released_before_older_pending_cluster():
    """Um cluster mais NOVO (measurement timestamp maior) pode ter seu
    watermark de CHEGADA completar antes de um cluster mais ANTIGO que
    chegou depois na rede - liberar o mais novo primeiro faria o tracker
    processar t=10.0 antes de t=5.0. A ordenacao dentro de uma UNICA
    chamada nao basta: precisa se manter atraves de DUAS chamadas
    separadas de pop_ready_batches()."""
    config = AssociationConfig(synchronization_window_ms=50)
    clock = FakeClock()
    buffer = TrackletBuffer(config, clock=clock)

    buffer.add(_tracklet("station_a", "001", 10.0))  # chega primeiro, watermark fecha primeiro
    clock.advance(0.02)
    buffer.add(_tracklet("station_a", "002", 5.0))  # timestamp de MEDICAO anterior, chegou depois

    clock.advance(0.04)  # cluster de 10.0 (aberto em 0) pronto; cluster de 5.0 (aberto em 0.02) ainda nao
    first_call = buffer.pop_ready_batches()
    assert first_call == []  # nao pode liberar 10.0 enquanto 5.0 (mais antigo) ainda pende

    clock.advance(0.05)  # agora os dois estao prontos
    second_call = buffer.pop_ready_batches()
    timestamps = [b[0].timestamp for b in second_call]
    assert timestamps == [5.0, 10.0]

    # monotonicidade: a sequencia COMPLETA de batches entregues ao tracker
    # (entre as duas chamadas) nunca pode regredir no tempo.
    all_released_timestamps = [b[0].timestamp for b in first_call] + [b[0].timestamp for b in second_call]
    assert all_released_timestamps == sorted(all_released_timestamps)


def test_tracklet_joins_nearest_compatible_cluster_not_first_match():
    """Quando um tracklet cabe na janela de mais de um cluster, precisa
    entrar no MAIS PROXIMO por timestamp de medicao - nao simplesmente no
    primeiro que aceitar."""
    config = AssociationConfig(synchronization_window_ms=100)
    clock = FakeClock()
    buffer = TrackletBuffer(config, clock=clock)

    buffer.add(_tracklet("station_a", "001", 0.00))  # cluster 1
    buffer.add(_tracklet("station_a", "001", 0.10))  # mesma fonte -> forca cluster 2 separado

    buffer.add(_tracklet("station_b", "007", 0.09))  # 0.09s do cluster 1, so 0.01s do cluster 2

    clock.advance(0.2)
    batches = buffer.pop_ready_batches()

    cluster_with_b = next(b for b in batches if any(t.station_id == "station_b" for t in b))
    other_timestamps = {t.timestamp for t in cluster_with_b if t.station_id != "station_b"}
    assert other_timestamps == {0.10}  # entrou no cluster mais proximo (0.10), nao no de 0.00


def test_batches_ordered_by_processing_timestamp_not_reference_timestamp():
    """Clusters ordenados por `reference_timestamp` (timestamp do PRIMEIRO
    membro) ainda podem sair de ordem para o Tracker, porque ele usa
    `max(tracklet.timestamp for tracklet in batch)` como timestamp efetivo
    do lote (`processing_timestamp`) - nao o do primeiro membro. Cenario:
    Cluster 1 (ref=0.00) acaba recebendo um membro tardio com
    timestamp=0.10 (o mesmo station_id ja estava no Cluster 2, entao o
    tracklet mais novo de Station B teve que entrar no Cluster 1); Cluster 2
    (ref=0.05) fecha com processing_timestamp=0.09. Por reference_timestamp,
    Cluster 1 (0.00) viria antes do Cluster 2 (0.05) - mas por
    processing_timestamp, 0.09 < 0.10, entao o Cluster 2 tem que sair
    primeiro."""
    config = AssociationConfig(synchronization_window_ms=100)
    clock = FakeClock()
    buffer = TrackletBuffer(config, clock=clock)

    buffer.add(_tracklet("station_a", "001", 0.00))  # cluster 1, reference_timestamp=0.00
    buffer.add(_tracklet("station_a", "001", 0.05))  # mesma fonte -> cluster 2, reference_timestamp=0.05
    buffer.add(_tracklet("station_b", "007", 0.09))  # mais perto do cluster 2 -> entra nele
    buffer.add(_tracklet("station_b", "007", 0.10))  # station_b ja no cluster 2 -> entra no cluster 1

    clock.advance(0.2)
    batches = buffer.pop_ready_batches()

    batch_timestamps = [max(t.timestamp for t in b) for b in batches]
    assert batch_timestamps == [0.09, 0.10]  # nunca 0.10 antes de 0.09
    assert batch_timestamps == sorted(batch_timestamps)


def test_tie_in_distance_breaks_towards_smaller_reference_timestamp():
    """Regra de desempate documentada em `TrackletBuffer.add`: quando um
    tracklet fica exatamente a mesma distancia de dois clusters compativeis,
    vence o de MENOR `reference_timestamp` - deterministico, independente da
    ordem de insercao dos clusters. Cria o cluster de referencia MAIOR
    primeiro de proposito, para provar que o desempate nao depende de ordem
    de criacao/insercao."""
    config = AssociationConfig(synchronization_window_ms=100)
    clock = FakeClock()
    buffer = TrackletBuffer(config, clock=clock)

    buffer.add(_tracklet("station_a", "002", 0.20))  # cluster de ref MAIOR, criado primeiro
    buffer.add(_tracklet("station_a", "001", 0.00))  # cluster de ref MENOR, criado depois

    buffer.add(_tracklet("station_b", "007", 0.10))  # exatamente 0.10s dos dois -> empate

    clock.advance(0.3)
    batches = buffer.pop_ready_batches()

    cluster_with_b = next(b for b in batches if any(t.station_id == "station_b" for t in b))
    other_timestamps = {t.timestamp for t in cluster_with_b if t.station_id != "station_b"}
    assert other_timestamps == {0.00}  # o de menor reference_timestamp venceu o empate
