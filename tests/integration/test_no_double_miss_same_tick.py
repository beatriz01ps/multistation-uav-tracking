"""Um GlobalTrack que participa de um lote real mas NAO consegue
associacao so pode receber UM miss neste tick() - nao um de
`_apply_misses` dentro de `process_batch` e outro do `_catch_up` no
final do mesmo `tick()`. O conjunto usado para saber "quem ja foi
processado neste tick" precisa incluir tambem quem foi avaliado e ficou
sem par, nao so quem gerou `TrackUpdateEvent`."""

import numpy as np

from config.models import AppConfig
from models.enums import TrackStatus
from models.local_tracklet import LocalTracklet
from tests.helpers import FakeClock
from tracking.tracker import Tracker


def _tracklet(station_id, local_id, t, x):
    return LocalTracklet(
        station_id=station_id, local_track_id=local_id, timestamp=t, state=[x, 0.0, 100.0, 10.0, 0.0, 0.0], covariance=np.eye(6)
    )


def test_unassociated_track_in_real_batch_misses_exactly_once_per_tick():
    clock = FakeClock()
    tracker = Tracker(AppConfig(), clock=clock)

    # confirma um track (3 hits, tentative_confirmation_hits default)
    for t in [0.0, 1.0, 2.0]:
        tracker.ingest(_tracklet("station_a", "001", t, x=t * 10))
        clock.advance(0.2)
        tracker.tick()

    (track,) = tracker.track_manager.tracks.values()
    global_track_id = track.global_track_id
    assert track.status is TrackStatus.CONFIRMED
    miss_count_before = track.miss_count
    assert miss_count_before == 0
    prediction_timestamp_before = track.last_prediction_timestamp

    # lote real, mas com um alvo muito longe: nao associa ao track existente
    # (vira candidato a um NOVO track) - o track original participa deste
    # tick (foi avaliado para associacao) mas fica sem par.
    tracker.ingest(_tracklet("station_a", "002", 3.0, x=99999.0))
    clock.advance(0.2)
    tracker.tick()

    track = tracker.track_manager.tracks[global_track_id]
    assert track.miss_count == miss_count_before + 1  # nunca +2
    assert track.status is TrackStatus.COASTING
    # a previsao temporal (catch-up) ainda deve acontecer normalmente - so a
    # segunda aplicacao de miss e que deve ser suprimida, nao o predict.
    assert track.last_prediction_timestamp > prediction_timestamp_before
