"""Um track que acabou de receber uma medicao valida (dentro de um
tick()) nao pode ser marcado como PREDICTION_ONLY/COASTING so porque o
mesmo tick() tambem avanca o relogio ate "agora" no passo de catch-up.

Com o TrackingClock (dominio de tempo separado), isso e garantido por
construcao: o anchor() e feito no exato instante em que o lote real e
processado, entao o catch-up que roda em seguida, no MESMO tick(), sempre
ve dt=0 para esse track (nenhum tempo monotonic decorreu entre o anchor e
o catch-up, dentro da mesma chamada sincrona)."""

import numpy as np

from config.models import AppConfig
from models.enums import TrackStatus, UpdateKind
from models.local_tracklet import LocalTracklet
from tests.helpers import FakeClock
from tracking.tracker import Tracker


def _tracklet(t, x):
    return LocalTracklet(
        station_id="station_a", local_track_id="001", timestamp=t, state=[x, 0.0, 100.0, 10.0, 0.0, 0.0], covariance=np.eye(6)
    )


def test_track_confirmed_by_measurement_does_not_flip_to_coasting_same_tick():
    clock = FakeClock()
    tracker = Tracker(AppConfig(), clock=clock)

    x = 0.0
    for t in [0.0, 1.0, 2.0]:
        tracker.ingest(_tracklet(t, x))
        clock.advance(0.2)  # passa a janela de sincronizacao -> lote fica pronto
        tracker.tick()
        x += 10.0

    (track,) = tracker.track_manager.tracks.values()
    assert track.status is TrackStatus.CONFIRMED
    assert track.miss_count == 0
    assert track.last_update_kind is UpdateKind.MEASUREMENT_UPDATED
