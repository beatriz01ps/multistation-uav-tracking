"""Uma mensagem atrasada (timestamp anterior ao ultimo lote ja liberado)
nunca pode ser aplicada - default e DROP, e LOG_ONLY tambem descarta (so
muda o nivel de detalhe do log)."""

import numpy as np
import pytest

from config.models import AppConfig, AssociationConfig
from models.enums import LateMessagePolicy
from models.local_tracklet import LocalTracklet
from synchronization.tracklet_buffer import TrackletBuffer
from tests.helpers import FakeClock


def _tracklet(t):
    return LocalTracklet(
        station_id="station_a", local_track_id="001", timestamp=t, state=[0.0] * 6, covariance=np.eye(6)
    )


@pytest.mark.parametrize("policy", [LateMessagePolicy.DROP, LateMessagePolicy.LOG_ONLY])
def test_late_message_is_never_applied_regardless_of_policy(policy):
    config = AssociationConfig(synchronization_window_ms=50, late_message_policy=policy)
    clock = FakeClock()
    buffer = TrackletBuffer(config, clock=clock)

    buffer.add(_tracklet(10.0))
    clock.advance(0.1)
    batches = buffer.pop_ready_batches()
    assert len(batches) == 1
    assert batches[0][0].timestamp == 10.0

    # tracklet atrasado: timestamp=9.0, mas o buffer ja liberou ate 10.0
    accepted = buffer.add(_tracklet(9.0))
    assert accepted is False

    clock.advance(0.1)
    assert buffer.pop_ready_batches() == []  # nada novo para liberar


def test_default_policy_is_drop():
    assert AppConfig().association.late_message_policy is LateMessagePolicy.DROP
