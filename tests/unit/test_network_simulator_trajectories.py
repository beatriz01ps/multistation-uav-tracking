"""Testes da geracao de trajetoria de verdade (ground truth) do gerador de
mensagens - Constant Velocity e Coordinated Turn por trechos."""

import numpy as np
import pytest

from network_simulator.trajectories import TurnSegment, true_state_at, true_state_at_turning


def test_zero_omega_schedule_matches_plain_constant_velocity():
    initial = [0.0, 0.0, 100.0, 10.0, 3.0, 0.0]
    schedule = [TurnSegment(duration_s=100.0, omega=0.0)]

    for t in [0.0, 1.0, 5.0, 12.5]:
        assert np.allclose(true_state_at_turning(initial, schedule, t), true_state_at(initial, t))


def test_turning_segment_curves_away_from_the_straight_line():
    initial = [0.0, 0.0, 100.0, 10.0, 0.0, 0.0]
    schedule = [TurnSegment(duration_s=10.0, omega=0.3)]

    curved = true_state_at_turning(initial, schedule, t=2.0)
    straight = true_state_at(initial, t=2.0)
    assert not np.allclose(curved[:2], straight[:2])


def test_s_curve_returns_close_to_straight_heading_after_opposite_segments():
    # omega positivo depois negativo, mesma duracao: a velocidade final
    # deve voltar a apontar na direcao original (curva em S "fecha").
    initial = [0.0, 0.0, 100.0, 10.0, 0.0, 0.0]
    schedule = [TurnSegment(duration_s=3.0, omega=0.4), TurnSegment(duration_s=3.0, omega=-0.4)]

    end_of_s = true_state_at_turning(initial, schedule, t=6.0)
    assert end_of_s[4] == pytest.approx(0.0, abs=1e-9)  # vy volta a 0
    assert end_of_s[3] == pytest.approx(10.0, abs=1e-9)  # vx volta ao modulo original
    assert end_of_s[1] > 0.0  # mas a posicao y ficou deslocada - e uma curva em S, nao um circulo fechado


def test_trajectory_continues_straight_after_the_last_scheduled_segment():
    initial = [0.0, 0.0, 100.0, 10.0, 0.0, 0.0]
    schedule = [TurnSegment(duration_s=2.0, omega=0.5)]

    state_at_end_of_schedule = true_state_at_turning(initial, schedule, t=2.0)
    state_later = true_state_at_turning(initial, schedule, t=5.0)

    # depois do ultimo trecho, e Constant Velocity com a velocidade que
    # sobrou no fim do trecho de giro - nunca continua girando nem para.
    expected = true_state_at(state_at_end_of_schedule, t=3.0)
    assert np.allclose(state_later, expected)


def test_query_time_before_any_segment_returns_the_initial_state():
    initial = [1.0, 2.0, 100.0, 10.0, 0.0, 0.0]
    schedule = [TurnSegment(duration_s=5.0, omega=0.3)]
    assert np.allclose(true_state_at_turning(initial, schedule, t=0.0), initial)


def test_multiple_segments_are_traversed_in_order():
    initial = [0.0, 0.0, 100.0, 10.0, 0.0, 0.0]
    schedule = [TurnSegment(duration_s=1.0, omega=0.5), TurnSegment(duration_s=1.0, omega=0.0)]

    # no segundo trecho (reto), a velocidade nao deve mudar mais - so a posicao.
    state_at_1 = true_state_at_turning(initial, schedule, t=1.0)
    state_at_2 = true_state_at_turning(initial, schedule, t=2.0)
    assert np.allclose(state_at_1[3:6], state_at_2[3:6])
    assert not np.allclose(state_at_1[:2], state_at_2[:2])
