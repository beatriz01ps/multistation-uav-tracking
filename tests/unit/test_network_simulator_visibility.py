"""Testes da oclusao geometrica de linha de visada (LOS)."""

from network_simulator.visibility import line_of_sight_blocked


def test_direct_hit_through_the_obstacle_center_is_blocked():
    # observador em (0,0), alvo em (100,0), obstaculo bem no meio do caminho
    assert line_of_sight_blocked((0.0, 0.0), (100.0, 0.0), (50.0, 0.0), obstacle_radius=10.0)


def test_obstacle_far_from_the_segment_does_not_block():
    assert not line_of_sight_blocked((0.0, 0.0), (100.0, 0.0), (50.0, 100.0), obstacle_radius=10.0)


def test_obstacle_exactly_at_the_radius_boundary_blocks():
    # distancia do centro ao segmento e exatamente o raio -> ainda bloqueia (<=)
    assert line_of_sight_blocked((0.0, 0.0), (100.0, 0.0), (50.0, 10.0), obstacle_radius=10.0)


def test_obstacle_just_outside_the_radius_does_not_block():
    assert not line_of_sight_blocked((0.0, 0.0), (100.0, 0.0), (50.0, 10.01), obstacle_radius=10.0)


def test_obstacle_beyond_the_target_does_not_block_even_if_collinear():
    # obstaculo esta na mesma reta, mas ALEM do alvo - fora do segmento
    # observador->alvo, entao nao bloqueia (projecao clampada em [0,1])
    assert not line_of_sight_blocked((0.0, 0.0), (10.0, 0.0), (50.0, 0.0), obstacle_radius=5.0)


def test_obstacle_behind_the_observer_does_not_block():
    assert not line_of_sight_blocked((0.0, 0.0), (10.0, 0.0), (-50.0, 0.0), obstacle_radius=5.0)


def test_observer_and_target_at_the_same_point():
    # segmento degenerado (comprimento 0) - so importa a distancia do
    # observador/alvo ao obstaculo
    assert line_of_sight_blocked((5.0, 5.0), (5.0, 5.0), (5.0, 5.0), obstacle_radius=1.0)
    assert not line_of_sight_blocked((5.0, 5.0), (5.0, 5.0), (50.0, 50.0), obstacle_radius=1.0)
