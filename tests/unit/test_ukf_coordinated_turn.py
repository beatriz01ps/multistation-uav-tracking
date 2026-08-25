"""O teste que realmente importa para o Coordinated Turn: prova que,
depois de observar um alvo curvando, o UKF continua extrapolando a curva
durante COASTING (predict puro, sem medicao) - nao volta a virar reta so
porque as observacoes pararam. Sem isso, `state_dim`/`augment_initial_state`
seriam só "matrizes do tamanho certo" sem nunca provar o comportamento que
motivou a feature (ver conversa: "durante coasting, o UKF consegue prever
manobra ou so reta?")."""

import numpy as np
import pytest

from filtering.motion_models import CoordinatedTurnModel, coordinated_turn_step
from filtering.ukf import UkfTracker


def test_ukf_projects_state_and_covariance_back_to_6_dimensions():
    # o resto do sistema (GlobalTrack, associacao, fusao) nunca deve saber
    # que o Coordinated Turn usa uma 7a dimensao internamente.
    model = CoordinatedTurnModel(turn_rate_process_noise_std=0.05, initial_turn_rate_std=0.3)
    tracker = UkfTracker(
        initial_state=np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0]),
        initial_covariance=np.eye(6) * 4.0,
        motion_model=model,
        process_noise_acceleration_std=1.0,
    )
    tracker.predict(1.0)
    assert tracker.state.shape == (6,)
    assert tracker.covariance.shape == (6, 6)


def test_new_track_without_any_turning_measurement_stays_close_to_constant_velocity():
    # antes de qualquer medicao real de curva, omega comeca em 0 (prior) -
    # coasting de um track RECEM-CRIADO em Coordinated Turn deve ficar
    # PERTO de Constant Velocity, mas nao e obrigado a ser IDENTICO bit a
    # bit: a incerteza inicial sobre omega (initial_turn_rate_std) nao e
    # zero, e como coordinated_turn_step e nao-linear em omega, a media
    # ponderada pelos sigma points de uma funcao nao-linear nao coincide
    # exatamente com a funcao avaliada na media (efeito de Jensen, o motivo
    # de existir "U" de Unscented no UKF). Confirmado empiricamente: esse
    # desvio cresce ~quadraticamente com initial_turn_rate_std (0.01 rad/s
    # -> desvio de 0.001; 0.3 rad/s -> desvio de ~1.2, ambos num predict de
    # 2s) - pequeno e prevesivel, nao um erro de implementacao.
    model = CoordinatedTurnModel(turn_rate_process_noise_std=0.05, initial_turn_rate_std=0.3)
    tracker = UkfTracker(
        initial_state=np.array([0.0, 0.0, 100.0, 10.0, 3.0, 0.0]),
        initial_covariance=np.eye(6) * 4.0,
        motion_model=model,
        process_noise_acceleration_std=1.0,
    )
    tracker.predict(2.0)
    assert tracker.state[0] == pytest.approx(20.0, abs=2.0)  # CV exato seria 20.0
    assert tracker.state[1] == pytest.approx(6.0, abs=2.0)  # CV exato seria 6.0


def test_coasting_after_observed_turn_extrapolates_the_curve_not_a_straight_line():
    """Cenario: um alvo faz uma curva de verdade (omega_true=0.3 rad/s).
    Alimentamos 10 medicoes reais ao longo do arco (5s de curva observada),
    depois simulamos COASTING puro por mais 2s (so predict, sem update) -
    exatamente o que Tracker._catch_up/_apply_misses fazem quando nenhuma
    estacao observa o alvo num ciclo.

    Esperado: a posicao prevista ao final do coasting fica MUITO mais perto
    da posicao verdadeira (que continuou curvando) do que uma extrapolacao
    ingenua em linha reta a partir da ultima velocidade medida - prova que
    o Coordinated Turn realmente aprendeu a taxa de giro e continua
    aplicando ela mesmo sem observacao nova."""
    omega_true = 0.3
    dt = 0.5
    rng = np.random.default_rng(0)
    measurement_covariance = np.eye(6) * 0.25

    true_state = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0, omega_true])

    model = CoordinatedTurnModel(turn_rate_process_noise_std=0.05, initial_turn_rate_std=0.3)
    tracker = UkfTracker(
        initial_state=true_state[:6],
        initial_covariance=np.eye(6) * 4.0,
        motion_model=model,
        process_noise_acceleration_std=1.0,
    )

    # fase 1: 10 ciclos de medicao real ao longo da curva (5s observando)
    for _ in range(10):
        true_state = coordinated_turn_step(true_state, dt)
        tracker.predict(dt)
        noisy_measurement = true_state[:6] + rng.normal(0.0, 0.5, size=6)
        tracker.update(noisy_measurement, measurement_covariance)

    last_measured_position = tracker.state[:3].copy()
    last_measured_velocity = tracker.state[3:6].copy()

    # fase 2: COASTING puro por 2s (4 ciclos de 0.5s), nenhuma medicao nova
    coast_steps = 4
    for _ in range(coast_steps):
        true_state = coordinated_turn_step(true_state, dt)
        tracker.predict(dt)

    true_position_after_coasting = true_state[:3]
    ct_predicted_position = tracker.state[:3]
    naive_straight_line_position = last_measured_position + last_measured_velocity * (dt * coast_steps)

    error_coordinated_turn = np.linalg.norm(ct_predicted_position - true_position_after_coasting)
    error_naive_straight_line = np.linalg.norm(naive_straight_line_position - true_position_after_coasting)

    assert error_coordinated_turn < error_naive_straight_line
    assert error_coordinated_turn < 2.0  # convergiu para perto da curva de verdade, nao so "menos errado"
