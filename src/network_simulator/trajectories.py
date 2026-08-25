"""Trajetoria de verdade (ground truth) para o gerador de mensagens.

Diferente de `simulation/trajectories.py` (que precomputa uma lista de
amostras num `dt` fixo, para o simulador em lockstep): aqui cada estacao
amostra a verdade no SEU proprio instante, de forma independente das
outras - por isso a trajetoria e uma funcao analitica do tempo, nao uma
lista precomputada.

Dois modelos, ambos reusando a MESMA matematica ja usada pelo filtro
central (`filtering/motion_models.py`), para nao ter uma segunda formula
de "o que e Constant Velocity"/"o que e Coordinated Turn" no projeto:

  true_state_at          - Constant Velocity puro (reta), para UAVs sem
                            manobra.
  true_state_at_turning   - trajetoria por TRECHOS de taxa de giro
                            constante (`TurnSegment`), cada trecho resolvido
                            em forma fechada por `coordinated_turn_step`.
                            Encadear um trecho com omega>0 e outro com
                            omega<0 produz uma curva em S; um trecho com
                            omega=0.0 e equivalente a reta. Depois do
                            ultimo trecho da lista, a trajetoria continua
                            reta (omega=0) indefinidamente - nunca para
                            nem "teleporta".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from filtering.motion_models import constant_velocity_transition_matrix, coordinated_turn_step


def true_state_at(initial_state, t: float) -> np.ndarray:
    """Estado verdadeiro no instante `t`, medido a partir do inicio do
    cenario (t=0), assumindo Constant Velocity perfeito desde entao."""
    initial = np.asarray(initial_state, dtype=float)
    return constant_velocity_transition_matrix(t) @ initial


@dataclass(frozen=True)
class TurnSegment:
    """Um trecho de `duration_s` segundos com taxa de giro `omega`
    (rad/s) CONSTANTE dentro do trecho - omega=0.0 e um trecho reto."""

    duration_s: float
    omega: float


def true_state_at_turning(initial_state, schedule: list, t: float) -> np.ndarray:
    """Estado verdadeiro no instante `t`, percorrendo `schedule` em ordem
    a partir de t=0. Cada trecho e resolvido em forma fechada (a mesma
    formula de `coordinated_turn_step`, exata para omega constante) -
    nunca uma integracao numerica aproximada."""
    state = np.asarray(initial_state, dtype=float).copy()
    remaining = t

    for segment in schedule:
        if remaining <= 0:
            return state
        segment_dt = min(segment.duration_s, remaining)
        state = coordinated_turn_step(np.concatenate([state, [segment.omega]]), segment_dt)[:6]
        remaining -= segment_dt

    if remaining > 0:
        # depois do ultimo trecho declarado, continua reto (omega=0).
        state = coordinated_turn_step(np.concatenate([state, [0.0]]), remaining)[:6]

    return state


def true_state_for_uav(uav, t: float) -> np.ndarray:
    """Escolhe `true_state_at` ou `true_state_at_turning` conforme
    `uav.turn_schedule` - unica fonte da verdade fisica de um UAV, usada
    tanto por quem gera OBSERVACOES (station_runner.py, com ruido) quanto
    por quem gera o REGISTRO DE VERDADE em si (main.py::_truth_loop, sem
    ruido nenhum) - ver docs/arquitetura.md, "ground truth desacoplado das
    estacoes": o registro de verdade nao pode depender de qual estacao
    esta observando, entao nao pode viver dentro de StationRunner."""
    if uav.turn_schedule:
        return true_state_at_turning(uav.initial_state, uav.turn_schedule, t)
    return true_state_at(uav.initial_state, t)
