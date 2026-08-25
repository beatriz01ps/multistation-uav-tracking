"""Oclusao geometrica de linha de visada (LOS): uma estacao para de
observar um UAV enquanto o segmento estacao->UAV passa perto demais de um
obstaculo circular - diferente do dropout por JANELA DE TEMPO
(`scenario.DropoutWindow`), que desliga uma estacao num intervalo fixo
independente de onde o alvo esta. Aqui a visibilidade e derivada da
POSICAO relativa entre estacao, alvo e obstaculo a cada instante -
util para cenarios como "um predio/morro bloqueia a visada de uma estacao
enquanto o alvo passa atras dele", sem precisar cravar manualmente em que
segundos isso acontece.

So plano-XY (2D): a posicao Z do alvo/estacao nao entra na checagem -
suficiente para os cenarios desta bateria (obstaculo modelado como um
cilindro infinito vertical, nao uma esfera 3D)."""

from __future__ import annotations

import numpy as np


def line_of_sight_blocked(observer_xy, target_xy, obstacle_center_xy, obstacle_radius: float) -> bool:
    """True se o segmento observer->target passa a `obstacle_radius` ou
    menos do centro do obstaculo - distancia de ponto a segmento, com o
    parametro de projecao `t` limitado a [0, 1] (o ponto mais proximo tem
    que estar DENTRO do segmento, nao na reta infinita que o contem)."""
    observer = np.asarray(observer_xy, dtype=float)
    target = np.asarray(target_xy, dtype=float)
    center = np.asarray(obstacle_center_xy, dtype=float)

    segment = target - observer
    segment_length_sq = float(segment @ segment)

    if segment_length_sq == 0.0:
        closest_point = observer
    else:
        t = float((center - observer) @ segment / segment_length_sq)
        t_clamped = max(0.0, min(1.0, t))
        closest_point = observer + t_clamped * segment

    distance = float(np.linalg.norm(closest_point - center))
    return distance <= obstacle_radius
