"""Ruido de processo do modelo Constant Velocity - parametrizado, nunca
hardcoded no filtro.

Modelo assumido: "piecewise constant white acceleration" (tambem chamado
Discrete Wiener Process Acceleration - ver Bar-Shalom, Li & Kirubarajan,
"Estimation with Applications to Tracking and Navigation", sec. 6.3.2).
Fisicamente: o alvo se move a velocidade constante DENTRO de cada
intervalo `dt`, mas sofre uma pequena aceleracao aleatoria `a ~ N(0,
sigma_a^2)`, CONSTANTE durante aquele intervalo e independente entre
intervalos - representa manobras/aceleracao nao modeladas pelo CV puro.

Por que este modelo (e nao "continuous white noise acceleration"): a
outra formulacao classica usa Q = q~ * [[dt^3/3, dt^2/2],[dt^2/2, dt]],
onde `q~` e uma densidade espectral de potencia (unidade: m^2/s^3), NAO
uma variancia de aceleracao em m^2/s^4 (m/s^2 ao quadrado). O parametro de
configuracao deste projeto chama-se `process_noise_acceleration_std` e e
descrito como um desvio padrao de aceleracao em m/s^2 - usar a formula de
"continuous white noise" com esse parametro misturaria unidades
incompativeis - a formula de um modelo com a interpretacao/unidade de
outro. A formula abaixo ("piecewise constant")
e a que faz `sigma_a` ser literalmente um desvio padrao de aceleracao,
consistente com o nome e a unidade do parametro.

Para uma unica coordenada [posicao, velocidade], com Gamma = [dt^2/2, dt]^T:

    Q_axis = Gamma @ Gamma.T * sigma_a^2
           = [[dt^4/4, dt^3/2],
              [dt^3/2, dt^2  ]] * sigma_a^2

Efeito pratico em coasting (sem medicao): Q pequeno demais deixa o filtro
excessivamente confiante na predicao (covariancia cresce devagar demais,
pode rejeitar uma reaquisicao valida no gate); Q grande demais faz a
covariancia explodir rapido demais (gate fica permissivo demais, mais
alvos podem casar por acaso). `process_noise_acceleration_std` e sempre
configuravel - nunca fixo no algoritmo.
"""

from __future__ import annotations

import numpy as np


def constant_velocity_process_noise(dt: float, acceleration_std: float) -> np.ndarray:
    """`acceleration_std` em m/s^2: desvio padrao da aceleracao nao
    modelada assumida constante dentro de cada intervalo `dt`."""
    variance = acceleration_std**2
    block = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]) * variance
    q = np.zeros((6, 6))
    for axis in range(3):
        idx = [axis, axis + 3]
        q[np.ix_(idx, idx)] = block
    return q
