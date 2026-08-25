"""Modelos de movimento: funcao de transicao de estado + ruido de processo
associado. O UKF (filtering/ukf.py) e agnostico ao modelo - so chama
`step`/`process_noise`/`augment_initial_state`. Isso e o que permite trocar
Constant Velocity por Coordinated Turn (ou, no futuro, Constant
Acceleration) sem tocar no filtro.

V1 implementava so Constant Velocity. Coordinated Turn foi adicionado
depois (ver `CoordinatedTurnModel` abaixo), para que o rastreador consiga
extrapolar uma manobra durante COASTING - nao so uma reta. Constant
Acceleration continua reservado, ainda nao implementado (mesmo motivo de
sempre: "preparar a arquitetura para permitir depois", sem implementar
antes do necessario).

Estado AUMENTADO (Coordinated Turn): o contrato externo
[x,y,z,vx,vy,vz] (6 dimensoes - LocalTracklet, GlobalTrack,
FusedMeasurement) NUNCA muda. Internamente, porem, o Coordinated Turn
precisa de uma setima dimensao - a taxa de giro `omega` (rad/s) - que e
ESTIMADA pelo filtro a partir das medicoes, nunca configurada fixa (um UAV
real pode nao girar, girar pra um lado ou pro outro, em momentos
diferentes - um valor fixo nao serviria pra rastrear isso). `state_dim`
informa ao UKF (filtering/ukf.py) quantas dimensoes internas usar;
`augment_initial_state` sabe como inicializar a(s) dimensao(oes) extra(s)
(o "prior"); `UkfTracker.state`/`.covariance` sempre projetam de volta
para as 6 dimensoes do contrato externo antes de expor o resultado pra
fora da classe - o resto do sistema (associacao, fusao, lifecycle,
serializacao) nunca fica sabendo que existe uma setima dimensao interna.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from filtering.process_noise import constant_velocity_process_noise
from models.enums import MotionModelName

STATE_DIM = 6  # dimensao do contrato EXTERNO - nunca muda, mesmo com estado interno aumentado


class MotionModel(Protocol):
    state_dim: int

    def step(self, state: np.ndarray, dt: float) -> np.ndarray: ...

    def process_noise(self, dt: float, acceleration_std: float) -> np.ndarray: ...

    def augment_initial_state(
        self, state: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def jacobian(self, state: np.ndarray, dt: float) -> np.ndarray:
        """d(step)/d(state) em torno do `state` atual - so usada pelo EKF
        (filtering/ekf.py); o UKF nunca chama isso, ele lineariza
        implicitamente via sigma points. Para um modelo LINEAR (Constant
        Velocity), o Jacobiano e a propria matriz de transicao, constante,
        independente de `state`."""
        ...


def constant_velocity_transition_matrix(dt: float) -> np.ndarray:
    """F tal que state(t+dt) = F @ state(t) para o modelo CV. Exposta como
    funcao publica (nao so dentro de ConstantVelocityModel.step) porque
    synchronization/temporal_alignment.py precisa dela tambem, para
    propagar a COVARIANCIA (F P F^T) de um tracklet local ao alinha-lo a
    um timestamp de referencia comum - nao so o estado."""
    transition = np.eye(STATE_DIM)
    for axis in range(3):
        transition[axis, axis + 3] = dt
    return transition


class ConstantVelocityModel:
    """x(k+1) = x(k) + vx(k)*dt (idem y, z); velocidade inalterada."""

    state_dim = STATE_DIM

    def step(self, state: np.ndarray, dt: float) -> np.ndarray:
        return constant_velocity_transition_matrix(dt) @ state

    def process_noise(self, dt: float, acceleration_std: float) -> np.ndarray:
        return constant_velocity_process_noise(dt, acceleration_std)

    def augment_initial_state(
        self, state: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return state, covariance  # nenhuma dimensao extra

    def jacobian(self, state: np.ndarray, dt: float) -> np.ndarray:
        return constant_velocity_transition_matrix(dt)  # modelo linear: Jacobiano = a propria F


# Abaixo de |omega| < _OMEGA_EPSILON, as formulas do Coordinated Turn tem
# uma divisao por omega que fica numericamente instavel (mesmo o LIMITE
# matematico sendo exatamente o Constant Velocity) - troca pela formula CV
# diretamente nesse caso, em vez de dividir por algo perto de zero.
_OMEGA_EPSILON = 1e-6


def coordinated_turn_step(state: np.ndarray, dt: float) -> np.ndarray:
    """Modelo Coordinated Turn classico (Bar-Shalom, Li & Kirubarajan,
    "Estimation with Applications to Tracking and Navigation", sec. 11.7):
    giro coordenado no plano xy, com taxa de giro `omega` (7a posicao do
    estado) constante DENTRO do intervalo `dt` - mas ela propria faz parte
    do estado, reestimada a cada `update()`, nao e um parametro fixo. z/vz
    seguem Constant Velocity puro (giro assumido horizontal - mudanca de
    altitude e independente da manobra em xy)."""
    x, y, z, vx, vy, vz, omega = state

    if abs(omega) < _OMEGA_EPSILON:
        new_x = x + vx * dt
        new_y = y + vy * dt
        new_vx = vx
        new_vy = vy
    else:
        sin_wt = np.sin(omega * dt)
        cos_wt = np.cos(omega * dt)
        new_x = x + (sin_wt / omega) * vx - ((1.0 - cos_wt) / omega) * vy
        new_y = y + ((1.0 - cos_wt) / omega) * vx + (sin_wt / omega) * vy
        new_vx = cos_wt * vx - sin_wt * vy
        new_vy = sin_wt * vx + cos_wt * vy

    return np.array([new_x, new_y, z + vz * dt, new_vx, new_vy, vz, omega])


def coordinated_turn_jacobian(state: np.ndarray, dt: float) -> np.ndarray:
    """Jacobiano analitico de `coordinated_turn_step` em relacao ao estado
    [x,y,z,vx,vy,vz,omega] - usado SO pelo EKF (filtering/ekf.py); o UKF
    nunca chama isso (lineariza implicitamente via sigma points, sem
    precisar de derivada fechada). Derivado a mao e VERIFICADO contra
    diferenca finita em tests/unit/test_coordinated_turn_motion_model.py -
    mas so LONGE da fronteira |omega| ~ _OMEGA_EPSILON: bem perto dela,
    `coordinated_turn_step` tem um corte artificial (por estabilidade
    numerica) que deixa a saida LOCALMENTE CONSTANTE em omega dentro desse
    raio - diferenciar numericamente o `step()` ali mediria uma derivada
    falsa (~0), nao a derivada real do modelo continuo. Este Jacobiano usa
    a derivada ANALITICA do modelo continuo verdadeiro (fisicamente
    correto), com a formula limite confirmada por continuidade: calculando
    a formula GERAL em omegas cada vez menores (1e-2 ate 1e-5, sem usar o
    atalho do limite), o resultado converge suavemente para os valores do
    limite usado aqui - prova que e o valor assintotico correto, nao um
    numero inventado.

    No limite omega->0, as derivadas parciais em relacao a omega tem forma
    fechada propria (0/0 na formula ingenua) - usa o limite analitico
    (mesmo _OMEGA_EPSILON de coordinated_turn_step), nao diferenca finita
    em tempo de execucao."""
    _, _, _, vx, vy, _, omega = state
    jacobian = np.eye(STATE_DIM + 1)
    jacobian[2, 5] = dt  # dz'/dvz

    if abs(omega) < _OMEGA_EPSILON:
        # limite omega->0 (mesmas series de Taylor documentadas no
        # docstring do modulo/PR que introduziu esta funcao):
        #   a = sin(w dt)/w -> dt          da/dw -> 0
        #   b = (1-cos(w dt))/w -> 0        db/dw -> dt^2/2
        a, da_dw = dt, 0.0
        b, db_dw = 0.0, dt**2 / 2.0
        c, s = 1.0, 0.0  # cos(0), sin(0)
    else:
        phi = omega * dt
        s, c = np.sin(phi), np.cos(phi)
        a = s / omega
        b = (1.0 - c) / omega
        da_dw = (dt * c * omega - s) / omega**2
        db_dw = (dt * s * omega - (1.0 - c)) / omega**2

    # x' = x + a*vx - b*vy
    jacobian[0, 3] = a
    jacobian[0, 4] = -b
    jacobian[0, 6] = da_dw * vx - db_dw * vy
    # y' = y + b*vx + a*vy
    jacobian[1, 3] = b
    jacobian[1, 4] = a
    jacobian[1, 6] = db_dw * vx + da_dw * vy
    # vx' = c*vx - s*vy
    jacobian[3, 3] = c
    jacobian[3, 4] = -s
    jacobian[3, 6] = -dt * (s * vx + c * vy)
    # vy' = s*vx + c*vy
    jacobian[4, 3] = s
    jacobian[4, 4] = c
    jacobian[4, 6] = dt * (c * vx - s * vy)
    # vz' = vz, omega' = omega: ja cobertos pela identidade inicial

    return jacobian


def coordinated_turn_process_noise(
    dt: float, acceleration_std: float, turn_rate_process_noise_std: float
) -> np.ndarray:
    """Reusa o MESMO bloco de ruido cinematico do Constant Velocity (a
    incerteza de "manobra nao modelada" em x/y/z continua fazendo sentido
    aqui - o Coordinated Turn so acrescenta a MEDIA da curva, nao elimina a
    necessidade desse termo) e soma um termo extra de passeio aleatorio
    para `omega`: variancia = turn_rate_process_noise_std^2 * dt (cresce
    linearmente com o tempo, como qualquer processo de Wiener - e o que
    permite a taxa de giro estimada se ajustar aos poucos em vez de ficar
    presa no valor inicial)."""
    q = np.zeros((STATE_DIM + 1, STATE_DIM + 1))
    q[:STATE_DIM, :STATE_DIM] = constant_velocity_process_noise(dt, acceleration_std)
    q[STATE_DIM, STATE_DIM] = (turn_rate_process_noise_std**2) * dt
    return q


class CoordinatedTurnModel:
    """Estado interno aumentado [x,y,z,vx,vy,vz,omega]. `omega` e ESTIMADA
    pelo filtro a partir das medicoes (nunca fixa/configurada) - e o ponto
    inteiro do modelo: um track novo comeca assumindo omega=0 (sem giro
    conhecido) e, se o alvo realmente estiver curvando, cada `update()` vai
    puxando a estimativa de omega pra longe de zero. So DEPOIS disso que
    `predict()` (usado durante COASTING) passa a extrapolar a curva em vez
    de uma reta - antes da primeira medicao real de uma curva, um track
    novo em Coordinated Turn se comporta como Constant Velocity puro."""

    state_dim = STATE_DIM + 1

    def __init__(self, turn_rate_process_noise_std: float, initial_turn_rate_std: float) -> None:
        self._turn_rate_process_noise_std = turn_rate_process_noise_std
        self._initial_turn_rate_variance = initial_turn_rate_std**2

    def step(self, state: np.ndarray, dt: float) -> np.ndarray:
        return coordinated_turn_step(state, dt)

    def process_noise(self, dt: float, acceleration_std: float) -> np.ndarray:
        return coordinated_turn_process_noise(dt, acceleration_std, self._turn_rate_process_noise_std)

    def augment_initial_state(
        self, state: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        augmented_state = np.concatenate([state, [0.0]])
        augmented_covariance = np.zeros((self.state_dim, self.state_dim))
        augmented_covariance[:STATE_DIM, :STATE_DIM] = covariance
        augmented_covariance[STATE_DIM, STATE_DIM] = self._initial_turn_rate_variance
        return augmented_state, augmented_covariance

    def jacobian(self, state: np.ndarray, dt: float) -> np.ndarray:
        return coordinated_turn_jacobian(state, dt)


def create_motion_model(
    name: MotionModelName,
    *,
    turn_rate_process_noise_std: float = 0.05,
    initial_turn_rate_std: float = 0.3,
) -> MotionModel:
    if name == MotionModelName.CONSTANT_VELOCITY:
        return ConstantVelocityModel()
    if name == MotionModelName.COORDINATED_TURN:
        return CoordinatedTurnModel(turn_rate_process_noise_std, initial_turn_rate_std)
    raise NotImplementedError(
        f"modelo de movimento '{name.value}' e uma extensao futura prevista na arquitetura "
        "(ver filtering/motion_models.py) mas ainda nao implementada nesta versao."
    )
