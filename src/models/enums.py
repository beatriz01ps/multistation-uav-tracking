"""Enumeracoes compartilhadas pelo dominio de rastreamento."""

from __future__ import annotations

from enum import Enum


class TrackStatus(str, Enum):
    TENTATIVE = "tentative"  # candidato a alvo, ainda não confirmado
    CONFIRMED = "confirmed"  # alvo confirmado, recebendo medições
    COASTING = "coasting"    # sem medição recente (imediato, não pegamos nesse ciclo), estado extrapolado por predição
    LOST = "lost"            # sem medição por tempo suficiente (10s na estimação - coasting) para reduzir a confiança
    DELETED = "deleted"      # perdido por tempo suficine (+/- 20 segundos como lost)


class UpdateKind(str, Enum):
    """Distingue explicitamente uma atualizacao com medicao real de uma
    predicao pura. Nunca apresentar uma predicao como se fosse medicao -
    regra cientifica basica para nao inflar artificialmente a confianca do
    rastreamento."""

    MEASUREMENT_UPDATED = "measurement_updated"
    PREDICTION_ONLY = "prediction_only"


class LateMessagePolicy(str, Enum):
    DROP = "drop"
    LOG_ONLY = "log_only"


class AssociationMode(str, Enum):
    POSITION_ONLY = "position_only"
    FULL_STATE = "full_state"


class FusionStrategyName(str, Enum):
    INFORMATION = "information"
    COVARIANCE_INTERSECTION = "covariance_intersection"


class MotionModelName(str, Enum):
    CONSTANT_VELOCITY = "constant_velocity"
    CONSTANT_ACCELERATION = "constant_acceleration"
    COORDINATED_TURN = "coordinated_turn"


class FilterType(str, Enum):
    """UKF (Unscented) e EKF (Extended) - mesma interface publica
    (filtering/base.py::FilterTracker), escolhidos por config sem mudar
    mais nada no resto do sistema. Ver filtering/factory.py."""

    UKF = "ukf"
    EKF = "ekf"
