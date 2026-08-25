"""Escolhe UKF ou EKF por config (`filter.type`), atras da mesma interface
(filtering/base.py::FilterTracker) - mesmo padrao de
`fusion/base.py::create_fusion_strategy` e
`filtering/motion_models.py::create_motion_model`."""

from __future__ import annotations

import numpy as np

from filtering.base import FilterTracker
from filtering.ekf import EkfTracker
from filtering.motion_models import MotionModel
from filtering.ukf import UkfTracker
from models.enums import FilterType


def create_filter_tracker(
    filter_type: FilterType,
    *,
    initial_state: np.ndarray,
    initial_covariance: np.ndarray,
    motion_model: MotionModel,
    process_noise_acceleration_std: float,
) -> FilterTracker:
    if filter_type is FilterType.UKF:
        return UkfTracker(initial_state, initial_covariance, motion_model, process_noise_acceleration_std)
    if filter_type is FilterType.EKF:
        return EkfTracker(initial_state, initial_covariance, motion_model, process_noise_acceleration_std)
    raise ValueError(f"tipo de filtro desconhecido: {filter_type}")
