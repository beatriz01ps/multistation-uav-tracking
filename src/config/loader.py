"""Carrega AppConfig de um arquivo YAML (ou usa os defaults se nenhum
caminho for passado)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import yaml

from config.models import AppConfig


def load_config(path: Optional[Union[str, Path]] = None) -> AppConfig:
    if path is None:
        return AppConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(raw)
