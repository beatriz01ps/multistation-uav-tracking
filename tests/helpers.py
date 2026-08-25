"""Utilitarios compartilhados entre testes."""

from __future__ import annotations


class FakeClock:
    """Clock controlavel manualmente, para testes deterministicos de
    codigo que depende de tempo (TrackletBuffer, Tracker)."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt
