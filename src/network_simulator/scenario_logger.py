"""Registro em disco do que o cenario faz, em tres arquivos SEPARADOS:

  sent_messages.jsonl  - exatamente o que foi enviado ao tracker pela rede
                          (o tracker nunca ve mais que isso).
  ground_truth.jsonl   - a identidade/estado verdadeiro por instante, usada
                          so para avaliacao offline depois - nunca trafega
                          pela rede.
  responses.jsonl      - a equivalencia local->global que o tracker
                          devolveu por UDP.

Escrito por multiplas threads (uma por estacao + a de resposta) - protegido
por lock, uma linha por evento, com flush imediato (throughput baixo o
suficiente nesta primeira versao para nao precisar de buffering)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Union


class ScenarioLogger:
    def __init__(self, out_dir: Union[str, Path]) -> None:
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._sent = (self._out_dir / "sent_messages.jsonl").open("w", encoding="utf-8")
        self._truth = (self._out_dir / "ground_truth.jsonl").open("w", encoding="utf-8")
        self._responses = (self._out_dir / "responses.jsonl").open("w", encoding="utf-8")

    def log_sent(self, message: dict[str, Any]) -> None:
        self._write(self._sent, message)

    def log_truth(self, true_target_id: str, timestamp: float, true_state) -> None:
        self._write(
            self._truth,
            {"timestamp": timestamp, "true_target_id": true_target_id, "true_state": list(map(float, true_state))},
        )

    def log_response(self, response: dict[str, Any]) -> None:
        self._write(self._responses, response)

    def _write(self, handle, record: dict[str, Any]) -> None:
        with self._lock:
            handle.write(json.dumps(record) + "\n")
            handle.flush()

    def close(self) -> None:
        with self._lock:
            self._sent.close()
            self._truth.close()
            self._responses.close()
