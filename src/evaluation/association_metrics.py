"""Metricas de associacao. Usa SOMENTE as associacoes que o tracker
retornou e o `truth_labels` produzido pelo simulador - nunca o inverso; o
tracker em si nunca ve `truth_labels`.

Nota de nomenclatura (licao de um projeto anterior - ver
projeto_antigo/docs/feedback.md): `track_fragmentation` e `id_switches` sao conceitos
DIFERENTES. Fragmentacao conta quantos global_track_id distintos um UAV
recebeu no TOTAL; ID switch e um evento temporal - a troca detectada entre
uma observacao e a proxima, em ordem cronologica. As duas sao implementadas
separadamente aqui, cada uma com o nome certo.
"""

from __future__ import annotations

from collections import Counter
from typing import NamedTuple

TruthLabels = dict[tuple[str, str, float], str]


class AssociationEvent(NamedTuple):
    station_id: str
    local_track_id: str
    global_track_id: int
    timestamp: float


def majority_label_per_global_id(events: list[AssociationEvent], truth_labels: TruthLabels) -> dict[int, str]:
    votes: dict[int, Counter] = {}
    for event in events:
        true_label = truth_labels.get((event.station_id, event.local_track_id, event.timestamp))
        if true_label is None:
            continue
        votes.setdefault(event.global_track_id, Counter())[true_label] += 1
    return {gid: counter.most_common(1)[0][0] for gid, counter in votes.items()}


def association_accuracy(
    events: list[AssociationEvent], truth_labels: TruthLabels, majority_labels: dict[int, str]
) -> float:
    total = correct = 0
    for event in events:
        true_label = truth_labels.get((event.station_id, event.local_track_id, event.timestamp))
        if true_label is None:
            continue
        total += 1
        if majority_labels.get(event.global_track_id) == true_label:
            correct += 1
    return correct / total if total else float("nan")


def track_fragmentation(majority_labels: dict[int, str]) -> dict[str, int]:
    """Quantos global_track_id distintos cada UAV real recebeu no total."""
    counts: dict[str, int] = {}
    for uav in majority_labels.values():
        counts[uav] = counts.get(uav, 0) + 1
    return counts


def id_switches(events: list[AssociationEvent], truth_labels: TruthLabels) -> int:
    """Troca de identidade no sentido temporal formal: para cada UAV real,
    quantas vezes o global_track_id associado a ele mudou entre uma
    observacao e a proxima, em ordem cronologica."""
    by_uav: dict[str, list[tuple[float, int]]] = {}
    for event in events:
        true_label = truth_labels.get((event.station_id, event.local_track_id, event.timestamp))
        if true_label is None:
            continue
        by_uav.setdefault(true_label, []).append((event.timestamp, event.global_track_id))

    switches = 0
    for observations in by_uav.values():
        observations.sort(key=lambda item: item[0])
        for (_, previous_gid), (_, current_gid) in zip(observations, observations[1:]):
            if current_gid != previous_gid:
                switches += 1
    return switches
