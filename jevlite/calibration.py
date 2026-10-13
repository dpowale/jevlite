"""Temperature scaling + calibration metrics, dependency-free."""
from __future__ import annotations

import json
import math
from pathlib import Path


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    ex = [math.exp(x - m) for x in logits]
    t = sum(ex)
    return [e / t for e in ex]


def apply_temperature(probs: list[float], temperature: float) -> list[float]:
    """Soften (T>1) or sharpen (T<1) a distribution. T=1 is a no-op."""
    if temperature == 1.0:
        return list(probs)
    logs = [math.log(max(p, 1e-12)) / temperature for p in probs]
    m = max(logs)
    ex = [math.exp(x - m) for x in logs]
    s = sum(ex)
    return [e / s for e in ex]


def fit_temperature(samples: list[tuple[list[float], int]]) -> float:
    """Pick T minimizing negative log-likelihood. samples = [(probs, true_index)]."""
    if not samples:
        return 1.0

    def nll(t: float) -> float:
        total = 0.0
        for probs, y in samples:
            total -= math.log(max(apply_temperature(probs, t)[y], 1e-12))
        return total / len(samples)

    grid = [math.exp(math.log(0.2) + i * (math.log(20) - math.log(0.2)) / 199) for i in range(200)]
    return min(grid, key=nll)


def ece(confidences: list[float], correct: list[bool], bins: int = 10) -> float:
    """Expected calibration error: weighted gap between confidence and accuracy per bin."""
    n = len(confidences)
    if n == 0:
        return 0.0
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if (lo < c <= hi) or (b == 0 and c == 0)]
        if not idx:
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        conf = sum(confidences[i] for i in idx) / len(idx)
        total += len(idx) / n * abs(acc - conf)
    return total


def brier(probs: list[list[float]], labels: list[int]) -> float:
    """Multiclass Brier score (lower is better)."""
    if not probs:
        return 0.0
    s = 0.0
    for p, y in zip(probs, labels):
        s += sum((pi - (1.0 if i == y else 0.0)) ** 2 for i, pi in enumerate(p))
    return s / len(probs)


def risk_coverage(confidences: list[float], correct: list[bool], points: int = 5) -> list[tuple[float, float]]:
    """(coverage, accuracy) when answering only the most confident fraction."""
    order = sorted(range(len(confidences)), key=lambda i: -confidences[i])
    out = []
    for k in range(1, points + 1):
        m = max(1, round(len(order) * k / points))
        kept = order[:m]
        out.append((m / len(order), sum(correct[i] for i in kept) / m))
    return out


class Calibration:
    """Temperatures keyed by task name, then question type, default 1.0."""

    def __init__(self, temps: dict[str, float] | None = None):
        self.temps = temps or {}

    def temperature(self, task: str | None, qtype: str) -> float:
        if task and task in self.temps:
            return self.temps[task]
        return self.temps.get(qtype, 1.0)

    @classmethod
    def load(cls, path: str | Path | None) -> "Calibration":
        if path and Path(path).exists():
            return cls(json.loads(Path(path).read_text()))
        return cls()

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.temps, indent=2, sort_keys=True))
