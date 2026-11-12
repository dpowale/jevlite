"""Temperature scaling, Split Conformal Prediction (APS), PriDe debiasing, and metrics.

All routines are pure Python / dependency-free so inference and evaluation run anywhere.

Key research techniques implemented here:
  * MAP-regularised Temperature Scaling (Guo et al., 2017 + log-normal prior):
    prevents boundary collapse (T -> 0) on small or 100%-accurate calibration splits.
  * Adaptive Prediction Sets / Split Conformal Prediction (Romano et al., 2020;
    Angelopoulos & Bates, 2023), including optional Mondrian (per-language) quantiles.
  * PriDe single-pass position-bias prior estimation & removal (Zheng et al., ICLR 2024).
  * Ordinal uncertainty (expected value, spread, and Ranked Probability Score).
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def logaddexp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b)) for folding colliding subword tokens."""
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    hi, lo = (a, b) if a >= b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))


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
    return softmax(logs)


def fit_temperature(
    samples: list[tuple[list[float], int]],
    prior_strength: float = 0.5,
    bounds: tuple[float, float] = (0.2, 20.0),
) -> float:
    """Pick T minimizing MAP negative log-likelihood with a log-normal prior around T=1.

    Objective:
        mean_NLL(T) + (prior_strength / N) * (log T)^2

    Without the prior term, small splits where empirical accuracy is 100% drive T to the
    hard lower bound (e.g. T=0.2), severely inflating confidence on unseen errors.
    Scaling the penalty by 1/N corresponds to a Gaussian prior log(T) ~ N(0, 1/(2*lambda))
    on the total log-likelihood: small splits shrink safely toward T=1, while large splits
    converge to the exact MLE temperature.
    """
    if not samples:
        return 1.0
    n = len(samples)
    lo_log, hi_log = math.log(bounds[0]), math.log(bounds[1])

    def objective(log_t: float) -> float:
        t = math.exp(log_t)
        total = 0.0
        for probs, y in samples:
            total -= math.log(max(apply_temperature(probs, t)[y], 1e-12))
        return (total + prior_strength * (log_t * log_t)) / n

    # Coarse log-spaced bracket followed by golden-section refinement.
    steps = 100
    grid = [lo_log + i * (hi_log - lo_log) / (steps - 1) for i in range(steps)]
    best_idx = min(range(steps), key=lambda i: objective(grid[i]))
    a = grid[max(0, best_idx - 1)]
    b = grid[min(steps - 1, best_idx + 1)]
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = objective(c), objective(d)
    for _ in range(32):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = objective(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = objective(d)
    return math.exp((a + b) / 2.0)


# ---- PriDe: single-pass option-position debiasing ---------------------------
def estimate_prior(dists: list[list[float]]) -> list[float]:
    """Estimate empirical position prior pi_j proportional to geometric mean across queries.

    Follows PriDe (Zheng et al., ICLR 2024): averaging log-probabilities across examples
    isolates systematic option-letter bias (e.g. favouring 'A') at 1x inference cost.
    """
    if not dists:
        return []
    k = len(dists[0])
    same_k = [d for d in dists if len(d) == k]
    if not same_k:
        return []
    mean_logs = [
        sum(math.log(max(d[j], 1e-12)) for d in same_k) / len(same_k)
        for j in range(k)
    ]
    return softmax(mean_logs)


def debias_prior(probs: list[float], prior: list[float] | None, beta: float = 1.0) -> list[float]:
    """Remove position bias in a single pass: p_tilde_i proportional to p_i / prior_i^beta."""
    if not prior or len(prior) != len(probs) or beta == 0.0:
        return list(probs)
    logits = [
        math.log(max(p, 1e-12)) - beta * math.log(max(pi, 1e-12))
        for p, pi in zip(probs, prior)
    ]
    return softmax(logits)


# ---- Split Conformal Prediction (Adaptive Prediction Sets) ------------------
def aps_nonconformity(probs: list[float], y: int) -> float:
    """Cumulative probability mass needed to include true class `y` when sorted descending."""
    order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
    cum = 0.0
    for idx in order:
        cum += probs[idx]
        if idx == y:
            return min(cum, 1.0)
    return 1.0


def fit_conformal_quantile(samples: list[tuple[list[float], int]], alpha: float = 0.10) -> float:
    """Finite-sample Split Conformal quantile q_hat guaranteeing P(Y in C(X)) >= 1 - alpha."""
    if not samples:
        return 1.0 - alpha
    n = len(samples)
    scores = sorted(aps_nonconformity(probs, y) for probs, y in samples)
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return 1.0
    return scores[max(0, rank - 1)]


def prediction_set_indices(probs: list[float], qhat: float) -> list[int]:
    """Smallest prefix of classes (most probable first) whose cumulative mass >= qhat."""
    if not probs:
        return []
    order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
    chosen: list[int] = []
    cum = 0.0
    # Tiny tolerance avoids floating-point overshoot after rounding.
    target = min(max(qhat, 0.0), 1.0) - 1e-9
    for idx in order:
        chosen.append(idx)
        cum += probs[idx]
        if cum >= target:
            break
    return chosen


# ---- Uncertainty & calibration metrics --------------------------------------
def normalized_entropy(probs: list[float]) -> float:
    """Shannon entropy scaled to [0, 1] by log(K); useful for active-learning ranking."""
    k = len(probs)
    if k <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in probs if p > 1e-15)
    return min(max(h / math.log(k), 0.0), 1.0)


def ordinal_stats(probs: list[float]) -> tuple[float, float]:
    """Return (expected_level, spread_std) for an ordered 1..K rubric distribution."""
    if not probs:
        return 0.0, 0.0
    mean = sum((i + 1) * p for i, p in enumerate(probs))
    var = sum(p * ((i + 1 - mean) ** 2) for i, p in enumerate(probs))
    return mean, math.sqrt(max(var, 0.0))


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


def rps(probs: list[list[float]], labels: list[int]) -> float:
    """Ranked Probability Score: ordinal Brier score over cumulative distributions.

    Unlike standard Brier, RPS penalises probability mass further from the true rubric
    level more heavily than off-by-one neighbours.
    """
    if not probs:
        return 0.0
    total = 0.0
    for p, y in zip(probs, labels):
        k = len(p)
        if k <= 1:
            continue
        cum_p, s = 0.0, 0.0
        for j in range(k - 1):
            cum_p += p[j]
            cum_y = 1.0 if y <= j else 0.0
            s += (cum_p - cum_y) ** 2
        total += s / (k - 1)
    return total / len(probs)


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
    """Calibration artifact storing per-task temperatures, conformal quantiles, and PriDe priors.

    Backward-compatible with flat JSON files (`{"sentiment": 1.2}`) as well as structured
    artifacts (`{"temps": {...}, "conformal": {...}, "priors": {...}, "alpha": 0.1}`).
    """

    def __init__(
        self,
        temps: dict[str, float] | None = None,
        conformal: dict[str, float] | None = None,
        priors: dict[str, list[float]] | None = None,
        alpha: float = 0.10,
    ):
        self.temps = temps or {}
        self.conformal = conformal or {}
        self.priors = priors or {}
        self.alpha = alpha

    def temperature(self, task: str | None, qtype: str) -> float:
        if task and task in self.temps:
            return self.temps[task]
        return self.temps.get(qtype, 1.0)

    def quantile(self, task: str | None, qtype: str, lang: str | None = None) -> float | None:
        """Lookup conformal APS quantile: Mondrian (task|lang) -> task -> qtype -> None."""
        if task and lang and f"{task}|{lang}" in self.conformal:
            return self.conformal[f"{task}|{lang}"]
        if lang and f"*|{lang}" in self.conformal:
            return self.conformal[f"*|{lang}"]
        if task and task in self.conformal:
            return self.conformal[task]
        return self.conformal.get(qtype)

    def prior(self, task: str | None, k: int) -> list[float] | None:
        """Lookup PriDe position prior matching task and option count `k`."""
        if task:
            scoped = self.priors.get(f"{task}:{k}") or self.priors.get(task)
            if scoped and len(scoped) == k:
                return scoped
        generic = self.priors.get(f"choice:{k}")
        return generic if generic and len(generic) == k else None

    @classmethod
    def load(cls, path: str | Path | None) -> "Calibration":
        if not path or not Path(path).exists():
            return cls()
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw, dict) and (
            "temps" in raw or "conformal" in raw or "priors" in raw
        ):
            return cls(
                temps=raw.get("temps", {}),
                conformal=raw.get("conformal", {}),
                priors=raw.get("priors", {}),
                alpha=float(raw.get("alpha", 0.10)),
            )
        return cls(temps=raw)

    def save(self, path: str | Path) -> None:
        if not self.conformal and not self.priors:
            payload: dict = self.temps
        else:
            payload = {
                "alpha": self.alpha,
                "temps": dict(sorted(self.temps.items())),
                "conformal": dict(sorted(self.conformal.items())),
                "priors": dict(sorted(self.priors.items())),
            }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
