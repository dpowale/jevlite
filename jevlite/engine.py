"""Decision engine: typed question -> prompt -> first-token logprobs -> calibrated distribution.

Multilingual design choices
  * The answer alphabet is language-neutral (Yes/No, A/B/C..., 1..N), so the model
    never has to emit a word in the input's language.
  * Instructions are English; the state can be any language and is never translated.
  * Candidate tokens are NFKC-normalised and case-folded, so full-width forms
    (Ａ, １) common in CJK text fold onto their ASCII equivalents.
  * Option text may be in any language; it is only ever *read* by the model.
"""
from __future__ import annotations

import asyncio
import json
import math
import string
import unicodedata

from .backend import Backend
from .calibration import Calibration, apply_temperature, softmax
from .schemas import Answer, Choice, DecideRequest, DecideResponse, Noul, Score

SYSTEM = (
    "You are a fast decision classifier inside a software system. "
    "The text you evaluate may be in any language: understand it in its original "
    "language and do not translate it. Reply with exactly one allowed symbol and "
    "nothing else - no explanation, no punctuation."
)


def _norm(tok: str) -> str:
    return unicodedata.normalize("NFKC", tok).strip().casefold()


def fold_distribution(top: dict[str, float], symbols: list[set[str]]) -> tuple[list[float], float]:
    """Sum probability over token variants per symbol, renormalise over valid symbols.

    Returns (distribution, coverage) where coverage is the raw probability mass
    the model placed on valid answers (low coverage = the model wanted to say
    something else, so don't trust the distribution).
    """
    masses = [0.0] * len(symbols)
    for tok, lp in top.items():
        key = _norm(tok)
        for i, variants in enumerate(symbols):
            if key in variants:
                masses[i] += math.exp(lp)
                break
    total = sum(masses)
    if total <= 0:
        return [1.0 / len(symbols)] * len(symbols), 0.0
    return [m / total for m in masses], min(total, 1.0)


def render_state(state: str | dict) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)  # keep non-ASCII readable


def build_prompt(state: str, q: Noul | Choice | Score, options: list[str] | None = None) -> str:
    # State first, question last: Ollama can reuse the cached KV prefix when several
    # questions share one state.
    head = f"<state>\n{state}\n</state>\n\nQuestion: {q.prompt}\n"
    if isinstance(q, Noul):
        return head + "Answer with exactly one word: Yes or No.\nAnswer:"
    if isinstance(q, Choice):
        opts = options if options is not None else q.options
        lines = "\n".join(f"{string.ascii_uppercase[i]}. {o}" for i, o in enumerate(opts))
        return head + f"Options:\n{lines}\nAnswer with the single letter of the best option.\nAnswer:"
    labels = q.rubric or [None] * q.levels
    lines = "\n".join(f"{i + 1}. {lab}" if lab else f"{i + 1}" for i, lab in enumerate(labels))
    return (
        head
        + f"Scale from 1 (lowest) to {q.levels} (highest):\n{lines}\n"
        + f"Answer with a single digit from 1 to {q.levels}.\nAnswer:"
    )


def _symbols(q: Noul | Choice | Score) -> list[set[str]]:
    if isinstance(q, Noul):
        return [{"yes", "y"}, {"no", "n"}]
    if isinstance(q, Choice):
        return [{string.ascii_uppercase[i].casefold()} for i in range(len(q.options))]
    return [{str(i + 1)} for i in range(q.levels)]


def option_texts(q: Noul | Choice | Score) -> list[str]:
    """Option strings handed to a pointer model (non-generative backends)."""
    if isinstance(q, Noul):
        return ["yes", "no"]
    if isinstance(q, Choice):
        return list(q.options)
    labels = q.rubric or [None] * q.levels
    return [f"{i + 1}: {lab}" if lab else str(i + 1) for i, lab in enumerate(labels)]


class Engine:
    def __init__(
        self,
        backend: Backend,
        calibration: Calibration | None = None,
        act_threshold: float = 0.85,
        confirm_threshold: float = 0.60,
        min_coverage: float = 0.50,
        concurrency: int = 4,
    ):
        self.backend = backend
        self.calibration = calibration or Calibration()
        self.act_threshold = act_threshold
        self.confirm_threshold = confirm_threshold
        self.min_coverage = min_coverage
        self._sem = asyncio.Semaphore(concurrency)
        self._cache: dict[tuple[str, str], dict[str, float]] = {}

    async def _top(self, user: str) -> dict[str, float]:
        key = (SYSTEM, user)
        if key not in self._cache:
            async with self._sem:
                self._cache[key] = await self.backend.first_token_logprobs(SYSTEM, user)
            if len(self._cache) > 2048:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[key]

    async def raw_distribution(
        self, state: str, q: Noul | Choice | Score, permutations: int = 1
    ) -> tuple[list[float], float]:
        """Uncalibrated probabilities in canonical order (yes,no | options | levels).

        Two backend styles are supported:
          * token backends  (first_token_logprobs): prompt + answer letters/digits
          * pointer backends (score_options): score the option texts directly; there is no
            answer alphabet, so coverage is always 1.0 and permutations are not needed.
        """
        score_options = getattr(self.backend, "score_options", None)
        if score_options is not None:
            async with self._sem:
                logits = await score_options(state, q.prompt, option_texts(q))
            return softmax(list(logits)), 1.0
        if isinstance(q, Choice):
            n = len(q.options)
            k = max(1, min(permutations, n))
            probs, cov = [0.0] * n, 0.0
            for r in range(k):
                order = list(range(r, n)) + list(range(r))  # cyclic rotation
                user = build_prompt(state, q, [q.options[i] for i in order])
                dist, c = fold_distribution(await self._top(user), _symbols(q))
                for pos, idx in enumerate(order):
                    probs[idx] += dist[pos] / k  # map back to original option order
                cov += c / k
            return probs, cov
        return fold_distribution(await self._top(build_prompt(state, q)), _symbols(q))

    async def calibrated_distribution(
        self, state: str, q: Noul | Choice | Score, permutations: int = 1
    ) -> list[float]:
        raw, _ = await self.raw_distribution(state, q, permutations)
        return apply_temperature(raw, self.calibration.temperature(q.task, q.type))

    def _route(self, confidence: float, coverage: float) -> str:
        if coverage < self.min_coverage:
            return "escalate"
        if confidence >= self.act_threshold:
            return "act"
        if confidence >= self.confirm_threshold:
            return "confirm"
        return "escalate"

    async def _answer(self, q: Noul | Choice | Score, state: str, permutations: int) -> Answer:
        raw, coverage = await self.raw_distribution(state, q, permutations)
        probs = apply_temperature(raw, self.calibration.temperature(q.task, q.type))
        ranked = sorted(probs, reverse=True)
        conf, margin = ranked[0], ranked[0] - ranked[1]
        top = max(range(len(probs)), key=probs.__getitem__)
        expected = None
        if isinstance(q, Noul):
            value, keyed = probs[0] >= probs[1], {"true": probs[0], "false": probs[1]}
        elif isinstance(q, Choice):
            value, keyed = q.options[top], dict(zip(q.options, probs))
        else:
            value = top + 1
            keyed = {str(i + 1): p for i, p in enumerate(probs)}
            expected = sum((i + 1) * p for i, p in enumerate(probs))
        return Answer(
            value=value,
            probabilities={k: round(v, 6) for k, v in keyed.items()},
            confidence=round(conf, 6),
            margin=round(margin, 6),
            coverage=round(coverage, 6),
            action=self._route(conf, coverage),
            expected=None if expected is None else round(expected, 4),
        )

    async def decide(self, req: DecideRequest) -> DecideResponse:
        state = render_state(req.state)
        ids = list(req.questions)
        results = await asyncio.gather(
            *(self._answer(req.questions[i], state, req.permutations) for i in ids)
        )
        return DecideResponse(model=self.backend.name, answers=dict(zip(ids, results)))
