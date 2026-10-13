import asyncio
import math
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from pydantic import ValidationError

from jevlite.backend import StubBackend
from jevlite.calibration import Calibration, apply_temperature, ece, fit_temperature
from jevlite.engine import Engine, fold_distribution
from jevlite.evaluate import collect, report
from jevlite.schemas import DecideRequest

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "multilingual_sample.jsonl"


def lp(p: float) -> float:
    return math.log(p)


def run(coro):
    return asyncio.run(coro)


def test_huggingface_loaders_respect_offline_mode(monkeypatch):
    calls = []

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(kwargs)
            return cls()

    class FakeModel:
        config = SimpleNamespace(hidden_size=4)

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(kwargs)
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = FakeTokenizer
    transformers.AutoModel = FakeModel
    transformers.AutoModelForCausalLM = FakeModel
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    from jevlite.encoders import HFEncoder
    from jevlite.hf_backend import HFBackend

    HFBackend("local-model")
    HFEncoder("local-model")

    assert len(calls) == 4
    assert all(call["local_files_only"] is True for call in calls)


def test_fold_sums_token_variants_and_reports_coverage():
    top = {"Yes": lp(0.6), " yes": lp(0.2), "No": lp(0.1), "The": lp(0.1)}
    dist, cov = fold_distribution(top, [{"yes", "y"}, {"no", "n"}])
    assert dist[0] == pytest.approx(0.8 / 0.9)
    assert cov == pytest.approx(0.9)


def test_fullwidth_tokens_fold_to_ascii():
    top = {"Ａ": lp(0.7), "Ｂ": lp(0.3)}  # full-width letters, common in CJK output
    dist, cov = fold_distribution(top, [{"a"}, {"b"}])
    assert dist == pytest.approx([0.7, 0.3])
    assert cov == pytest.approx(1.0)


def test_no_valid_tokens_gives_uniform_and_zero_coverage():
    dist, cov = fold_distribution({"Sure": lp(0.9)}, [{"yes"}, {"no"}])
    assert dist == [0.5, 0.5] and cov == 0.0


def test_noul_end_to_end():
    backend = StubBackend(lambda s, u: {"Yes": lp(0.93), "No": lp(0.05)})
    req = DecideRequest(state="muy bueno", questions={"q": {"type": "noul", "prompt": "positive?"}})
    ans = run(Engine(backend).decide(req)).answers["q"]
    assert ans.value is True
    assert ans.probabilities["true"] == pytest.approx(0.93 / 0.98, abs=1e-4)
    assert ans.action == "act"


def test_score_end_to_end_expected_value():
    backend = StubBackend(lambda s, u: {"1": lp(0.1), "2": lp(0.2), "3": lp(0.7)})
    req = DecideRequest(state="x", questions={"u": {"type": "score", "prompt": "urgent?", "levels": 3}})
    ans = run(Engine(backend).decide(req)).answers["u"]
    assert ans.value == 3
    assert ans.expected == pytest.approx(0.1 + 0.4 + 2.1)


def test_permutations_cancel_position_bias():
    # A model that always says "A" regardless of content has pure position bias.
    backend = StubBackend(lambda s, u: {"A": lp(0.97), "B": lp(0.01), "C": lp(0.01)})
    q = {"type": "choice", "prompt": "which?", "options": ["x", "y", "z"]}
    one = run(Engine(backend).decide(DecideRequest(state="s", questions={"q": q}, permutations=1)))
    three = run(Engine(backend).decide(DecideRequest(state="s", questions={"q": q}, permutations=3)))
    assert max(one.answers["q"].probabilities.values()) > 0.9
    assert max(three.answers["q"].probabilities.values()) == pytest.approx(1 / 3, abs=0.02)


def test_permutation_maps_back_to_original_options():
    # Model picks whichever letter is next to "billing" -> answer must stay "billing"
    def fn(system, user):
        for line in user.splitlines():
            if line.endswith(". billing"):
                return {line[0]: lp(0.95), "Z": lp(0.01)}
        return {}

    q = {"type": "choice", "prompt": "topic?", "options": ["shipping", "billing", "other"]}
    ans = run(Engine(StubBackend(fn)).decide(DecideRequest(state="s", questions={"q": q}, permutations=3))).answers["q"]
    assert ans.value == "billing" and ans.confidence > 0.99


def test_low_coverage_forces_escalation():
    backend = StubBackend(lambda s, u: {"Yes": lp(0.2), "No": lp(0.01), "I": lp(0.7)})
    req = DecideRequest(state="s", questions={"q": {"type": "noul", "prompt": "p?"}})
    assert run(Engine(backend).decide(req)).answers["q"].action == "escalate"


def test_temperature_softens_and_fit_recovers_overconfidence():
    assert apply_temperature([0.95, 0.05], 2.0)[0] < 0.95
    # model says 95% but is right only 70% of the time -> T should be > 1
    samples = [([0.95, 0.05], 0)] * 70 + [([0.95, 0.05], 1)] * 30
    t = fit_temperature(samples)
    assert t > 1.5
    assert apply_temperature([0.95, 0.05], t)[0] == pytest.approx(0.7, abs=0.03)


def test_ece_zero_when_calibrated():
    conf = [0.8] * 10
    correct = [True] * 8 + [False] * 2
    assert ece(conf, correct) == pytest.approx(0.0, abs=1e-9)


def test_calibration_lookup_prefers_task_then_type():
    c = Calibration({"noul": 2.0, "sentiment": 3.0})
    assert c.temperature("sentiment", "noul") == 3.0
    assert c.temperature("other", "noul") == 2.0
    assert c.temperature(None, "choice") == 1.0


def test_report_handles_empty_dataset():
    text, cal = report([])
    assert "empty" in text
    assert cal.temps == {}


def _generic_answer(system: str, user: str) -> dict[str, float]:
    """Answers any noul/choice/score prompt plausibly, without knowing the question in advance."""
    if "Yes or No" in user:
        return {"Yes": lp(0.7), "No": lp(0.3)}
    m = re.search(r"single digit from 1 to (\d+)", user)
    if m:
        n = int(m.group(1))
        probs = [0.6] + [0.4 / (n - 1)] * (n - 1) if n > 1 else [1.0]
        return {str(i + 1): lp(p) for i, p in enumerate(probs)}
    letters = re.findall(r"(?m)^([A-Z])\. ", user)
    if letters:
        n = len(letters)
        probs = [0.6] + [0.4 / (n - 1)] * (n - 1) if n > 1 else [1.0]
        return {letter: lp(p) for letter, p in zip(letters, probs)}
    return {}


def test_multilingual_sample_dataset_is_supported():
    # Exercises every row in data/multilingual_sample.jsonl end-to-end (schema -> engine -> calibration),
    # including edge cases: dict state, missing `task`, 5-option choice, 5-level/no-rubric score.
    rows = run(collect(Engine(StubBackend(_generic_answer)), DATA_PATH))
    assert len(rows) == 49
    assert all(0.0 <= r["coverage"] <= 1.0 for r in rows)
    text, cal = report(rows)
    assert "fitted temperatures" in text
    assert set(cal.temps) <= {"intent", "sentiment", "urgency"}


def test_validation_rejects_bad_requests():
    with pytest.raises(ValidationError):
        DecideRequest(state="s", questions={"q": {"type": "choice", "prompt": "p", "options": ["a", "a"]}})
    with pytest.raises(ValidationError):
        DecideRequest(state="s", questions={"q": {"type": "score", "prompt": "p", "levels": 3, "rubric": ["a"]}})
    with pytest.raises(ValidationError):
        DecideRequest(state="s", questions={})
