import asyncio
import json
import random

import numpy as np
import pytest

from jevlite.backend import StubBackend
from jevlite.calibration import Calibration
from jevlite.distill import distill
from jevlite.encoders import HashEncoder
from jevlite.engine import Engine
from jevlite.pointer import Batch, PointerHead, PointerScorer, build_batch, read_jsonl
from jevlite.schemas import DecideRequest


def run(coro):
    return asyncio.run(coro)


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(1)
    d, n, k = 6, 5, 4
    head = PointerHead(d, hidden=7, key_dim=5, seed=3)
    mask = np.ones((n, k), dtype=bool)
    mask[:, 3] = rng.random(n) > 0.5  # exercise padding
    target = rng.random((n, k)) * mask
    target /= target.sum(1, keepdims=True)
    batch = Batch(rng.normal(size=(n, 3 * d)), rng.normal(size=(n, k, d)) * mask[..., None], mask, target)

    _, grads = head.loss_and_grads(batch)
    eps = 1e-6
    for name, g in grads.items():
        flat = np.atleast_1d(head.p[name]).reshape(-1)
        idx = rng.choice(flat.size, size=min(6, flat.size), replace=False)
        for i in idx:
            orig = head.p[name]
            plus = np.array(orig, dtype=float).copy()
            minus = plus.copy()
            plus.reshape(-1)[i] += eps
            minus.reshape(-1)[i] -= eps
            head.p[name] = plus
            lp, _ = head.loss_and_grads(batch)
            head.p[name] = minus
            lm, _ = head.loss_and_grads(batch)
            head.p[name] = orig
            numeric = (lp - lm) / (2 * eps)
            analytic = np.atleast_1d(g).reshape(-1)[i]
            assert analytic == pytest.approx(numeric, rel=1e-4, abs=1e-8), name


def test_hash_encoder_is_deterministic_normalised_and_handles_cjk():
    enc = HashEncoder(128)
    a, b = enc.encode(["手机拍照效果非常好", "手机拍照效果非常好"])
    assert np.allclose(a, b)
    assert np.linalg.norm(a) == pytest.approx(1.0)
    assert not np.allclose(enc.encode(["hola"])[0], enc.encode(["hello"])[0])


POS = ["great", "excelente", "magnifique", "wunderbar", "素晴らしい", "ممتاز"]
NEG = ["awful", "terrible", "horrible", "schrecklich", "最悪", "سيء"]
FILL = ["the", "product", "was", "really", "this", "item", "arrived", "today"]


def _sentiment_records(n, seed):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        pos = rng.random() < 0.5
        words = rng.sample(FILL, 3) + [rng.choice(POS if pos else NEG)]
        rng.shuffle(words)
        out.append(
            {"id": str(i), "task": "sentiment", "state": " ".join(words),
             "question": {"type": "noul", "prompt": "Is the opinion positive?"}, "label": pos}
        )
    return out


def test_pointer_learns_and_generalises_to_heldout_examples():
    enc = HashEncoder(256)
    train = build_batch(_sentiment_records(300, 0), enc)
    val = build_batch(_sentiment_records(100, 1), enc)
    head = PointerHead(enc.dim, hidden=64, key_dim=32, seed=0)
    stats = head.fit(train, val, epochs=120, lr=5e-3, patience=30)
    assert stats["val_acc"] > 0.9


def test_save_load_roundtrip(tmp_path):
    enc = HashEncoder(64)
    batch = build_batch(_sentiment_records(20, 2), enc)
    head = PointerHead(enc.dim, hidden=16, key_dim=8, seed=5)
    head.fit(batch, None, epochs=5)
    path = tmp_path / "h.npz"
    head.save(path, {"encoder": "hash:64"})
    loaded, meta = PointerHead.load(path)
    assert meta["encoder"] == "hash:64"
    assert np.allclose(head.logits(batch.ctx, batch.opts, batch.mask), loaded.logits(batch.ctx, batch.opts, batch.mask))
    assert PointerScorer.load(path).name == "pointer:hash:64"


def test_engine_with_pointer_backend_all_three_primitives():
    enc = HashEncoder(64)
    scorer = PointerScorer(enc, PointerHead(enc.dim, hidden=16, key_dim=8))
    req = DecideRequest(
        state="El producto llegó roto",
        questions={
            "neg": {"type": "noul", "prompt": "unhappy?"},
            "topic": {"type": "choice", "prompt": "topic?", "options": ["billing", "shipping", "other"]},
            "urg": {"type": "score", "prompt": "urgent?", "levels": 3, "rubric": ["low", "mid", "high"]},
        },
    )
    resp = run(Engine(scorer).decide(req))
    assert resp.model.startswith("pointer")
    for ans in resp.answers.values():
        assert sum(ans.probabilities.values()) == pytest.approx(1.0, abs=1e-4)
        assert ans.coverage == 1.0
    assert resp.answers["topic"].value in {"billing", "shipping", "other"}
    assert resp.answers["urg"].value in {1, 2, 3}
    assert set(resp.answers["urg"].probabilities) == {"1", "2", "3"}


def test_distill_then_train_on_soft_labels_matches_teacher(tmp_path):
    # Teacher: a stub "LLM" that reads sentiment words and answers Yes/No with calibrated-ish confidence.
    import math

    def teacher_fn(system, user):
        pos = any(w in user.split("<state>")[1].split("</state>")[0] for w in POS)
        hi, lo = (0.92, 0.05) if pos else (0.05, 0.92)
        return {"Yes": math.log(hi), "No": math.log(lo)}

    raw = _sentiment_records(300, 3)
    unlabeled = tmp_path / "raw.jsonl"
    unlabeled.write_text("\n".join(json.dumps({k: v for k, v in r.items() if k != "label"}) for r in raw), encoding="utf-8")
    soft = tmp_path / "soft.jsonl"
    kept, total = run(distill(Engine(StubBackend(teacher_fn), Calibration()), unlabeled, soft, min_conf=0.8))
    assert kept == total == 300

    recs = read_jsonl(soft)
    assert "soft" in recs[0] and len(recs[0]["soft"]) == 2
    enc = HashEncoder(256)
    head = PointerHead(enc.dim, hidden=64, key_dim=32)
    head.fit(build_batch(recs[:240], enc), build_batch(recs[240:], enc), epochs=120, lr=5e-3)

    # Student agrees with the teacher on fresh data
    fresh = _sentiment_records(100, 9)
    scorer = PointerScorer(enc, head)
    agree = 0
    for r in fresh:
        req = DecideRequest(state=r["state"], questions={"q": r["question"]})
        agree += run(Engine(scorer).decide(req)).answers["q"].value == r["label"]
    assert agree >= 90
