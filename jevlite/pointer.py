"""Non-generative pointer model.

Instead of asking an LLM to *write* an answer symbol, a frozen encoder embeds the state, the
question, and each candidate option; a small trainable head scores ("points at") the options.
There is no vocabulary, no sampling, and no way to emit text.

    ctx    = [h_state, h_question, h_state * h_question]          (3d)
    q      = W2 . tanh(W1 . ctx + b1) + b2                        (key_dim)
    k_j    = Wk . h_option_j                                      (key_dim)
    logit_j = exp(ls) * (q . k_j) / sqrt(key_dim)

The head is plain NumPy (about 0.7M parameters at d=768) with hand-written backprop, so training on
cached embeddings needs no GPU and no torch. Only the *encoder* (HFEncoder) needs torch.
"""
from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import TypeAdapter

from .encoders import Encoder, make_encoder
from .engine import option_texts, render_state
from .evaluate import _label_index
from .schemas import Question

_QUESTION = TypeAdapter(Question)
_PARAMS = ("W1", "b1", "W2", "b2", "Wk", "ls")


def ctx_vector(h_state: np.ndarray, h_question: np.ndarray) -> np.ndarray:
    return np.concatenate([h_state, h_question, h_state * h_question], axis=-1)


@dataclass
class Batch:
    ctx: np.ndarray  # (N, 3d)
    opts: np.ndarray  # (N, K, d), zero-padded
    mask: np.ndarray  # (N, K) bool, True = real option
    target: np.ndarray  # (N, K) rows sum to 1 over valid options

    def __len__(self) -> int:
        return len(self.ctx)

    def take(self, idx) -> "Batch":
        return Batch(self.ctx[idx], self.opts[idx], self.mask[idx], self.target[idx])


class PointerHead:
    def __init__(self, dim: int, hidden: int = 256, key_dim: int = 128, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.dim, self.hidden, self.key_dim = dim, hidden, key_dim
        self.p = {
            "W1": rng.normal(0, 1 / math.sqrt(3 * dim), (3 * dim, hidden)),
            "b1": np.zeros(hidden),
            "W2": rng.normal(0, 1 / math.sqrt(hidden), (hidden, key_dim)),
            "b2": np.zeros(key_dim),
            "Wk": rng.normal(0, 1 / math.sqrt(dim), (dim, key_dim)),
            "ls": np.array(math.log(8.0)),
        }

    # ---- forward / backward -------------------------------------------------
    def _forward(self, ctx, opts, mask):
        p = self.p
        a = np.tanh(ctx @ p["W1"] + p["b1"])
        q = a @ p["W2"] + p["b2"]
        kk = opts @ p["Wk"]
        raw = np.einsum("bk,bjk->bj", q, kk) / math.sqrt(self.key_dim)
        s = math.exp(float(p["ls"]))
        logits = np.where(mask, s * raw, -1e9)
        z = logits - logits.max(1, keepdims=True)
        e = np.exp(z) * mask
        probs = e / e.sum(1, keepdims=True)
        return probs, (a, q, kk, raw, s)

    def logits(self, ctx, opts, mask) -> np.ndarray:
        p = self.p
        a = np.tanh(ctx @ p["W1"] + p["b1"])
        q = a @ p["W2"] + p["b2"]
        raw = np.einsum("bk,bjk->bj", q, opts @ p["Wk"]) / math.sqrt(self.key_dim)
        return np.where(mask, math.exp(float(p["ls"])) * raw, -1e9)

    def loss_and_grads(self, b: Batch):
        probs, (a, q, kk, raw, s) = self._forward(b.ctx, b.opts, b.mask)
        n = len(b)
        loss = float(-(b.target * np.log(probs + 1e-12)).sum(1).mean())
        u = (probs - b.target) / n  # dL/dlogits (zero on masked entries)
        draw = u * s
        sq = math.sqrt(self.key_dim)
        dq = np.einsum("bj,bjk->bk", draw, kk) / sq
        dkk = np.einsum("bj,bk->bjk", draw, q) / sq
        da = dq @ self.p["W2"].T
        dz = da * (1.0 - a * a)
        g = {
            "Wk": np.einsum("bjd,bjk->dk", b.opts, dkk),
            "W2": a.T @ dq,
            "b2": dq.sum(0),
            "W1": b.ctx.T @ dz,
            "b1": dz.sum(0),
            "ls": np.array(float(np.sum(u * s * raw))),
        }
        return loss, g

    def evaluate(self, b: Batch) -> tuple[float, float]:
        """(negative log-likelihood, accuracy vs argmax of target)."""
        probs, _ = self._forward(b.ctx, b.opts, b.mask)
        nll = float(-(b.target * np.log(probs + 1e-12)).sum(1).mean())
        acc = float((probs.argmax(1) == b.target.argmax(1)).mean())
        return nll, acc

    # ---- training -----------------------------------------------------------
    def fit(
        self,
        train: Batch,
        val: Batch | None = None,
        epochs: int = 300,
        lr: float = 3e-3,
        batch_size: int = 64,
        label_smoothing: float = 0.05,
        weight_decay: float = 1e-4,
        patience: int = 40,
        seed: int = 0,
        log=None,
    ) -> dict:
        rng = np.random.default_rng(seed)
        k_valid = train.mask.sum(1, keepdims=True)
        smoothed = Batch(
            train.ctx,
            train.opts,
            train.mask,
            (1 - label_smoothing) * train.target + label_smoothing * train.mask / k_valid,
        )
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v2 = {k: np.zeros_like(v) for k, v in self.p.items()}
        step, best, best_epoch, best_params = 0, math.inf, 0, None
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(train))
            for i in range(0, len(order), batch_size):
                _, g = self.loss_and_grads(smoothed.take(order[i : i + batch_size]))
                step += 1
                for k in _PARAMS:
                    m[k] = 0.9 * m[k] + 0.1 * g[k]
                    v2[k] = 0.999 * v2[k] + 0.001 * g[k] ** 2
                    mh, vh = m[k] / (1 - 0.9**step), v2[k] / (1 - 0.999**step)
                    self.p[k] = self.p[k] - lr * mh / (np.sqrt(vh) + 1e-8)
                    if k in ("W1", "W2", "Wk"):
                        self.p[k] = self.p[k] * (1 - lr * weight_decay)
            if val is not None:
                nll, acc = self.evaluate(val)
                if log and epoch % 20 == 0:
                    log(f"epoch {epoch:4d}  val_nll {nll:.4f}  val_acc {acc:.3f}")
                if nll < best - 1e-5:
                    best, best_epoch = nll, epoch
                    best_params = {k: v.copy() for k, v in self.p.items()}
                elif epoch - best_epoch >= patience:
                    break
        if best_params is not None:
            self.p = best_params
        out = {"epochs_run": epoch, "best_epoch": best_epoch or epoch}
        if val is not None:
            out["val_nll"], out["val_acc"] = self.evaluate(val)
        out["train_nll"], out["train_acc"] = self.evaluate(train)
        return out

    # ---- persistence --------------------------------------------------------
    def save(self, path: str | Path, meta: dict | None = None) -> None:
        info = {"dim": self.dim, "hidden": self.hidden, "key_dim": self.key_dim, **(meta or {})}
        np.savez(path, meta=json.dumps(info), **self.p)

    @classmethod
    def load(cls, path: str | Path) -> tuple["PointerHead", dict]:
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        head = cls(meta["dim"], meta["hidden"], meta["key_dim"])
        head.p = {k: np.array(z[k]) for k in _PARAMS}
        return head, meta


# ---- data -------------------------------------------------------------------
def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def build_batch(records: list[dict], encoder: Encoder) -> Batch:
    """records need state + question and either `label` (hard) or `soft` (teacher probs)."""
    parsed = []
    for r in records:
        q = _QUESTION.validate_python(r["question"])
        parsed.append((render_state(r["state"]), q, option_texts(q), r))
    texts = sorted({t for st, q, ops, _ in parsed for t in [st, q.prompt, *ops]})
    emb = dict(zip(texts, encoder.encode(texts)))
    n, kmax, d = len(parsed), max(len(ops) for _, _, ops, _ in parsed), encoder.dim
    ctx = np.zeros((n, 3 * d))
    opts = np.zeros((n, kmax, d))
    mask = np.zeros((n, kmax), dtype=bool)
    target = np.zeros((n, kmax))
    for i, (st, q, ops, r) in enumerate(parsed):
        ctx[i] = ctx_vector(emb[st], emb[q.prompt])
        for j, o in enumerate(ops):
            opts[i, j], mask[i, j] = emb[o], True
        if "soft" in r:
            target[i, : len(ops)] = r["soft"]
        else:
            target[i, _label_index(q, r["label"])] = 1.0
    return Batch(ctx, opts, mask, target)


# ---- inference --------------------------------------------------------------
class PointerScorer:
    """Backend with `score_options`: what Engine uses instead of first_token_logprobs."""

    def __init__(self, encoder: Encoder, head: PointerHead, name: str = "pointer"):
        self.encoder, self.head, self.name = encoder, head, name
        self._emb: dict[str, np.ndarray] = {}

    def _embed(self, texts: list[str]) -> list[np.ndarray]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._emb]
        if missing:
            for t, v in zip(missing, self.encoder.encode(missing)):
                self._emb[t] = v
            if len(self._emb) > 20000:  # crude bound
                for k in list(self._emb)[:5000]:
                    self._emb.pop(k)
        return [self._emb[t] for t in texts]

    def _score(self, state: str, prompt: str, options: list[str]) -> list[float]:
        hs, hq, *ho = self._embed([state, prompt, *options])
        ctx = ctx_vector(hs, hq)[None]
        opts = np.stack(ho)[None]
        mask = np.ones((1, len(options)), dtype=bool)
        return self.head.logits(ctx, opts, mask)[0].tolist()

    async def score_options(self, state: str, prompt: str, options: list[str]) -> list[float]:
        return await asyncio.to_thread(self._score, state, prompt, options)

    @classmethod
    def load(cls, path: str | Path, encoder: Encoder | None = None) -> "PointerScorer":
        head, meta = PointerHead.load(path)
        enc = encoder or make_encoder(meta.get("encoder", "hash"), meta.get("pooling", "mean"), meta.get("prefix", ""))
        if enc.dim != head.dim:
            raise ValueError(f"encoder dim {enc.dim} != head dim {head.dim}")
        return cls(enc, head, name=f"pointer:{meta.get('encoder', '?')}")
