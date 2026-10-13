"""Text encoders used by the pointer model.

An encoder maps a list of strings to an (N, dim) float64 array of L2-normalised vectors.

  HashEncoder - dependency-free char n-gram hashing. Deterministic. For tests/demos only;
                it has no real semantic or cross-lingual knowledge.
  HFEncoder   - any Hugging Face encoder (multilingual-e5, bge-m3, xlm-roberta, mdeberta ...).
                Needs `pip install torch transformers`.
"""
from __future__ import annotations

import os
import unicodedata
import zlib
from typing import Protocol

import numpy as np


class Encoder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashEncoder:
    def __init__(self, dim: int = 512, ngrams: tuple[int, ...] = (1, 2, 3)):
        self.dim = dim
        self.ngrams = ngrams

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim)
        t = unicodedata.normalize("NFKC", text).casefold()
        grams = [f" {w} " for w in t.split()]  # word-level features
        padded = f" {t} "
        for n in self.ngrams:
            grams += [padded[i : i + n] for i in range(max(len(padded) - n + 1, 0))]
        for g in grams:
            h = zlib.crc32(g.encode("utf-8"))
            v[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0  # signed hashing
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim))


class HFEncoder:
    """Frozen Hugging Face encoder with mean (or CLS) pooling."""

    def __init__(
        self,
        path: str,
        device: str | None = None,
        max_length: int = 256,
        pooling: str = "mean",
        prefix: str = "",
        batch_size: int = 32,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        local_files_only = os.getenv("HF_HUB_OFFLINE") == "1"
        self.tok = AutoTokenizer.from_pretrained(path, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(path, local_files_only=local_files_only).to(self.device).eval()
        self.dim = int(self.model.config.hidden_size)
        self.max_length, self.pooling, self.prefix, self.batch_size = max_length, pooling, prefix, batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = [self.prefix + t for t in texts[i : i + self.batch_size]]
            enc = self.tok(
                batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            ).to(self.device)
            with torch.inference_mode():
                h = self.model(**enc).last_hidden_state
                if self.pooling == "cls":
                    v = h[:, 0]
                else:
                    m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                    v = (h * m).sum(1) / m.sum(1).clamp(min=1)
                v = torch.nn.functional.normalize(v.float(), dim=-1)
            out.append(v.cpu().numpy())
        return np.concatenate(out).astype(np.float64) if out else np.zeros((0, self.dim))


def make_encoder(spec: str, pooling: str = "mean", prefix: str = "") -> Encoder:
    """spec: 'hash' | 'hash:1024' | a Hugging Face model id / local path."""
    if spec == "hash" or spec.startswith("hash:"):
        return HashEncoder(int(spec.split(":")[1]) if ":" in spec else 512)
    return HFEncoder(spec, pooling=pooling, prefix=prefix)
