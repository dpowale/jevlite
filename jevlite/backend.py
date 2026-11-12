"""Backends return the first-token log-probabilities for a prompt.

A backend only has to implement:

    async def first_token_logprobs(system, user) -> dict[str, float]

mapping candidate token strings to natural-log probabilities.
"""
from __future__ import annotations

from typing import Callable, Protocol

import httpx

from .calibration import logaddexp


class BackendError(RuntimeError):
    pass


class Backend(Protocol):
    name: str

    async def first_token_logprobs(self, system: str, user: str) -> dict[str, float]: ...


class OllamaBackend:
    """Reads next-token probabilities from Ollama's /api/chat.

    Needs an Ollama version that supports `logprobs` / `top_logprobs`
    (added in late 2025). Run `python -m jevlite.cli check` to verify.
    """

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        top_logprobs: int = 20,
        think: bool | None = False,
        timeout: float = 120.0,
    ):
        self.name = model
        # Ollama's /api/chat enforces 0 <= top_logprobs <= 20 (OpenAI-compatible cap).
        self.top_logprobs = min(max(top_logprobs, 1), 20)
        self.think = think
        self._client = httpx.AsyncClient(base_url=host, timeout=timeout)

    async def first_token_logprobs(self, system: str, user: str) -> dict[str, float]:
        body: dict = {
            "model": self.name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            "options": {"num_predict": 1, "temperature": 0},
        }
        if self.think is not None:
            body["think"] = self.think  # stops reasoning models from opening with <think>

        try:
            r = await self._client.post("/api/chat", json=body)
            if r.status_code == 400 and "think" in r.text.lower() and "think" in body:
                body.pop("think")  # model doesn't support the flag; retry without it
                r = await self._client.post("/api/chat", json=body)
            if r.status_code >= 400:
                detail = r.text.strip()
                raise BackendError(f"Ollama HTTP {r.status_code} (model={self.name!r}): {detail}")
        except httpx.HTTPError as e:
            raise BackendError(f"Ollama request failed: {e}") from e

        entries = r.json().get("logprobs")
        if not entries:
            raise BackendError(
                "Ollama returned no logprobs. Upgrade Ollama, or use a llama.cpp/vLLM "
                "server behind a custom Backend."
            )
        first = entries[0]
        top_list = first.get("top_logprobs") or []
        out: dict[str, float] = {}
        if top_list:
            # Ollama includes the sampled `first["token"]` inside `top_logprobs`;
            # summing across `top_logprobs` avoids double-counting `first` while
            # properly merging distinct token IDs that decode to the same string.
            for t in top_list:
                tok, lp = t["token"], t["logprob"]
                out[tok] = logaddexp(out[tok], lp) if tok in out else lp
            if first["token"] not in out:
                out[first["token"]] = first["logprob"]
        else:
            out[first["token"]] = first["logprob"]
        return out

    async def aclose(self) -> None:
        await self._client.aclose()


class StubBackend:
    """Deterministic backend for tests: fn(system, user) -> {token: logprob}."""

    def __init__(self, fn: Callable[[str, str], dict[str, float]], name: str = "stub"):
        self.name = name
        self._fn = fn
        self.calls = 0

    async def first_token_logprobs(self, system: str, user: str) -> dict[str, float]:
        self.calls += 1
        return self._fn(system, user)
