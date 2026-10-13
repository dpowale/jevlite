"""Step 1 of going non-generative: one forward pass, read the last-position logits, never sample.

Same Backend interface as OllamaBackend (first_token_logprobs), but no generate() call and no
server: the model never produces a token, we only look at its next-token distribution.
Needs `pip install torch transformers`.
"""
from __future__ import annotations

import asyncio
import os


class HFBackend:
    def __init__(self, path: str, device: str | None = None, top_k: int = 50, dtype: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.name = path
        self.top_k = top_k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        local_files_only = os.getenv("HF_HUB_OFFLINE") == "1"
        self.tok = AutoTokenizer.from_pretrained(path, local_files_only=local_files_only)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, local_files_only=local_files_only
        ).to(self.device).eval()

    def _run(self, system: str, user: str) -> dict[str, float]:
        torch = self._torch
        enc = self.tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,  # ignored by templates that don't use it (stops Qwen3 opening <think>)
        ).to(self.device)
        with torch.inference_mode():
            logits = self.model(**enc).logits[0, -1].float()
            lp = torch.log_softmax(logits, dim=-1)
            top = lp.topk(self.top_k)
        out: dict[str, float] = {}
        for v, i in zip(top.values.tolist(), top.indices.tolist()):
            tok = self.tok.decode([i])
            out[tok] = max(out.get(tok, float("-inf")), v)
        return out

    async def first_token_logprobs(self, system: str, user: str) -> dict[str, float]:
        return await asyncio.to_thread(self._run, system, user)

    async def aclose(self) -> None:
        pass
