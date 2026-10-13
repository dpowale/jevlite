"""Distillation: label unlabeled records with a teacher Engine (e.g. the Ollama-backed jevlite)
and train the pointer model on its calibrated probabilities (soft labels).

Input JSONL:  {"id", "lang"?, "task"?, "state", "question": {...}}   (no label needed)
Output JSONL: same + "soft": [p_1..p_K], "teacher_conf": max(p)
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pydantic import TypeAdapter

from .engine import Engine, render_state
from .schemas import Question

_QUESTION = TypeAdapter(Question)


async def distill(
    teacher: Engine, in_path: str | Path, out_path: str | Path, permutations: int = 1, min_conf: float = 0.0
) -> tuple[int, int]:
    records = [json.loads(l) for l in Path(in_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    questions = []
    for r in records:
        q = _QUESTION.validate_python(r["question"])
        q.task = r.get("task")
        questions.append(q)
    dists = await asyncio.gather(
        *(teacher.calibrated_distribution(render_state(r["state"]), q, permutations) for r, q in zip(records, questions))
    )
    kept = 0
    with Path(out_path).open("w", encoding="utf-8") as f:
        for r, d in zip(records, dists):
            if max(d) >= min_conf:
                f.write(json.dumps({**r, "soft": [round(x, 6) for x in d], "teacher_conf": round(max(d), 6)}, ensure_ascii=False) + "\n")
                kept += 1
    return kept, len(records)
