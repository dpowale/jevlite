"""Distillation: label unlabeled records with a teacher Engine (e.g. the Ollama-backed jevlite)
and train the pointer model on its calibrated probabilities (soft labels).

Uncertainty-aware distillation design:
  * Filter primarily by `min_coverage` (did the teacher emit valid answer symbols?) rather
    than aggressively discarding low-confidence boundary examples (`min_conf` defaults to 0.0).
  * Persist `teacher_cov` and `teacher_entropy` so student training weights examples by
    valid-symbol coverage without losing calibrated uncertainty on ambiguous inputs, and
    so human labelers can prioritise high-entropy examples for active learning.

Input JSONL:  {"id", "lang"?, "task"?, "state", "question": {...}}   (no label needed)
Output JSONL: same + "soft": [p_1..p_K], "teacher_conf", "teacher_cov", "teacher_entropy"
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pydantic import TypeAdapter

from .calibration import apply_temperature, normalized_entropy
from .engine import Engine, render_state
from .schemas import Question

_QUESTION = TypeAdapter(Question)


async def distill(
    teacher: Engine,
    in_path: str | Path,
    out_path: str | Path,
    permutations: int = 1,
    min_conf: float = 0.0,
    min_coverage: float = 0.50,
) -> tuple[int, int]:
    records = [json.loads(l) for l in Path(in_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    questions = []
    for r in records:
        q = _QUESTION.validate_python(r["question"])
        q.task = r.get("task")
        questions.append(q)
    raw_covs = await asyncio.gather(
        *(teacher.raw_distribution(render_state(r["state"]), q, permutations) for r, q in zip(records, questions))
    )
    kept = 0
    with Path(out_path).open("w", encoding="utf-8") as f:
        for r, q, (raw, cov) in zip(records, questions, raw_covs):
            d = apply_temperature(raw, teacher.calibration.temperature(q.task, q.type))
            if cov >= min_coverage and max(d) >= min_conf:
                row = {
                    **r,
                    "soft": [round(x, 6) for x in d],
                    "teacher_conf": round(max(d), 6),
                    "teacher_cov": round(cov, 6),
                    "teacher_entropy": round(normalized_entropy(d), 6),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
    return kept, len(records)
