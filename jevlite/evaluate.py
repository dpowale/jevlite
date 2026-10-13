"""Evaluate accuracy / calibration per language and fit temperature scaling.

Dataset: JSONL, one record per line:
  {"id": "...", "lang": "es", "task": "sentiment", "state": "...",
   "question": {"type": "noul", "prompt": "..."}, "label": true}

label: bool (noul) | option string (choice) | int level 1..N (score)
"""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from pathlib import Path

from pydantic import TypeAdapter

from .calibration import Calibration, apply_temperature, brier, ece, fit_temperature, risk_coverage
from .engine import Engine, render_state
from .schemas import Question

_QUESTION = TypeAdapter(Question)


def _label_index(q, label) -> int:
    if q.type == "noul":
        return 0 if label else 1
    if q.type == "choice":
        return q.options.index(label)
    return int(label) - 1


async def collect(engine: Engine, path: str | Path, permutations: int = 1) -> list[dict]:
    records = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    questions = []
    for r in records:
        q = _QUESTION.validate_python(r["question"])
        q.task = r.get("task")
        questions.append(q)
    dists = await asyncio.gather(
        *(engine.raw_distribution(render_state(r["state"]), q, permutations) for r, q in zip(records, questions))
    )
    rows = []
    for r, q, (probs, cov) in zip(records, questions, dists):
        rows.append(
            {
                "id": r["id"],
                "lang": r.get("lang", "?"),
                "task": r.get("task") or q.type,
                "probs": probs,
                "coverage": cov,
                "y": _label_index(q, r["label"]),
            }
        )
    return rows


def metrics(rows: list[dict], temps: dict[str, float]) -> dict:
    probs = [apply_temperature(r["probs"], temps.get(r["task"], 1.0)) for r in rows]
    labels = [r["y"] for r in rows]
    conf = [max(p) for p in probs]
    correct = [max(range(len(p)), key=p.__getitem__) == y for p, y in zip(probs, labels)]
    return {
        "n": len(rows),
        "accuracy": sum(correct) / max(len(rows), 1),
        "ece": ece(conf, correct, bins=5),
        "brier": brier(probs, labels),
        "mean_conf": sum(conf) / max(len(conf), 1),
        "risk_coverage": risk_coverage(conf, correct) if rows else [],
    }


def fit(rows: list[dict]) -> Calibration:
    by_task: dict[str, list] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append((r["probs"], r["y"]))
    return Calibration({t: round(fit_temperature(s), 3) for t, s in by_task.items()})


def report(rows: list[dict]) -> tuple[str, Calibration]:
    if not rows:
        return "no rows to evaluate (empty dataset)", Calibration()
    fit_rows, test_rows = rows[0::2], rows[1::2]  # deterministic split
    cal = fit(fit_rows)
    before, after = metrics(test_rows, {}), metrics(test_rows, cal.temps)
    out = []
    out.append(f"examples: {len(rows)}  (fit {len(fit_rows)} / held-out {len(test_rows)})")
    out.append(f"mean coverage on valid answer tokens: {sum(r['coverage'] for r in rows) / len(rows):.3f}")
    out.append("")
    out.append("per-language accuracy (uncalibrated argmax, all rows):")
    langs: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        langs[r["lang"]].append(max(range(len(r["probs"])), key=r["probs"].__getitem__) == r["y"])
    for lang in sorted(langs):
        v = langs[lang]
        out.append(f"  {lang:>3}  {sum(v)}/{len(v)}")
    out.append("")
    out.append(f"fitted temperatures: {cal.temps}")
    out.append("held-out metrics        before   after")
    for k in ("accuracy", "ece", "brier", "mean_conf"):
        out.append(f"  {k:<18}  {before[k]:.3f}   {after[k]:.3f}")
    out.append("")
    out.append("risk-coverage after calibration (answer only the most confident X%):")
    for cov, acc in after["risk_coverage"]:
        out.append(f"  coverage {cov:4.0%}  accuracy {acc:.3f}")
    if len(rows) < 100:
        out.append("")
        out.append("NOTE: tiny sample - treat as a smoke test. Use 100+ labeled examples per task.")
    return "\n".join(out), cal
