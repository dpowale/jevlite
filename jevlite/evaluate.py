"""Evaluate accuracy / calibration per language and fit temperature, conformal, and PriDe artifacts.

Dataset: JSONL, one record per line:
  {"id": "...", "lang": "es", "task": "sentiment", "state": "...",
   "question": {"type": "noul", "prompt": "..."}, "label": true}

label: bool (noul) | option string (choice) | int level 1..N (score)
"""
from __future__ import annotations

import asyncio
import json
import math
from collections import defaultdict
from pathlib import Path

from pydantic import TypeAdapter

from .calibration import (
    Calibration,
    apply_temperature,
    brier,
    debias_prior,
    ece,
    estimate_prior,
    fit_conformal_quantile,
    fit_temperature,
    prediction_set_indices,
    risk_coverage,
    rps,
)
from .engine import Engine, render_state
from .schemas import Question

_QUESTION = TypeAdapter(Question)


def _label_index(q, label) -> int:
    if q.type == "noul":
        return 0 if label else 1
    if q.type == "choice":
        return q.options.index(label)
    return int(label) - 1


def stratified_split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deterministic 50/50 split stratified by `task` so every task is balanced.

    Singleton task groups (n=1, e.g. an untagged edge-case question) go to `test_rows`
    rather than fitting a 1-sample parameter with no held-out evaluation.
    """
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    fit_rows, test_rows = [], []
    for task in sorted(by_task):
        grp = by_task[task]
        if len(grp) == 1:
            test_rows.extend(grp)
        else:
            fit_rows.extend(grp[0::2])
            test_rows.extend(grp[1::2])
    return fit_rows, test_rows


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
                "qtype": q.type,
                "probs": probs,
                "coverage": cov,
                "y": _label_index(q, r["label"]),
            }
        )
    return rows


def _calibrated_probs(r: dict, cal: Calibration) -> list[float]:
    p = r["probs"]
    if r.get("qtype") == "choice":
        p = debias_prior(p, cal.prior(r["task"], len(p)))
    return apply_temperature(p, cal.temperature(r["task"], r.get("qtype", r["task"])))


def metrics(rows: list[dict], cal_or_temps: Calibration | dict[str, float], alpha: float = 0.10) -> dict:
    cal = cal_or_temps if isinstance(cal_or_temps, Calibration) else Calibration(temps=cal_or_temps, alpha=alpha)
    probs = [_calibrated_probs(r, cal) for r in rows]
    labels = [r["y"] for r in rows]
    conf = [max(p) for p in probs]
    correct = [max(range(len(p)), key=p.__getitem__) == y for p, y in zip(probs, labels)]

    score_idx = [i for i, r in enumerate(rows) if r.get("qtype") == "score"]
    score_rps = (
        rps([probs[i] for i in score_idx], [labels[i] for i in score_idx])
        if score_idx
        else None
    )

    psets = [
        prediction_set_indices(
            p,
            cal.quantile(r["task"], r.get("qtype", r["task"]), r.get("lang")) or (1.0 - cal.alpha),
        )
        for r, p in zip(rows, probs)
    ]
    set_cov = sum(y in ps for ps, y in zip(psets, labels)) / max(len(rows), 1)
    mean_set_size = sum(len(ps) for ps in psets) / max(len(rows), 1)

    return {
        "n": len(rows),
        "accuracy": sum(correct) / max(len(rows), 1),
        "ece": ece(conf, correct, bins=5),
        "brier": brier(probs, labels),
        "score_rps": score_rps,
        "mean_conf": sum(conf) / max(len(conf), 1),
        "conformal_cov": set_cov,
        "mean_set_size": mean_set_size,
        "risk_coverage": risk_coverage(conf, correct) if rows else [],
    }


def fit(rows: list[dict], alpha: float = 0.10, mondrian_min_n: int = 10) -> Calibration:
    """Fit PriDe choice priors, MAP temperatures, and Split Conformal quantiles."""
    # 1. Fit PriDe position priors per (task, K) on choice questions.
    priors: dict[str, list[float]] = {}
    choice_groups: dict[tuple[str, int], list[list[float]]] = defaultdict(list)
    for r in rows:
        if r.get("qtype") == "choice":
            choice_groups[(r["task"], len(r["probs"]))].append(r["probs"])
    for (task, k), dists in choice_groups.items():
        if len(dists) >= 4:
            priors[f"{task}:{k}"] = [round(x, 6) for x in estimate_prior(dists)]

    # 2. Fit MAP-regularised temperature per task on PriDe-debiased distributions.
    by_task: dict[str, list[tuple[list[float], int]]] = defaultdict(list)
    for r in rows:
        p = r["probs"]
        if r.get("qtype") == "choice":
            p = debias_prior(p, priors.get(f"{r['task']}:{len(p)}"))
        by_task[r["task"]].append((p, r["y"]))
    temps = {t: round(fit_temperature(s), 3) for t, s in by_task.items()}

    # 3. Fit Split Conformal APS quantiles per task (+ Mondrian per-language if enough data).
    temp_cal = Calibration(temps=temps, priors=priors, alpha=alpha)
    conformal: dict[str, float] = {}
    cal_by_task: dict[str, list[tuple[list[float], int]]] = defaultdict(list)
    cal_by_task_lang: dict[tuple[str, str], list[tuple[list[float], int]]] = defaultdict(list)
    for r in rows:
        cp = _calibrated_probs(r, temp_cal)
        cal_by_task[r["task"]].append((cp, r["y"]))
        lang = r.get("lang")
        if lang and lang != "?":
            cal_by_task_lang[(r["task"], lang)].append((cp, r["y"]))

    min_conformal_n = max(10, math.ceil(1.0 / max(alpha, 1e-6)))
    for t, s in cal_by_task.items():
        if len(s) >= min_conformal_n:
            conformal[t] = round(fit_conformal_quantile(s, alpha=alpha), 4)
    for (t, lang), s in cal_by_task_lang.items():
        if len(s) >= max(mondrian_min_n, min_conformal_n):
            conformal[f"{t}|{lang}"] = round(fit_conformal_quantile(s, alpha=alpha), 4)

    return Calibration(temps=temps, conformal=conformal, priors=priors, alpha=alpha)


def report(rows: list[dict], alpha: float = 0.10) -> tuple[str, Calibration]:
    if not rows:
        return "no rows to evaluate (empty dataset)", Calibration()
    fit_rows, test_rows = stratified_split(rows)
    if not test_rows:
        test_rows = fit_rows
    cal = fit(fit_rows, alpha=alpha)
    before = metrics(test_rows, Calibration(alpha=alpha), alpha=alpha)
    after = metrics(test_rows, cal, alpha=alpha)
    out = []
    out.append(f"examples: {len(rows)}  (stratified fit {len(fit_rows)} / held-out {len(test_rows)})")
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
    if cal.priors:
        out.append(f"fitted PriDe priors: {cal.priors}")
    out.append(f"conformal quantiles (alpha={alpha:.2f}): {cal.conformal}")
    out.append("held-out metrics        before   after")
    for k in ("accuracy", "ece", "brier", "mean_conf"):
        out.append(f"  {k:<18}  {before[k]:.3f}   {after[k]:.3f}")
    if before["score_rps"] is not None and after["score_rps"] is not None:
        out.append(f"  {'score_rps':<18}  {before['score_rps']:.3f}   {after['score_rps']:.3f}")
    out.append(
        f"  {'conformal_cov':<18}  {before['conformal_cov']:.3f}   {after['conformal_cov']:.3f}"
        f"   (mean set size {after['mean_set_size']:.2f})"
    )
    out.append("")
    out.append("risk-coverage after calibration (answer only the most confident X%):")
    for cov, acc in after["risk_coverage"]:
        out.append(f"  coverage {cov:4.0%}  accuracy {acc:.3f}")
    if len(rows) < 100:
        out.append("")
        out.append("NOTE: tiny sample - treat as a smoke test. Use 100+ labeled examples per task.")
    return "\n".join(out), cal
