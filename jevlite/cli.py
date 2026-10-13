"""CLI.

Backends (pick one; default is Ollama):
  --pointer FILE.npz   non-generative pointer model (trained with `train`)
  --hf-model PATH      local Hugging Face causal LM, logits only (no generate)
  --model NAME         Ollama model (default)

  python -m jevlite.cli check
  python -m jevlite.cli decide '{"state": "...", "questions": {...}}'
  python -m jevlite.cli eval    --data data/multilingual_sample.jsonl --out calibration.json
  python -m jevlite.cli distill --unlabeled raw.jsonl --out soft.jsonl --min-conf 0.8      (teacher = LLM backend)
  python -m jevlite.cli train   --data soft.jsonl --encoder intfloat/multilingual-e5-base --prefix "query: " --out pointer_head.npz
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys

from .backend import BackendError, OllamaBackend
from .calibration import Calibration
from .engine import SYSTEM, Engine, _symbols, build_prompt, fold_distribution
from .evaluate import collect, report
from .schemas import DecideRequest, Noul


def _backend(args):
    if args.pointer:
        from .pointer import PointerScorer

        return PointerScorer.load(args.pointer)
    if args.hf_model:
        from .hf_backend import HFBackend

        return HFBackend(args.hf_model)
    return OllamaBackend(args.model, host=args.host)


async def _close(backend) -> None:
    close = getattr(backend, "aclose", None)
    if close:
        await close()


async def cmd_check(args) -> int:
    if args.pointer:
        print("check probes token logprobs; the pointer model has none. Use `eval` instead.")
        return 0
    backend = _backend(args)
    probes = {
        "en": "I absolutely love this product, it works perfectly.",
        "ja": "この製品は本当に素晴らしい、大満足です。",
        "ar": "المنتج سيئ للغاية ولن أشتريه مرة أخرى.",
    }
    q = Noul(type="noul", prompt="Is the sentiment positive?")
    try:
        for lang, text in probes.items():
            top = await backend.first_token_logprobs(SYSTEM, build_prompt(text, q))
            dist, cov = fold_distribution(top, _symbols(q))
            best = sorted(top.items(), key=lambda kv: -kv[1])[:4]
            shown = ", ".join(f"{t!r}:{math.exp(lp):.2f}" for t, lp in best)
            print(f"[{lang}] P(yes)={dist[0]:.3f} coverage={cov:.3f}  top tokens: {shown}")
    except BackendError as e:
        print(f"FAILED: {e}")
        return 1
    finally:
        await _close(backend)
    print("OK - logprobs available. Coverage should be close to 1.0; if not, try another model.")
    return 0


async def cmd_decide(args) -> int:
    raw = args.json if args.json != "-" else sys.stdin.read()
    req = DecideRequest.model_validate_json(raw)
    backend = _backend(args)
    try:
        resp = await Engine(backend, Calibration.load(args.calibration)).decide(req)
    finally:
        await _close(backend)
    print(json.dumps(resp.model_dump(), ensure_ascii=False, indent=2))
    return 0


async def cmd_eval(args) -> int:
    backend = _backend(args)
    try:
        rows = await collect(Engine(backend, concurrency=args.concurrency), args.data, args.permutations)
    finally:
        await _close(backend)
    text, cal = report(rows)
    print(text)
    if args.out:
        cal.save(args.out)
        print(f"\nwrote {args.out}")
    return 0


async def cmd_distill(args) -> int:
    from .distill import distill

    backend = _backend(args)
    try:
        teacher = Engine(backend, Calibration.load(args.calibration), concurrency=args.concurrency)
        kept, total = await distill(teacher, args.unlabeled, args.out, args.permutations, args.min_conf)
    finally:
        await _close(backend)
    print(f"kept {kept}/{total} records with teacher confidence >= {args.min_conf} -> {args.out}")
    return 0


async def cmd_train(args) -> int:
    from .encoders import make_encoder
    from .pointer import PointerHead, build_batch, read_jsonl

    records = read_jsonl(args.data)
    random.Random(args.seed).shuffle(records)
    n_val = int(len(records) * args.val_frac)
    val_recs, train_recs = records[:n_val], records[n_val:]
    if not train_recs:
        print("no training records")
        return 1
    encoder = make_encoder(args.encoder, args.pooling, args.prefix)
    train = build_batch(train_recs, encoder)
    val = build_batch(val_recs, encoder) if val_recs else None
    head = PointerHead(encoder.dim, hidden=args.hidden, key_dim=args.key_dim, seed=args.seed)
    n_params = sum(v.size for v in head.p.values())
    print(f"encoder={args.encoder} dim={encoder.dim}  head params={n_params:,}  train={len(train)} val={len(val) if val else 0}")
    stats = head.fit(train, val, epochs=args.epochs, lr=args.lr, log=print)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()}))
    head.save(args.out, {"encoder": args.encoder, "pooling": args.pooling, "prefix": args.prefix})
    print(f"wrote {args.out}. Now run `eval --pointer {args.out}` on data NOT used for training, then fit calibration.")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(prog="jevlite")
    p.add_argument("--model", default=os.getenv("JEV_MODEL", "gemma:latest"))
    p.add_argument("--host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    p.add_argument("--pointer", default=os.getenv("JEV_POINTER"))
    p.add_argument("--hf-model", default=os.getenv("JEV_HF_MODEL"))
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check")

    d = sub.add_parser("decide")
    d.add_argument("json", help="request JSON, or - for stdin")
    d.add_argument("--calibration", default="calibration.json")

    e = sub.add_parser("eval")
    e.add_argument("--data", required=True)
    e.add_argument("--out")
    e.add_argument("--permutations", type=int, default=1)
    e.add_argument("--concurrency", type=int, default=4)

    s = sub.add_parser("distill")
    s.add_argument("--unlabeled", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--calibration", default="calibration.json")
    s.add_argument("--min-conf", type=float, default=0.0)
    s.add_argument("--permutations", type=int, default=1)
    s.add_argument("--concurrency", type=int, default=4)

    t = sub.add_parser("train")
    t.add_argument("--data", required=True, help="JSONL with `label` or `soft`")
    t.add_argument("--encoder", default="hash", help="'hash' (demo) or a Hugging Face model id/path")
    t.add_argument("--pooling", default="mean", choices=["mean", "cls"])
    t.add_argument("--prefix", default="", help='e.g. "query: " for multilingual-e5')
    t.add_argument("--out", default="pointer_head.npz")
    t.add_argument("--val-frac", type=float, default=0.2)
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--hidden", type=int, default=256)
    t.add_argument("--key-dim", type=int, default=128)
    t.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    fn = {"check": cmd_check, "decide": cmd_decide, "eval": cmd_eval, "distill": cmd_distill, "train": cmd_train}[args.cmd]
    sys.exit(asyncio.run(fn(args)))


if __name__ == "__main__":
    main()
