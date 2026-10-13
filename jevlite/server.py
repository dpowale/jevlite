"""FastAPI wrapper. Run: uvicorn jevlite.server:app --port 8000

Env vars: JEV_POINTER (pointer .npz; non-generative), JEV_HF_MODEL (logits-only HF LM),
          JEV_MODEL + OLLAMA_HOST (Ollama, default), JEV_CALIBRATION, JEV_ACT, JEV_CONFIRM
Backend precedence: JEV_POINTER > JEV_HF_MODEL > Ollama.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .backend import BackendError, OllamaBackend
from .calibration import Calibration
from .engine import Engine
from .schemas import DecideRequest, DecideResponse

LOG_PATH = Path(os.getenv("JEV_LOG", "decisions.jsonl"))


def make_backend():
    if os.getenv("JEV_POINTER"):
        from .pointer import PointerScorer

        return PointerScorer.load(os.environ["JEV_POINTER"])
    if os.getenv("JEV_HF_MODEL"):
        from .hf_backend import HFBackend

        return HFBackend(os.environ["JEV_HF_MODEL"])
    return OllamaBackend(
        os.getenv("JEV_MODEL", "gemma:latest"),
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )


def make_engine() -> Engine:
    return Engine(
        make_backend(),
        calibration=Calibration.load(os.getenv("JEV_CALIBRATION", "calibration.json")),
        act_threshold=float(os.getenv("JEV_ACT", "0.85")),
        confirm_threshold=float(os.getenv("JEV_CONFIRM", "0.60")),
    )


app = FastAPI(title="jevlite", version="0.1")
engine = make_engine()


@app.get("/healthz")
async def healthz():
    return {"ok": True, "model": engine.backend.name}


@app.post("/v1/decide", response_model=DecideResponse)
async def decide(req: DecideRequest) -> DecideResponse:
    t0 = time.time()
    try:
        resp = await engine.decide(req)
    except BackendError as e:
        raise HTTPException(status_code=502, detail=str(e))
    # Audit trail + future training data (needed for the later classifier phase).
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "ts": t0,
                    "ms": round((time.time() - t0) * 1000),
                    "request": req.model_dump(),
                    "response": resp.model_dump(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    return resp
