"""FastAPI wrapper. Run: uvicorn jevlite.server:app --port 8000

Env vars: JEV_POINTER (pointer .npz; non-generative), JEV_HF_MODEL (logits-only HF LM),
          JEV_MODEL + OLLAMA_HOST (Ollama, default), JEV_CALIBRATION, JEV_ACT, JEV_CONFIRM,
          JEV_MAX_CONFIRM_SET
Backend precedence: JEV_POINTER > JEV_HF_MODEL > Ollama.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from .backend import BackendError, OllamaBackend
from .calibration import Calibration
from .engine import Engine
from .schemas import DecideRequest, DecideResponse


class HealthResponse(BaseModel):
    ok: bool = Field(default=True, description="True when the decision engine is initialised and ready.")
    model: str = Field(description="Active backend model name (pointer, HF path, or Ollama tag).")

LOG_PATH = Path(os.getenv("JEV_LOG", "decisions.jsonl"))
_LOG_LOCK = asyncio.Lock()


def make_backend():
    if os.getenv("JEV_POINTER"):
        from .pointer import PointerScorer

        return PointerScorer.load(os.environ["JEV_POINTER"])
    if os.getenv("JEV_HF_MODEL"):
        from .hf_backend import HFBackend

        return HFBackend(os.environ["JEV_HF_MODEL"])
    return OllamaBackend(
        os.getenv("JEV_MODEL", "gemma3:12b"),
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )


def make_engine() -> Engine:
    return Engine(
        make_backend(),
        calibration=Calibration.load(os.getenv("JEV_CALIBRATION", "calibration.json")),
        act_threshold=float(os.getenv("JEV_ACT", "0.85")),
        confirm_threshold=float(os.getenv("JEV_CONFIRM", "0.60")),
        max_confirm_set_size=int(os.getenv("JEV_MAX_CONFIRM_SET", "2")),
    )


def _write_log_line(line: str) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


engine = make_engine()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    yield
    close = getattr(engine.backend, "aclose", None)
    if close:
        await close()


app = FastAPI(
    title="jevlite API",
    version="0.2.0",
    summary="Calibrated System-1 decision layer for multilingual text triage.",
    description=(
        "Evaluates untrusted multilingual text (`state`) against typed non-generative questions "
        "(`noul`, `choice`, `score`) and returns MAP-calibrated probabilities, Split Conformal "
        "Adaptive Prediction Sets (`prediction_set`), and routing actions (`act` | `confirm` | `escalate`)."
    ),
    openapi_tags=[
        {"name": "triage", "description": "Structured multilingual decision & routing endpoint."},
        {"name": "system", "description": "Service health and active model inspection."},
    ],
    lifespan=_lifespan,
)

_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>jevlite — Multilingual Triage Console</title>
  <style>
    :root { --bg:#0f172a; --card:#1e293b; --border:#334155; --text:#f8fafc; --muted:#94a3b8;
            --act:#16a34a; --confirm:#d97706; --escalate:#dc2626; --accent:#3b82f6; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: system-ui, -apple-system, sans-serif; background:var(--bg); color:var(--text); }
    main { max-width: 960px; margin: 2rem auto; padding: 0 1.25rem; }
    header { display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem; margin-bottom:1.5rem; }
    h1 { margin:0; font-size:1.5rem; }
    .sub { color:var(--muted); font-size:0.9rem; margin-top:0.25rem; }
    .links a { color:#93c5fd; text-decoration:none; margin-left:1rem; font-size:0.9rem; }
    .links a:hover { text-decoration:underline; }
    .card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:1.25rem; margin-bottom:1.25rem; }
    label { display:block; font-size:0.85rem; color:var(--muted); margin-bottom:0.4rem; font-weight:600; }
    textarea, select { width:100%; background:#0b1120; color:var(--text); border:1px solid var(--border);
                       border-radius:8px; padding:0.65rem; font-family:inherit; font-size:0.92rem; }
    textarea.code { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size:0.84rem; }
    .row { display:flex; gap:0.75rem; align-items:center; flex-wrap:wrap; margin-top:0.9rem; }
    button { background:var(--accent); color:#fff; border:none; border-radius:8px; padding:0.6rem 1.2rem;
             font-weight:600; cursor:pointer; font-size:0.92rem; }
    button:disabled { opacity:0.6; cursor:wait; }
    .badge { display:inline-block; padding:0.2rem 0.55rem; border-radius:999px; font-size:0.75rem; font-weight:700; text-transform:uppercase; }
    .act { background:rgba(22,163,74,.2); color:#4ade80; border:1px solid var(--act); }
    .confirm { background:rgba(217,119,6,.2); color:#fbbf24; border:1px solid var(--confirm); }
    .escalate { background:rgba(220,38,38,.2); color:#f87171; border:1px solid var(--escalate); }
    .ans-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:0.9rem; margin-bottom:1rem; }
    .ans-box { background:#0b1120; border:1px solid var(--border); border-radius:8px; padding:0.9rem; }
    .ans-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem; }
    .meta { color:var(--muted); font-size:0.8rem; line-height:1.5; }
    pre { margin:0; overflow:auto; font-size:0.8rem; color:#cbd5e1; }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>jevlite <span style="font-weight:400;color:var(--muted);font-size:1rem">System-1 Multilingual Triage</span></h1>
      <div class="sub">Active backend: <strong id="model-name">__MODEL__</strong></div>
    </div>
    <div class="links">
      <a href="/docs">OpenAPI /docs</a>
      <a href="/healthz">/healthz</a>
    </div>
  </header>

  <section class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem">
      <label for="state" style="margin:0">Input State (any language)</label>
      <select id="preset" style="width:auto;padding:0.3rem 0.6rem;font-size:0.82rem">
        <option value="es">Sample: Spanish (shipping delay)</option>
        <option value="ja">Sample: Japanese (login outage)</option>
        <option value="ar">Sample: Arabic (server down)</option>
        <option value="de">Sample: German (double charge)</option>
      </select>
    </div>
    <textarea id="state" rows="3">Mi paquete lleva dos semanas sin llegar y nadie me contesta.</textarea>

    <label for="questions" style="margin-top:0.9rem">Typed Questions JSON (noul | choice | score)</label>
    <textarea id="questions" class="code" rows="9">{
  "sentiment": {"type": "noul", "prompt": "Is the customer unhappy?", "task": "sentiment"},
  "topic": {"type": "choice", "prompt": "What is this about?", "options": ["billing", "shipping", "technical_support", "other"], "task": "intent"},
  "urgency": {"type": "score", "prompt": "How urgent is this?", "levels": 3, "rubric": ["not urgent", "somewhat urgent", "urgent"], "task": "urgency"}
}</textarea>

    <div class="row">
      <button id="run-btn" type="button">Decide (/v1/decide)</button>
      <span id="status" class="sub"></span>
    </div>
  </section>

  <section class="card" id="out-card" style="display:none">
    <label>Triage Decisions</label>
    <div id="cards" class="ans-grid"></div>
    <label>Raw JSON Response</label>
    <pre id="raw"></pre>
  </section>
</main>
<script>
  const samples = {
    es: "Mi paquete lleva dos semanas sin llegar y nadie me contesta.",
    ja: "ログインしようとするとエラーが出て、社内の誰も作業できません。至急確認してください。",
    ar: "الموقع لا يعمل ونحن نخسر المبيعات الآن!",
    de: "Mir wurde mein Abonnement diesen Monat zweimal berechnet. Bitte um Rückerstattung."
  };
  document.getElementById("preset").addEventListener("change", (e) => {
    document.getElementById("state").value = samples[e.target.value] || "";
  });
  document.getElementById("run-btn").addEventListener("click", async () => {
    const btn = document.getElementById("run-btn");
    const status = document.getElementById("status");
    btn.disabled = true;
    status.textContent = "Scoring...";
    const t0 = performance.now();
    try {
      const questions = JSON.parse(document.getElementById("questions").value);
      const body = { state: document.getElementById("state").value, lang: document.getElementById("preset").value, questions };
      const r = await fetch("/v1/decide", { method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify(body) });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || JSON.stringify(data));
      const ms = Math.round(performance.now() - t0);
      status.textContent = `Done in ${ms} ms`;
      document.getElementById("out-card").style.display = "block";
      document.getElementById("raw").textContent = JSON.stringify(data, null, 2);
      const cards = document.getElementById("cards");
      cards.innerHTML = "";
      for (const [k, a] of Object.entries(data.answers)) {
        const extra = a.expected != null ? `<br/>expected=${a.expected} spread=${a.spread}` : "";
        cards.innerHTML += `<div class="ans-box">
          <div class="ans-head"><strong>${k}</strong><span class="badge ${a.action}">${a.action}</span></div>
          <div style="font-size:1.15rem;font-weight:700;margin-bottom:0.35rem">${JSON.stringify(a.value)}</div>
          <div class="meta">conf=${a.confidence} &middot; margin=${a.margin} &middot; cov=${a.coverage}<br/>set=${JSON.stringify(a.prediction_set)}${extra}</div>
        </div>`;
      }
    } catch (err) {
      status.textContent = "Error: " + err.message;
    } finally {
      btn.disabled = false;
    }
  });
</script>
</body>
</html>"""


@app.get("/", include_in_schema=False)
async def root(request: Request):
    accept = request.headers.get("accept", "")
    if "application/json" in accept and "text/html" not in accept:
        return JSONResponse(
            {"service": "jevlite", "model": engine.backend.name, "endpoints": ["/v1/decide", "/healthz", "/docs"]}
        )
    return HTMLResponse(_INDEX_HTML.replace("__MODEL__", engine.backend.name))


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)


@app.get(
    "/healthz",
    response_model=HealthResponse,
    tags=["system"],
    summary="Liveness and active model check",
)
async def healthz() -> HealthResponse:
    return HealthResponse(ok=True, model=engine.backend.name)


@app.post(
    "/v1/decide",
    response_model=DecideResponse,
    tags=["triage"],
    summary="Evaluate typed multilingual triage questions",
    responses={502: {"description": "Upstream model backend error (e.g. Ollama unreachable or missing logprobs)."}},
)
async def decide(req: DecideRequest) -> DecideResponse:
    t0 = time.time()
    try:
        resp = await engine.decide(req)
    except BackendError as e:
        raise HTTPException(status_code=502, detail=str(e))
    # Offload JSONL audit append off the asyncio event loop; serialised via lock.
    line = (
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
    async with _LOG_LOCK:
        await asyncio.to_thread(_write_log_line, line)
    return resp
