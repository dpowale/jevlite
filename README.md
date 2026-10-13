# jevlite

A small, local, JEV-style **System One** decision layer for multilingual text, running on top of [Ollama](https://ollama.com).

Typed questions in, calibrated probabilities out. No generated text ever leaves the system.

```
state + typed questions ──► prompt per question ──► LLM first-token logprobs
        ──► fold token variants ──► renormalise over valid answers
        ──► (option-order averaging) ──► temperature scaling ──► value + probs + confidence + action
```

## What it is (and isn't)

jevlite *approximates* a System One model by reading the next-token distribution of an instruct LLM and restricting it to a fixed answer alphabet. Real Jev is a purpose-trained non-generative model, so expect lower accuracy and speed than the real thing. Calibration (below) is what keeps the probabilities honest.

## How it works

1. You give it a `state` (e.g. a customer message, in any language) and one or more typed questions: `noul` (yes/no), `choice` (pick one of N options), or `score` (a 1..N rubric rating).
2. For each question it builds a prompt and asks a backend for only the **next-token probabilities** - it reads the distribution over the model's vocabulary instead of letting it write a sentence.
3. It folds those raw token probabilities onto the valid answer symbols (e.g. "Yes"/"yes"/full-width "Ａ" all map to the same answer), renormalises, and (for `choice`) averages over several option orderings to cancel position bias.
4. A per-task temperature (fit via `calibration.json`) rescales the result so confidence reflects actual accuracy, since raw LLM probabilities tend to be overconfident.
5. The answer comes back with `value`, `probabilities`, `confidence`, `margin`, `coverage` (how much probability mass landed on valid answers at all - a sanity check the model didn't try to go off-script), and an `action` of `act` / `confirm` / `escalate` based on configurable thresholds.

Three interchangeable backends share the same engine, API, and calibration (see [Going non-generative](#going-non-generative)): Ollama (default, reads `logprobs` from a running Ollama model), a local Hugging Face causal LM (one forward pass, no `generate()`), and a non-generative pointer model (frozen encoder + small trained head that scores option text directly, no sampled tokens at all).

## Quickstart

```bash
pip install fastapi httpx pydantic uvicorn pytest
ollama pull gemma:latest            # see the note below: `gemma3` is the newer, more multilingual family
export JEV_MODEL=gemma:latest

python -m jevlite.cli check         # verifies Ollama returns logprobs; probes en/ja/ar
python -m jevlite.cli eval --data data/multilingual_sample.jsonl --out calibration.json
uvicorn jevlite.server:app --port 8000
```

```bash
curl -s localhost:8000/v1/decide -H 'content-type: application/json' -d '{
  "state": "Mi paquete lleva dos semanas sin llegar y nadie me contesta.",
  "permutations": 4,
  "questions": {
    "sentiment": {"type": "noul", "prompt": "Is the customer unhappy?", "task": "sentiment"},
    "topic": {"type": "choice", "prompt": "What is this about?",
              "options": ["billing", "shipping", "technical_support", "other"], "task": "intent"},
    "urgency": {"type": "score", "prompt": "How urgent is this?", "levels": 3,
                "rubric": ["not urgent", "somewhat urgent", "urgent"], "task": "urgency"}
  }
}'
```

Each answer returns `value`, `probabilities`, `confidence` (max probability), `margin`, `coverage`, and `action` (`act` / `confirm` / `escalate`).

## Usage

### CLI

```bash
python -m jevlite.cli check                                               # probe backend, check logprobs/coverage
python -m jevlite.cli decide '{"state": "...", "questions": {...}}'       # one-off decision, prints JSON
python -m jevlite.cli eval --data FILE.jsonl [--out calibration.json]      # accuracy/ECE/Brier + fit calibration
python -m jevlite.cli distill --unlabeled raw.jsonl --out soft.jsonl       # label data with the LLM teacher
python -m jevlite.cli train --data soft.jsonl --out pointer_head.npz      # train the pointer head
```

`--model` / `--host` (Ollama), `--pointer FILE.npz`, and `--hf-model PATH` pick the backend for any subcommand (precedence: pointer > hf > Ollama).

### HTTP API

```bash
uvicorn jevlite.server:app --port 8000
```

- `POST /v1/decide` - body is a `DecideRequest` (`state`, `questions`, optional `permutations`), returns a `DecideResponse` with one `Answer` per question. Every request/response pair is appended to `decisions.jsonl` for audit and future training data.
- `GET /healthz` - `{"ok": true, "model": "<backend name>"}`, useful for liveness checks.

Backend and thresholds are chosen via env vars: `JEV_POINTER`, `JEV_HF_MODEL`, `JEV_MODEL` / `OLLAMA_HOST`, `JEV_CALIBRATION`, `JEV_ACT`, `JEV_CONFIRM`.

## Air-gapped installs

Prepare dependencies and model files on a connected machine, then transfer them into the isolated environment. Build the wheelhouse for the target operating system and Python version; include `torch` and `transformers` there when using either Hugging Face backend.

```bash
python -m pip download -r requirements.txt -d wheelhouse
python -m pip download torch transformers -d wheelhouse  # optional Hugging Face support
python -m pip install --no-index --find-links=wheelhouse -r requirements.txt torch transformers
```

Transfer the complete Hugging Face model snapshot (configuration, tokenizer files, and all weight shards) and use its local directory. Set `HF_HUB_OFFLINE=1` before starting JEVlite; Hugging Face loaders then use local files only and fail instead of trying the Hub if anything is missing.

```bash
export HF_HUB_OFFLINE=1
python -m jevlite.cli --hf-model /models/my-causal-lm check
```

On PowerShell, set it with `$env:HF_HUB_OFFLINE = "1"`. The default Ollama backend also works offline when Ollama and the selected model are already installed locally.

> **About `gemma:latest`.** To my knowledge this tag is the first-generation Gemma 7B, which is mostly English-focused. The newer family is `gemma3` (`gemma3:4b`, `gemma3:12b`), which is much stronger on other languages. Check what you have with `ollama show gemma:latest`, and compare both with `eval` on your languages. Switching is just `JEV_MODEL=gemma3:12b`. Gemma has no thinking mode, so Ollama rejects the `think` flag; the backend retries without it automatically.

## Question primitives

| type | answer alphabet | returns |
|---|---|---|
| `noul` | Yes / No | `true` / `false` probabilities |
| `choice` | A, B, C ... | one probability per option |
| `score` | 1 ... N (max 9), optional rubric labels | probability per level + expected level |

## Multilingual design choices

- **Language-neutral answer alphabet.** The model answers `A`, `2`, or `Yes`, never a word in the input language.
- **English instructions, original-language state.** The state is never translated.
- **Unicode folding.** Candidate tokens are NFKC-normalised and case-folded, so full-width `Ａ` / `１` map to `A` / `1`.
- **Position-bias control.** `permutations: k` averages a `choice` over k cyclic option orderings.
- **Coverage check.** If the model puts little probability on valid answers (it wanted to write prose), the answer is escalated regardless of confidence.
- **Prefix caching friendly.** State comes first and the question last, so several questions about one state can reuse Ollama's KV cache.

## Calibration

Raw LLM probabilities are usually overconfident. `eval` fits one temperature per `task` on half your labeled data and reports accuracy, ECE, Brier, and a risk-coverage curve on the other half. The server loads `calibration.json` and applies it automatically.

### How it works

- **Temperature scaling** ([calibration.py](jevlite/calibration.py)) divides each answer's log-probabilities by a single scalar `T` before renormalising: `T > 1` softens an overconfident distribution, `T < 1` sharpens an underconfident one, `T = 1` is a no-op. It doesn't change which answer wins (argmax is unaffected), only how honest the confidence number is.
- **Fitting `T`** (`fit_temperature`) does a grid search over `T` from 0.2 to 20, picking the value that minimises negative log-likelihood of the true labels under the temperature-scaled probabilities. `eval` fits one `T` per `task` (falling back to one per question `type` if a question has no `task`) and writes the result to `calibration.json`.
- **Held-out split**: `report()` fits on the even-indexed rows and evaluates on the odd-indexed rows (a deterministic 50/50 split), so the reported before/after metrics are never measured on the same examples used to fit `T`.
- **Metrics reported**: `accuracy` (argmax vs. label), `ece` (expected calibration error - the weighted gap between confidence and actual accuracy across confidence bins), `brier` (mean squared error between the full probability vector and the one-hot label), and `risk_coverage` (accuracy if you only acted on the most-confident X% of answers, for X in 20/40/60/80/100%).
- **Applied at inference**: `Engine.calibrated_distribution` looks up `Calibration.temperature(task, type)` and applies it after folding/renormalising the raw model output, before confidence/margin/action are computed.

The bundled `data/multilingual_sample.jsonl` (49 examples, 17 languages, 3 tasks, plus edge cases like dict-valued state, a question with no `task`, a 5-option `choice`, and a 5-level `score` with no rubric) is only a smoke test. For real use, hand-label **100+ examples per task**, covering the languages you expect, and re-run `eval`.

`confidence` measures how concentrated the distribution is, not whether the answer is right. Pick the `act` / `confirm` thresholds (`JEV_ACT`, `JEV_CONFIRM`) from the risk-coverage output, not by guessing.

### Example run

```
$ python -m jevlite.cli --model gemma4:latest eval --data data/multilingual_sample.jsonl --out calibration.json
examples: 49  (fit 25 / held-out 24)
mean coverage on valid answer tokens: 0.961

per-language accuracy (uncalibrated argmax, all rows):
   ar  3/3
   de  3/3
   en  9/10
   es  3/4
   ...

fitted temperatures: {'intent': 0.2, 'urgency': 6.288, 'sentiment': 0.2}
held-out metrics        before   after
  accuracy            0.792   0.792
  ece                 0.183   0.113
  brier               0.378   0.267
  mean_conf           0.975   0.904

risk-coverage after calibration (answer only the most confident X%):
  coverage  21%  accuracy 1.000
  coverage  42%  accuracy 1.000
  coverage  58%  accuracy 1.000
  coverage  79%  accuracy 0.947
  coverage 100%  accuracy 0.792

NOTE: tiny sample - treat as a smoke test. Use 100+ labeled examples per task.
wrote calibration.json
```

How to read this, and what to do next:

- **`mean coverage`** close to 1.0 means the model is almost always answering inside the allowed alphabet (Yes/No, a letter, a digit) instead of writing prose. If it's noticeably lower, try a different model or simplify the prompt/options.
- **`per-language accuracy`** tells you where the model struggles. Single-digit sample sizes (e.g. `1/1`) aren't meaningful yet - add more examples in that language before trusting the number either way.
- **`accuracy` doesn't change** between "before" and "after" calibration - temperature scaling only reshapes confidence, it never changes which answer wins (argmax is the same with or without `T`).
- **`ece` and `brier` going down** after calibration means confidence now better reflects how often the model is actually right - that's the whole point of `eval`.
- **A fitted temperature far from 1.0** (like `urgency=6.288` above) is a signal in itself: the model is very overconfident on that task and you should weight its raw confidence accordingly, or collect more labeled data for that task specifically.
- **`risk-coverage`** is the actionable part: it tells you the accuracy you'd get if you only acted automatically on the most-confident X% of answers and escalated the rest. Pick `JEV_ACT` / `JEV_CONFIRM` so that the "act" band matches an accuracy you're willing to trust unattended.
- Commit to the model/calibration pair you validated: point the server at it with `JEV_MODEL` + `JEV_CALIBRATION=calibration.json` (the file `eval --out` just wrote).

## Layout

```
jevlite/schemas.py      request/response models
jevlite/backend.py      OllamaBackend (logprobs) + StubBackend (tests)
jevlite/engine.py       prompts, token folding, permutations, routing
jevlite/calibration.py  temperature scaling, ECE, Brier, risk-coverage
jevlite/evaluate.py     dataset eval + calibration fitting
jevlite/server.py       FastAPI /v1/decide (logs decisions.jsonl)
jevlite/cli.py          check / decide / eval
jevlite/hf_backend.py   logits-only Hugging Face backend (one forward pass, no generate)
jevlite/encoders.py     HashEncoder (demo) + HFEncoder (frozen multilingual encoder)
jevlite/pointer.py      non-generative pointer head (NumPy) + training + PointerScorer
jevlite/distill.py      label unlabeled data with an LLM teacher (soft labels)
tests/                  21 tests, no Ollama / torch / downloads needed
```

## Known limits

- One Ollama call per question (times `permutations` for `choice`), not a single parallel pass as in real Jev.
- Requires an Ollama version that returns `logprobs`; `cli check` tells you. Otherwise implement `Backend` for llama.cpp or vLLM, which is about 20 lines.
- Reasoning models should have thinking disabled (`think=False` is sent by default for Ollama).
- Lower-resource languages may show lower accuracy and lower `coverage`. Check per-language results in `eval`.

## Going non-generative

Three backends share one engine, API, and calibration. Pick with a flag or env var (precedence: pointer > hf > Ollama).

| backend | how | generates tokens? |
|---|---|---|
| Ollama (default) | `num_predict=1`, reads logprobs | samples 1 token |
| `--hf-model PATH` / `JEV_HF_MODEL` | one forward pass, reads last-position logits | no |
| `--pointer FILE.npz` / `JEV_POINTER` | frozen encoder + trained pointer head scores the option texts | no, and has no vocabulary to emit |

### Pointer model (closest to a real System One model)

A frozen multilingual encoder embeds the state, the question, and each option. A small head (about 0.7M parameters at d=768, plain NumPy, trains on CPU) scores each option. Options are read as text at runtime, so there are no letter labels and no position bias, and `coverage` is always 1.0.

```bash
pip install numpy torch transformers      # torch/transformers only for the encoder

# 1. (optional) label raw, unlabeled data with the LLM teacher
python -m jevlite.cli distill --unlabeled raw.jsonl --out soft.jsonl --min-conf 0.8

# 2. train the head on cached embeddings (hard `label` or teacher `soft` targets)
python -m jevlite.cli train --data soft.jsonl --out pointer_head.npz \
    --encoder intfloat/multilingual-e5-base --prefix "query: "

# 3. calibrate on data NOT used for training, then serve
python -m jevlite.cli --pointer pointer_head.npz eval --data heldout.jsonl --out calibration.json
JEV_POINTER=pointer_head.npz uvicorn jevlite.server:app
```

Encoder choices: `intfloat/multilingual-e5-base` (use `--prefix "query: "`), `BAAI/bge-m3` (`--pooling cls`), `xlm-roberta-base`. `--encoder hash` is a char n-gram demo encoder with no cross-lingual knowledge; it only proves the pipeline.

Notes:
- Fit calibration on examples the head never trained on. Training data scores will look too good.
- Unlabeled records for `distill` use the same format as the dataset, minus `label`.
- Soft labels carry the teacher's calibrated uncertainty, which usually trains a better-calibrated student than hard labels.

### Test status

All 21 unit tests pass, including a finite-difference check of the head's gradients, an end-to-end distill-then-train test, and a dataset-support regression test that runs every row of `data/multilingual_sample.jsonl` through `collect`/`report`. `HFEncoder` and `HFBackend` need torch/transformers; their `from_pretrained` calls are covered by a mocked test for `HF_HUB_OFFLINE` behavior, but the real adapters were **not run** in the environment this was built in. Expect to adjust those two adapters for your transformers version.

## Next steps

1. Collect `decisions.jsonl` plus human corrections and add them as hard labels.
2. Retrain the pointer head as data grows and re-fit calibration; compare per-language accuracy against the LLM teacher.
3. Keep the LLM backend as the fallback for the `escalate` band.

## License

[Apache License 2.0](LICENSE)
