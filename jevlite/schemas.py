"""Request/response schemas.

Three question primitives, named after Jev's public ones:
  noul   - yes/no, returns P(yes)
  choice - pick one of N options, returns a distribution
  score  - position on an ordered 1..N rubric, returns a distribution
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _QBase(BaseModel):
    prompt: str = Field(min_length=1)
    # Optional calibration key. Lets you keep one temperature per task
    # (e.g. "sentiment") instead of one per question type.
    task: str | None = None


class Noul(_QBase):
    type: Literal["noul"]


class Choice(_QBase):
    type: Literal["choice"]
    options: list[str] = Field(min_length=2, max_length=26)

    @model_validator(mode="after")
    def _unique(self):
        if len(set(self.options)) != len(self.options):
            raise ValueError("options must be unique")
        return self


class Score(_QBase):
    type: Literal["score"]
    levels: int = Field(default=5, ge=2, le=9)
    rubric: list[str] | None = None  # optional label per level, lowest first

    @model_validator(mode="after")
    def _rubric_len(self):
        if self.rubric is not None and len(self.rubric) != self.levels:
            raise ValueError("rubric length must equal levels")
        return self


Question = Annotated[Union[Noul, Choice, Score], Field(discriminator="type")]


class DecideRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "state": "ログインしようとするとエラーが出て、社内の誰も作業できません。至急確認してください。",
                    "lang": "ja",
                    "permutations": 1,
                    "questions": {
                        "sentiment": {
                            "type": "noul",
                            "prompt": "Is the customer unhappy?",
                            "task": "sentiment",
                        },
                        "topic": {
                            "type": "choice",
                            "prompt": "What is this about?",
                            "options": ["billing", "shipping", "technical_support", "other"],
                            "task": "intent",
                        },
                        "urgency": {
                            "type": "score",
                            "prompt": "How urgent is this?",
                            "levels": 3,
                            "rubric": ["not urgent", "somewhat urgent", "urgent"],
                            "task": "urgency",
                        },
                    },
                }
            ]
        }
    )

    state: str | dict = Field(description="Untrusted input text or structured JSON object in any language.")
    questions: dict[str, Question] = Field(
        min_length=1,
        max_length=64,
        description="Map of question IDs to typed question definitions (`noul`, `choice`, `score`).",
    )
    # Optional ISO language tag (e.g. "es", "ar") for Mondrian (group-conditional)
    # conformal calibration when language-specific quantiles are available.
    lang: str | None = Field(
        default=None,
        description="Optional ISO 639-1 language tag (e.g. 'es', 'ja', 'ar') for Mondrian conformal quantiles.",
    )
    # For `choice` questions: average over this many balanced option orderings
    # to cancel position bias. 1 = single pass (uses PriDe prior if calibrated).
    permutations: int = Field(
        default=1,
        ge=1,
        le=26,
        description="Number of option-order rotations for `choice` (1 uses single-pass PriDe prior if calibrated).",
    )


class Answer(BaseModel):
    value: bool | int | str = Field(description="Top-1 predicted answer (`bool` for noul, `str` for choice, `int` for score).")
    probabilities: dict[str, float] = Field(description="Calibrated probability distribution over valid options.")
    confidence: float = Field(description="Maximum calibrated probability (concentration, not raw correctness).")
    margin: float = Field(description="Probability gap between top-1 and top-2 candidates.")
    coverage: float = Field(description="Raw probability mass placed on valid answer tokens before renormalisation.")
    action: Literal["act", "confirm", "escalate"] = Field(description="Automated triage routing decision.")
    # Adaptive Prediction Set (APS) of plausible answers at target coverage (1 - alpha).
    # Singletons support unattended `act`; small subsets power 1-click human `confirm`.
    prediction_set: list[bool | int | str] | None = Field(
        default=None,
        description="Split Conformal Adaptive Prediction Set (APS) at target coverage (1 - alpha).",
    )
    # Normalized Shannon entropy in [0, 1] for active-learning / escalation ranking.
    entropy: float | None = Field(
        default=None,
        description="Normalized Shannon entropy in [0, 1] for active-learning prioritisation.",
    )
    expected: float | None = Field(default=None, description="`score` only: expected rubric level E[Y].")
    spread: float | None = Field(default=None, description="`score` only: ordinal standard deviation sqrt(Var[Y]).")


class DecideResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "model": "gemma3:12b",
                    "answers": {
                        "sentiment": {
                            "value": True,
                            "probabilities": {"true": 0.999937, "false": 0.000063},
                            "confidence": 0.999937,
                            "margin": 0.999873,
                            "coverage": 0.999997,
                            "action": "act",
                            "prediction_set": [True],
                            "entropy": 0.000976,
                            "expected": None,
                            "spread": None,
                        },
                        "topic": {
                            "value": "technical_support",
                            "probabilities": {
                                "billing": 0.0,
                                "shipping": 0.0,
                                "technical_support": 1.0,
                                "other": 0.0,
                            },
                            "confidence": 1.0,
                            "margin": 0.999999,
                            "coverage": 1.0,
                            "action": "act",
                            "prediction_set": ["technical_support"],
                            "entropy": 0.000012,
                            "expected": None,
                            "spread": None,
                        },
                        "urgency": {
                            "value": 3,
                            "probabilities": {"1": 0.0052, "2": 0.019721, "3": 0.975079},
                            "confidence": 0.975079,
                            "margin": 0.955358,
                            "coverage": 0.999995,
                            "action": "act",
                            "prediction_set": [3],
                            "entropy": 0.1184,
                            "expected": 2.9699,
                            "spread": 0.2008,
                        },
                    },
                }
            ]
        }
    )

    model: str = Field(description="Identifier of the backend model that produced the decision.")
    answers: dict[str, Answer] = Field(description="Calibrated triage answers keyed by question ID.")
