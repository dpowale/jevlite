"""Request/response schemas.

Three question primitives, named after Jev's public ones:
  noul   - yes/no, returns P(yes)
  choice - pick one of N options, returns a distribution
  score  - position on an ordered 1..N rubric, returns a distribution
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator


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
    state: str | dict
    questions: dict[str, Question] = Field(min_length=1, max_length=64)
    # For `choice` questions: average over this many cyclic option orderings
    # to cancel position bias. 1 = off. Capped at len(options).
    permutations: int = Field(default=1, ge=1, le=26)


class Answer(BaseModel):
    value: bool | int | str
    probabilities: dict[str, float]
    confidence: float  # max calibrated probability (concentration, NOT correctness)
    margin: float  # top1 - top2
    coverage: float  # prob. mass the LM put on valid answer tokens (sanity check)
    action: Literal["act", "confirm", "escalate"]
    expected: float | None = None  # score only: expected level


class DecideResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
