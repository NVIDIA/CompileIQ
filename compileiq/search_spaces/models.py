from __future__ import annotations

import math
from collections.abc import Mapping
from decimal import Decimal
from numbers import Real
from typing import Any, Literal, Union

from pydantic import AliasChoices, BaseModel, Field, model_validator


def _probability_to_threshold(knockout_prob: float | None) -> float | None:
    """Convert omission probability without introducing a binary-float artifact."""
    if knockout_prob is None:
        return None
    return float(Decimal("1") - Decimal(str(knockout_prob)))


class ParamConfigBase(BaseModel, populate_by_name=True):
    """Shared fields for all search space parameter types."""

    knockout_prob: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_knockout_probability(cls, value: Any) -> Any:
        """Canonicalize the legacy complementary threshold into omission probability."""

        if not isinstance(value, Mapping):
            return value

        normalized = dict(value)
        probability = cls._validate_knockout_value(
            normalized.get("knockout_prob"),
            name="knockout_prob",
        )
        threshold = cls._validate_knockout_value(
            normalized.get("knockout_threshold"),
            name="knockout_threshold",
        )

        if probability is not None and threshold is not None:
            if not math.isclose(
                probability,
                1.0 - threshold,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "knockout_prob and knockout_threshold must be complementary values"
                )
        elif probability is None and threshold is not None:
            # Legacy thresholds were themselves produced by a floating-point
            # complement. Remove that representational noise so equivalent old
            # and new construction paths have the same checkpoint identity.
            probability = round(1.0 - threshold, 15)

        if "knockout_prob" in normalized or "knockout_threshold" in normalized:
            normalized["knockout_prob"] = probability
        normalized.pop("knockout_threshold", None)
        return normalized

    @staticmethod
    def _validate_knockout_value(value: Any, *, name: str) -> float | None:
        """Validate probability-shaped compatibility inputs without bool coercion."""

        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a number between 0.0 and 1.0")
        parsed = float(value)
        if not 0.0 <= parsed <= 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0")
        return parsed

    @property
    def knockout_threshold(self) -> float | None:
        """Return the legacy complementary knockout threshold for compatibility reads."""

        return _probability_to_threshold(self.knockout_prob)


class RangeParamConfig(ParamConfigBase):
    type: Literal["range"] = "range"
    low: int | float
    high: int | float
    step: int | float = 1
    seed_low: int | float | None = Field(
        default=None,
        validation_alias=AliasChoices("seed-low", "seed_low"),
        serialization_alias="seed-low",
    )
    seed_high: int | float | None = Field(
        default=None,
        validation_alias=AliasChoices("seed-high", "seed_high"),
        serialization_alias="seed-high",
    )


class ChoiceParamConfig(ParamConfigBase):
    type: Literal["enum"] = "enum"
    vals: list[Any]


class LiteralParamConfig(ParamConfigBase):
    type: Literal["literal"] = "literal"
    value: int | float | str


ParamConfig = Union[RangeParamConfig, ChoiceParamConfig, LiteralParamConfig]
