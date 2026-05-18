from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import AliasChoices, BaseModel, Field


class ParamConfigBase(BaseModel, populate_by_name=True):
    """Shared fields for all search space parameter types."""

    knockout_threshold: float | None = None


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


class SearchSpaceFileModel(BaseModel, populate_by_name=True):
    format: str = "compileiq-search-space-v1"
    classes: dict[str, ParamConfig]
    parameter_layout: list[str]
