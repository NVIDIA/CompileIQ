from typing import Any, Optional, Annotated, Sequence
from annotated_types import Gt
import numpy as np

from compileiq.search_spaces.models import (
    RangeParamConfig,
    ChoiceParamConfig,
    LiteralParamConfig,
)


def range(
    start: int | float,
    end: int | float,
    step: Annotated[int | float, Gt(0)] = 1,
    knockout_prob: Optional[float] = None,
    seed_low: Optional[int | float] = None,
    seed_high: Optional[int | float] = None,
) -> RangeParamConfig:
    """
    Search space that follows a range-like format.

    Example:
        `range(0,8,2)` will produce the inclusive search space `[0,2,4,6,8]`.

    Args:
        start:
            Similar to a loop, start of the range.

        end:
            Inclusive upper bound. Values advance from ``start`` by ``step`` without exceeding
            this bound; the bound itself does not need to fall on that stepped grid.

        step:
            Similar to a loop, how to slide the range window.

        knockout_prob:
            Probability from 0.0 to 1.0 that the parameter is omitted from an initially sampled
            candidate. Search engines may then evolve the parameter's on/off state, so this
            is an initialization prior rather than a guaranteed ratio in every candidate batch.

        seed_low:
            Range parameters can also specify a secondary subset through `seed_low` and `seed_high`.
            Instead of sampling from the full range, CompileIQ will initialize with values
            inside this smaller range. This can speed your search if your problem has too many
            failures in a specific range. The percentage of candidates initialized this way
            is determined by the `SearchConfiguration` parameter
            `init_with_true_random_threshold`.

        seed_high:
            Range parameters can also specify a secondary subset through `seed_low` and `seed_high`.
            Instead of sampling from the full range, CompileIQ will initialize with values
            inside this smaller range. This can speed your search if your problem has too many
            failures in a specific range. The percentage of candidates initialized this way
            is determined by the `SearchConfiguration` parameter
            `init_with_true_random_threshold`.
    """

    if step <= 0:
        raise ValueError("Range parameters require a positive step size")

    if end <= start:
        raise ValueError("Please provide a proper range value where start < end.")

    # A single seed bound narrows only that side. Both current engines implement these as
    # initialization bounds, so the bounds do not need to be emitted values on the stepped grid.
    if seed_low is not None and not start <= seed_low <= end:
        raise ValueError("`seed_low` falls outside the configured range.")
    if seed_high is not None and not start <= seed_high <= end:
        raise ValueError("`seed_high` falls outside the configured range.")
    if seed_low is not None and seed_high is not None and seed_high < seed_low:
        raise ValueError("`seed_high` must be greater than or equal to `seed_low`.")
    return RangeParamConfig(
        low=start,
        high=end,
        step=step,
        knockout_prob=knockout_prob,
        seed_low=seed_low,
        seed_high=seed_high,
    )


def choice(
    choice_list: Sequence[int | float | bool | str] | np.ndarray,
    knockout_prob: Optional[float] = None,
) -> ChoiceParamConfig:
    """
    Search space that will sample from a list.
    Legacy version called this Enum

    Example:
        `choice([1,2,3])` will pick only one out of `1`, `2` or `3` at a time

    Args:
        choice_list:
            A list of values to sample from.

        knockout_prob:
            Probability from 0.0 to 1.0 that the parameter is omitted from an initially sampled
            candidate. Search engines may then evolve the parameter's on/off state, so this
            is an initialization prior rather than a guaranteed ratio in every candidate batch.

    """
    if len(choice_list) == 0:
        raise ValueError("choice_list must contain at least one value.")

    processed: list[Any] = []
    for val in choice_list:
        if isinstance(val, bool):
            processed.append(int(val))
        elif isinstance(val, np.floating):
            processed.append(float(val))
        elif isinstance(val, np.integer):
            processed.append(int(val))
        else:
            processed.append(val)

    return ChoiceParamConfig(
        vals=processed,
        knockout_prob=knockout_prob,
    )


def literal(
    const_value: str | int | float,
    knockout_prob: Optional[float] = None,
) -> LiteralParamConfig:
    """
    Represents a constant value with an optional knockout threshold.
    Helpful if you have constant parameters that need to take
    advantage of knockout, like on and off flags from compilers.

    Example:
        `{'x': literal(5)}` sets `config['x']` to `5`

    Args:
        const_value:
            A constant value that will be forwarded to your evaluation.

        knockout_prob:
            Probability from 0.0 to 1.0 that the parameter is omitted from an initially sampled
            candidate. Search engines may then evolve the parameter's on/off state, so this
            is an initialization prior rather than a guaranteed ratio in every candidate batch.
    """
    if not isinstance(const_value, (str, int, float)):
        raise ValueError("Literal `const_value` must be a str, int, or float.")

    if isinstance(const_value, bool):
        const_value = int(const_value)

    return LiteralParamConfig(
        value=const_value,
        knockout_prob=knockout_prob,
    )


def log_sampling(
    start: Annotated[float, Gt(0)],
    end: Annotated[float, Gt(0)],
    total: int = 10,
    knockout_prob: Optional[float] = None,
) -> ChoiceParamConfig:
    """
    Search space similar to `range()` but will follow a logarithmic distribution

    Args:
        start:
            Similar to a loop, start of the range.

        end:
            Similar to a loop, end of the range. Must respect the step size to not overflow or
            underflow

        total:
            The larger the total number the bigger the sampler space CompileIQ will use.

        knockout_prob:
            Probability from 0.0 to 1.0 that the parameter is omitted from an initially sampled
            candidate. Search engines may then evolve the parameter's on/off state, so this
            is an initialization prior rather than a guaranteed ratio in every candidate batch.
    """
    # Search spaces have no native logarithmic range, so we emulate one with a
    # choice over values that follow a log scale.
    sampling_list = np.geomspace(start=start, stop=end, num=total)
    return choice(sampling_list, knockout_prob)
