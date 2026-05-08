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
        `range(0,9,2)` will produce the search space of `[1,3,5,7,9]`

    Args:
        start:
            Similar to a loop, start of the range.

        end:
            Similar to a loop, end of the range. Must respect the step size to not overflow or
            underflow

        step:
            Similar to a loop, how to slide the range window.

        knockout_prob:
            The `knockout_prob` is a percentage value representing the likelihood of the value
            being dropped and not forwarded to the user objective function.

        seed_low:
            Range genes can also specify some secondary subset through `seed_low` and `seed_high`
            Instead of sampling from the full range, CompileIQ will initialize with values
            inside this smaller range. This can speed your search if your problem has too many
            failures in a specific range. The percentage of DNA strands user would like to be
            initialized this way is determined by the `SearchConfiguration` parameter
            `init_with_true_random_threshold`.

        seed_high:
            Range genes can also specify some secondary subset through `seed_low` and `seed_high`
            Instead of sampling from the full range, CompileIQ will initialize with values
            inside this smaller range. This can speed your search if your problem has too many
            failures in a specific range. The percentage of DNA strands user would like to be
            initialized this way is determined by the `SearchConfiguration` parameter
            `init_with_true_random_threshold`.
    """

    # Workaround: Core Limitations
    if step < 0:
        raise ValueError("Core only accepts positive step sizes for range genes")

    if end <= start:
        raise ValueError("Please provide a proper range value where start < end.")

    knockout_threshold = (1.0 - knockout_prob) if knockout_prob is not None else None

    # Validating seeding option
    if seed_low is not None and seed_high is not None:
        if seed_low < start or seed_high > end or seed_high < seed_low:
            raise ValueError(
                "`seed_low` and `seed_high` is falling outside the correct range."
                " Make sure low > start, high < end and high > low."
            )

    return RangeParamConfig(
        low=start,
        high=end,
        step=step,
        knockout_threshold=knockout_threshold,
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
            The `knockout_prob` is a percentage value representing the likelihood of the value
            being dropped and not forwarded to the user objective function.

    """
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

    knockout_threshold = (1.0 - knockout_prob) if knockout_prob is not None else None

    return ChoiceParamConfig(
        vals=processed,
        knockout_threshold=knockout_threshold,
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
            The `knockout_prob` is a percentage value representing the likelihood of the value
            being dropped and not forwarded to the user objective function.
    """
    if not isinstance(const_value, (str, int, float)):
        raise ValueError("Literal `const_value` must be a str, int, or float.")

    if isinstance(const_value, bool):
        const_value = int(const_value)

    knockout_threshold = (1.0 - knockout_prob) if knockout_prob is not None else None

    return LiteralParamConfig(
        value=const_value,
        knockout_threshold=knockout_threshold,
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
            The `knockout_prob` is a percentage value representing the likelihood of the value
            being dropped and not forwarded to the user objective function.
    """
    #  Core does not have native support for logarithmic sampling,
    # so we emulate it by creating a list that follows a log scale.
    sampling_list = np.geomspace(start=start, stop=end, num=total)
    return choice(sampling_list, knockout_prob)
