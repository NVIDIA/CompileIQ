import math
from pydantic import (
    BaseModel,
    model_validator,
    TypeAdapter,
    ValidationError,
    StrictInt,
    StrictFloat,
    Field,
)
from pydantic.functional_validators import AfterValidator
from typing import Any, Annotated, Tuple, List, Literal

# DNA representing the baseline measurement
BASELINE_DNA = {}
INVALID_SCORE = "*"


def validate_inf_nan(val: float) -> float | str:
    """
    Communication with Core is done through json which does not handle 'inf' and 'nan' when dumped.
    """
    if math.isnan(val) or math.isinf(val):
        val = INVALID_SCORE

    return val


SingleScore = StrictInt | Annotated[StrictFloat, AfterValidator(validate_inf_nan)] | Literal["*"]

MultiScore = List[SingleScore] | Tuple[SingleScore, ...]


class Score(BaseModel):
    """
    This class represents returned values from the objective function.
    """

    score: MultiScore | SingleScore = Field(
        description="The raw score returned by the objective function."
    )
    norm_score: MultiScore | SingleScore | None = Field(
        None,
        description="The normalized score computed by the Worker based on the baseline score"
        "and the score returned by the objective function."
        "Only needed if `normalization=True`",
    )
    params: str | dict | list[str | dict] = Field(
        description="The parameters sent to the objective function that generated this score"
    )
    metadata: str = Field(
        "", description="Additional metadata that will be forward to the resulting dataframe"
    )
    param_id: int | str = Field(
        description="The unique identifier for the parameters used in the objective function"
    )
    num_objectives: int = Field(description="The number of objectives in the optimization problem")
    is_baseline: bool = Field(False, description="Indicates if this score is a baseline score")

    @property
    def failed(self) -> bool:
        """
        Checks if the score indicates a failure based on the presence of INVALID_SCORE.
        """
        if self.num_objectives == 1:
            return INVALID_SCORE == self.score or (
                self.norm_score is not None and self.norm_score == INVALID_SCORE
            )
        else:
            norm_failed = False
            if self.norm_score is not None:
                norm_failed = any(s == INVALID_SCORE for s in self.norm_score)
            return any(s == INVALID_SCORE for s in self.score) or norm_failed

    @model_validator(mode="after")
    def validate_score_length(self):
        """
        Validates that the length of the score matches the number of objectives.
        """
        # This validation also transforms `SingleScore` into a list of 1 score
        self.score = validate_scores(self.score, self.num_objectives)

        return self


def validate_scores(func_return: Any, num_objectives: int) -> List:
    """
    Check if the objective function return matches the expected type
    and `num_objectives` reported by the user at `SearchConfiguration`.
    """

    # Create type here to set num_objectives inside Tuple typehint
    # and let pydantic validate it himself
    if num_objectives > 1:
        if INVALID_SCORE == func_return:
            # Adjusting for the case where the function returns a single invalid score
            func_return = [INVALID_SCORE for _ in range(num_objectives)]

        multi_score = [SingleScore] * num_objectives
        score_typer = TypeAdapter(Tuple[*multi_score])
    else:
        score_typer = TypeAdapter(SingleScore)

    try:
        # Validate the score and update its values
        fixed_func_return = score_typer.validate_python(func_return)
    except ValidationError:
        raise ValueError(
            "One or more return types of your objective function does not match expected types: "
            "int | float | '*'"
        )

    return fixed_func_return


def _manual_validation(func_return: Any, num_objectives: int) -> List:
    if not isinstance(func_return, list) and not isinstance(func_return, tuple):
        raise ValueError(
            f"Number of returns (1) is not matching num_objectives ({(num_objectives)})."
        )

    elif len(func_return) != num_objectives:
        raise ValueError(
            f"Number of returns ({len(func_return)}) is not matching num_objectives "
            f"({(num_objectives)})."
        )

    # Validating each of the returns individually
    score_typer = TypeAdapter(SingleScore)
    fixed_return = [score_typer.validate_python(val) for val in func_return]

    return fixed_return
