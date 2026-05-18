from pydantic import BaseModel
from typing import List


class SingleDNA(BaseModel):
    id: int
    knobs: str


class ParameterSet(BaseModel):
    params: List[SingleDNA]
    invocation_id: int
    generation_num: int


class CompletionMessage(BaseModel):
    complete: bool


class EvaluatedParamResponse(BaseModel):
    id: int
    scores: List[int | float | str]


class ResponseTemplate(BaseModel):
    evaluated_params: List[EvaluatedParamResponse] | CompletionMessage
