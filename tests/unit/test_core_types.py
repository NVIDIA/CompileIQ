"""
Tests for compileiq/core/core_types.py.
"""

import pytest

from compileiq.core.core_types import EvaluatedParamResponse, ResponseTemplate


class TestResponseTemplateLength:
    def test_response_template_has_no_len(self):
        response = ResponseTemplate(evaluated_params=[])
        with pytest.raises(TypeError, match="has no len"):
            len(response)  # pyright: ignore[reportArgumentType]

    def test_evaluated_params_list_is_lengthable(self):
        response = ResponseTemplate(evaluated_params=[])
        assert isinstance(response.evaluated_params, list)
        assert len(response.evaluated_params) == 0

    def test_response_template_serializes_evaluated_params(self):
        response = ResponseTemplate(
            evaluated_params=[EvaluatedParamResponse(id=1, scores=[2.0])]
        )

        payload = response.model_dump()

        assert "evaluated_params" in payload
        assert "evaluated_dna" not in payload
