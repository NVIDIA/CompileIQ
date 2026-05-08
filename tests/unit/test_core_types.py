"""
Tests for compileiq/core/core_types.py.
"""

import pytest

from compileiq.core.core_types import ResponseTemplate


class TestResponseTemplateLength:
    def test_response_template_has_no_len(self):
        response = ResponseTemplate(evaluated_dna=[])
        with pytest.raises(TypeError, match="has no len"):
            len(response)  # pyright: ignore[reportArgumentType]

    def test_evaluated_dna_list_is_lengthable(self):
        response = ResponseTemplate(evaluated_dna=[])
        assert isinstance(response.evaluated_dna, list)
        assert len(response.evaluated_dna) == 0
