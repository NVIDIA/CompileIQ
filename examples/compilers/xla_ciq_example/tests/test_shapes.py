from __future__ import annotations

import pytest

from xla_ciq.shapes import ShapeParseError, is_all_shapes, parse_shapes


def test_parse_single_shape():
    assert parse_shapes("f32[1024,2048]") == ["f32[1024,2048]"]


def test_parse_list_with_spaces():
    assert parse_shapes("f32[1024,2048], f32[1024], s8[32,128]") == [
        "f32[1024,2048]",
        "f32[1024]",
        "s8[32,128]",
    ]


def test_is_all_shapes():
    assert is_all_shapes("all")
    assert is_all_shapes("ALL")
    assert not is_all_shapes("f32[1]")


def test_parse_all_direct_errors():
    with pytest.raises(ShapeParseError, match="dump"):
        parse_shapes("all")


def test_reject_all_mixed_with_shapes():
    with pytest.raises(ShapeParseError, match="alone"):
        parse_shapes("all,f32[1024]")


def test_reject_empty():
    with pytest.raises(ShapeParseError):
        parse_shapes("  ")


def test_reject_invalid_token():
    with pytest.raises(ShapeParseError):
        parse_shapes("f32[]")


def test_reject_non_dtype_prefix():
    # HLO text contains brackets like args[0] / data[1] that are not shapes.
    for token in ("args[0]", "data[1]"):
        with pytest.raises(ShapeParseError, match="not an HLO element type"):
            parse_shapes(token)
