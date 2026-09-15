from __future__ import annotations

import pytest

from xla_ciq.search.shape_launch import grid_for_numel, parse_shape_token


def test_parse_shape_token():
    shape = parse_shape_token("f32[1024,2048]")
    assert shape.dtype == "f32"
    assert shape.dims == (1024, 2048)
    assert shape.numel == 1024 * 2048
    assert shape.nbytes == shape.numel * 4


def test_grid_for_numel():
    assert grid_for_numel(1, 256) == 1
    assert grid_for_numel(256, 256) == 1
    assert grid_for_numel(257, 256) == 2


def test_parse_shape_rejects_bad():
    with pytest.raises(ValueError):
        parse_shape_token("f32[]")
