"""
Tests for parameter parsing edge cases in ciq.py _load_params / parse_param_payload.
Covers: json5 fallback, raw-string fallback, non-dict JSON, and nested restoration.
"""

import base64
import json
import warnings
from unittest.mock import MagicMock

from compileiq.ciq import Search
from compileiq.search_spaces import base as ss
from compileiq.types import SearchConfiguration
from compileiq.core.core_types import ParameterSet, SingleCandidate


def _make_search(mocker, tmp_path):
    mocker.patch("compileiq.core.core_comms.socket.socket", return_value=MagicMock())
    mocker.patch("compileiq.worker.multiprocessing.Manager", return_value=MagicMock())
    return Search(
        objective_function=lambda p: 1.0,
        search_space={"lr": ss.range(start=0, end=10)},
        search_config=SearchConfiguration(generations=2),
        cache_folder=tmp_path,
    )


def _b64(s: str) -> str:
    """Base64-encode a string, matching core's key encoding."""
    return base64.b64encode(s.encode()).decode()


def _param_set(*knobs_list):
    """Build a ParameterSet with the given knobs strings."""
    return ParameterSet(
        params=[SingleCandidate(id=i, knobs=k) for i, k in enumerate(knobs_list)],
        invocation_id=0,
        generation_num=0,
    )


# ---------------------------------------------------------------------------
# Standard JSON parsing
# ---------------------------------------------------------------------------


def test_valid_json_dict_parsed(mocker, tmp_path):
    search = _make_search(mocker, tmp_path)
    # Core sends base64-encoded keys; restore_nested_search_space decodes them
    knobs = json.dumps({_b64("lr"): 0.1, _b64("wd"): 0.01})
    result = search._load_params(_param_set(knobs))
    assert result == [{"lr": 0.1, "wd": 0.01}]


def test_valid_json_returns_numeric_string_as_string(mocker, tmp_path):
    """A bare JSON string (not a dict) should be returned as-is, not passed
    to restore_nested_search_space."""
    search = _make_search(mocker, tmp_path)
    result = search._load_params(_param_set('"just a string"'))
    assert result == ["just a string"]


def test_valid_json_number_returned_directly(mocker, tmp_path):
    """A bare JSON number should be returned, not crash in restore_nested."""
    search = _make_search(mocker, tmp_path)
    result = search._load_params(_param_set("42"))
    assert result == [42]


def test_valid_json_list_returned_directly(mocker, tmp_path):
    """A JSON array should be returned, not crash in restore_nested."""
    search = _make_search(mocker, tmp_path)
    result = search._load_params(_param_set("[1, 2, 3]"))
    assert result == [[1, 2, 3]]


# ---------------------------------------------------------------------------
# json5 fallback
# ---------------------------------------------------------------------------


def test_json5_fallback_for_trailing_comma(mocker, tmp_path):
    """json5 handles trailing commas which standard json rejects."""
    search = _make_search(mocker, tmp_path)
    lr_key, wd_key = _b64("lr"), _b64("wd")
    # Trailing comma is invalid JSON but valid json5
    knobs = f'{{"{lr_key}": 0.1, "{wd_key}": 0.01,}}'
    result = search._load_params(_param_set(knobs))
    assert len(result) == 1
    assert isinstance(result[0], dict)
    assert result[0]["lr"] == 0.1


def test_json5_fallback_for_single_quotes(mocker, tmp_path):
    """json5 handles single-quoted strings."""
    search = _make_search(mocker, tmp_path)
    lr_key = _b64("lr")
    # Single quotes are invalid JSON but valid json5
    knobs = f"{{'{lr_key}': 0.1}}"
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = search._load_params(_param_set(knobs))
    assert len(result) == 1
    assert isinstance(result[0], dict)
    assert result[0]["lr"] == 0.1


# ---------------------------------------------------------------------------
# Raw string fallback
# ---------------------------------------------------------------------------


def test_unparseable_string_returned_raw(mocker, tmp_path):
    """If neither json nor json5 can parse the string, return it raw."""
    search = _make_search(mocker, tmp_path)
    raw = "this is not json at all {{{{"
    result = search._load_params(_param_set(raw))
    assert result == [raw]


# ---------------------------------------------------------------------------
# Non-dict JSON with json5 fallback
# ---------------------------------------------------------------------------


def test_json5_non_dict_result_not_passed_to_restore(mocker, tmp_path):
    """If json5 parses to a non-dict, restore_nested_search_space should be skipped."""
    search = _make_search(mocker, tmp_path)
    # json5 can parse bare identifiers like Infinity
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = search._load_params(_param_set("Infinity"))
    assert len(result) == 1
    # Should not crash — the result is a float, not a dict
