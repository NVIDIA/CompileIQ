"""
Tests for bundled-core IPC parameter parsing edge cases.
Covers: json5 fallback, raw-string fallback, non-dict JSON, and nested restoration.
"""

import base64
import json
import warnings
import pytest

from compileiq.bundled_backend import BundledCoreBackend
from compileiq.core.core_types import ParameterSet, SingleCandidate
from compileiq.types import InternalSearchConfiguration


def _make_backend(tmp_path, *, file_backed: bool = False):
    """Construct the adapter without starting the bundled process or IPC socket."""

    return BundledCoreBackend(
        search_space=tmp_path / "space.bin" if file_backed else {"lr": {}},
        search_config=InternalSearchConfiguration(generations=2),
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


def test_valid_json_dict_parsed(tmp_path):
    backend = _make_backend(tmp_path)
    # Core sends base64-encoded keys; restore_nested_search_space decodes them
    knobs = json.dumps({_b64("lr"): 0.1, _b64("wd"): 0.01})
    result = backend._load_params(_param_set(knobs))
    assert result == [{"lr": 0.1, "wd": 0.01}]


@pytest.mark.parametrize(
    "knobs",
    [
        '"just a string"',
        "42",
        "[1, 2, 3]",
    ],
)
def test_dictionary_search_space_json_non_object_rejected(tmp_path, knobs):
    """Dictionary-defined search spaces should always decode to a JSON object."""
    backend = _make_backend(tmp_path)
    with pytest.raises(RuntimeError, match="non-object parameter payload"):
        backend._load_params(_param_set(knobs))


def test_file_backed_json_string_returned_as_string(tmp_path):
    """File-backed spaces may return scalar JSON strings."""
    backend = _make_backend(tmp_path, file_backed=True)

    result = backend._load_params(_param_set('"just a string"'))
    assert result == ["just a string"]


# ---------------------------------------------------------------------------
# json5 fallback
# ---------------------------------------------------------------------------


def test_json5_fallback_for_trailing_comma(tmp_path):
    """json5 handles trailing commas which standard json rejects."""
    backend = _make_backend(tmp_path)
    lr_key, wd_key = _b64("lr"), _b64("wd")
    # Trailing comma is invalid JSON but valid json5
    knobs = f'{{"{lr_key}": 0.1, "{wd_key}": 0.01,}}'
    result = backend._load_params(_param_set(knobs))
    assert len(result) == 1
    assert isinstance(result[0], dict)
    assert result[0]["lr"] == 0.1


def test_json5_fallback_for_single_quotes(tmp_path):
    """json5 handles single-quoted strings."""
    backend = _make_backend(tmp_path)
    lr_key = _b64("lr")
    # Single quotes are invalid JSON but valid json5
    knobs = f"{{'{lr_key}': 0.1}}"
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = backend._load_params(_param_set(knobs))
    assert len(result) == 1
    assert isinstance(result[0], dict)
    assert result[0]["lr"] == 0.1


# ---------------------------------------------------------------------------
# Raw string fallback
# ---------------------------------------------------------------------------


def test_dictionary_search_space_unparseable_string_rejected(tmp_path):
    """Dictionary-defined search spaces should receive parseable JSON from core."""
    backend = _make_backend(tmp_path)
    raw = "this is not json at all {{{{"
    with pytest.raises(RuntimeError, match="unparseable parameter payload"):
        backend._load_params(_param_set(raw))


def test_file_backed_unparseable_string_returned_raw(tmp_path):
    """File-backed spaces may return opaque strings."""
    backend = _make_backend(tmp_path, file_backed=True)
    raw = "this is not json at all {{{{"

    result = backend._load_params(_param_set(raw))
    assert result == [raw]


# ---------------------------------------------------------------------------
# Non-dict JSON with json5 fallback
# ---------------------------------------------------------------------------


def test_json5_non_dict_result_rejected_for_dictionary_search_space(tmp_path):
    """If json5 parses dictionary params to a non-dict, reject the core payload."""
    backend = _make_backend(tmp_path)
    # json5 can parse bare identifiers like Infinity
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="non-object parameter payload"):
            backend._load_params(_param_set("Infinity"))
